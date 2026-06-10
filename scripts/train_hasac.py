"""
Clean HASAC — Heterogeneous-Agent SAC (Liu et al., ICLR 2024 / HARL)

完全按照論文規定：
  - N_BS 個獨立 actor（separate policies，NOT parameter-shared）
  - Sequential Soft Policy Decomposition：每步隨機排列，一次只更新 π^i
  - 更新 π^i 時：其他 agent 的 policy 不動（用 replay buffer 的 a_ 固定）
  - Centralized critic Q(share_obs, joint_action, onehot_i)
  - Auto-tuned per-agent temperature α
  - 無 BC warm-start、無 oracle_z、無 z encoder

可選工程修正：
  - --mu_bound 5: mu=mu_bound*tanh(mu_raw)，防止 tanh 飽和零功率崩潰
    （HASAC 論文無此需求；本 5G+logpf 環境的數值必要條件）
"""
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_chasac as E

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


# ── networks ──────────────────────────────────────────────────────────────────

def mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]),
                   act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Per-BS actor: permutation-equivariant SetActor over that BS's UEs.
    Takes full [N_UE, ue_feat] obs + membership mask; outputs actions for all UEs
    (caller uses mask to extract only this BS's UEs)."""
    def __init__(self, ue_feat=3, hidden=256):
        super().__init__()
        self.enc      = mlp([ue_feat, hidden, hidden])
        self.head     = mlp([hidden * 2, hidden, hidden])
        self.mu       = nn.Linear(hidden, 1)
        self.lsg      = nn.Linear(hidden, 1)
        self.mu_bound = 0.0   # >0: mu=mu_bound*tanh(mu_raw), prevents tanh saturation collapse

    def forward(self, o, mask):
        # o: [B, N_UE, F]; mask: [B, N_BS, N_UE]
        emb  = self.enc(o)                                      # [B, N_UE, H]
        cnt  = mask.sum(-1, keepdim=True).clamp_min(1.)         # [B, N_BS, 1]
        ctx  = torch.einsum("biu,buh->bih", mask, emb) / cnt    # [B, N_BS, H]
        ue_ctx = torch.einsum("biu,bih->buh", mask, ctx)        # [B, N_UE, H]
        feat = torch.cat([emb, ue_ctx], dim=-1)                 # [B, N_UE, 2H]
        h    = self.head(feat)
        mu   = self.mu(h).squeeze(-1)                           # [B, N_UE]
        if self.mu_bound > 0:
            mu = self.mu_bound * torch.tanh(mu)                 # bound mean -> no -inf saturation
        lsg  = self.lsg(h).squeeze(-1).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, lsg

    def sample(self, o, mask):
        mu, lsg = self.forward(o, mask)
        std = lsg.exp()
        x   = torch.distributions.Normal(mu, std).rsample()
        a   = torch.tanh(x)
        lp  = (torch.distributions.Normal(mu, std).log_prob(x)
               - torch.log(1 - a.pow(2) + 1e-6))               # [B, N_UE]
        logp_bs = torch.einsum("biu,bu->bi", mask, lp)         # [B, N_BS]
        return a, logp_bs

    @torch.no_grad()
    def act(self, o, mask, deterministic=True):
        mu, lsg = self.forward(o, mask)
        if deterministic:
            return torch.tanh(mu)
        return torch.tanh(torch.distributions.Normal(mu, lsg.exp()).sample())


class Critic(nn.Module):
    """Agent-conditioned twin-Q: Q(share_obs, joint_action, onehot_i)."""
    def __init__(self, share_dim, act_dim, n_bs, hidden=256):
        super().__init__()
        d = share_dim + act_dim + n_bs
        self.q1 = mlp([d, hidden, hidden, 1])
        self.q2 = mlp([d, hidden, hidden, 1])

    def forward(self, share, a, onehot):
        x = torch.cat([share, a, onehot], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


# ── replay ────────────────────────────────────────────────────────────────────

class Replay:
    def __init__(self, cap, n_ue, n_bs, ue_feat, share_dim):
        self.cap, self.ptr, self.size = cap, 0, 0
        self.o    = np.zeros((cap, n_ue, ue_feat), np.float32)
        self.mask = np.zeros((cap, n_bs, n_ue), np.float32)
        self.sh   = np.zeros((cap, share_dim), np.float32)
        self.a    = np.zeros((cap, n_ue), np.float32)
        self.r    = np.zeros((cap, n_bs), np.float32)
        self.no   = np.zeros_like(self.o)
        self.nmask= np.zeros_like(self.mask)
        self.nsh  = np.zeros_like(self.sh)

    def add(self, o, mask, sh, a, r, no, nmask, nsh):
        i = self.ptr
        self.o[i], self.mask[i], self.sh[i] = o, mask, sh
        self.a[i], self.r[i] = a, r
        self.no[i], self.nmask[i], self.nsh[i] = no, nmask, nsh
        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, bs):
        idx = np.random.randint(0, self.size, size=bs)
        t   = lambda x: torch.as_tensor(x[idx], device=DEVICE)
        return t(self.o), t(self.mask), t(self.sh), t(self.a), t(self.r), \
               t(self.no), t(self.nmask), t(self.nsh)


# ── helpers ───────────────────────────────────────────────────────────────────

def build_obs(env, n_bs, n_ue, ue_feat=3):
    cfg = env.cfg
    w   = env._weights()
    rate, _, _ = E.rates_from_power(env.p, env.g, env.serv, n_bs)
    o   = np.stack([rate, w, env.p], axis=1).astype(np.float32)  # [N_UE, 3]
    mask = np.zeros((n_bs, n_ue), np.float32)
    mask[env.serv, np.arange(n_ue)] = 1.0
    share = E.obs_share(env.p, env.g, env.serv, env.bs).astype(np.float32)
    return o, mask, share


def action_to_powerlist(a, serv, n_bs, pmax):
    frac = (a + 1.0) / 2.0
    return [frac[serv == i] * pmax for i in range(n_bs)]


def onehots(n_bs):
    return torch.eye(n_bs, device=DEVICE)


# ── eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_policy(actors, cfg, n_eval=20, T=10, seed=2024):
    pmax = E.dbm_to_w(cfg.Pmax_dBm)
    n_bs, n_ue = cfg.N_BS, cfg.N_UE
    rng  = np.random.default_rng(seed)
    Us   = []
    for _ in range(n_eval):
        env  = E.Env(cfg, reward_mode="difference", seed=int(rng.integers(1 << 30)))
        env.reset()
        rate_sum = np.zeros(n_ue)
        for _ in range(T):
            o, mask, sh = build_obs(env, n_bs, n_ue)
            ot  = torch.as_tensor(o[None],    device=DEVICE)
            mt  = torch.as_tensor(mask[None], device=DEVICE)
            # combine actions from all separate actors
            a = np.zeros(n_ue, np.float32)
            for i, actor_i in enumerate(actors):
                ai = actor_i.act(ot, mt, deterministic=True)[0].cpu().numpy()
                a += mask[i] * ai          # only BS i's UEs
            pl = action_to_powerlist(a, env.serv, n_bs, pmax)
            _, _, _, info = env.step(pl)
            rate_sum += info["rate"]
        Us.append(np.log(rate_sum / T + 1e-6).sum())
    return float(np.mean(Us)), float(np.std(Us))


@torch.no_grad()
def eval_baseline(fn, cfg, n_eval=20, T=10, seed=2024):
    pmax = E.dbm_to_w(cfg.Pmax_dBm)
    rng  = np.random.default_rng(seed)
    Us   = []
    for _ in range(n_eval):
        env  = E.Env(cfg, reward_mode="difference", seed=int(rng.integers(1 << 30)))
        env.reset()
        rate_sum = np.zeros(cfg.N_UE)
        for _ in range(T):
            w  = env._weights()
            p  = fn(env.g, env.serv, w, cfg.N_BS)
            pl = [p[env.serv == i] for i in range(cfg.N_BS)]
            _, _, _, info = env.step(pl)
            rate_sum += info["rate"]
        Us.append(np.log(rate_sum / T + 1e-6).sum())
    return float(np.mean(Us)), float(np.std(Us))


# ── train ─────────────────────────────────────────────────────────────────────

def train(args):
    cfg   = E.Cfg()
    pmax  = E.dbm_to_w(cfg.Pmax_dBm)
    n_bs, n_ue = cfg.N_BS, cfg.N_UE
    ue_feat  = 3
    n_bs_pairs = n_bs * (n_bs - 1) // 2
    share_dim  = n_bs * n_ue + n_ue + n_ue + n_bs_pairs   # 63

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ── N_BS separate actors (HASAC: per-agent independent policy) ──
    actors   = [Actor(ue_feat, args.hidden).to(DEVICE) for _ in range(n_bs)]
    for actor_i in actors:
        actor_i.mu_bound = args.mu_bound
    opts_a   = [torch.optim.Adam(actors[i].parameters(), lr=args.lr) for i in range(n_bs)]

    critic   = Critic(share_dim, n_ue, n_bs, args.hidden).to(DEVICE)
    critic_t = Critic(share_dim, n_ue, n_bs, args.hidden).to(DEVICE)
    critic_t.load_state_dict(critic.state_dict())
    opt_c    = torch.optim.Adam(critic.parameters(), lr=args.lr)

    log_alpha  = torch.tensor(np.log(args.alpha_init), requires_grad=True, device=DEVICE)
    opt_alpha  = torch.optim.Adam([log_alpha], lr=args.lr)
    target_H   = -float(n_ue) / n_bs   # per-BS target entropy

    OH  = onehots(n_bs)
    rb  = Replay(args.replay, n_ue, n_bs, ue_feat, share_dim)
    env = E.Env(cfg, reward_mode=args.reward, seed=args.seed)
    env.reset()

    logf = open(args.log, "w")
    def logp(*a):
        s = " ".join(str(x) for x in a); print(s); logf.write(s+"\n"); logf.flush()

    logp(f"# HASAC (separate policies) | reward={args.reward} steps={args.steps} "
         f"hidden={args.hidden} alpha_init={args.alpha_init} mu_bound={args.mu_bound} device={DEVICE}")

    best_U, best_state = -1e9, None
    t0 = time.time()

    for step in range(1, args.steps + 1):
        o, mask, sh = build_obs(env, n_bs, n_ue)
        if step < args.warmup:
            a = np.random.uniform(-1, 1, n_ue).astype(np.float32)
        else:
            ot = torch.as_tensor(o[None],    device=DEVICE)
            mt = torch.as_tensor(mask[None], device=DEVICE)
            a  = np.zeros(n_ue, np.float32)
            with torch.no_grad():
                for i, actor_i in enumerate(actors):
                    ai = actor_i.act(ot, mt, deterministic=False)[0].cpu().numpy()
                    a += mask[i] * ai

        pl = action_to_powerlist(a, env.serv, n_bs, pmax)
        _, r, _, _ = env.step(pl)
        no, nmask, nsh = build_obs(env, n_bs, n_ue)
        rb.add(o, mask, sh, a.astype(np.float32), r.astype(np.float32),
               no, nmask, nsh)

        if step % args.ep_len == 0:
            env.reset()

        # ── updates ──
        if rb.size >= args.batch and step >= args.warmup:
            o_, m_, s_, a_, r_, no_, nm_, ns_ = rb.sample(args.batch)
            alpha = log_alpha.exp().detach()
            B = o_.shape[0]

            # ── critic ──
            with torch.no_grad():
                # next joint action: each separate actor contributes its UEs
                na = torch.zeros(B, n_ue, device=DEVICE)
                nlogp_bs = torch.zeros(B, n_bs, device=DEVICE)
                for i, actor_i in enumerate(actors):
                    na_i, nlp_i = actor_i.sample(no_, nm_)
                    na       += nm_[:, i, :] * na_i
                    nlogp_bs[:, i] = nlp_i[:, i]
                yq = []
                for i in range(n_bs):
                    oh   = OH[i][None].expand(B, -1)
                    q1, q2 = critic_t(ns_, na, oh)
                    yq.append(r_[:, i] + args.gamma * (torch.min(q1,q2) - alpha * nlogp_bs[:,i]))
                y = torch.stack(yq, dim=1)             # [B, N_BS]

            loss_c = 0.0
            for i in range(n_bs):
                oh   = OH[i][None].expand(B, -1)
                q1, q2 = critic(s_, a_, oh)
                loss_c = loss_c + F.mse_loss(q1, y[:,i]) + F.mse_loss(q2, y[:,i])
            opt_c.zero_grad(); loss_c.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
            opt_c.step()

            # ── actors: Sequential Soft Policy Decomposition ──
            # For agent i: replace BS-i's portion of joint action with actor_i's live output.
            # Other BSes' actions come from replay buffer (fixed, no gradient).
            logp_all = [None] * n_bs
            for i in torch.randperm(n_bs).tolist():
                pa_i, plogp_i = actors[i].sample(o_, m_)   # [B,N_UE], [B,N_BS]
                # joint action: actor_i's UEs (live) + replay actions for other UEs
                bs_mask_i = m_[:, i, :]                     # [B, N_UE]
                pa_full   = a_.detach() * (1 - bs_mask_i) + pa_i * bs_mask_i
                oh = OH[i][None].expand(B, -1)
                q1, q2 = critic(s_, pa_full, oh)
                loss_i = (alpha * plogp_i[:, i] - torch.min(q1, q2)).mean()
                opts_a[i].zero_grad(); loss_i.backward()
                nn.utils.clip_grad_norm_(actors[i].parameters(), 10.0)
                opts_a[i].step()
                logp_all[i] = plogp_i[:, i].detach()

            # ── alpha ──
            avg_logp  = torch.stack(logp_all, dim=1).mean(dim=1)
            loss_alp  = -(log_alpha * (avg_logp + target_H)).mean()
            opt_alpha.zero_grad(); loss_alp.backward(); opt_alpha.step()

            # ── polyak ──
            with torch.no_grad():
                for p, pt in zip(critic.parameters(), critic_t.parameters()):
                    pt.mul_(1 - args.tau).add_(args.tau * p)

        # ── eval / log ──
        if step % args.eval_every == 0:
            U, Us = eval_policy(actors, cfg, n_eval=args.n_eval, T=args.ep_len)
            tag   = ""
            if U > best_U:
                best_U    = U
                best_state = {f"actor_{i}": {k: v.cpu().clone()
                               for k,v in actors[i].state_dict().items()}
                              for i in range(n_bs)}
                best_state["critic"] = {k: v.cpu().clone()
                                        for k,v in critic.state_dict().items()}
                tag = " *"
                torch.save(best_state, f"results/{args.tag}_best.pt")
            a_now = float(log_alpha.exp().detach())
            logp(f"step {step:>7} | PF-U {U:8.3f} ± {Us:5.3f} | alpha {a_now:6.4f} | "
                 f"best {best_U:8.3f} | {time.time()-t0:5.0f}s{tag}")

    # ── FINAL ──
    if best_state:
        for i in range(n_bs):
            actors[i].load_state_dict(best_state[f"actor_{i}"])

    logp("\n=== FINAL (best ckpt) ===")
    U, Us = eval_policy(actors, cfg, n_eval=args.n_eval_final, T=args.ep_len)
    logp(f"{'policy':<22}{U:8.3f} ± {Us:.3f}")
    floorU, _ = eval_baseline(lambda g,s,w,N: E.bl_equal_power(g,s,w,N),
                              cfg, n_eval=args.n_eval_final, T=args.ep_len)
    ceilU,  _ = eval_baseline(lambda g,s,w,N: E.pf_wsr_ceiling(g,s,w,N,cfg.ceiling_grid),
                              cfg, n_eval=args.n_eval_final, T=args.ep_len)
    logp(f"{'equal_power (floor)':<22}{floorU:8.3f}")
    logp(f"{'PF-WSR (ceiling)':<22}{ceilU:8.3f}")
    np.save(args.out, dict(policy=U, floor=floorU, ceiling=ceilU, best_U=best_U),
            allow_pickle=True)
    logf.close()
    return best_U


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward",     default="difference", choices=["difference","team","logpf"])
    ap.add_argument("--steps",      type=int,   default=200000)
    ap.add_argument("--warmup",     type=int,   default=5000)
    ap.add_argument("--batch",      type=int,   default=256)
    ap.add_argument("--replay",     type=int,   default=1000000)
    ap.add_argument("--gamma",      type=float, default=0.99)
    ap.add_argument("--tau",        type=float, default=0.005)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--hidden",     type=int,   default=256)
    ap.add_argument("--ep_len",     type=int,   default=10)
    ap.add_argument("--eval_every", type=int,   default=5000)
    ap.add_argument("--n_eval",     type=int,   default=20)
    ap.add_argument("--n_eval_final",type=int,  default=50)
    ap.add_argument("--alpha_init", type=float, default=0.2)
    ap.add_argument("--mu_bound",   type=float, default=0.0,
                    help=">0: mu=mu_bound*tanh(mu_raw); prevents tanh-saturation zero-power collapse")
    ap.add_argument("--seed",       type=int,   default=0)
    ap.add_argument("--tag",        default="hasac_clean")
    args = ap.parse_args()
    args.log = f"results/{args.tag}_log.txt"
    args.out = f"results/{args.tag}_result.npy"
    train(args)
