"""
Centralized full-CSI SAC — decisive structural test for C-HASAC.

A single centralized actor sees the full share_obs (channel matrix g + power + serv)
and outputs all per-UE powers. This is the upper bound on what coordination can buy:
if even a full-CSI centralized SAC cannot beat the trivial fixed-power baseline (~-4.17),
the env is only solvable by the WF oracle; if it breaks through toward the PF-WSR ceiling
(+23.5), then the bottleneck is decentralization (local obs + broadcast z), not the env.

Eval metric = canonical PF utility U = Σ_u log(R̄_u) on held-out scenarios (same as train_chasac).
"""
import os, sys, time, argparse
sys.path.insert(0, "/home/hyc1014/DL/FinalProject")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import env_chasac as E

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)


class CentralActor(nn.Module):
    def __init__(self, share_dim, n_ue, hidden=256, mu_bound=5.0):
        super().__init__()
        self.body = mlp([share_dim, hidden, hidden])
        self.mu = nn.Linear(hidden, n_ue)
        self.log_std = nn.Linear(hidden, n_ue)
        self.mu_bound = mu_bound

    def forward(self, s):
        h = self.body(s)
        mu = self.mu(h)
        if self.mu_bound > 0:
            mu = self.mu_bound * torch.tanh(mu)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, s):
        mu, log_std = self.forward(s)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        a = torch.tanh(x)
        logp = (dist.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a, logp

    @torch.no_grad()
    def act(self, s, deterministic=True):
        mu, log_std = self.forward(s)
        if deterministic:
            return torch.tanh(mu)
        return torch.tanh(torch.distributions.Normal(mu, log_std.exp()).sample())


class Critic(nn.Module):
    def __init__(self, share_dim, n_ue, hidden=256):
        super().__init__()
        d = share_dim + n_ue
        self.q1 = mlp([d, hidden, hidden, 1])
        self.q2 = mlp([d, hidden, hidden, 1])

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class Replay:
    def __init__(self, cap, share_dim, n_ue):
        self.s = np.zeros((cap, share_dim), np.float32)
        self.a = np.zeros((cap, n_ue), np.float32)
        self.r = np.zeros(cap, np.float32)
        self.ns = np.zeros((cap, share_dim), np.float32)
        self.cap, self.idx, self.size = cap, 0, 0

    def add(self, s, a, r, ns):
        i = self.idx
        self.s[i], self.a[i], self.r[i], self.ns[i] = s, a, r, ns
        self.idx = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        t = lambda x: torch.as_tensor(x, device=DEVICE)
        return t(self.s[idx]), t(self.a[idx]), t(self.r[idx]), t(self.ns[idx])


def share_of(env):
    return E.obs_share(env.p, env.g, env.serv).astype(np.float32)


def act_to_power(a, env, pmax):
    """a in (-1,1)^N_UE -> per-BS power list with Σ ≤ PMAX (same projection as env)."""
    frac = (a + 1.0) / 2.0
    desired = frac * pmax
    return [desired[env.serv == i] for i in range(env.cfg.N_BS)]


@torch.no_grad()
def eval_policy(actor, cfg, n_eval=20, T=10, seed=2024):
    pmax = E.dbm_to_w(cfg.Pmax_dBm)
    rng = np.random.default_rng(seed)
    Us, fr = [], []
    for _ in range(n_eval):
        env = E.Env(cfg, reward_mode="logpf", seed=int(rng.integers(1 << 30)))
        env.reset()
        rate_sum = np.zeros(cfg.N_UE)
        for _t in range(T):
            s = torch.as_tensor(share_of(env)[None], device=DEVICE)
            a = actor.act(s, deterministic=True)[0].cpu().numpy()
            fr.append(float(((a + 1) / 2).mean()))
            _, _, _, info = env.step(act_to_power(a, env, pmax))
            rate_sum += info["rate"]
        Us.append(np.log(rate_sum / T + 1e-6).sum())
    return float(np.mean(Us)), float(np.std(Us)), float(np.mean(fr))


def train(args):
    cfg = E.Cfg()
    pmax = E.dbm_to_w(cfg.Pmax_dBm)
    n_ue = cfg.N_UE
    share_dim = cfg.N_BS * cfg.N_UE + cfg.N_UE + cfg.N_UE
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    actor = CentralActor(share_dim, n_ue, args.hidden, args.mu_bound).to(DEVICE)
    critic = Critic(share_dim, n_ue, args.hidden).to(DEVICE)
    critic_t = deepcopy(critic)
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr)
    log_alpha = torch.tensor(np.log(args.alpha_init), requires_grad=True, device=DEVICE)
    opt_alpha = torch.optim.Adam([log_alpha], lr=args.lr)
    target_H = -float(n_ue)

    rb = Replay(args.replay, share_dim, n_ue)
    env = E.Env(cfg, reward_mode="logpf", seed=args.seed); env.reset()

    logf = open(args.log, "w")
    def logp(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()
    logp(f"# CENTRAL SAC | share_dim={share_dim} mu_bound={args.mu_bound} reward=logpf(global) "
         f"steps={args.steps} device={DEVICE}")

    best_U, best_state, t0 = -1e9, None, time.time()
    for step in range(1, args.steps + 1):
        s = share_of(env)
        if step < args.warmup:
            a = np.random.uniform(-1, 1, size=n_ue).astype(np.float32)
        else:
            st = torch.as_tensor(s[None], device=DEVICE)
            a = actor.act(st, deterministic=False)[0].cpu().numpy().astype(np.float32)
        _, r, _, _ = env.step(act_to_power(a, env, pmax))
        r_scalar = float(r.sum())                 # global ΔΦ
        ns = share_of(env)
        rb.add(s, a, r_scalar, ns)
        if step % args.ep_len == 0:
            env.reset()

        if rb.size >= args.batch and step >= args.warmup:
            S, A, R, NS = rb.sample(args.batch)
            alpha = log_alpha.exp().detach()
            with torch.no_grad():
                na, nlogp = actor.sample(NS)
                q1t, q2t = critic_t(NS, na)
                y = R + args.gamma * (torch.min(q1t, q2t) - alpha * nlogp.squeeze(-1))
            q1, q2 = critic(S, A)
            loss_c = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            opt_c.zero_grad(); loss_c.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 10.0); opt_c.step()

            pa, plogp = actor.sample(S)
            q1p, q2p = critic(S, pa)
            loss_a = (alpha * plogp.squeeze(-1) - torch.min(q1p, q2p)).mean()
            opt_a.zero_grad(); loss_a.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 10.0); opt_a.step()

            loss_alpha = -(log_alpha * (plogp.squeeze(-1).detach() + target_H)).mean()
            opt_alpha.zero_grad(); loss_alpha.backward(); opt_alpha.step()

            with torch.no_grad():
                for p, pt in zip(critic.parameters(), critic_t.parameters()):
                    pt.mul_(1 - args.tau).add_(args.tau * p)

        if step % args.eval_every == 0:
            U, Us, pwr = eval_policy(actor, cfg, n_eval=args.n_eval, T=args.ep_len)
            tag = ""
            if U > best_U:
                best_U = U; best_state = deepcopy(actor.state_dict()); tag = " *"
            a_now = float(log_alpha.exp().detach())
            logp(f"step {step:>7} | PF-U {U:8.3f} ± {Us:5.3f} | alpha {a_now:6.4f} | "
                 f"pwr {pwr:5.3f} | best {best_U:8.3f} | {time.time()-t0:5.0f}s{tag}")

    if best_state is not None:
        actor.load_state_dict(best_state)
    U, Us, pwr = eval_policy(actor, cfg, n_eval=args.n_eval_final, T=args.ep_len)
    logp("\n=== FINAL (best ckpt) ===")
    logp(f"central policy   {U:8.3f} ± {Us:.3f}  (pwr {pwr:.3f})")
    # references
    eq = eval_baseline_fixed(cfg, 0.0, args.n_eval_final)   # equal_power-ish handled below
    logp(f"fixed-0.75       {eval_baseline_fixed(cfg, 0.75, args.n_eval_final):8.3f}")
    logp(f"PF-WSR ceiling   (full-CSI) ~ run env _sanity for exact; prior ≈ +23.5")
    np.save(args.out, np.array([U, Us]))


@torch.no_grad()
def eval_baseline_fixed(cfg, frac, n=20, T=10, seed=2024):
    pmax = E.dbm_to_w(cfg.Pmax_dBm); rng = np.random.default_rng(seed); Us = []
    for _ in range(n):
        env = E.Env(cfg, reward_mode="logpf", seed=int(rng.integers(1 << 30))); env.reset()
        rs = np.zeros(cfg.N_UE)
        for _t in range(T):
            pl = []
            for i in range(cfg.N_BS):
                idx = np.where(env.serv == i)[0]
                pl.append(np.full(len(idx), frac * pmax / max(len(idx), 1)))
            _, _, _, info = env.step(pl); rs += info["rate"]
        Us.append(np.log(rs / T + 1e-6).sum())
    return float(np.mean(Us))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--replay", type=int, default=1000000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--ep_len", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--n_eval", type=int, default=15)
    ap.add_argument("--n_eval_final", type=int, default=20)
    ap.add_argument("--alpha_init", type=float, default=0.1)
    ap.add_argument("--mu_bound", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="chasac_central")
    args = ap.parse_args()
    args.log = f"results/{args.tag}_log.txt"
    args.out = f"results/{args.tag}_result.npy"
    train(args)
