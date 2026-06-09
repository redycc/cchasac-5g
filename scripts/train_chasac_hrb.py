"""
H-RB-style C-HASAC — the structural breakthrough for env_chasac.

Insight: the PF-WSR oracle (+23) reaches the ceiling by GRID-SEARCHING each BS's discrete
TOTAL POWER LEVEL (cooperative power back-off) and doing per-BS weighted water-filling inside
each budget. Flat SAC over continuous per-UE power cannot find this (proven: caps ~ -4 even
centralized full-CSI). A discrete coordinator can — exactly H-RB's lesson.

This is a contextual bandit: state = scenario (channel g, static within an episode),
action = one of GRID^N_BS per-BS power-level combos, reward = PF score of (combo + analytic
weighted water-filling). A learned amortized argmax approaches the oracle's grid search.

Learner: bandit DQN over the flat combo space (gamma=0 -> Q(s,combo) regresses the PF score;
argmax at eval). Worker = analytic weighted water-filling (deployable, no RL).
Eval = canonical PF utility U=Σ_u log(R̄_u) on held-out scenarios (floor ≈ -6.3, oracle ≈ +23.3).
"""
import os, sys, time, argparse, itertools
sys.path.insert(0, "/home/hyc1014/DL/FinalProject")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import env_chasac as E

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)


def manager_powerlist(env, combo, levels):
    """combo[N_BS] (level idx per BS) -> per-BS analytic weighted water-filling power list.
    Replicates the oracle inner loop (deployable: local CSI + measurable inter-cell interference)."""
    g, serv, N0, N_BS = env.g, env.serv, E.N0, env.cfg.N_BS
    w = env._weights()
    B = np.array([levels[combo[i]] for i in range(N_BS)])
    total = (B[:, None] * g).sum(axis=0)
    p = np.zeros(g.shape[1])
    for i in range(N_BS):
        idx = np.where(serv == i)[0]
        I_u = total[idx] - B[i] * g[i, idx] + N0
        a_u = g[i, idx] / I_u
        p[idx] = E._weighted_waterfill(w[idx], a_u, B[i])
    return [p[serv == i] for i in range(N_BS)]


class QNet(nn.Module):
    def __init__(self, share_dim, n_combo, hidden=256):
        super().__init__()
        self.net = mlp([share_dim, hidden, hidden, n_combo])

    def forward(self, s):
        return self.net(s)


def share_of(env):
    return E.obs_share(env.p, env.g, env.serv).astype(np.float32)


def combo_scores(env, combos, levels):
    """single-step PF score Σ_u log(rate_u) for every combo at current env state (no mutation)."""
    g, serv, N0, N_BS = env.g, env.serv, E.N0, env.cfg.N_BS
    w = env._weights()
    out = np.empty(len(combos), np.float32)
    for k in range(len(combos)):
        combo = combos[k]
        B = np.array([levels[combo[i]] for i in range(N_BS)])
        total = (B[:, None] * g).sum(0)
        p = np.zeros(g.shape[1])
        for i in range(N_BS):
            idx = np.where(serv == i)[0]
            I_u = total[idx] - B[i] * g[i, idx] + N0
            p[idx] = E._weighted_waterfill(w[idx], g[i, idx] / I_u, B[i])
        rate, _, _ = E.rates_from_power(p, g, serv, N_BS)
        out[k] = np.log(rate + 1e-6).sum()
    return out


def sup_pretrain(q, combos, levels, cfg, n_data, steps, batch, lr, logp, seed=777):
    """amortize the oracle grid-search via CLASSIFICATION: label = argmax-combo (oracle's pick),
    train Q logits with cross-entropy. argmax Q ≈ oracle. (Regression fails: -165 outlier combos
    dominate MSE and wash out the ranking among good combos.)"""
    rng = np.random.default_rng(seed)
    Sd = np.zeros((n_data, q.net[0].in_features), np.float32)
    Yd = np.zeros(n_data, np.int64)
    for n in range(n_data):
        env = E.Env(cfg, reward_mode="difference", seed=int(rng.integers(1 << 30))); env.reset()
        Sd[n] = share_of(env)
        Yd[n] = int(np.argmax(combo_scores(env, combos, levels)))
    Sd = torch.as_tensor(Sd, device=DEVICE); Yd = torch.as_tensor(Yd, device=DEVICE)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    logp(f"[SUP] oracle-argmax classification pretrain | n_data={n_data} steps={steps} "
         f"n_classes={len(combos)}")
    for it in range(1, steps + 1):
        bi = torch.randint(0, n_data, (batch,), device=DEVICE)
        logits = q(Sd[bi])
        loss = F.cross_entropy(logits, Yd[bi])
        opt.zero_grad(); loss.backward(); opt.step()
        if it == 1 or it % 1000 == 0:
            acc = (logits.argmax(-1) == Yd[bi]).float().mean().item()
            logp(f"[SUP] it {it:>5}/{steps} | CE {loss.item():.4f} | train-acc {acc:.3f}")


@torch.no_grad()
def eval_policy(qnet, combos, levels, cfg, n_eval=20, T=10, seed=2024, mode="greedy"):
    rng = np.random.default_rng(seed)
    Us = []
    for _ in range(n_eval):
        env = E.Env(cfg, reward_mode="difference", seed=int(rng.integers(1 << 30)))
        env.reset()
        rate_sum = np.zeros(cfg.N_UE)
        for _t in range(T):
            if mode == "random":
                c = int(rng.integers(len(combos)))
            else:
                q = qnet(torch.as_tensor(share_of(env)[None], device=DEVICE))
                c = int(q.argmax(-1)[0].item())
            _, _, _, info = env.step(manager_powerlist(env, combos[c], levels))
            rate_sum += info["rate"]
        Us.append(np.log(rate_sum / T + 1e-6).sum())
    return float(np.mean(Us)), float(np.std(Us))


@torch.no_grad()
def eval_oracle(cfg, n_eval=20, T=10, seed=2024):
    rng = np.random.default_rng(seed); Us = []
    for _ in range(n_eval):
        env = E.Env(cfg, reward_mode="difference", seed=int(rng.integers(1 << 30))); env.reset()
        rs = np.zeros(cfg.N_UE)
        for _t in range(T):
            w = env._weights()
            p = E.pf_wsr_ceiling(env.g, env.serv, w, cfg.N_BS, cfg.ceiling_grid)
            _, _, _, info = env.step([p[env.serv == i] for i in range(cfg.N_BS)])
            rs += info["rate"]
        Us.append(np.log(rs / T + 1e-6).sum())
    return float(np.mean(Us))


def train(args):
    cfg = E.Cfg()
    pmax = E.dbm_to_w(cfg.Pmax_dBm)
    n_bs = cfg.N_BS
    grid = args.grid
    levels = np.linspace(0.0, pmax, grid)
    combos = np.array(list(itertools.product(range(grid), repeat=n_bs)), dtype=np.int64)  # [n_combo, N_BS]
    n_combo = len(combos)
    share_dim = n_bs * cfg.N_UE + cfg.N_UE + cfg.N_UE
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    q = QNet(share_dim, n_combo, args.hidden).to(DEVICE)
    qt = deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)

    cap = args.replay
    S = np.zeros((cap, share_dim), np.float32); A = np.zeros(cap, np.int64); R = np.zeros(cap, np.float32)
    idx = size = 0

    env = E.Env(cfg, reward_mode="difference", seed=args.seed); env.reset()
    logf = open(args.log, "w")
    def logp(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()
    logp(f"# H-RB C-HASAC (bandit-DQN) | share_dim={share_dim} grid={grid} n_combo={n_combo} "
         f"steps={args.steps} device={DEVICE}")

    if args.sup_data > 0:
        sup_pretrain(q, combos, levels, cfg, args.sup_data, args.sup_steps,
                     args.batch, args.lr, logp)
        qt = deepcopy(q)
        U0, Us0 = eval_policy(q, combos, levels, cfg, n_eval=args.n_eval, T=args.ep_len, mode="greedy")
        logp(f"[SUP] post-pretrain greedy PF-U = {U0:.3f} ± {Us0:.3f}")

    eps = args.eps_start
    best_U, best_state, t0 = -1e9, None, time.time()
    for step in range(1, args.steps + 1):
        s = share_of(env)
        if step < args.warmup or np.random.rand() < eps:
            c = np.random.randint(n_combo)
        else:
            with torch.no_grad():
                c = int(q(torch.as_tensor(s[None], device=DEVICE)).argmax(-1)[0].item())
        _, _, _, info = env.step(manager_powerlist(env, combos[c], levels))
        r = float(np.log(info["rate"] + 1e-6).sum())          # per-step PF score (matches eval)
        S[idx], A[idx], R[idx] = s, c, r
        idx = (idx + 1) % cap; size = min(size + 1, cap)
        if step % args.ep_len == 0:
            env.reset()
        eps = max(args.eps_end, eps - (args.eps_start - args.eps_end) / args.eps_decay)

        if size >= args.batch and step >= args.warmup:
            bi = np.random.randint(0, size, args.batch)
            t = lambda x: torch.as_tensor(x, device=DEVICE)
            s_, a_, r_ = t(S[bi]), t(A[bi]), t(R[bi])
            qv = q(s_).gather(-1, a_[:, None]).squeeze(-1)
            # gamma=0 contextual bandit: Q(s,a) regresses the immediate PF score
            loss = F.mse_loss(qv, r_)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()

        if step % args.eval_every == 0:
            U, Us = eval_policy(q, combos, levels, cfg, n_eval=args.n_eval, T=args.ep_len, mode="greedy")
            tag = ""
            if U > best_U:
                best_U = U; best_state = deepcopy(q.state_dict()); tag = " *"
            logp(f"step {step:>7} | PF-U {U:8.3f} ± {Us:5.3f} | eps {eps:5.3f} | "
                 f"best {best_U:8.3f} | {time.time()-t0:5.0f}s{tag}")

    if best_state is not None:
        q.load_state_dict(best_state)
    logp("\n=== FINAL (best ckpt) ===")
    U, Us = eval_policy(q, combos, levels, cfg, n_eval=args.n_eval_final, T=args.ep_len, mode="greedy")
    Ur, _ = eval_policy(q, combos, levels, cfg, n_eval=args.n_eval_final, T=args.ep_len, mode="random")
    Uo = eval_oracle(cfg, n_eval=args.n_eval_final, T=args.ep_len)
    logp(f"learned manager  {U:8.3f} ± {Us:.3f}")
    logp(f"random manager   {Ur:8.3f}")
    logp(f"PF-WSR oracle    {Uo:8.3f}")
    logp(f"(ref) fixed-power best ~ -4.17 | orig flat C-HASAC ~ -4.26 | floor ~ -6.3")
    np.save(args.out, np.array([U, Us, Ur, Uo]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--warmup", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--replay", type=int, default=500000)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--ep_len", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--n_eval", type=int, default=15)
    ap.add_argument("--n_eval_final", type=int, default=30)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay", type=float, default=40000)
    ap.add_argument("--sup_data", type=int, default=0, help="oracle-regression pretrain scenarios (0=off)")
    ap.add_argument("--sup_steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="chasac_hrb")
    args = ap.parse_args()
    args.log = f"results/{args.tag}_log.txt"
    args.out = f"results/{args.tag}_result.npy"
    train(args)
