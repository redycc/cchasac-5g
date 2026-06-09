"""
H-RB-style C-HASAC v2 — Deployment-compliant Manager with Transformer encoder.

Deployment fix over v1 (train_chasac_hrb.py):
  v1 violation: share_of(env) = obs_share() = concat(g.flatten(), p, serv)  ← full CSI (C-layer)
  v2 fix:       BSEncoder( obs_kpm() [N_BS,3]  +  bs_pos [N_BS,2] )         ← KPM + O1 location (B-layer)

Manager architecture:
  Node features:  obs_kpm [N_BS, 3] = [cell_load/N_UE, throughput/10, P_bs/Pmax]
                  (B-layer, RIC-KPM — measurable at every BS, reported via RIC with short delay)
  Location info:  BS positions [N_BS, 2] / area  (O1 config, static infra, fully deployable)
  Encoder:        Transformer w/ distance-biased attention  ← plays role of GAT/GNN
                    att_bias[i,j] = -d(i,j)/d_ref  (nearby BSs attend more strongly)
                  This is equivalent to a 1-hop Graph Attention Network over the BS topology.
  Output:         z_global [Z_DIM] = mean-pool over BS embeddings
  Q-head:         z_global → Q[n_combo]

Worker (unchanged from v1):
  Analytic weighted water-filling within each BS's discrete power budget.

tanh saturation note:
  No tanh-squashed continuous head in this script → no mu saturation.
  LayerNorm in encoder prevents attention score blow-up.
  If a future continuous action head is added, use: mu = mu_bound * tanh(mu_raw).
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
Z_DIM  = 32   # Manager embedding dimension


# ── Utilities ─────────────────────────────────────────────────────────────────

def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)


# ── Deployment-compliant BSEncoder (Transformer/GAT over BS graph) ────────────

class BSEncoder(nn.Module):
    """Graph-Attention-style Transformer with O1-location distance bias.

    Acts as a 1-hop GAT: each BS node aggregates neighbour KPM information
    weighted by learned attention + spatial distance bias from O1 config positions.

    Input:
      kpm  [B, N_BS, 3]  — (B-layer) normalized RIC KPM per cell
      pos  [B, N_BS, 2]  — (O1 config) BS positions normalized to [0, 1]
    Output:
      z_global [B, Z_DIM] — mean-pooled BS embeddings (global Manager state)
    """

    def __init__(self, kpm_dim=3, z_dim=Z_DIM, d_ref=0.5, hidden=32):
        super().__init__()
        self.d_ref   = d_ref
        in_dim       = kpm_dim + 2   # KPM features + (x, y)
        self.embed   = nn.Linear(in_dim, hidden)
        self.q_proj  = nn.Linear(hidden, hidden)
        self.k_proj  = nn.Linear(hidden, hidden)
        self.v_proj  = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.norm1   = nn.LayerNorm(hidden)   # prevents attention score explosion
        self.ff      = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm2   = nn.LayerNorm(hidden)
        self.proj    = nn.Linear(hidden, z_dim)

    def forward(self, kpm, pos):
        # kpm: [B, N_BS, 3]   pos: [B, N_BS, 2]
        x = torch.cat([kpm, pos], dim=-1)              # [B, N_BS, 5]
        x = self.embed(x)                              # [B, N_BS, H]
        Q = self.q_proj(x); K = self.k_proj(x); V = self.v_proj(x)
        scale  = x.shape[-1] ** -0.5
        scores = torch.bmm(Q, K.transpose(1, 2)) * scale   # [B, N_BS, N_BS]
        # spatial bias: closer BSs attend more strongly
        dist   = torch.cdist(pos, pos)                 # [B, N_BS, N_BS]
        scores = scores + (-dist / self.d_ref)
        w = torch.softmax(scores, dim=-1)
        h = torch.bmm(w, V)
        x = self.norm1(x + self.out_proj(h))
        x = self.norm2(x + self.ff(x))
        z = self.proj(x)                               # [B, N_BS, Z_DIM]
        return z.mean(dim=1)                           # [B, Z_DIM]


class QNet(nn.Module):
    """Manager Q-network: deployment-compliant state → Q values over BS power combos."""

    def __init__(self, z_dim, n_combo, hidden=256):
        super().__init__()
        self.enc  = BSEncoder(z_dim=z_dim)
        self.head = mlp([z_dim, hidden, hidden, n_combo])

    def forward(self, kpm, pos):
        z = self.enc(kpm, pos)    # [B, Z_DIM]
        return self.head(z)       # [B, n_combo]


# ── Observation extraction (B-layer + O1 config — no full CSI) ────────────────

def kpm_of(env):
    """(B-layer) RIC KPM, normalized to ~[0,1]. Returns [N_BS, 3] float32."""
    kpm  = E.obs_kpm(env.p, env.g, env.serv, env.cfg.N_BS).astype(np.float32)
    pmax = E.dbm_to_w(env.cfg.Pmax_dBm)
    kpm[:, 0] /= float(env.cfg.N_UE)   # cell load → fraction
    kpm[:, 1] /= 10.0                   # throughput proxy (bps/Hz range)
    kpm[:, 2] /= pmax                   # BS total power → fraction
    return kpm


def pos_of(env):
    """(O1 config) BS positions normalized by area. Returns [N_BS, 2] float32."""
    return (env.bs / env.cfg.area).astype(np.float32)


# ── Worker: analytic weighted water-filling (deployable, unchanged) ───────────

def manager_powerlist(env, combo, levels):
    """combo[N_BS] level-idx per BS → per-BS water-filled power allocation."""
    g, serv, N0, N_BS = env.g, env.serv, E.N0, env.cfg.N_BS
    w = env._weights()
    B     = np.array([levels[combo[i]] for i in range(N_BS)])
    total = (B[:, None] * g).sum(axis=0)
    p     = np.zeros(g.shape[1])
    for i in range(N_BS):
        idx   = np.where(serv == i)[0]
        I_u   = total[idx] - B[i] * g[i, idx] + N0
        a_u   = g[i, idx] / I_u
        p[idx] = E._weighted_waterfill(w[idx], a_u, B[i])
    return [p[serv == i] for i in range(N_BS)]


def combo_scores(env, combos, levels):
    """Oracle: PF score for every combo at current env state."""
    g, serv, N0, N_BS = env.g, env.serv, E.N0, env.cfg.N_BS
    w   = env._weights()
    out = np.empty(len(combos), np.float32)
    for k in range(len(combos)):
        combo = combos[k]
        B     = np.array([levels[combo[i]] for i in range(N_BS)])
        total = (B[:, None] * g).sum(0)
        p     = np.zeros(g.shape[1])
        for i in range(N_BS):
            idx   = np.where(serv == i)[0]
            I_u   = total[idx] - B[i] * g[i, idx] + N0
            p[idx] = E._weighted_waterfill(w[idx], g[i, idx] / I_u, B[i])
        rate, _, _ = E.rates_from_power(p, g, serv, N_BS)
        out[k] = np.log(rate + 1e-6).sum()
    return out


# ── Supervised pre-train (oracle-argmax classification) ───────────────────────

def sup_pretrain(q, combos, levels, cfg, n_data, steps, batch, lr, logp, seed=777):
    """Amortize oracle grid-search via CE classification (same as v1, adapted for v2 state)."""
    rng  = np.random.default_rng(seed)
    n_bs = cfg.N_BS
    Skpm = np.zeros((n_data, n_bs, 3), np.float32)
    Spos = np.zeros((n_data, n_bs, 2), np.float32)
    Yd   = np.zeros(n_data, np.int64)
    for n in range(n_data):
        env = E.Env(cfg, reward_mode="difference", seed=int(rng.integers(1 << 30)))
        env.reset()
        Skpm[n] = kpm_of(env)
        Spos[n] = pos_of(env)
        Yd[n]   = int(np.argmax(combo_scores(env, combos, levels)))
    Skpm = torch.as_tensor(Skpm, device=DEVICE)
    Spos = torch.as_tensor(Spos, device=DEVICE)
    Yd   = torch.as_tensor(Yd,   device=DEVICE)
    opt  = torch.optim.Adam(q.parameters(), lr=lr)
    logp(f"[SUP] oracle-argmax classification pretrain | n_data={n_data} steps={steps} "
         f"n_classes={len(combos)}")
    for it in range(1, steps + 1):
        bi     = torch.randint(0, n_data, (batch,), device=DEVICE)
        logits = q(Skpm[bi], Spos[bi])
        loss   = F.cross_entropy(logits, Yd[bi])
        opt.zero_grad(); loss.backward(); opt.step()
        if it == 1 or it % 1000 == 0:
            acc = (logits.argmax(-1) == Yd[bi]).float().mean().item()
            logp(f"[SUP] it {it:>5}/{steps} | CE {loss.item():.4f} | train-acc {acc:.3f}")


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_policy(qnet, combos, levels, cfg, n_eval=20, T=10, seed=2024, mode="greedy"):
    rng = np.random.default_rng(seed)
    Us  = []
    for _ in range(n_eval):
        env = E.Env(cfg, reward_mode="difference", seed=int(rng.integers(1 << 30)))
        env.reset()
        rate_sum = np.zeros(cfg.N_UE)
        for _t in range(T):
            if mode == "random":
                c = int(rng.integers(len(combos)))
            else:
                kpm_t = torch.as_tensor(kpm_of(env)[None], device=DEVICE)
                pos_t = torch.as_tensor(pos_of(env)[None], device=DEVICE)
                c = int(qnet(kpm_t, pos_t).argmax(-1)[0].item())
            _, _, _, info = env.step(manager_powerlist(env, combos[c], levels))
            rate_sum += info["rate"]
        Us.append(np.log(rate_sum / T + 1e-6).sum())
    return float(np.mean(Us)), float(np.std(Us))


@torch.no_grad()
def eval_oracle(cfg, n_eval=20, T=10, seed=2024):
    rng = np.random.default_rng(seed); Us = []
    for _ in range(n_eval):
        env = E.Env(cfg, reward_mode="difference", seed=int(rng.integers(1 << 30))); env.reset()
        rs  = np.zeros(cfg.N_UE)
        for _t in range(T):
            w = env._weights()
            p = E.pf_wsr_ceiling(env.g, env.serv, w, cfg.N_BS, cfg.ceiling_grid)
            _, _, _, info = env.step([p[env.serv == i] for i in range(cfg.N_BS)])
            rs += info["rate"]
        Us.append(np.log(rs / T + 1e-6).sum())
    return float(np.mean(Us))


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args):
    cfg    = E.Cfg()
    pmax   = E.dbm_to_w(cfg.Pmax_dBm)
    n_bs   = cfg.N_BS
    grid   = args.grid
    levels = np.linspace(0.0, pmax, grid)
    combos = np.array(list(itertools.product(range(grid), repeat=n_bs)), dtype=np.int64)
    n_combo = len(combos)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    q   = QNet(Z_DIM, n_combo, args.hidden).to(DEVICE)
    qt  = deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)

    cap  = args.replay
    Skpm = np.zeros((cap, n_bs, 3), np.float32)
    Spos = np.zeros((cap, n_bs, 2), np.float32)
    A    = np.zeros(cap, np.int64)
    R    = np.zeros(cap, np.float32)
    idx  = size = 0

    env = E.Env(cfg, reward_mode="difference", seed=args.seed); env.reset()
    os.makedirs("results", exist_ok=True)
    logf = open(args.log, "w")

    def logp(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    logp(f"# H-RB C-HASAC v2 (deployment-compliant Transformer Manager) | "
         f"Z_DIM={Z_DIM} grid={grid} n_combo={n_combo} steps={args.steps} device={DEVICE}")
    logp(f"# State: KPM [N_BS,3] + BS_pos [N_BS,2]/area  (NO full CSI g)")

    if args.sup_data > 0:
        sup_pretrain(q, combos, levels, cfg, args.sup_data, args.sup_steps,
                     args.batch, args.lr, logp)
        qt = deepcopy(q)
        U0, Us0 = eval_policy(q, combos, levels, cfg,
                               n_eval=args.n_eval, T=args.ep_len, mode="greedy")
        logp(f"[SUP] post-pretrain greedy PF-U = {U0:.3f} ± {Us0:.3f}")

    eps = args.eps_start
    best_U, best_state, t0 = -1e9, None, time.time()

    for step in range(1, args.steps + 1):
        kpm = kpm_of(env); pos = pos_of(env)
        if step < args.warmup or np.random.rand() < eps:
            c = np.random.randint(n_combo)
        else:
            with torch.no_grad():
                kpm_t = torch.as_tensor(kpm[None], device=DEVICE)
                pos_t = torch.as_tensor(pos[None], device=DEVICE)
                c = int(q(kpm_t, pos_t).argmax(-1)[0].item())

        _, _, _, info = env.step(manager_powerlist(env, combos[c], levels))
        r = float(np.log(info["rate"] + 1e-6).sum())
        Skpm[idx], Spos[idx], A[idx], R[idx] = kpm, pos, c, r
        idx  = (idx + 1) % cap
        size = min(size + 1, cap)
        if step % args.ep_len == 0:
            env.reset()
        eps = max(args.eps_end,
                  eps - (args.eps_start - args.eps_end) / args.eps_decay)

        if size >= args.batch and step >= args.warmup:
            bi   = np.random.randint(0, size, args.batch)
            t    = lambda x: torch.as_tensor(x, device=DEVICE)
            kpm_b, pos_b = t(Skpm[bi]), t(Spos[bi])
            a_, r_       = t(A[bi]), t(R[bi])
            qv   = q(kpm_b, pos_b).gather(-1, a_[:, None]).squeeze(-1)
            loss = F.mse_loss(qv, r_)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()

        if step % args.eval_every == 0:
            U, Us = eval_policy(q, combos, levels, cfg,
                                n_eval=args.n_eval, T=args.ep_len)
            tag = ""
            if U > best_U:
                best_U = U; best_state = deepcopy(q.state_dict()); tag = " *"
            logp(f"step {step:>7} | PF-U {U:8.3f} ± {Us:5.3f} | eps {eps:5.3f} | "
                 f"best {best_U:8.3f} | {time.time()-t0:5.0f}s{tag}")

    if best_state is not None:
        q.load_state_dict(best_state)

    logp("\n=== FINAL (best ckpt) ===")
    U,  Us  = eval_policy(q, combos, levels, cfg,
                           n_eval=args.n_eval_final, T=args.ep_len, mode="greedy")
    Ur, _   = eval_policy(q, combos, levels, cfg,
                           n_eval=args.n_eval_final, T=args.ep_len, mode="random")
    Uo      = eval_oracle(cfg, n_eval=args.n_eval_final, T=args.ep_len)
    logp(f"learned manager  {U:8.3f} ± {Us:.3f}")
    logp(f"random manager   {Ur:8.3f}")
    logp(f"PF-WSR oracle    {Uo:8.3f}")
    logp(f"(ref) fixed-power best ~ -4.17 | orig flat C-HASAC ~ -4.26 | floor ~ -6.3")
    np.save(args.out, np.array([U, Us, Ur, Uo]))
    logf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",        type=int,   default=150000)
    ap.add_argument("--warmup",       type=int,   default=8000)
    ap.add_argument("--batch",        type=int,   default=256)
    ap.add_argument("--replay",       type=int,   default=500000)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--hidden",       type=int,   default=256)
    ap.add_argument("--grid",         type=int,   default=6)
    ap.add_argument("--ep_len",       type=int,   default=10)
    ap.add_argument("--eval_every",   type=int,   default=5000)
    ap.add_argument("--n_eval",       type=int,   default=15)
    ap.add_argument("--n_eval_final", type=int,   default=30)
    ap.add_argument("--eps_start",    type=float, default=1.0)
    ap.add_argument("--eps_end",      type=float, default=0.05)
    ap.add_argument("--eps_decay",    type=float, default=40000)
    ap.add_argument("--sup_data",     type=int,   default=0,
                    help="oracle-argmax pretrain scenarios (0=off); "
                         "recommended: 2500 (same as v1)")
    ap.add_argument("--sup_steps",    type=int,   default=8000)
    ap.add_argument("--seed",         type=int,   default=0)
    ap.add_argument("--tag",          default="chasac_hrb_v2")
    args        = ap.parse_args()
    args.log    = f"results/{args.tag}_log.txt"
    args.out    = f"results/{args.tag}_result.npy"
    train(args)
