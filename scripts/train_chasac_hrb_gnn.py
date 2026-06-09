"""
H-RB C-HASAC + GraphQNet Manager.

Motivation: centralized SAC failure (-5.7) showed that a flat MLP Manager
cannot leverage full information — NOT that full information is useless.
A GNN that treats BSs as nodes and interference channels as edges provides
spatial inductive bias: "BSs with strong mutual interference should cooperate
on power back-off."

Architecture:
  Manager (xApp, centralized) — GraphQNet over N_BS=3 nodes
    Node features : g[i,:] (channel to all UEs) + bs_pos[i] (normalized xy)
    Edge weights  : A[i,j] = Σ_{u∈i} g[j,u]  (interference BS j causes to i's UEs)
    2-layer message passing → n_combo logits
  Worker (analytical water-filling, deployable)

Deployment Line: GraphQNet runs on xApp → can use full g, positions. OK.
Tanh saturation: DQN + analytic worker → NOT an issue here.
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


# ──────────────────────────── GNN components ───────────────────────────────

def _interference_adj(g, serv, N_BS):
    """A[i,j] = Σ_{u: serv[u]=i} g[j,u]  — interference BS j imposes on BS i's UEs."""
    A = np.zeros((N_BS, N_BS), np.float32)
    for i in range(N_BS):
        idx = np.where(serv == i)[0]
        if len(idx):
            A[i] = g[:, idx].sum(-1)
    np.fill_diagonal(A, 0.0)
    return A


class GraphQNet(nn.Module):
    """2-layer interference-weighted message passing over N_BS BS nodes."""

    def __init__(self, n_bs, n_ue, n_combo, hidden=256):
        super().__init__()
        self.n_bs = n_bs
        # node feature dim: g[i,:] (N_UE) + normalised bs_pos (2)
        node_in = n_ue + 2
        self.node_enc = nn.Linear(node_in, hidden)

        # layer 1
        self.msg1 = nn.Linear(hidden, hidden)
        self.upd1 = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # layer 2
        self.msg2 = nn.Linear(hidden, hidden)
        self.upd2 = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
        )
        # output
        self.head = nn.Sequential(
            nn.Linear(n_bs * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_combo),
        )

    def _mp(self, h, adj, msg_layer, upd_layer):
        # adj: [B, N, N],  h: [B, N, hidden]
        w = adj / (adj.sum(-1, keepdim=True) + 1e-10)  # row-normalise
        # weighted sum of neighbour messages
        nb_msg = torch.bmm(w, msg_layer(h))             # [B, N, hidden]
        return upd_layer(torch.cat([h, nb_msg], -1))    # [B, N, hidden]

    def forward(self, g_mat, bs_pos, adj):
        """
        g_mat : [B, N_BS, N_UE]   channel gains (already float)
        bs_pos: [B, N_BS, 2]      BS xy positions
        adj   : [B, N_BS, N_BS]   interference adjacency
        """
        g_norm   = g_mat  / (g_mat.amax(dim=(1, 2), keepdim=True) + 1e-30)
        pos_norm = bs_pos / 500.0                                  # area=500 m
        x = torch.cat([g_norm, pos_norm], -1)                      # [B, N, N_UE+2]
        h = F.relu(self.node_enc(x))                               # [B, N, hidden]
        h = self._mp(h, adj, self.msg1, self.upd1)
        h = self._mp(h, adj, self.msg2, self.upd2)
        return self.head(h.flatten(1))                             # [B, n_combo]


# ──────────────────────────── helpers ──────────────────────────────────────

def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)


def manager_powerlist(env, combo, levels):
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


def combo_scores(env, combos, levels):
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


def _env_gnn_input(env):
    """Return (g_mat, bs_pos, adj) as float32 numpy arrays for current env state."""
    g  = env.g.astype(np.float32)                                # [N_BS, N_UE]
    bp = env.bs[:, :2].astype(np.float32)                        # [N_BS, 2]
    A  = _interference_adj(g, env.serv, env.cfg.N_BS)            # [N_BS, N_BS]
    return g, bp, A


def _to_gnn_tensor(g_arr, bp_arr, A_arr):
    t = lambda x: torch.FloatTensor(x).unsqueeze(0).to(DEVICE)
    return t(g_arr), t(bp_arr), t(A_arr)


# ──────────────────────────── supervised pre-train ─────────────────────────

def sup_pretrain(q, combos, levels, cfg, n_data, steps, batch, lr, logp, seed=777):
    rng = np.random.default_rng(seed)
    G_buf  = np.zeros((n_data, cfg.N_BS, cfg.N_UE), np.float32)
    BP_buf = np.zeros((n_data, cfg.N_BS, 2),        np.float32)
    A_buf  = np.zeros((n_data, cfg.N_BS, cfg.N_BS), np.float32)
    Y_buf  = np.zeros(n_data, np.int64)

    for n in range(n_data):
        env = E.Env(cfg, reward_mode="difference", seed=int(rng.integers(1 << 30)))
        env.reset()
        g, bp, A = _env_gnn_input(env)
        G_buf[n], BP_buf[n], A_buf[n] = g, bp, A
        Y_buf[n] = int(np.argmax(combo_scores(env, combos, levels)))

    Gt = torch.as_tensor(G_buf,  device=DEVICE)
    Bt = torch.as_tensor(BP_buf, device=DEVICE)
    At = torch.as_tensor(A_buf,  device=DEVICE)
    Yt = torch.as_tensor(Y_buf,  device=DEVICE)

    opt = torch.optim.Adam(q.parameters(), lr=lr)
    logp(f"[SUP-GNN] oracle-argmax pretraining | n_data={n_data} steps={steps} "
         f"n_classes={len(combos)}")
    for it in range(1, steps + 1):
        bi = torch.randint(0, n_data, (batch,), device=DEVICE)
        logits = q(Gt[bi], Bt[bi], At[bi])
        loss = F.cross_entropy(logits, Yt[bi])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
        if it == 1 or it % 1000 == 0:
            acc = (logits.argmax(-1) == Yt[bi]).float().mean().item()
            logp(f"[SUP-GNN] it {it:>5}/{steps} | CE {loss.item():.4f} | train-acc {acc:.3f}")


# ──────────────────────────── evaluation ───────────────────────────────────

@torch.no_grad()
def eval_policy(q, combos, levels, cfg, n_eval=20, T=10, seed=2024, mode="greedy"):
    rng = np.random.default_rng(seed)
    Us = []
    for _ in range(n_eval):
        env = E.Env(cfg, reward_mode="difference", seed=int(rng.integers(1 << 30)))
        env.reset()
        rate_sum = np.zeros(cfg.N_UE)
        for _ in range(T):
            if mode == "random":
                c = int(rng.integers(len(combos)))
            else:
                g, bp, A = _env_gnn_input(env)
                gt, bt, at = _to_gnn_tensor(g, bp, A)
                c = int(q(gt, bt, at).argmax(-1)[0].item())
            _, _, _, info = env.step(manager_powerlist(env, combos[c], levels))
            rate_sum += info["rate"]
        Us.append(np.log(rate_sum / T + 1e-6).sum())
    return float(np.mean(Us)), float(np.std(Us))


@torch.no_grad()
def eval_oracle(cfg, n_eval=20, T=10, seed=2024):
    rng = np.random.default_rng(seed); Us = []
    for _ in range(n_eval):
        env = E.Env(cfg, reward_mode="difference", seed=int(rng.integers(1 << 30)))
        env.reset()
        rs = np.zeros(cfg.N_UE)
        for _ in range(T):
            w = env._weights()
            p = E.pf_wsr_ceiling(env.g, env.serv, w, cfg.N_BS, cfg.ceiling_grid)
            _, _, _, info = env.step([p[env.serv == i] for i in range(cfg.N_BS)])
            rs += info["rate"]
        Us.append(np.log(rs / T + 1e-6).sum())
    return float(np.mean(Us))


# ──────────────────────────── training loop ────────────────────────────────

def train(args):
    cfg = E.Cfg()
    pmax  = E.dbm_to_w(cfg.Pmax_dBm)
    n_bs  = cfg.N_BS; n_ue = cfg.N_UE
    grid  = args.grid
    levels = np.linspace(0.0, pmax, grid)
    combos = np.array(list(itertools.product(range(grid), repeat=n_bs)), dtype=np.int64)
    n_combo = len(combos)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    q  = GraphQNet(n_bs, n_ue, n_combo, args.hidden).to(DEVICE)
    qt = deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)

    cap = args.replay
    G_buf  = np.zeros((cap, n_bs, n_ue), np.float32)
    BP_buf = np.zeros((cap, n_bs, 2),    np.float32)
    A_buf  = np.zeros((cap, n_bs, n_bs), np.float32)
    A_act  = np.zeros(cap, np.int64)
    R_buf  = np.zeros(cap, np.float32)
    idx = size = 0

    env = E.Env(cfg, reward_mode="difference", seed=args.seed); env.reset()
    os.makedirs("results", exist_ok=True)
    logf = open(args.log, "w")
    def logp(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()

    logp(f"# H-RB C-HASAC + GraphQNet | grid={grid} n_combo={n_combo} "
         f"steps={args.steps} hidden={args.hidden} device={DEVICE}")
    logp(f"# Node features: g[i,:] ({n_ue}-dim) + bs_pos (2-dim) = {n_ue+2}-dim")
    logp(f"# Edges: interference adj A[i,j]=Σ_{{u∈i}} g[j,u]")

    if args.sup_data > 0:
        sup_pretrain(q, combos, levels, cfg, args.sup_data, args.sup_steps,
                     args.batch, args.lr, logp)
        qt = deepcopy(q)
        U0, Us0 = eval_policy(q, combos, levels, cfg, n_eval=args.n_eval, T=args.ep_len)
        logp(f"[SUP-GNN] post-pretrain greedy PF-U = {U0:.3f} ± {Us0:.3f}")

    eps = args.eps_start
    best_U, best_state, t0 = -1e9, None, time.time()

    for step in range(1, args.steps + 1):
        g, bp, A = _env_gnn_input(env)
        if step < args.warmup or np.random.rand() < eps:
            c = np.random.randint(n_combo)
        else:
            with torch.no_grad():
                gt, bt, at = _to_gnn_tensor(g, bp, A)
                c = int(q(gt, bt, at).argmax(-1)[0].item())

        _, _, _, info = env.step(manager_powerlist(env, combos[c], levels))
        r = float(np.log(info["rate"] + 1e-6).sum())

        G_buf[idx], BP_buf[idx], A_buf[idx] = g, bp, A
        A_act[idx] = c; R_buf[idx] = r
        idx = (idx + 1) % cap; size = min(size + 1, cap)
        if step % args.ep_len == 0:
            env.reset()
        eps = max(args.eps_end, eps - (args.eps_start - args.eps_end) / args.eps_decay)

        if size >= args.batch and step >= args.warmup:
            bi = np.random.randint(0, size, args.batch)
            t = lambda x: torch.as_tensor(x, device=DEVICE)
            g_ = t(G_buf[bi]); bp_ = t(BP_buf[bi]); a_ = t(A_buf[bi])
            c_ = t(A_act[bi]); r_ = t(R_buf[bi])
            qv = q(g_, bp_, a_).gather(-1, c_[:, None]).squeeze(-1)
            loss = F.mse_loss(qv, r_)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()

            # soft target update
            with torch.no_grad():
                for p_q, p_qt in zip(q.parameters(), qt.parameters()):
                    p_qt.data.mul_(1 - args.tau).add_(args.tau * p_q.data)

        if step % args.eval_every == 0:
            U, Us = eval_policy(q, combos, levels, cfg, n_eval=args.n_eval, T=args.ep_len)
            tag = ""
            if U > best_U:
                best_U = U; best_state = deepcopy(q.state_dict()); tag = " *"
            logp(f"step {step:>7} | PF-U {U:8.3f} ± {Us:5.3f} | eps {eps:5.3f} | "
                 f"best {best_U:8.3f} | {time.time()-t0:5.0f}s{tag}")

    if best_state is not None:
        q.load_state_dict(best_state)
    logp("\n=== FINAL (best ckpt) ===")
    U,  Us  = eval_policy(q, combos, levels, cfg, n_eval=args.n_eval_final, T=args.ep_len)
    Ur, _   = eval_policy(q, combos, levels, cfg, n_eval=args.n_eval_final, T=args.ep_len, mode="random")
    Uo      = eval_oracle(cfg, n_eval=args.n_eval_final, T=args.ep_len)
    logp(f"GraphQNet learned  {U:8.3f} ± {Us:.3f}")
    logp(f"random manager     {Ur:8.3f}")
    logp(f"PF-WSR oracle      {Uo:8.3f}")
    logp(f"(ref) fixed-power best ~ -4.17 | orig flat C-HASAC ~ -4.26 | flat H-RB random ~ -0.71")
    np.save(args.out, np.array([U, Us, Ur, Uo]))
    logf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",      type=int,   default=150000)
    ap.add_argument("--warmup",     type=int,   default=8000)
    ap.add_argument("--batch",      type=int,   default=256)
    ap.add_argument("--replay",     type=int,   default=500000)
    ap.add_argument("--tau",        type=float, default=0.005)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--hidden",     type=int,   default=256)
    ap.add_argument("--grid",       type=int,   default=6)
    ap.add_argument("--ep_len",     type=int,   default=10)
    ap.add_argument("--eval_every", type=int,   default=5000)
    ap.add_argument("--n_eval",     type=int,   default=15)
    ap.add_argument("--n_eval_final", type=int, default=30)
    ap.add_argument("--eps_start",  type=float, default=1.0)
    ap.add_argument("--eps_end",    type=float, default=0.05)
    ap.add_argument("--eps_decay",  type=float, default=40000)
    ap.add_argument("--sup_data",   type=int,   default=0)
    ap.add_argument("--sup_steps",  type=int,   default=8000)
    ap.add_argument("--seed",       type=int,   default=0)
    ap.add_argument("--tag",                    default="chasac_hrb_gnn")
    args = ap.parse_args()
    args.log = f"results/{args.tag}_log.txt"
    args.out = f"results/{args.tag}_result.npy"
    train(args)
