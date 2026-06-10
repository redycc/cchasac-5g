"""
train_reproduce.py — RL/BC training per tasks/files/REPRODUCE.md §3, §6.

Canonical comparison (fixed topology):
  HASAC   : python3 scripts/train_reproduce.py --use_z 0 --actor separate --seed 0 --tag rep_hasac_s0
  C-HASAC : python3 scripts/train_reproduce.py --use_z 1 --actor separate --seed 0 --tag rep_chasac_s0

Spec essentials implemented:
- per-cell scalar squashed-Gaussian action; UE-embed → segment-mean pool per cell → head
- actor variants: shared (parameter sharing) / separate (pure HASAC, canonical)
- C-HASAC: per-cell z_i (mean-pool encoder over OTHER cells' KPM) concat to pooled embedding
- centralized twin-Q on share_obs ⊕ joint_action ⊕ onehot; never given z
- sequential HAML update (random order, only agent i's action carries gradient,
  already-updated agents' actions resampled & detached)
- gamma = 0 (contextual bandit): Q target = r, no bootstrap/target net
- auto-tuned alpha, target entropy −1/agent; optional alpha floor (0 = pure HASAC)
- team reward (canonical) or difference reward; optional running-std reward normalization
- eval: deterministic, held-out seeds (in-loop 10000+, final 20000+), T=150, discard first T/3,
  report mean delivered goodput and % of floor→ceiling gap
"""
import os, sys, time, argparse, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_reproduce as E

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
UE_FEAT = 4
KPM_DIM = 3


def mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]),
                   act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


class CellActor(nn.Module):
    """One cell's policy: UE-embed → (pool outside) → head(pooled ⊕ z_i) → scalar μ, log_std."""
    def __init__(self, hidden=256, z_dim=0):
        super().__init__()
        self.emb = mlp([UE_FEAT, hidden, hidden])
        self.head = mlp([hidden + z_dim, hidden, hidden])
        self.mu = nn.Linear(hidden, 1)
        self.log_std = nn.Linear(hidden, 1)
        self.z_dim = z_dim

    def pooled(self, o, cell_mask):
        """o [B,N_UE,4], cell_mask [B,N_UE] → [B,H] segment-mean over own UEs."""
        e = self.emb(o)                                        # [B,N_UE,H]
        cnt = cell_mask.sum(-1, keepdim=True).clamp_min(1.0)
        return (cell_mask[..., None] * e).sum(1) / cnt

    def dist_params(self, o, cell_mask, z_i=None):
        h = self.pooled(o, cell_mask)
        if self.z_dim > 0:
            h = torch.cat([h, z_i], dim=-1)
        h = self.head(h)
        return self.mu(h).squeeze(-1), self.log_std(h).squeeze(-1).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, o, cell_mask, z_i=None):
        mu, log_std = self.dist_params(o, cell_mask, z_i)
        dist = torch.distributions.Normal(mu, log_std.exp())
        x = dist.rsample()
        a = torch.tanh(x)
        logp = dist.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)
        return a, logp                                         # [B], [B]

    @torch.no_grad()
    def act(self, o, cell_mask, z_i=None, deterministic=True):
        mu, log_std = self.dist_params(o, cell_mask, z_i)
        if deterministic:
            return torch.tanh(mu)
        return torch.tanh(torch.distributions.Normal(mu, log_std.exp()).sample())


class Encoder(nn.Module):
    """Mean-pool encoder: z_i = rho(mean_{j != i} phi(kpm_j)). Permutation-invariant, no CSI."""
    def __init__(self, hidden=128, z_dim=16):
        super().__init__()
        self.phi = mlp([KPM_DIM, hidden, hidden])
        self.rho = mlp([hidden, hidden, z_dim])

    def forward(self, kpm):                                    # [B,N_BS,3] → [B,N_BS,z_dim]
        h = self.phi(kpm)                                      # [B,N_BS,H]
        n = kpm.shape[1]
        zs = []
        for i in range(n):
            others = [j for j in range(n) if j != i]
            zs.append(self.rho(h[:, others].mean(1)))
        return torch.stack(zs, dim=1)


class Critic(nn.Module):
    def __init__(self, share_dim, n_bs, hidden=256):
        super().__init__()
        d = share_dim + n_bs + n_bs                            # share ⊕ joint_a ⊕ onehot
        self.q1 = mlp([d, hidden, hidden, 1])
        self.q2 = mlp([d, hidden, hidden, 1])

    def forward(self, share, a_joint, onehot):
        x = torch.cat([share, a_joint, onehot], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


# --------------------------- replay (gamma=0: no next state) -------
class Replay:
    def __init__(self, cap, n_ue, n_bs, share_dim):
        self.cap, self.ptr, self.size = cap, 0, 0
        self.o = np.zeros((cap, n_ue, UE_FEAT), np.float32)
        self.m = np.zeros((cap, n_bs, n_ue), np.float32)
        self.k = np.zeros((cap, n_bs, KPM_DIM), np.float32)
        self.s = np.zeros((cap, share_dim), np.float32)
        self.a = np.zeros((cap, n_bs), np.float32)
        self.r = np.zeros((cap, n_bs), np.float32)

    def add(self, o, m, k, s, a, r):
        i = self.ptr
        self.o[i], self.m[i], self.k[i], self.s[i], self.a[i], self.r[i] = o, m, k, s, a, r
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, bs):
        idx = np.random.randint(0, self.size, size=bs)
        t = lambda x: torch.as_tensor(x[idx], device=DEVICE)
        return t(self.o), t(self.m), t(self.k), t(self.s), t(self.a), t(self.r)


# --------------------------- helpers -------------------------------
def serv_mask(env, cfg):
    m = np.zeros((cfg.N_BS, cfg.N_UE), np.float32)
    m[env.serv, np.arange(cfg.N_UE)] = 1.0
    return m


def normalize_kpm(kpm, cfg):
    """[load, mean R̄, mean Q] → comparable scales."""
    k = kpm.copy()
    k[..., 0] /= cfg.N_UE
    k[..., 1] /= 5.0
    k[..., 2] /= cfg.q_max
    return k


def policy_levels(actors, encoder, o, m, kpm, n_bs, use_z, deterministic=True):
    """numpy obs → per-cell levels [N_BS] in [0,1]."""
    ot = torch.as_tensor(o[None], device=DEVICE)
    mt = torch.as_tensor(m[None], device=DEVICE)
    z = None
    if use_z:
        kt = torch.as_tensor(kpm[None], device=DEVICE)
        z = encoder(kt)                                        # [1,N_BS,z]
    a = np.zeros(n_bs, np.float32)
    for i in range(n_bs):
        actor = actors[i if len(actors) > 1 else 0]
        zi = z[:, i] if z is not None else None
        a[i] = float(actor.act(ot, mt[:, i], zi, deterministic=deterministic)[0])
    return (a + 1.0) / 2.0, a


# --------------------------- eval ----------------------------------
@torch.no_grad()
def eval_policy(actors, encoder, cfg, use_z, n_trials=3, T=150, seed0=10000):
    means = []
    for k in range(n_trials):
        env = E.Env(cfg, seed=seed0 + k)
        env.reset()
        gp = []
        for t in range(T):
            o = env.obs_local()
            m = serv_mask(env, cfg)
            kpm = normalize_kpm(env.obs_kpm(), cfg)
            levels, _ = policy_levels(actors, encoder, o, m, kpm, cfg.N_BS, use_z)
            _, _, _, info = env.step(levels)
            gp.append(info["goodput"])
        means.append(np.mean(gp[T // 3:]))
    return float(np.mean(means)), float(np.std(means))


def eval_reference(cfg, policy, n_trials=6, T=150, seed0=20000):
    means = []
    for k in range(n_trials):
        env = E.Env(cfg, seed=seed0 + k)
        env.reset()
        gp = []
        for t in range(T):
            _, _, _, info = env.step(policy(env, t))
            gp.append(info["goodput"])
        means.append(np.mean(gp[T // 3:]))
    return float(np.mean(means)), float(np.std(means))


# --------------------------- BC dataset (§6.2) ---------------------
def collect_oracle_dataset(cfg, n_trials, T, logp, spatial=False,
                           cache="results/rep_oracle_dataset.npz"):
    tag = "spatial" if spatial else "full"
    cache = cache.replace(".npz", f"_{tag}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        if int(d["n_trials"]) == n_trials and int(d["topo_seed"]) == cfg.topo_seed:
            logp(f"[BC] oracle dataset cache hit ({cache})")
            return d["O"], d["M"], d["K"], d["L"]
    logp(f"[BC] rolling out {tag} oracle: {n_trials} trials x {T} slots ...")
    O, M, K, L = [], [], [], []
    for k in range(n_trials):
        env = E.Env(cfg, seed=30000 + k)
        env.reset()
        for t in range(T):
            lv = env.spatial_oracle_action() if spatial else env.oracle_action()
            O.append(env.obs_local()); M.append(serv_mask(env, cfg))
            K.append(normalize_kpm(env.obs_kpm(), cfg)); L.append(lv.astype(np.float32))
            env.step(lv)
    O, M, K, L = map(lambda x: np.stack(x).astype(np.float32), (O, M, K, L))
    np.savez(cache, O=O, M=M, K=K, L=L, n_trials=n_trials, topo_seed=cfg.topo_seed)
    logp(f"[BC] dataset saved -> {cache} ({O.shape[0]} samples)")
    return O, M, K, L


# --------------------------- gate×base (§3.4, §5, §6.3) ------------
class CellNet(nn.Module):
    """Deterministic per-cell net: UE-embed → segment-mean pool → head → sigmoid scalar."""
    def __init__(self, hidden=256):
        super().__init__()
        self.emb = mlp([UE_FEAT, hidden, hidden])
        self.head = mlp([hidden, hidden, hidden, 1])

    def forward(self, o, cell_mask):
        e = self.emb(o)
        cnt = cell_mask.sum(-1, keepdim=True).clamp_min(1.0)
        pooled = (cell_mask[..., None] * e).sum(1) / cnt
        return torch.sigmoid(self.head(pooled)).squeeze(-1)    # [B] in (0,1)


def collect_gate_dataset(cfg, n_trials, T, logp, cache="results/rep_gate_dataset.npz"):
    """Roll out FULL oracle; at each state record (obs, mask, full level, spatial level)."""
    if cfg.hot_init < 1.0:
        cache = cache.replace(".npz", "_bursty.npz")
    if getattr(cfg, "geom_file", None):
        cache = cache.replace(".npz", "_geom.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        if int(d["n_trials"]) == n_trials and int(d["topo_seed"]) == cfg.topo_seed:
            logp(f"[GATE] dataset cache hit ({cache})")
            return d["O"], d["M"], d["LF"], d["LS"]
    logp(f"[GATE] rolling out full+spatial oracle: {n_trials} trials x {T} slots ...")
    O, M, LF, LS = [], [], [], []
    for k in range(n_trials):
        env = E.Env(cfg, seed=30000 + k)
        env.reset()
        for t in range(T):
            lf = env.oracle_action()
            ls = env.spatial_oracle_action()
            O.append(env.obs_local()); M.append(serv_mask(env, cfg))
            LF.append(lf.astype(np.float32)); LS.append(ls.astype(np.float32))
            env.step(lf)
    O, M, LF, LS = map(lambda x: np.stack(x).astype(np.float32), (O, M, LF, LS))
    np.savez(cache, O=O, M=M, LF=LF, LS=LS, n_trials=n_trials, topo_seed=cfg.topo_seed)
    logp(f"[GATE] dataset saved -> {cache} ({O.shape[0]} samples)")
    return O, M, LF, LS


def apply_traffic(cfg, traffic):
    """full-buffer (always hot) or bursty (hot/cold 2-state Markov, spec §2.2)."""
    if traffic == "bursty":
        cfg.hot_init, cfg.p_on_off, cfg.p_off_on = 0.5, 0.05, 0.05


def apply_geom(cfg, args):
    if getattr(args, "geom_file", ""):
        cfg.geom_file = args.geom_file


def train_gate(args):
    """§6.3 modular recipe: gate→spatial oracle, base→residual, FIXED multiply, eval."""
    cfg = E.Cfg()
    cfg.fixed_topology = bool(args.fixed_topology)
    cfg.topo_seed = args.topo_seed
    apply_traffic(cfg, args.traffic)
    apply_geom(cfg, args)
    n_bs = cfg.N_BS
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    logf = open(args.log, "w")
    def logp(*a):
        s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush()
    logp(f"# REPRODUCE gate×base | fixed_topology={cfg.fixed_topology} "
         f"topo_seed={cfg.topo_seed} geom={getattr(cfg, 'geom_file', None)} "
         f"seed={args.seed} bc_iters={args.bc_iters} device={DEVICE}")

    O, M, LF, LS = collect_gate_dataset(cfg, args.gate_trials, 150, logp)
    # residual target: base_gt = clip(full/spatial, 0, 1); spatial==0 → gate kills power, base free (=1)
    base_gt = np.where(LS > 1e-6, np.clip(LF / np.maximum(LS, 1e-6), 0.0, 1.0), 1.0)
    Ot = torch.as_tensor(O, device=DEVICE); Mt = torch.as_tensor(M, device=DEVICE)
    LSt = torch.as_tensor(LS, device=DEVICE)
    BGt = torch.as_tensor(base_gt.astype(np.float32), device=DEVICE)

    gates = nn.ModuleList([CellNet(args.hidden).to(DEVICE) for _ in range(n_bs)])
    bases = nn.ModuleList([CellNet(args.hidden).to(DEVICE) for _ in range(n_bs)])
    opt_g = torch.optim.Adam(gates.parameters(), lr=args.lr)
    opt_b = torch.optim.Adam(bases.parameters(), lr=args.lr)
    n = Ot.shape[0]
    for it in range(1, args.bc_iters + 1):
        idx = torch.randint(0, n, (args.batch,), device=DEVICE)
        lg = sum(F.mse_loss(gates[i](Ot[idx], Mt[idx][:, i]), LSt[idx][:, i])
                 for i in range(n_bs))
        opt_g.zero_grad(); lg.backward(); opt_g.step()
        lb = sum(F.mse_loss(bases[i](Ot[idx], Mt[idx][:, i]), BGt[idx][:, i])
                 for i in range(n_bs))
        opt_b.zero_grad(); lb.backward(); opt_b.step()
        if it == 1 or it % 500 == 0:
            logp(f"[BC] it {it:>5}/{args.bc_iters} | gate MSE {lg.item()/n_bs:.5f} | "
                 f"base MSE {lb.item()/n_bs:.5f}")

    # ---- eval: power = gate × base (fixed multiply) ----
    @torch.no_grad()
    def gate_policy(env, t):
        o = torch.as_tensor(env.obs_local()[None], device=DEVICE)
        m = torch.as_tensor(serv_mask(env, cfg)[None], device=DEVICE)
        return np.array([float(gates[i](o, m[:, i]) * bases[i](o, m[:, i]))
                         for i in range(n_bs)])

    floor, _ = eval_reference(cfg, lambda e, t: E.bl_equal(cfg))
    ceil, _ = eval_reference(cfg, lambda e, t: e.oracle_action(), n_trials=3)
    gp, gps = eval_reference(cfg, gate_policy, n_trials=6)
    pct = 100.0 * (gp - floor) / (ceil - floor)
    logp("\n=== FINAL (gate×base, fixed multiply, eval seeds 20000+) ===")
    logp(f"goodput          {gp:7.3f} ± {gps:.3f}   ({pct:.1f}% of gap)")
    logp(f"floor (equal)    {floor:7.3f}")
    logp(f"ceiling (oracle) {ceil:7.3f}")
    np.save(args.out, dict(goodput=gp, floor=floor, ceiling=ceil, pct=pct), allow_pickle=True)
    torch.save(dict(gates=[g.state_dict() for g in gates],
                    bases=[b.state_dict() for b in bases]), f"results/{args.tag}_best.pt")
    logf.close()


# --------------------------- learnable combine (§5, §6.3) ----------
def train_combine(args):
    """Claim #3: BC a combine NN into a verified multiplier (uniform pairs), wire
    power = combine(gate, base), then RL-refine → combine degrades.
    --combine fixed: multiply is hard-wired (no combine params) → RL-refine holds."""
    cfg = E.Cfg()
    cfg.fixed_topology = bool(args.fixed_topology)
    cfg.topo_seed = args.topo_seed
    apply_traffic(cfg, args.traffic)
    apply_geom(cfg, args)
    n_bs = cfg.N_BS
    share_dim = E.SHARE_DIM(cfg)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    logf = open(args.log, "w")
    def logp(*a):
        s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush()
    logp(f"# REPRODUCE combine | variant={args.combine} freeze_gate={args.freeze_gate} "
         f"seed={args.seed} rl_steps={args.steps} device={DEVICE}")

    # ---- load BC'd gate/base from the gate run ----
    ck = torch.load(args.gate_ckpt, map_location=DEVICE, weights_only=True)
    gates = nn.ModuleList([CellNet(args.gate_hidden).to(DEVICE) for _ in range(n_bs)])
    bases = nn.ModuleList([CellNet(args.gate_hidden).to(DEVICE) for _ in range(n_bs)])
    for g, sd in zip(gates, ck["gates"]):
        g.load_state_dict(sd)
    for b, sd in zip(bases, ck["bases"]):
        b.load_state_dict(sd)

    # ---- combine NN: BC on UNIFORM (z,b) pairs → z·b, then verify (§5) ----
    combine = None
    if args.combine == "learnable":
        combine = mlp([2, 64, 64, 1], out_act=nn.Sigmoid).to(DEVICE)
        opt_cb = torch.optim.Adam(combine.parameters(), lr=1e-3)
        for it in range(1, 5001):
            zb = torch.rand(512, 2, device=DEVICE)
            loss = F.mse_loss(combine(zb).squeeze(-1), zb[:, 0] * zb[:, 1])
            opt_cb.zero_grad(); loss.backward(); opt_cb.step()
        with torch.no_grad():                                   # verify on grid
            gx = torch.linspace(0, 1, 101, device=DEVICE)
            gz, gb = torch.meshgrid(gx, gx, indexing="ij")
            pairs = torch.stack([gz.flatten(), gb.flatten()], dim=1)
            pred = combine(pairs).squeeze(-1)
            true = pairs[:, 0] * pairs[:, 1]
            mae = float((pred - true).abs().mean())
            corr = float(torch.corrcoef(torch.stack([pred, true]))[0, 1])
        logp(f"[COMBINE] multiplier verified: MAE={mae:.5f} corr={corr:.5f}")

    def wired_level(o, m):
        """[B,N_UE,4],[B,N_BS,N_UE] → levels [B,N_BS] via gate ∘ combine ∘ base."""
        lv = []
        for i in range(n_bs):
            g = gates[i](o, m[:, i]); b = bases[i](o, m[:, i])
            if combine is not None:
                lv.append(combine(torch.stack([g, b], dim=-1)).squeeze(-1))
            else:
                lv.append(g * b)                               # fixed multiply
            i += 0
        return torch.stack(lv, dim=1).clamp(1e-3, 1 - 1e-3)

    @torch.no_grad()
    def wired_policy(env, t):
        o = torch.as_tensor(env.obs_local()[None], device=DEVICE)
        m = torch.as_tensor(serv_mask(env, cfg)[None], device=DEVICE)
        return wired_level(o, m)[0].cpu().numpy()

    floor, _ = eval_reference(cfg, lambda e, t: E.bl_equal(cfg))
    ceil, _ = eval_reference(cfg, lambda e, t: e.oracle_action(), n_trials=3)
    pct = lambda x: 100.0 * (x - floor) / (ceil - floor)
    gp0, _ = eval_reference(cfg, wired_policy, n_trials=6)
    logp(f"[WIRED-BC] goodput {gp0:.3f} ({pct(gp0):.1f}% of gap) | "
         f"floor {floor:.3f} ceiling {ceil:.3f}")

    # ---- RL: spatial-gate × RL-worker (spec construction) ----
    # Worker is a fresh squashed-Gaussian SAC actor trained FROM SCRATCH per cell.
    # The (frozen) gate multiplies OUTSIDE the squash: applied = gate(o) × (a+1)/2 — no
    # atanh inversion anywhere (first refine attempt saturated exactly like the old −165 bug).
    # Degrade arm: applied = combine(gate, base) with the BC'd multiplier trainable by RL.
    workers = nn.ModuleList([CellActor(args.hidden, 0).to(DEVICE) for _ in range(n_bs)])
    opt_w = [torch.optim.Adam(w.parameters(), lr=args.lr) for w in workers]
    opt_cb2 = (torch.optim.Adam(combine.parameters(), lr=args.lr)
               if combine is not None else None)
    for p in gates.parameters():
        p.requires_grad_(False)                                # gate stays frozen (oracle role)

    def apply_gate(g, base01_i):
        """one cell: frozen gate level × worker base (or learnable combine of the two)."""
        if combine is not None:
            return combine(torch.stack([g, base01_i], dim=-1)).squeeze(-1).clamp(0, 1)
        return (g * base01_i).clamp(0, 1)

    @torch.no_grad()
    def rl_policy(env, t):
        o = torch.as_tensor(env.obs_local()[None], device=DEVICE)
        m = torch.as_tensor(serv_mask(env, cfg)[None], device=DEVICE)
        lv = []
        for i in range(n_bs):
            g = gates[i](o, m[:, i])
            base01 = (workers[i].act(o, m[:, i], deterministic=True) + 1.0) / 2.0
            lv.append(float(apply_gate(g, base01)))
        return np.array(lv)

    critic = Critic(share_dim, n_bs, args.hidden).to(DEVICE)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr)
    log_alpha = torch.tensor(np.log(args.alpha_init), requires_grad=True, device=DEVICE)
    opt_alpha = torch.optim.Adam([log_alpha], lr=args.lr)
    rb = Replay(args.replay, cfg.N_UE, n_bs, share_dim)
    OH = torch.eye(n_bs, device=DEVICE)
    env = E.Env(cfg, reward_mode=args.reward, seed=args.seed)
    env.reset()
    r_std_ema = 1.0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        o = env.obs_local(); m = serv_mask(env, cfg); sh = env.obs_share()
        ot = torch.as_tensor(o[None], device=DEVICE)
        mt = torch.as_tensor(m[None], device=DEVICE)
        with torch.no_grad():
            lv = np.zeros(n_bs, np.float32)
            for i in range(n_bs):
                g = gates[i](ot, mt[:, i])
                if step < args.warmup:
                    base01 = torch.rand(1, device=DEVICE)
                else:
                    base01 = (workers[i].act(ot, mt[:, i], deterministic=False) + 1.0) / 2.0
                lv[i] = float(apply_gate(g, base01))
        _, r, _, info = env.step(lv)
        if args.norm_reward:
            r_std_ema = 0.999 * r_std_ema + 0.001 * float(np.mean(r ** 2))
            r = np.clip(r / (np.sqrt(r_std_ema) + 1e-8), -10.0, 10.0)
        rb.add(o, m, normalize_kpm(env.obs_kpm(), cfg), sh, lv, r.astype(np.float32))
        if step % cfg.ep_len == 0:
            env.reset()

        if rb.size >= args.batch and step >= args.warmup:
            o_, m_, k_, s_, a_, r_ = rb.sample(args.batch)   # a_ = APPLIED levels
            B = o_.shape[0]
            alpha = log_alpha.exp().detach()
            loss_c = 0.0
            for i in range(n_bs):
                oh = OH[i][None].expand(B, -1)
                q1, q2 = critic(s_, a_, oh)
                loss_c = loss_c + F.mse_loss(q1, r_[:, i]) + F.mse_loss(q2, r_[:, i])
            opt_c.zero_grad(); loss_c.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
            opt_c.step()

            # current applied levels from all workers (held fixed for others)
            with torch.no_grad():
                lv_cur = []
                for j in range(n_bs):
                    gj = gates[j](o_, m_[:, j])
                    aj, _ = workers[j].sample(o_, m_[:, j])
                    lv_cur.append(apply_gate(gj, (aj + 1.0) / 2.0))
            if opt_cb2 is not None:
                opt_cb2.zero_grad()
            logps = [None] * n_bs
            for i in torch.randperm(n_bs).tolist():
                ai, logp_i = workers[i].sample(o_, m_[:, i])
                gi = gates[i](o_, m_[:, i]).detach()
                lv_i = apply_gate(gi, (ai + 1.0) / 2.0)        # grads → worker (+combine)
                a_joint = torch.stack(
                    [lv_i if j == i else lv_cur[j] for j in range(n_bs)], dim=1)
                oh = OH[i][None].expand(B, -1)
                q1, q2 = critic(s_, a_joint, oh)
                loss_i = (alpha * logp_i - torch.min(q1, q2)).mean()
                opt_w[i].zero_grad(); loss_i.backward()
                nn.utils.clip_grad_norm_(workers[i].parameters(), 10.0)
                opt_w[i].step()
                with torch.no_grad():
                    ai2, _ = workers[i].sample(o_, m_[:, i])
                    lv_cur[i] = apply_gate(gates[i](o_, m_[:, i]), (ai2 + 1.0) / 2.0)
                logps[i] = logp_i.detach()
            if opt_cb2 is not None:                            # combine: accumulated grads
                nn.utils.clip_grad_norm_(combine.parameters(), 10.0)
                opt_cb2.step()
            loss_alpha = -(log_alpha * (torch.stack(logps, 1).mean() + (-1.0)))
            opt_alpha.zero_grad(); loss_alpha.backward(); opt_alpha.step()

        if step % args.eval_every == 0:
            gp, gps = eval_reference(cfg, rl_policy, n_trials=3, seed0=10000 + args.seed)
            if combine is not None:
                with torch.no_grad():
                    pred = combine(pairs).squeeze(-1)
                    mae_now = float((pred - true).abs().mean())
                    corr_now = float(torch.corrcoef(torch.stack([pred, true]))[0, 1])
                cstr = f" | mult MAE {mae_now:.4f} corr {corr_now:.4f}"
            else:
                cstr = ""
            logp(f"step {step:>6} | goodput {gp:7.3f} ({pct(gp):5.1f}% gap) | "
                 f"alpha {float(log_alpha.exp()):6.4f}{cstr} | {time.time()-t0:5.0f}s")

    gp, gps = eval_reference(cfg, rl_policy, n_trials=6)
    logp(f"\n=== FINAL (spatial-gate × RL-worker, {args.combine} combine) ===")
    logp(f"wired BC reference    {gp0:7.3f}   ({pct(gp0):.1f}% of gap)")
    logp(f"gate × RL-worker      {gp:7.3f}   ({pct(gp):.1f}% of gap)")
    logp(f"floor {floor:.3f} / ceiling {ceil:.3f}")
    np.save(args.out, dict(bc=gp0, rl=gp, floor=floor, ceiling=ceil), allow_pickle=True)
    logf.close()


# --------------------------- train (RL, §6.1) ----------------------
def train(args):
    cfg = E.Cfg()
    cfg.fixed_topology = bool(args.fixed_topology)
    cfg.topo_seed = args.topo_seed
    apply_traffic(cfg, args.traffic)
    apply_geom(cfg, args)
    n_bs, n_ue = cfg.N_BS, cfg.N_UE
    share_dim = E.SHARE_DIM(cfg)
    z_dim = args.z_dim if args.use_z else 0

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    n_actors = n_bs if args.actor == "separate" else 1
    actors = nn.ModuleList([CellActor(args.hidden, z_dim).to(DEVICE) for _ in range(n_actors)])
    opt_a = [torch.optim.Adam(a.parameters(), lr=args.lr) for a in actors]
    encoder = Encoder(128, z_dim).to(DEVICE) if args.use_z else None
    opt_enc = torch.optim.Adam(encoder.parameters(), lr=args.lr) if encoder else None
    critic = Critic(share_dim, n_bs, args.hidden).to(DEVICE)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr)
    log_alpha = torch.tensor(np.log(args.alpha_init), requires_grad=True, device=DEVICE)
    opt_alpha = torch.optim.Adam([log_alpha], lr=args.lr)
    target_H = -1.0                                            # per agent, 1-dim action

    rb = Replay(args.replay, n_ue, n_bs, share_dim)
    OH = torch.eye(n_bs, device=DEVICE)
    env = E.Env(cfg, reward_mode=args.reward, seed=args.seed)
    env.reset()

    logf = open(args.log, "w")
    def logp(*a):
        s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush()

    logp(f"# REPRODUCE train | use_z={args.use_z} actor={args.actor} reward={args.reward} "
         f"steps={args.steps} fixed_topology={cfg.fixed_topology} topo_seed={cfg.topo_seed} "
         f"seed={args.seed} gamma=0 device={DEVICE}")

    # references on final eval seeds
    floor, _ = eval_reference(cfg, lambda e, t: E.bl_equal(cfg))
    ceil, _ = eval_reference(cfg, lambda e, t: e.oracle_action(), n_trials=3)
    logp(f"floor(equal)={floor:.3f}  ceiling(oracle)={ceil:.3f}")
    pct = lambda x: 100.0 * (x - floor) / (ceil - floor)

    best, best_state = -1e9, None
    r_std_ema = 1.0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        o = env.obs_local(); m = serv_mask(env, cfg)
        kpm = normalize_kpm(env.obs_kpm(), cfg); sh = env.obs_share()
        if step < args.warmup:
            a = np.random.uniform(-1, 1, size=n_bs).astype(np.float32)
        else:
            _, a = policy_levels(actors, encoder, o, m, kpm, n_bs, args.use_z,
                                 deterministic=False)
        _, r, _, info = env.step((a + 1.0) / 2.0)
        if args.norm_reward:
            r_std_ema = 0.999 * r_std_ema + 0.001 * float(np.mean(r ** 2))
            r = np.clip(r / (np.sqrt(r_std_ema) + 1e-8), -10.0, 10.0)
        rb.add(o, m, kpm, sh, a, r.astype(np.float32))
        if step % cfg.ep_len == 0:
            env.reset()

        if rb.size >= args.batch and step >= args.warmup:
            o_, m_, k_, s_, a_, r_ = rb.sample(args.batch)
            B = o_.shape[0]
            alpha = log_alpha.exp().detach()

            # ---- critic: gamma=0 → y = r (no bootstrap) ----
            loss_c = 0.0
            for i in range(n_bs):
                oh = OH[i][None].expand(B, -1)
                q1, q2 = critic(s_, a_, oh)
                loss_c = loss_c + F.mse_loss(q1, r_[:, i]) + F.mse_loss(q2, r_[:, i])
            opt_c.zero_grad(); loss_c.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
            opt_c.step()

            # ---- sequential HAML actor update (z frozen; encoder updated after) ----
            z_b = encoder(k_).detach() if encoder else None
            with torch.no_grad():
                a_cur = []
                for j in range(n_bs):
                    actor = actors[j if n_actors > 1 else 0]
                    zj = z_b[:, j] if z_b is not None else None
                    aj, _ = actor.sample(o_, m_[:, j], zj)
                    a_cur.append(aj)
            logps = [None] * n_bs
            for i in torch.randperm(n_bs).tolist():
                actor = actors[i if n_actors > 1 else 0]
                zi = z_b[:, i] if z_b is not None else None
                ai, logp_i = actor.sample(o_, m_[:, i], zi)
                a_joint = torch.stack(
                    [ai if j == i else a_cur[j] for j in range(n_bs)], dim=1)
                oh = OH[i][None].expand(B, -1)
                q1, q2 = critic(s_, a_joint, oh)
                loss_i = (alpha * logp_i - torch.min(q1, q2)).mean()
                if args.act_reg > 0:
                    loss_i = loss_i + args.act_reg * (ai ** 2).mean()
                oa = opt_a[i if n_actors > 1 else 0]
                oa.zero_grad(); loss_i.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                oa.step()
                with torch.no_grad():                          # already-updated action, held fixed
                    a_cur[i], _ = actor.sample(o_, m_[:, i], zi)
                logps[i] = logp_i.detach()

            # ---- alpha (one update per step) ----
            avg_logp = torch.stack(logps, dim=1).mean()
            loss_alpha = -(log_alpha * (avg_logp + target_H))
            opt_alpha.zero_grad(); loss_alpha.backward(); opt_alpha.step()
            if args.alpha_min > 0:
                with torch.no_grad():
                    log_alpha.clamp_(min=float(np.log(args.alpha_min)))

            # ---- encoder: one end-to-end update through actor loss (live z) ----
            if encoder:
                z_live = encoder(k_)
                loss_e = 0.0
                for i in range(n_bs):
                    actor = actors[i if n_actors > 1 else 0]
                    ai, logp_i = actor.sample(o_, m_[:, i], z_live[:, i])
                    a_joint = torch.stack(
                        [ai if j == i else a_cur[j] for j in range(n_bs)], dim=1)
                    oh = OH[i][None].expand(B, -1)
                    q1, q2 = critic(s_, a_joint, oh)
                    loss_e = loss_e + (alpha * logp_i - torch.min(q1, q2)).mean()
                opt_enc.zero_grad()
                for oa in opt_a:
                    oa.zero_grad()
                loss_e.backward()
                nn.utils.clip_grad_norm_(encoder.parameters(), 10.0)
                opt_enc.step()

        if step % args.eval_every == 0:
            gp, gps = eval_policy(actors, encoder, cfg, args.use_z, seed0=10000 + args.seed)
            tag = ""
            if gp > best:
                best = gp
                best_state = dict(
                    actors=[copy.deepcopy(a.state_dict()) for a in actors],
                    encoder=copy.deepcopy(encoder.state_dict()) if encoder else None)
                tag = " *"
            a_now = float(log_alpha.exp())
            logp(f"step {step:>6} | goodput {gp:7.3f} ± {gps:5.3f} ({pct(gp):5.1f}% gap) | "
                 f"alpha {a_now:6.4f} | best {best:7.3f} | {time.time()-t0:5.0f}s{tag}")

    # ---- final ----
    if best_state is not None:
        for a, sd in zip(actors, best_state["actors"]):
            a.load_state_dict(sd)
        if encoder and best_state["encoder"]:
            encoder.load_state_dict(best_state["encoder"])
    gp, gps = eval_policy(actors, encoder, cfg, args.use_z, n_trials=6, T=150,
                          seed0=20000)
    logp("\n=== FINAL (best ckpt, eval seeds 20000+) ===")
    logp(f"goodput          {gp:7.3f} ± {gps:.3f}   ({pct(gp):.1f}% of gap)")
    logp(f"floor (equal)    {floor:7.3f}")
    logp(f"ceiling (oracle) {ceil:7.3f}")
    np.save(args.out, dict(goodput=gp, floor=floor, ceiling=ceil, pct=pct(gp), best=best),
            allow_pickle=True)
    torch.save(best_state, f"results/{args.tag}_best.pt")
    logf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rl", "gate", "combine"], default="rl")
    ap.add_argument("--bc_iters", type=int, default=3000)
    ap.add_argument("--gate_trials", type=int, default=60)
    ap.add_argument("--combine", choices=["learnable", "fixed"], default="learnable")
    ap.add_argument("--gate_ckpt", default="results/rep_gate_s0_best.pt")
    ap.add_argument("--gate_hidden", type=int, default=256)
    ap.add_argument("--traffic", choices=["full", "bursty"], default="full",
                    help="full-buffer (always hot) or bursty hot/cold Markov (spec §2.2)")
    ap.add_argument("--freeze_gate", type=int, default=1,
                    help="combine mode: 1 = spatial gate frozen during RL refine (spec hold-arm); "
                         "0 = gate also trainable (RL destroys it — stronger negative result)")
    ap.add_argument("--geom_file", default="",
                    help="path to authors' geometry npz (g_ls/serv/bs/ue/N0/Pmax); "
                         "loads exact placement, skipping local RNG draws")
    ap.add_argument("--use_z", type=int, default=0)
    ap.add_argument("--actor", choices=["shared", "separate"], default="separate")
    ap.add_argument("--reward", choices=["team", "difference"], default="team")
    ap.add_argument("--fixed_topology", type=int, default=1)
    ap.add_argument("--topo_seed", type=int, default=12345)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--replay", type=int, default=1000000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--z_dim", type=int, default=16)
    ap.add_argument("--alpha_init", type=float, default=0.1)
    ap.add_argument("--alpha_min", type=float, default=0.0, help="0 = pure HASAC")
    ap.add_argument("--act_reg", type=float, default=0.0, help="0 = pure HASAC")
    ap.add_argument("--norm_reward", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="rep")
    args = ap.parse_args()
    args.log = f"results/{args.tag}_log.txt"
    args.out = f"results/{args.tag}_result.npy"
    if args.mode == "gate":
        train_gate(args)
    elif args.mode == "combine":
        train_combine(args)
    else:
        train(args)
