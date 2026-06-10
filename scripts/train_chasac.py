"""
C-HASAC / vanilla HASAC 訓練 (對應 HANDOFF_CHASAC_IMPL.md §5–§6, §10 步驟 3–4)

唯一真相來源 = env_chasac.Env (channel->SINR->rate->difference reward).
HASAC 與 C-HASAC 的「唯一差別」= actor 有沒有吃 encoder 學出的 z (--use_z).
reward / critic / 演算法 / 超參完全一致 (HANDOFF 原則 #7).

設計:
  Actor   : parameter-shared, permutation-equivariant set head over UE.
            對全 N_UE 一次前向, 用 membership mask 分 BS 做 intra-cell 池化.
            squashed Gaussian (reparam) -> raw a in (-1,1); env 投影到 sum-power<=Pmax.
  Critic  : agent-conditioned twin-Q. Q(share_obs, joint_a, onehot_i) -> scalar.
            critic 用 share_obs (已含全域), **不吃 z** (HANDOFF #6).
  Encoder : z = f(kpm[N_BS,3]); per-cell MLP -> mean pool (permutation-invariant) -> z.
            只餵 actor; 梯度經由 actor loss 回傳訓練 encoder (HANDOFF §6.1).
  alpha   : 自動 entropy tuning, target H = -|A|/N_BS.

用法:
  python scripts/train_chasac.py --use_z 0 --steps 100000 --tag hasac
  python scripts/train_chasac.py --use_z 1 --steps 100000 --tag chasac
"""
import os, sys, time, argparse, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_chasac as E

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


# --------------------------- networks -----------------------------
def mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]),
                   act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


def mlp_ln(sizes, act=nn.ReLU, out_act=nn.Identity):
    """MLP with LayerNorm after each hidden activation — stabilises Q-value scale."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
            layers.append(nn.LayerNorm(sizes[i + 1]))
        else:
            layers.append(out_act())
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """z = f({kpm_c}); per-cell MLP -> masked mean pool (permutation-invariant)."""
    def __init__(self, kpm_dim=3, hidden=128, z_dim=16):
        super().__init__()
        self.phi = mlp([kpm_dim, hidden, hidden])
        self.rho = mlp([hidden, hidden, z_dim])

    def forward(self, kpm):                 # kpm: [B, N_BS, kpm_dim]
        h = self.phi(kpm)                   # [B, N_BS, hidden]
        pooled = h.mean(dim=1)              # permutation-invariant
        return self.rho(pooled)            # [B, z_dim]


def encode_kpm(enc, kpm, n_bs, remove_own=False):
    """kpm: [B, N_BS, kpm_dim].
    remove_own=False → global z [B, z_dim].
    remove_own=True  → per-BS z [B, N_BS, z_dim], each BS i sees only neighbors' KPM.
    """
    if not remove_own:
        return enc(kpm)
    z_list = []
    for i in range(n_bs):
        others = [j for j in range(n_bs) if j != i]
        z_list.append(enc(kpm[:, others, :]))   # [B, z_dim]
    return torch.stack(z_list, dim=1)           # [B, N_BS, z_dim]


class SetActor(nn.Module):
    """
    permutation-equivariant per-BS actor (parameter-shared).
    對全 N_UE 一次前向; mask[B,N_BS,N_UE] 標記 UE 歸屬, 用來做 intra-cell 池化 + log_prob 分組.
    C-HASAC: 把 global z broadcast concat 到每個 UE embedding.
    """
    def __init__(self, ue_feat=3, hidden=256, z_dim=0):
        super().__init__()
        self.enc = mlp([ue_feat, hidden, hidden])      # per-UE embedding
        head_in = hidden * 2 + z_dim                   # emb + intra-cell ctx [+ z]
        self.head = mlp([head_in, hidden, hidden])
        self.mu = nn.Linear(hidden, 1)
        self.log_std = nn.Linear(hidden, 1)
        self.z_dim = z_dim
        self.mu_bound = 0.0          # >0: mu = mu_bound*tanh(mu_raw), prevents tanh saturation collapse

    def forward(self, o, mask, z=None):
        # o: [B,N_UE,ue_feat]; mask: [B,N_BS,N_UE]; z: [B,z_dim] or None
        emb = self.enc(o)                              # [B,N_UE,H]
        cnt = mask.sum(-1, keepdim=True).clamp_min(1.) # [B,N_BS,1]
        ctx_bs = torch.einsum("biu,buh->bih", mask, emb) / cnt   # [B,N_BS,H]
        ue_ctx = torch.einsum("biu,bih->buh", mask, ctx_bs)      # [B,N_UE,H]
        feat = torch.cat([emb, ue_ctx], dim=-1)
        if self.z_dim > 0:
            if z.dim() == 3:  # [B, N_BS, z_dim] — per-BS z; route each UE to its BS's z
                zb = torch.einsum("biu,biz->buz", mask, z)
            else:             # [B, z_dim] — global z broadcast
                zb = z[:, None, :].expand(-1, o.shape[1], -1)
            feat = torch.cat([feat, zb], dim=-1)
        h = self.head(feat)
        mu = self.mu(h).squeeze(-1)                    # [B,N_UE]
        if self.mu_bound > 0:
            mu = self.mu_bound * torch.tanh(mu)        # bound mean -> no -inf saturation

        log_std = self.log_std(h).squeeze(-1).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, o, mask, z=None):
        mu, log_std = self.forward(o, mask, z)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        a = torch.tanh(x)                              # raw action in (-1,1)
        # tanh-corrected log prob, per UE
        logp_u = dist.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)   # [B,N_UE]
        # 分 BS 加總 -> per-BS log prob [B,N_BS]
        logp_bs = torch.einsum("biu,bu->bi", mask, logp_u)
        return a, logp_bs

    @torch.no_grad()
    def act(self, o, mask, z=None, deterministic=True):
        mu, log_std = self.forward(o, mask, z)
        return torch.tanh(mu) if deterministic else torch.tanh(
            torch.distributions.Normal(mu, log_std.exp()).sample())


class Critic(nn.Module):
    """agent-conditioned twin-Q. 輸入 [share_obs, joint_a, onehot_i]."""
    def __init__(self, share_dim, act_dim, n_bs, hidden=256, layer_norm=False):
        super().__init__()
        d = share_dim + act_dim + n_bs
        build = mlp_ln if layer_norm else mlp
        self.q1 = build([d, hidden, hidden, 1])
        self.q2 = build([d, hidden, hidden, 1])

    def forward(self, share, a, onehot):
        x = torch.cat([share, a, onehot], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


# --------------------------- replay -------------------------------
class Replay:
    def __init__(self, cap, n_ue, n_bs, ue_feat, kpm_dim, share_dim):
        self.cap, self.ptr, self.size = cap, 0, 0
        self.o   = np.zeros((cap, n_ue, ue_feat), np.float32)
        self.mask = np.zeros((cap, n_bs, n_ue), np.float32)
        self.kpm = np.zeros((cap, n_bs, kpm_dim), np.float32)
        self.sh  = np.zeros((cap, share_dim), np.float32)
        self.a   = np.zeros((cap, n_ue), np.float32)
        self.r   = np.zeros((cap, n_bs), np.float32)
        self.no  = np.zeros_like(self.o)
        self.nmask = np.zeros_like(self.mask)
        self.nkpm = np.zeros_like(self.kpm)
        self.nsh = np.zeros_like(self.sh)
        self.d   = np.zeros((cap,), np.float32)   # done: episode boundary, no bootstrap

    def add(self, o, mask, kpm, sh, a, r, no, nmask, nkpm, nsh, done):
        i = self.ptr
        self.o[i], self.mask[i], self.kpm[i], self.sh[i] = o, mask, kpm, sh
        self.a[i], self.r[i] = a, r
        self.no[i], self.nmask[i], self.nkpm[i], self.nsh[i] = no, nmask, nkpm, nsh
        self.d[i] = done
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, bs):
        idx = np.random.randint(0, self.size, size=bs)
        t = lambda x: torch.as_tensor(x[idx], device=DEVICE)
        return (t(self.o), t(self.mask), t(self.kpm), t(self.sh), t(self.a),
                t(self.r), t(self.no), t(self.nmask), t(self.nkpm), t(self.nsh),
                t(self.d))


class NStepBuffer:
    """Accumulates n transitions and computes n-step discounted returns before adding to Replay.
    At episode boundary call flush() to drain remaining transitions with truncated returns."""
    def __init__(self, n, gamma, n_bs):
        self.n = n
        self.gamma = gamma
        self.n_bs = n_bs
        self.buf = []  # list of (o, mask, kpm, sh, a, r, no, nmask, nkpm, nsh)

    def _make_transition(self, start, end):
        o, mask, kpm, sh, a = self.buf[start][:5]
        no, nmask, nkpm, nsh = self.buf[end - 1][6:10]
        done = self.buf[end - 1][10]
        r_n = np.zeros(self.n_bs, np.float32)
        for k in range(end - start):
            r_n += (self.gamma ** k) * self.buf[start + k][5]
        return (o, mask, kpm, sh, a, r_n, no, nmask, nkpm, nsh, done)

    def add(self, transition):
        """Add one transition; returns a ready (n-step) transition or None."""
        self.buf.append(transition)
        if len(self.buf) >= self.n:
            t = self._make_transition(0, self.n)
            self.buf.pop(0)
            return t
        return None

    def flush(self):
        """Drain remaining transitions (episode end) with truncated n-step returns."""
        results = []
        while self.buf:
            results.append(self._make_transition(0, len(self.buf)))
            self.buf.pop(0)
        return results


# --------------------------- helpers ------------------------------
def pad_local(local, n_ue, ue_feat):
    """env 的 local (list per BS of [n_ue_i,F]) -> full [n_ue,F] (依 serv 順序) + 我們用 serv 建 mask."""
    # 我們改用 serv 直接組 o_full / mask, 見 build_obs
    raise NotImplementedError


def build_obs(env, use_rsrp=False):
    """從 env 內部狀態組出 actor 輸入: o_full[N_UE, ue_feat], mask, kpm, share.
    use_rsrp=True: 在 ue_feat 附加每個 BS→UE 的 channel gain（正規化），ue_feat 3→3+N_BS。"""
    cfg = env.cfg
    w = env._weights()
    rate, sinr, P_bs = E.rates_from_power(env.p, env.g, env.serv, cfg.N_BS)
    o_full = np.stack([rate, w, env.p], axis=1).astype(np.float32)      # [N_UE,3]
    if use_rsrp:
        # g[N_BS, N_UE] → normalize by global max → append [N_UE, N_BS]
        g_norm = (env.g / (env.g.max() + 1e-9)).T.astype(np.float32)   # [N_UE, N_BS]
        o_full = np.concatenate([o_full, g_norm], axis=1)               # [N_UE, 3+N_BS]
    mask = np.zeros((cfg.N_BS, cfg.N_UE), np.float32)
    mask[env.serv, np.arange(cfg.N_UE)] = 1.0
    kpm = E.obs_kpm(env.p, env.g, env.serv, cfg.N_BS, env.bs).astype(np.float32)
    share = E.obs_share(env.p, env.g, env.serv, env.bs).astype(np.float32)
    return o_full, mask, kpm, share


def _oracle_kpm_np(env, cfg, pmax, grid=3):
    """Per-BS oracle power fractions [N_BS, 1] to store in replay kpm field (oracle_z mode).
    grid=3 (default) is fast (27 combos); use cfg.ceiling_grid only for final eval."""
    w = env._weights()
    p_o = E.pf_wsr_ceiling(env.g, env.serv, w, cfg.N_BS, grid)
    bs_pwr = np.array([p_o[env.serv == b].sum() for b in range(cfg.N_BS)], np.float32)
    return (bs_pwr / (pmax + 1e-9))[:, None]  # [N_BS, 1]


def action_to_powerlist(a, serv, n_bs, pmax):
    """raw a in (-1,1) -> 每 BS 的 desired power list (env._project 會再投影到 sum<=Pmax)."""
    frac = (a + 1.0) / 2.0                       # [0,1]
    desired = frac * pmax
    return [desired[serv == i] for i in range(n_bs)]


def onehots(n_bs):
    return torch.eye(n_bs, device=DEVICE)


# --------------------------- eval ---------------------------------
@torch.no_grad()
def eval_policy(actor, encoder, cfg, n_eval=20, T=10, seed=2024, zero_z=False, shuffle_z=False,
                use_rsrp=False, remove_own_kpm=False, oracle_z=False):
    """canonical PF utility U = Σ_u log(R̄_u); 在 held-out scenarios 上跑 policy."""
    pmax = E.dbm_to_w(cfg.Pmax_dBm)
    n_bs = cfg.N_BS
    rng = np.random.default_rng(seed)
    has_z = (encoder is not None) or oracle_z

    # Pre-collect z from n_eval DIFFERENT scenarios for shuffle_z.
    # Episode i receives z from scenario (i+1)%n_eval — guaranteed cross-scenario mismatch.
    z_wrong = None
    if shuffle_z and has_z:
        rng_sh = np.random.default_rng(seed + 54321)
        z_wrong = []
        for _ in range(n_eval):
            env_sh = E.Env(cfg, reward_mode="difference", seed=int(rng_sh.integers(1 << 30)))
            env_sh.reset()
            ep_z = []
            for _t in range(T):
                if oracle_z:
                    oz = _oracle_kpm_np(env_sh, cfg, pmax, grid=cfg.ceiling_grid)
                    z_sh = torch.as_tensor(oz[:, 0][None], device=DEVICE)  # [1, N_BS]
                else:
                    _, _, kpm_sh, _ = build_obs(env_sh, use_rsrp=use_rsrp)
                    z_sh = encode_kpm(encoder, torch.as_tensor(kpm_sh[None], device=DEVICE),
                                       n_bs, remove_own_kpm)
                ep_z.append(z_sh.clone())
                p_eq = E.bl_equal_power(env_sh.g, env_sh.serv, env_sh._weights(), n_bs)
                env_sh.step([p_eq[env_sh.serv == i] for i in range(n_bs)])
            z_wrong.append(ep_z)
        z_wrong = z_wrong[1:] + [z_wrong[0]]   # rotate: ep i gets z from scenario (i+1)%n_eval

    Us = []
    for ep_idx in range(n_eval):
        env = E.Env(cfg, reward_mode="difference", seed=int(rng.integers(1 << 30)))
        env.reset()
        rate_sum = np.zeros(cfg.N_UE)
        for t in range(T):
            o, mask, kpm, sh = build_obs(env, use_rsrp=use_rsrp)
            ot = torch.as_tensor(o[None], device=DEVICE)
            mt = torch.as_tensor(mask[None], device=DEVICE)
            z = None
            if has_z:
                if oracle_z:
                    oz = _oracle_kpm_np(env, cfg, pmax, grid=cfg.ceiling_grid)
                    z = torch.as_tensor(oz[:, 0][None], device=DEVICE)  # [1, N_BS]
                else:
                    z = encode_kpm(encoder, torch.as_tensor(kpm[None], device=DEVICE),
                                    n_bs, remove_own_kpm)
                if zero_z:
                    z = torch.zeros_like(z)
                elif shuffle_z:
                    z = z_wrong[ep_idx][t]
            a = actor.act(ot, mt, z, deterministic=True)[0].cpu().numpy()
            pl = action_to_powerlist(a, env.serv, n_bs, pmax)
            _, _, _, info = env.step(pl)
            rate_sum += info["rate"]
        Us.append(np.log(rate_sum / T + 1e-6).sum())
    return float(np.mean(Us)), float(np.std(Us))


@torch.no_grad()
def eval_power_frac(actor, encoder, cfg, n=5, T=10, seed=2024, use_rsrp=False, remove_own_kpm=False,
                    oracle_z=False):
    """diagnostic: mean deterministic power fraction (a+1)/2; ~0 => actor saturated to zero-power."""
    pmax = E.dbm_to_w(cfg.Pmax_dBm); rng = np.random.default_rng(seed); fr = []
    for _ in range(n):
        env = E.Env(cfg, reward_mode="difference", seed=int(rng.integers(1 << 30))); env.reset()
        for _t in range(T):
            o, mask, kpm, sh = build_obs(env, use_rsrp=use_rsrp)
            if oracle_z:
                oz = _oracle_kpm_np(env, cfg, pmax, grid=cfg.ceiling_grid)
                z = torch.as_tensor(oz[:, 0][None], device=DEVICE)
            else:
                z = (encode_kpm(encoder, torch.as_tensor(kpm[None], device=DEVICE),
                                cfg.N_BS, remove_own_kpm) if encoder else None)
            a = actor.act(torch.as_tensor(o[None], device=DEVICE),
                          torch.as_tensor(mask[None], device=DEVICE), z, deterministic=True)[0].cpu().numpy()
            fr.append(float(((a + 1.0) / 2.0).mean()))
            env.step(action_to_powerlist(a, env.serv, cfg.N_BS, pmax))
    return float(np.mean(fr))


@torch.no_grad()
def eval_baseline(fn, cfg, n_eval=20, T=10, seed=2024):
    pmax = E.dbm_to_w(cfg.Pmax_dBm)
    rng = np.random.default_rng(seed)
    Us = []
    for _ in range(n_eval):
        env = E.Env(cfg, reward_mode="difference", seed=int(rng.integers(1 << 30)))
        env.reset()
        rate_sum = np.zeros(cfg.N_UE)
        for t in range(T):
            w = env._weights()
            p = fn(env.g, env.serv, w, cfg.N_BS)
            pl = [p[env.serv == i] for i in range(cfg.N_BS)]
            _, _, _, info = env.step(pl)
            rate_sum += info["rate"]
        Us.append(np.log(rate_sum / T + 1e-6).sum())
    return float(np.mean(Us)), float(np.std(Us))


# --------------------------- BC pretrain --------------------------
def _bc_dataset(cfg, n_data, logp, seed=777, cache="results/bc_dataset.npz", use_rsrp=False,
                oracle_z=False):
    """一次性生成 BC 資料集 (expert=pf_wsr_ceiling); 依 (cfg, n_data, seed, use_rsrp) 快取到磁碟。
    oracle_z=True: K stores per-BS oracle power fracs [N_BS, 1] instead of KPM [N_BS, kpm_dim]."""
    pmax = E.dbm_to_w(cfg.Pmax_dBm)
    if oracle_z:
        cache = cache.replace(".npz", "_oracle.npz")
        kpm_dim_now = 1
    else:
        kpm_dim_now = 3 + cfg.N_BS - 1
    if os.path.exists(cache):
        d = np.load(cache)
        cached_kpm_dim = int(d["kpm_dim"]) if "kpm_dim" in d else 3
        cached_rsrp = bool(d["use_rsrp"]) if "use_rsrp" in d else False
        if (int(d["n_data"]) == n_data and int(d["seed"]) == seed
                and cached_kpm_dim == kpm_dim_now and cached_rsrp == use_rsrp):
            logp(f"[BC] dataset cache hit ({cache}, n={n_data})")
            return d["O"], d["M"], d["K"], d["A"]
    logp(f"[BC] building dataset n={n_data} (pf_wsr_ceiling expert, oracle_z={oracle_z})...")
    rng = np.random.default_rng(seed)
    O, M, K, A = [], [], [], []
    for _ in range(n_data):
        env = E.Env(cfg, reward_mode="difference", seed=int(rng.integers(1 << 30)))
        env.reset()
        w = env._weights()
        p_exp = E.pf_wsr_ceiling(env.g, env.serv, w, cfg.N_BS, cfg.ceiling_grid)
        o, mask, kpm, _ = build_obs(env, use_rsrp=use_rsrp)
        a_exp = np.clip(2.0 * (p_exp / pmax) - 1.0, -0.999, 0.999).astype(np.float32)
        if oracle_z:
            bs_pwr = np.array([p_exp[env.serv == b].sum() for b in range(cfg.N_BS)], np.float32)
            kpm = (bs_pwr / (pmax + 1e-9))[:, None]   # [N_BS, 1] oracle power fracs
        O.append(o); M.append(mask); K.append(kpm); A.append(a_exp)
    O, M, K, A = map(lambda x: np.stack(x).astype(np.float32), (O, M, K, A))
    np.savez(cache, O=O, M=M, K=K, A=A, n_data=n_data, seed=seed,
             kpm_dim=kpm_dim_now, use_rsrp=use_rsrp)
    logp(f"[BC] dataset built + cached -> {cache}")
    return O, M, K, A


def _bc_critic_dataset(cfg, n_eps, logp, seed=778, use_rsrp=False):
    """Collect expert (PF-WSR) full-episode rollouts for critic pre-training.
    Returns S[N, share_dim], A[N, N_UE], R[N, N_BS] (Monte Carlo returns)."""
    cache = "results/bc_critic_dataset.npz"
    kpm_dim_now = 3 + cfg.N_BS - 1
    if os.path.exists(cache):
        d = np.load(cache)
        cached_rsrp = bool(d["use_rsrp"]) if "use_rsrp" in d else False
        if (int(d["n_eps"]) == n_eps and int(d["seed"]) == seed
                and int(d["kpm_dim"]) == kpm_dim_now and cached_rsrp == use_rsrp):
            logp(f"[BC-Q] dataset cache hit ({cache})")
            return d["S"], d["A"], d["R"]
    logp(f"[BC-Q] building critic dataset n_eps={n_eps}...")
    pmax = E.dbm_to_w(cfg.Pmax_dBm)
    rng = np.random.default_rng(seed)
    ALL_S, ALL_A, ALL_R = [], [], []
    ep_len = 10
    gamma = 0.99
    for _ in range(n_eps):
        env = E.Env(cfg, reward_mode="logpf", seed=int(rng.integers(1 << 30)))
        env.reset()
        ep_buf = []
        for _ in range(ep_len):
            w = env._weights()
            p_exp = E.pf_wsr_ceiling(env.g, env.serv, w, cfg.N_BS, cfg.ceiling_grid)
            _, _, _, share = build_obs(env, use_rsrp=use_rsrp)
            a_exp = np.clip(2.0 * (p_exp / pmax) - 1.0, -0.999, 0.999).astype(np.float32)
            pl = action_to_powerlist(a_exp, env.serv, cfg.N_BS, pmax)
            _, r, _, _ = env.step(pl)
            ep_buf.append((share.copy(), a_exp.copy(), r.astype(np.float32)))
        G = np.zeros(cfg.N_BS, np.float32)
        for share, a, r in reversed(ep_buf):
            G = r + gamma * G
            ALL_S.append(share); ALL_A.append(a); ALL_R.append(G.copy())
    S = np.stack(ALL_S).astype(np.float32)
    A = np.stack(ALL_A).astype(np.float32)
    R = np.stack(ALL_R).astype(np.float32)
    np.savez(cache, S=S, A=A, R=R, n_eps=n_eps, seed=seed,
             kpm_dim=kpm_dim_now, use_rsrp=use_rsrp)
    logp(f"[BC-Q] dataset built + cached -> {cache} ({S.shape[0]} transitions)")
    return S, A, R


def bc_pretrain_critic(critic, critic_t, opt_c, cfg, steps, batch, logp,
                       n_eps=300, seed=778, use_rsrp=False):
    """Pre-train critic Q to Monte Carlo returns from PF-WSR expert rollouts."""
    S, A, R = _bc_critic_dataset(cfg, n_eps, logp, seed=seed, use_rsrp=use_rsrp)
    n_bs = cfg.N_BS
    OH = onehots(n_bs)
    St = torch.as_tensor(S, device=DEVICE)
    At = torch.as_tensor(A, device=DEVICE)
    Rt = torch.as_tensor(R, device=DEVICE)
    logp(f"[BC-Q] start | steps={steps} batch={batch} n_transitions={St.shape[0]}")
    for it in range(1, steps + 1):
        idx = torch.randint(0, St.shape[0], (batch,), device=DEVICE)
        loss_c = 0.0
        for i in range(n_bs):
            oh = OH[i][None].expand(batch, -1)
            q1, q2 = critic(St[idx], At[idx], oh)
            target = Rt[idx, i]
            loss_c = loss_c + F.mse_loss(q1, target) + F.mse_loss(q2, target)
        opt_c.zero_grad(); loss_c.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        opt_c.step()
        if it == 1 or it % 200 == 0:
            logp(f"[BC-Q] it {it:>5}/{steps} | loss {loss_c.item():.4f}")
    with torch.no_grad():
        critic_t.load_state_dict(critic.state_dict())
    logp("[BC-Q] critic_t synced")


def bc_pretrain(actor, encoder, params, cfg, steps, batch, lr, logp,
                n_data=3000, seed=777, use_rsrp=False, remove_own_kpm=False, oracle_z=False):
    """
    用 full-CSI PF-WSR ceiling 當 expert, 監督式 warm-start actor (+encoder).
    oracle_z=True: use K[:,:,0] as z directly (no encoder); actor learns to map oracle z → expert action.
    """
    O, M, K, A = _bc_dataset(cfg, n_data, logp, seed=seed, use_rsrp=use_rsrp, oracle_z=oracle_z)
    O, M, K, A = (torch.as_tensor(x, device=DEVICE) for x in (O, M, K, A))
    opt = torch.optim.Adam(params, lr=lr)
    logp(f"[BC] start | steps={steps} batch={batch} n_data={n_data} oracle_z={oracle_z}")
    for it in range(1, steps + 1):
        idx = torch.randint(0, O.shape[0], (batch,), device=DEVICE)
        if oracle_z:
            z = K[idx][:, :, 0]                        # [B, N_BS] oracle power fracs
        else:
            z = encode_kpm(encoder, K[idx], cfg.N_BS, remove_own_kpm) if encoder else None
        mu, _ = actor.forward(O[idx], M[idx], z)
        pred = torch.tanh(mu)                          # deterministic action
        loss = F.mse_loss(pred, A[idx])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        if it == 1 or it % 200 == 0:
            logp(f"[BC] it {it:>5}/{steps} | MSE {loss.item():.6f}")


# --------------------------- train --------------------------------
def train(args):
    cfg = E.Cfg()
    cfg.walk_speed = args.walk_speed   # UE randomwalk; 0 = static (default)
    pmax = E.dbm_to_w(cfg.Pmax_dBm)
    n_bs, n_ue = cfg.N_BS, cfg.N_UE
    ue_feat = 3 + (cfg.N_BS if args.use_rsrp else 0)   # +N_BS if RSRP_neighbor enabled
    kpm_dim = 3 + cfg.N_BS - 1      # 3 KPM + (N_BS-1) inter-BS distances = 5 for N_BS=3
    kpm_dim_buf = 1 if args.oracle_z else kpm_dim  # replay buffer kpm field (oracle: 1 float per BS)
    n_bs_pairs = cfg.N_BS * (cfg.N_BS - 1) // 2
    share_dim = n_bs * n_ue + n_ue + n_ue + n_bs_pairs  # g(36)+p(12)+serv(12)+dist(3)=63
    if args.oracle_z:
        z_dim = n_bs   # oracle z = per-BS power fractions [N_BS]
    else:
        z_dim = args.z_dim if args.use_z else 0

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    actor = SetActor(ue_feat, args.hidden, z_dim).to(DEVICE)
    actor.mu_bound = args.mu_bound
    critic = Critic(share_dim, n_ue, n_bs, args.hidden, layer_norm=args.critic_ln).to(DEVICE)
    critic_t = Critic(share_dim, n_ue, n_bs, args.hidden, layer_norm=args.critic_ln).to(DEVICE)
    critic_t.load_state_dict(critic.state_dict())
    encoder = Encoder(kpm_dim, 128, z_dim).to(DEVICE) if (args.use_z and not args.oracle_z) else None

    actor_only_params = list(actor.parameters())
    encoder_params = list(encoder.parameters()) if encoder else []
    actor_lr = args.actor_lr if args.actor_lr > 0 else args.lr
    opt_a = torch.optim.Adam(actor_only_params, lr=actor_lr)
    opt_enc = torch.optim.Adam(encoder_params, lr=actor_lr) if encoder_params else None
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr)
    log_alpha = torch.tensor(np.log(args.alpha_init), requires_grad=True, device=DEVICE)
    opt_alpha = torch.optim.Adam([log_alpha], lr=args.lr)
    target_H = -float(n_ue) / n_bs                  # per-BS target entropy

    rb = Replay(args.replay, n_ue, n_bs, ue_feat, kpm_dim_buf, share_dim)
    nsb = NStepBuffer(args.n_step, args.gamma, n_bs) if args.n_step > 1 else None
    gamma_n = args.gamma ** args.n_step   # discount for bootstrapped Q target
    OH = onehots(n_bs)                              # [N_BS,N_BS]
    env = E.Env(cfg, reward_mode=args.reward, seed=args.seed)
    env.reset()

    logf = open(args.log, "w")
    def logp(*a):
        s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush()

    logp(f"# C-HASAC train | use_z={args.use_z} oracle_z={args.oracle_z} reward={args.reward} steps={args.steps} "
         f"z_dim={z_dim} bc_steps={args.bc_steps} alpha_init={args.alpha_init} n_step={args.n_step} device={DEVICE}")

    # ---- BC warm-start (expert = PF-WSR full-CSI ceiling) ----
    if args.bc_steps > 0:
        bc_pretrain(actor, encoder, actor_only_params + encoder_params, cfg,
                    args.bc_steps, args.bc_batch, args.bc_lr, logp,
                    n_data=args.bc_data, use_rsrp=args.use_rsrp,
                    remove_own_kpm=args.remove_own_kpm, oracle_z=args.oracle_z)

    # ---- Critic BC warm-start (Monte Carlo returns from expert rollouts) ----
    if args.bc_critic_steps > 0:
        bc_pretrain_critic(critic, critic_t, opt_c, cfg,
                           args.bc_critic_steps, args.bc_batch, logp,
                           n_eps=args.bc_critic_eps, use_rsrp=args.use_rsrp)

    # ---- BC-anchor dataset (TD3+BC style, keeps actor from collapsing to zero-power) ----
    bc_O = bc_M = bc_K = bc_A = None
    if args.bc_reg > 0:
        O_, M_, K_, A_ = _bc_dataset(cfg, args.bc_data, logp, use_rsrp=args.use_rsrp)
        bc_O = torch.as_tensor(O_, device=DEVICE); bc_M = torch.as_tensor(M_, device=DEVICE)
        bc_K = torch.as_tensor(K_, device=DEVICE); bc_A = torch.as_tensor(A_, device=DEVICE)
        logp(f"[BC-REG] anchor on n={bc_O.shape[0]} expert samples | lambda={args.bc_reg}")

    # ---- EMA (Polyak-averaged) deployment copy of actor/encoder ----
    ema_actor = ema_encoder = None
    if args.ema_decay > 0:
        ema_actor = copy.deepcopy(actor)
        for p in ema_actor.parameters():
            p.requires_grad_(False)
        if encoder is not None:
            ema_encoder = copy.deepcopy(encoder)
            for p in ema_encoder.parameters():
                p.requires_grad_(False)

    # ---- top-K checkpoint pool; FINAL re-ranks on a held-out validation seed ----
    best_pool = []
    def pool_push(U_, step_, src, actor_m, enc_m):
        if args.topk <= 0:
            return
        sd_a = {k: v.detach().cpu().clone() for k, v in actor_m.state_dict().items()}
        sd_e = ({k: v.detach().cpu().clone() for k, v in enc_m.state_dict().items()}
                if enc_m is not None else None)
        best_pool.append(dict(U=U_, step=step_, src=src, actor=sd_a, enc=sd_e))
        best_pool.sort(key=lambda c: -c["U"])
        del best_pool[args.topk:]

    best_U, best_state = -1e9, None
    t0 = time.time()
    ep_len = args.ep_len
    r_sq_ema = 1.0          # running E[r^2] for optional reward normalization
    q_dbg_mean = q_dbg_max = 0.0   # critic Q diagnostics for eval log
    for step in range(1, args.steps + 1):
        # curriculum: isolated pretrain phase (no inter-cell interference)
        if args.isolate_pretrain_steps > 0:
            was_isolated = env.isolated
            env.isolated = (step <= args.isolate_pretrain_steps)
            if was_isolated and not env.isolated:
                logp(f"[curriculum] step {step}: switching to full interference (isolate phase done)")
        o, mask, kpm, sh = build_obs(env, use_rsrp=args.use_rsrp)
        if args.oracle_z:
            kpm = _oracle_kpm_np(env, cfg, pmax, grid=args.oracle_grid)   # [N_BS, 1] oracle power fracs
        if step < args.warmup:
            a = np.random.uniform(-1, 1, size=n_ue).astype(np.float32)
        else:
            ot = torch.as_tensor(o[None], device=DEVICE)
            mt = torch.as_tensor(mask[None], device=DEVICE)
            if args.oracle_z:
                z = torch.as_tensor(kpm[:, 0][None], device=DEVICE)   # [1, N_BS]
            else:
                z = (encode_kpm(encoder, torch.as_tensor(kpm[None], device=DEVICE),
                                 n_bs, args.remove_own_kpm) if encoder else None)
            a = actor.act(ot, mt, z, deterministic=False)[0].cpu().numpy().astype(np.float32)

        pl = action_to_powerlist(a, env.serv, n_bs, pmax)
        _, r, _, info = env.step(pl)
        if args.power_pen > 0:
            pbs = np.array([float(np.sum(pl[i])) for i in range(n_bs)], dtype=np.float32) / pmax
            r = r - args.power_pen * pbs                # discourage interference-causing full power
        if args.norm_reward:
            r_sq_ema = 0.999 * r_sq_ema + 0.001 * float(np.mean(r ** 2))
            r = np.clip(r / (np.sqrt(r_sq_ema) + 1e-8), -10.0, 10.0)
        no, nmask, nkpm, nsh = build_obs(env, use_rsrp=args.use_rsrp)
        if args.oracle_z:
            nkpm = _oracle_kpm_np(env, cfg, pmax, grid=args.oracle_grid)  # [N_BS, 1] oracle for next state
        done = 1.0 if step % ep_len == 0 else 0.0   # episode boundary: PF weights reset, no future
        transition = (o, mask, kpm, sh, a, r.astype(np.float32), no, nmask, nkpm, nsh, done)
        if nsb is not None:
            t = nsb.add(transition)
            if t is not None:
                rb.add(*t)
        else:
            rb.add(*transition)

        if step % ep_len == 0:
            if nsb is not None:
                for t in nsb.flush():
                    rb.add(*t)
            env.reset()

        # ---- updates ----
        if rb.size >= args.batch and step >= args.warmup:
            o_, m_, k_, s_, a_, r_, no_, nm_, nk_, ns_, d_ = rb.sample(args.batch)
            alpha = log_alpha.exp().detach()
            B = o_.shape[0]

            # ---- critic (args.critic_updates steps; resample each pass) ----
            for _cu in range(args.critic_updates):
                if _cu > 0:
                    o_, m_, k_, s_, a_, r_, no_, nm_, nk_, ns_, d_ = rb.sample(args.batch)
                    B = o_.shape[0]
                with torch.no_grad():
                    if args.oracle_z:
                        nz = nk_[:, :, 0]   # [B, N_BS] oracle power fracs
                    else:
                        nz = encode_kpm(encoder, nk_, n_bs, args.remove_own_kpm) if encoder else None
                    na, nlogp = actor.sample(no_, nm_, nz)     # na:[B,N_UE], nlogp:[B,N_BS]
                    yq = []
                    for i in range(n_bs):
                        oh = OH[i][None].expand(B, -1)
                        q1, q2 = critic_t(ns_, na, oh)
                        qmin = torch.min(q1, q2)
                        yq.append(r_[:, i] + gamma_n * (1.0 - d_) * (qmin - alpha * nlogp[:, i]))
                    y = torch.stack(yq, dim=1)                  # [B,N_BS]
                    if args.q_clamp > 0:   # logpf per-BS return is bounded; out-of-range targets are noise
                        y = y.clamp(-args.q_clamp, args.q_clamp)
                loss_c = 0.0
                for i in range(n_bs):
                    oh = OH[i][None].expand(B, -1)
                    q1, q2 = critic(s_, a_, oh)
                    loss_c = loss_c + F.mse_loss(q1, y[:, i]) + F.mse_loss(q2, y[:, i])
                q_dbg_mean, q_dbg_max = float(q1.mean()), float(q1.max())   # diagnostic for eval log
                opt_c.zero_grad(); loss_c.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
                opt_c.step()

            # ---- actor: Sequential Soft Policy Decomposition (HASAC §3.3) ----
            # delayed policy update (TD3-style): actor/alpha/encoder only every actor_every
            # critic steps — sequential loop gives actor 3 backwards/step vs critic 1,
            # letting actor exploit Q faster than Q corrects.
            # z is FROZEN (detached) during sequential loop so encoder gets one consistent
            # gradient signal after all agents update, not N_BS conflicting signals.
            if step % args.actor_every == 0:
                if args.oracle_z:
                    z_frozen = k_[:, :, 0]   # [B, N_BS] — no encoder, no gradient needed
                elif encoder:
                    z_frozen = encode_kpm(encoder, k_, n_bs, args.remove_own_kpm).detach()
                else:
                    z_frozen = None
                logp_all_ordered = [None] * n_bs
                for i in torch.randperm(n_bs).tolist():
                    pa, plogp = actor.sample(o_, m_, z_frozen)   # re-sample with current policy
                    oh = OH[i][None].expand(B, -1)
                    q1, q2 = critic(s_, pa, oh)
                    qmin = torch.min(q1, q2)
                    loss_i = (alpha * plogp[:, i] - qmin).mean()
                    # TD3+BC anchor per agent (z also frozen for BC)
                    if args.bc_reg > 0 and not args.oracle_z:
                        bidx = torch.randint(0, bc_O.shape[0], (B,), device=DEVICE)
                        zb = encode_kpm(encoder, bc_K[bidx], n_bs, args.remove_own_kpm) if encoder else None
                        if zb is not None:
                            zb = zb.detach()
                        mub, _ = actor.forward(bc_O[bidx], bc_M[bidx], zb)
                        loss_i = loss_i + args.bc_reg * F.mse_loss(torch.tanh(mub), bc_A[bidx])
                    opt_a.zero_grad(); loss_i.backward()
                    nn.utils.clip_grad_norm_(actor_only_params, 10.0)
                    opt_a.step()
                    logp_i = plogp[:, i].detach()
                    logp_all_ordered[i] = logp_i
                # ---- alpha: ONE update per actor cycle (avg logp across all agents) ----
                avg_logp = torch.stack(logp_all_ordered, dim=1).mean(dim=1)  # [B]
                loss_alpha = -(log_alpha * (avg_logp + target_H)).mean()
                opt_alpha.zero_grad(); loss_alpha.backward(); opt_alpha.step()
                if args.alpha_min > 0:   # entropy floor: failed runs die at alpha 0.0005-0.0007
                    with torch.no_grad():
                        log_alpha.clamp_(min=float(np.log(args.alpha_min)))
                alpha = log_alpha.exp().detach()
                # ---- encoder: one joint update after all agents (live z, combined loss) ----
                if encoder:
                    z_live = encode_kpm(encoder, k_, n_bs, args.remove_own_kpm)
                    pa_e, plogp_e = actor.sample(o_, m_, z_live)
                    loss_enc = 0.0
                    for i in range(n_bs):
                        oh = OH[i][None].expand(B, -1)
                        q1, q2 = critic(s_, pa_e, oh)
                        loss_enc = loss_enc + (alpha * plogp_e[:, i] - torch.min(q1, q2)).mean()
                    opt_a.zero_grad(); opt_enc.zero_grad(); loss_enc.backward()
                    nn.utils.clip_grad_norm_(actor_only_params, 10.0)
                    nn.utils.clip_grad_norm_(encoder_params, 10.0)
                    opt_a.step(); opt_enc.step()

            # ---- polyak ----
            with torch.no_grad():
                for p, pt in zip(critic.parameters(), critic_t.parameters()):
                    pt.mul_(1 - args.tau).add_(args.tau * p)
                if ema_actor is not None:
                    eta = 1.0 - args.ema_decay
                    for pe, p in zip(ema_actor.parameters(), actor.parameters()):
                        pe.mul_(args.ema_decay).add_(eta * p)
                    if ema_encoder is not None:
                        for pe, p in zip(ema_encoder.parameters(), encoder.parameters()):
                            pe.mul_(args.ema_decay).add_(eta * p)

        # ---- logging / eval ----
        if step % args.eval_every == 0:
            U, Us = eval_policy(actor, encoder, cfg, n_eval=args.n_eval, T=ep_len,
                                use_rsrp=args.use_rsrp, remove_own_kpm=args.remove_own_kpm,
                                oracle_z=args.oracle_z)
            pool_push(U, step, "raw", actor, encoder)
            ema_str = ""
            if ema_actor is not None:
                U_ema, _ = eval_policy(ema_actor, ema_encoder, cfg, n_eval=args.n_eval, T=ep_len,
                                       use_rsrp=args.use_rsrp, remove_own_kpm=args.remove_own_kpm,
                                       oracle_z=args.oracle_z)
                pool_push(U_ema, step, "ema", ema_actor, ema_encoder)
                ema_str = f" | ema {U_ema:8.3f}"
            tag = ""
            if U > best_U:
                best_U = U
                best_state = {
                    "actor": {k: v.cpu().clone() for k, v in actor.state_dict().items()},
                    "encoder": ({k: v.cpu().clone() for k, v in encoder.state_dict().items()}
                                if encoder else None)}
                tag = " *"
                torch.save(best_state, f"results/{args.tag}_best.pt")
            a_now = float(log_alpha.exp().detach())
            pfrac = eval_power_frac(actor, encoder, cfg, use_rsrp=args.use_rsrp,
                                    remove_own_kpm=args.remove_own_kpm, oracle_z=args.oracle_z)
            logp(f"step {step:>7} | PF-U {U:8.3f} ± {Us:5.3f} | alpha {a_now:6.4f} | "
                 f"pwr {pfrac:5.3f} | Q {q_dbg_mean:7.2f}/{q_dbg_max:7.2f} | "
                 f"best {best_U:8.3f}{ema_str} | {time.time()-t0:5.0f}s{tag}")

    # ---- final: re-rank top-K ckpts on held-out validation seed, pick true winner ----
    if best_pool:
        logp(f"\n=== VALIDATION re-rank (top-{len(best_pool)} ckpts, 50 eps, seed 5151) ===")
        best_val, best_cand = -1e9, None
        for c in best_pool:
            actor.load_state_dict(c["actor"])
            if encoder is not None and c["enc"] is not None:
                encoder.load_state_dict(c["enc"])
            Uv, _ = eval_policy(actor, encoder, cfg, n_eval=50, T=ep_len, seed=5151,
                                use_rsrp=args.use_rsrp, remove_own_kpm=args.remove_own_kpm,
                                oracle_z=args.oracle_z)
            logp(f"  ckpt step {c['step']:>7} [{c['src']}] | train_U {c['U']:8.3f} | val_U {Uv:8.3f}")
            if Uv > best_val:
                best_val, best_cand = Uv, c
        actor.load_state_dict(best_cand["actor"])
        if encoder is not None and best_cand["enc"] is not None:
            encoder.load_state_dict(best_cand["enc"])
        logp(f"  -> selected step {best_cand['step']} [{best_cand['src']}] val_U {best_val:.3f}")
        torch.save(dict(actor=best_cand["actor"], encoder=best_cand["enc"]),
                   f"results/{args.tag}_best.pt")
    elif best_state is not None:
        actor.load_state_dict(best_state["actor"])
        if encoder and best_state["encoder"]:
            encoder.load_state_dict(best_state["encoder"])

    logp("\n=== FINAL (best ckpt) ===")
    U, Us = eval_policy(actor, encoder, cfg, n_eval=args.n_eval_final, T=ep_len,
                        use_rsrp=args.use_rsrp, remove_own_kpm=args.remove_own_kpm,
                        oracle_z=args.oracle_z)
    logp(f"{'policy':<22}{U:8.3f} ± {Us:.3f}")
    U0 = Ush = None
    if encoder or args.oracle_z:
        U0, _ = eval_policy(actor, encoder, cfg, n_eval=args.n_eval_final, T=ep_len, zero_z=True,
                            use_rsrp=args.use_rsrp, remove_own_kpm=args.remove_own_kpm,
                            oracle_z=args.oracle_z)
        Ush, _ = eval_policy(actor, encoder, cfg, n_eval=args.n_eval_final, T=ep_len, shuffle_z=True,
                             use_rsrp=args.use_rsrp, remove_own_kpm=args.remove_own_kpm,
                             oracle_z=args.oracle_z)
        logp(f"{'policy z<-0':<22}{U0:8.3f}   (drop_zero   = {U - U0:+.3f})")
        logp(f"{'policy z<-shuffle':<22}{Ush:8.3f}   (drop_shuffle= {U - Ush:+.3f})")
    floorU, _ = eval_baseline(lambda g, s, w, N: E.bl_equal_power(g, s, w, N),
                              cfg, n_eval=args.n_eval_final, T=ep_len)
    ceilU, _ = eval_baseline(lambda g, s, w, N: E.pf_wsr_ceiling(g, s, w, N, cfg.ceiling_grid),
                             cfg, n_eval=args.n_eval_final, T=ep_len)
    logp(f"{'equal_power (floor)':<22}{floorU:8.3f}")
    logp(f"{'PF-WSR (ceiling)':<22}{ceilU:8.3f}")
    np.save(args.out, dict(policy=U, floor=floorU, ceiling=ceilU,
                           zero_z=U0, shuffle_z=Ush, best_U=best_U), allow_pickle=True)
    logf.close()
    return best_U


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_z", type=int, default=0)
    ap.add_argument("--oracle_z", type=int, default=0,
                    help="1: bypass encoder; feed PF-WSR per-BS power fracs directly as z (upper bound test)")
    ap.add_argument("--oracle_grid", type=int, default=3,
                    help="grid size for oracle_z during training (default=3=27 combos, fast); eval always uses cfg.ceiling_grid=6")
    ap.add_argument("--use_rsrp", type=int, default=0,
                    help="1: add g[i][u] for all BSes to actor obs (ue_feat 3->6)")
    ap.add_argument("--reward", default="difference", choices=["difference", "team", "logpf"])
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--replay", type=int, default=1000000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--lr", type=float, default=3e-4, help="critic & alpha lr")
    ap.add_argument("--actor_lr", type=float, default=0.0,
                    help="actor & encoder lr (0=same as --lr)")
    ap.add_argument("--critic_ln", type=int, default=0,
                    help="1: LayerNorm in Q networks (stabilises Q-value scale)")
    ap.add_argument("--critic_updates", type=int, default=1,
                    help="critic gradient steps per RL step (default=1)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--z_dim", type=int, default=16)
    ap.add_argument("--ep_len", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--n_eval", type=int, default=20)
    ap.add_argument("--n_eval_final", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="chasac")
    ap.add_argument("--bc_steps", type=int, default=0, help="BC warm-start iters (0=off)")
    ap.add_argument("--bc_critic_steps", type=int, default=0,
                    help="Critic BC warm-start iters: pre-train Q on MC returns from expert rollouts (0=off)")
    ap.add_argument("--bc_critic_eps", type=int, default=300,
                    help="Number of expert episodes for critic BC dataset")
    ap.add_argument("--bc_data", type=int, default=3000, help="BC expert dataset size")
    ap.add_argument("--bc_batch", type=int, default=256)
    ap.add_argument("--bc_lr", type=float, default=3e-4)
    ap.add_argument("--alpha_init", type=float, default=0.1, help="initial SAC temperature")
    ap.add_argument("--bc_reg", type=float, default=0.0,
                    help="TD3+BC-style anchor: lambda*MSE(actor, expert) added to actor loss (0=off)")
    ap.add_argument("--norm_reward", type=int, default=0,
                    help="1=standardize reward by running RMS + clip (tames team-reward scale spikes)")
    ap.add_argument("--mu_bound", type=float, default=0.0,
                    help=">0: mu=mu_bound*tanh(mu_raw); prevents tanh-saturation zero-power collapse")
    ap.add_argument("--power_pen", type=float, default=0.0,
                    help="subtract power_pen*(per-BS power/Pmax) from reward; keeps power moderate for PF fairness")
    ap.add_argument("--remove_own_kpm", type=int, default=0,
                    help="1: per-BS encoder — BS i's z excludes its own KPM row (only neighbor KPM); "
                         "forces z to be the sole cross-BS info channel")
    ap.add_argument("--isolate_pretrain_steps", type=int, default=0,
                    help="N>0: first N RL steps run with env.isolated=True (no inter-cell interference); "
                         "curriculum learning — actor learns single-cell optimum before coordination challenge")
    ap.add_argument("--walk_speed", type=float, default=0.0,
                    help=">0: UE randomwalk (m/step), BS topology fixed at env init")
    ap.add_argument("--n_step", type=int, default=1,
                    help="n-step returns: accumulate n transitions before adding to replay, "
                         "use gamma^n for bootstrapped Q target (1=standard 1-step SAC)")
    ap.add_argument("--alpha_min", type=float, default=0.0,
                    help=">0: floor for SAC temperature; failed runs die at alpha 0.0005-0.0007, "
                         "successful ones sit at 0.0015+ (suggest 0.001)")
    ap.add_argument("--ema_decay", type=float, default=0.0,
                    help=">0: keep EMA (Polyak) copy of actor/encoder, eval it alongside raw policy "
                         "and add to ckpt pool (suggest 0.9999); smooths crash-rebound oscillation")
    ap.add_argument("--topk", type=int, default=10,
                    help="keep top-K eval checkpoints; FINAL re-ranks them on a held-out "
                         "50-ep validation set (seed 5151) before test eval (0=old single-best)")
    ap.add_argument("--actor_every", type=int, default=1,
                    help="delayed policy update (TD3-style): actor/alpha/encoder update only "
                         "every K critic steps; counters the 3:1 actor:critic backward ratio "
                         "of the sequential loop (suggest 2-3)")
    ap.add_argument("--q_clamp", type=float, default=0.0,
                    help=">0: clamp critic targets to [-X, X]; logpf per-BS episodic return is "
                         "bounded (~[-20, +30]), out-of-range targets are noise (suggest 50)")
    args = ap.parse_args()
    args.log = f"results/{args.tag}_log.txt"
    args.out = f"results/{args.tag}_result.npy"
    train(args)
