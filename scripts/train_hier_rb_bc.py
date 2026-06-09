"""
H-RB — Hierarchical discrete RB-partition Manager + continuous power Workers.

NEW ARCHITECTURE (clean break from cc-HASAC v1..v24 / goodput v1..v5).

Motivation
----------
Across every prior version the single recurring bottleneck was: independent SAC
actors + a broadcast encoder z could NOT reliably converge to the "each BS uses
complementary (orthogonal) RBs" frequency-reuse pattern. The freq-reuse oracle
hits goodput≈29.7 / P99≈102 purely by giving each BS disjoint RBs; RL stalled at
24-26 because continuous per-RB power exploration almost never discovers a clean
orthogonal partition and SAC entropy destroys it once found.

H-RB attacks this directly by making the orthogonalisation an EXPLICIT, structured
decision owned by a slow-timescale central manager:

  Manager (every K steps, central / xApp)
      obs : global KPM (all BS: sinr, load, goodput, buf, n_ue, HOL)  [N_BS*9]
      act : per-RB categorical over N_BS  →  assignment[rb] = owning BS
      → builds a hard RB→BS partition mask (forces frequency reuse)
      reward : Σ env reward over the K-step hold (slow credit)
      algo : factored discrete SAC (per-RB head, twin Q, auto temperature)

  Workers (every step, decentralised, parameter-shared)
      obs : local KPM (9) + own RB-ownership mask (N_RB) + agent id (N_BS)
      act : power fraction per RB; non-owned RBs forced to 0 by the mask
      reward : per-BS env reward
      algo : continuous SAC, shared twin Q over the joint (CTDE)

The manager's job — dynamically reallocate RBs to whichever cells have queue
pressure — is exactly the sequential decision a myopic per-step optimiser (WMMSE)
cannot make. Workers just learn "push power on the RBs I own".

Env: envs/cc_env_goodput_v2.py reused as-is (mask applied in the training loop).

Targets: goodput > 26.15 (goodput v5, prior dynamic best) and ideally → freq-reuse
oracle 29.7 with P99 < 102.
"""
import sys
import os
sys.path.insert(0, "/home/hyc1014/DL/FinalProject")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import deque

from envs.cc_env_goodput_v2 import CCEnvGoodputV2

# ── Dimensions ────────────────────────────────────────────────────────────────
N_BS    = 3
N_RB    = 4
KPM_DIM = CCEnvGoodputV2.KPM_DIM          # 9

MGR_OBS    = N_BS * KPM_DIM                # 27 (global KPM, includes queues)
WORKER_OBS = KPM_DIM + N_RB + N_BS         # 9 + 4 + 3 = 16
SHARE_OBS  = N_BS * (KPM_DIM + N_RB)       # 39 (per-BS kpm + own mask)
HIDDEN     = 128

# ── Hyper-parameters ──────────────────────────────────────────────────────────
GAMMA_W   = 0.97          # worker discount (per-step, queue-drain horizon)
GAMMA_M   = 0.95          # manager discount (per-K-step decision)
POLYAK    = 0.005
LR        = 3e-4
ALPHA_W   = 0.05          # worker entropy init (auto-tuned)
ALPHA_M   = 0.5           # manager entropy init (auto-tuned)
BUFFER    = 200_000
BATCH     = 256

K_MGR        = 10         # manager acts every K_MGR steps
WARMUP       = 5_000      # random exploration before learning
TRAIN_EVERY  = 1
NUM_STEPS    = 300_000
LOG_EVERY    = 10_000
EP_LEN       = 200
N_EVAL_EPS   = 20

SEED        = 42
RESULTS_DIR = "/home/hyc1014/DL/FinalProject/results"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AGENT_IDS   = np.eye(N_BS, dtype=np.float32)


# ── Networks ──────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(HIDDEN, HIDDEN), out_bias=None):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        last = nn.Linear(prev, out_dim)
        if out_bias is not None:
            nn.init.constant_(last.bias, out_bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Replay buffers ────────────────────────────────────────────────────────────

class WorkerBuffer:
    # kpm[N_BS,9], mask[N_BS,4], acts[N_BS,4], rews[N_BS], kpm_n, mask_n, done
    FIELDS = 7

    def __init__(self, cap=BUFFER):
        self.buf = deque(maxlen=cap)

    def push(self, kpm, mask, acts, rews, kpm_n, mask_n, done):
        self.buf.append((kpm.astype(np.float32), mask.astype(np.float32),
                         acts.astype(np.float32), rews.astype(np.float32),
                         kpm_n.astype(np.float32), mask_n.astype(np.float32),
                         np.float32(done)))

    def sample(self, n):
        idx = np.random.choice(len(self.buf), n, replace=False)
        b   = [self.buf[i] for i in idx]
        return [np.stack([x[j] for x in b]) for j in range(self.FIELDS)]

    def __len__(self):
        return len(self.buf)


class ManagerBuffer:
    # gobs[27], assign[N_RB](int), r_K(scalar), gobs_n[27], done
    FIELDS = 5

    def __init__(self, cap=BUFFER):
        self.buf = deque(maxlen=cap)

    def push(self, gobs, assign, r_K, gobs_n, done):
        self.buf.append((gobs.astype(np.float32), assign.astype(np.int64),
                         np.float32(r_K), gobs_n.astype(np.float32),
                         np.float32(done)))

    def sample(self, n):
        idx = np.random.choice(len(self.buf), n, replace=False)
        b   = [self.buf[i] for i in idx]
        return [np.stack([x[j] for x in b]) for j in range(self.FIELDS)]

    def __len__(self):
        return len(self.buf)


# ── Hierarchical agent ────────────────────────────────────────────────────────

class HierRB:
    def __init__(self):
        # Manager — factored discrete SAC (per-RB categorical over N_BS)
        self.m_pi  = MLP(MGR_OBS, N_RB * N_BS).to(DEVICE)
        self.m_q1  = MLP(MGR_OBS, N_RB * N_BS).to(DEVICE)
        self.m_q2  = MLP(MGR_OBS, N_RB * N_BS).to(DEVICE)
        self.m_q1t = deepcopy(self.m_q1)
        self.m_q2t = deepcopy(self.m_q2)

        # Worker — continuous SAC (shared actor + twin Q over joint)
        self.w_pi  = MLP(WORKER_OBS, N_RB * 2, out_bias=1.0).to(DEVICE)
        self.w_q1  = MLP(SHARE_OBS + N_BS * N_RB, 1).to(DEVICE)
        self.w_q2  = MLP(SHARE_OBS + N_BS * N_RB, 1).to(DEVICE)
        self.w_q1t = deepcopy(self.w_q1)
        self.w_q2t = deepcopy(self.w_q2)

        self.m_pi_opt = torch.optim.Adam(self.m_pi.parameters(), lr=LR)
        self.m_q_opt  = torch.optim.Adam(
            list(self.m_q1.parameters()) + list(self.m_q2.parameters()), lr=LR)
        self.w_pi_opt = torch.optim.Adam(self.w_pi.parameters(), lr=LR)
        self.w_q_opt  = torch.optim.Adam(
            list(self.w_q1.parameters()) + list(self.w_q2.parameters()), lr=LR)

        self.m_log_alpha = torch.tensor(np.log(ALPHA_M), requires_grad=True, device=DEVICE)
        self.w_log_alpha = torch.tensor(np.log(ALPHA_W), requires_grad=True, device=DEVICE)
        self.m_alpha_opt = torch.optim.Adam([self.m_log_alpha], lr=LR)
        self.w_alpha_opt = torch.optim.Adam([self.w_log_alpha], lr=LR)
        # discrete target entropy per RB (encourage exploration, allow convergence)
        self.m_target_ent = 0.6 * np.log(N_BS)
        self.w_target_ent = -float(N_RB)

        self.w_buf = WorkerBuffer()
        self.m_buf = ManagerBuffer()
        self.env      = CCEnvGoodputV2(seed=SEED)
        self.eval_env = CCEnvGoodputV2(seed=SEED + 9999)

        self.best_gput = -float("inf")
        self.best_ckpt = None

    # ── manager action ──────────────────────────────────────────────────────

    @torch.no_grad()
    def manager_act(self, gobs_np, greedy=False):
        t = torch.FloatTensor(gobs_np).unsqueeze(0).to(DEVICE)
        logits = self.m_pi(t).view(1, N_RB, N_BS)
        if greedy:
            assign = logits.argmax(-1).squeeze(0).cpu().numpy()
        else:
            p = torch.softmax(logits, -1).squeeze(0)        # [N_RB, N_BS]
            assign = torch.multinomial(p, 1).squeeze(-1).cpu().numpy()
        return assign.astype(np.int64)

    @staticmethod
    def assign_to_mask(assign):
        """assign[N_RB] -> mask[N_BS, N_RB] (1 if BS owns that RB)."""
        mask = np.zeros((N_BS, N_RB), dtype=np.float32)
        for rb in range(N_RB):
            mask[assign[rb], rb] = 1.0
        return mask

    # ── worker action ─────────────────────────────────────────────────────────

    def _worker_obs(self, kpm, mask):
        # kpm[N_BS,9], mask[N_BS,4] -> [N_BS, WORKER_OBS]
        return np.concatenate([kpm, mask, AGENT_IDS], axis=-1)

    @torch.no_grad()
    def worker_act(self, kpm, mask, greedy=False):
        wobs = self._worker_obs(kpm, mask)
        t    = torch.FloatTensor(wobs).to(DEVICE)
        out  = self.w_pi(t)
        mean, log_std = out[:, :N_RB], out[:, N_RB:]
        if greedy:
            a = torch.tanh(mean)
        else:
            log_std = torch.clamp(log_std, -5, 2)
            a = torch.tanh(mean + log_std.exp() * torch.randn_like(mean))
        a = (a + 1) / 2                       # [N_BS, N_RB] in [0,1]
        a = a.cpu().numpy() * mask            # zero non-owned RBs
        return a.astype(np.float32)

    def _worker_forward(self, wobs_t):
        B   = wobs_t.shape[0]
        out = self.w_pi(wobs_t.reshape(B * N_BS, WORKER_OBS))
        mean, log_std = out[:, :N_RB], out[:, N_RB:]
        log_std = torch.clamp(log_std, -5, 2)
        std = log_std.exp()
        zs  = mean + std * torch.randn_like(mean)
        a   = torch.tanh(zs)
        lp  = (-((zs - mean) ** 2) / (2 * std ** 2 + 1e-8)
               - log_std - 0.5 * np.log(2 * np.pi)
               - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        a   = (a + 1) / 2
        return a.reshape(B, N_BS, N_RB), lp.reshape(B, N_BS, 1)

    @staticmethod
    def _share_obs(kpm_t, mask_t):
        B = kpm_t.shape[0]
        return torch.cat([kpm_t, mask_t], -1).reshape(B, -1)   # [B, SHARE_OBS]

    # ── manager update (factored discrete SAC) ──────────────────────────────────

    def update_manager(self):
        gobs, assign, rK, gobs_n, done = self.m_buf.sample(BATCH)
        s    = torch.FloatTensor(gobs).to(DEVICE)
        a    = torch.LongTensor(assign).to(DEVICE)            # [B, N_RB]
        r    = torch.FloatTensor(rK).unsqueeze(1).to(DEVICE)  # [B,1]
        s_n  = torch.FloatTensor(gobs_n).to(DEVICE)
        d    = torch.FloatTensor(done).unsqueeze(1).to(DEVICE)
        alpha = self.m_log_alpha.exp().detach()

        # ---- critic target ----
        with torch.no_grad():
            logits_n = self.m_pi(s_n).view(BATCH, N_RB, N_BS)
            logp_n   = F.log_softmax(logits_n, -1)
            p_n      = logp_n.exp()
            q1n = self.m_q1t(s_n).view(BATCH, N_RB, N_BS)
            q2n = self.m_q2t(s_n).view(BATCH, N_RB, N_BS)
            qn  = torch.min(q1n, q2n)
            v_n = (p_n * (qn - alpha * logp_n)).sum(-1)       # [B, N_RB]
            backup = r + GAMMA_M * (1 - d) * v_n              # broadcast r,d → [B,N_RB]

        q1 = self.m_q1(s).view(BATCH, N_RB, N_BS)
        q2 = self.m_q2(s).view(BATCH, N_RB, N_BS)
        q1a = q1.gather(-1, a.unsqueeze(-1)).squeeze(-1)      # [B, N_RB]
        q2a = q2.gather(-1, a.unsqueeze(-1)).squeeze(-1)
        q_loss = ((q1a - backup) ** 2).mean() + ((q2a - backup) ** 2).mean()
        if not q_loss.isnan():
            self.m_q_opt.zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.m_q1.parameters()) + list(self.m_q2.parameters()), 10.0)
            self.m_q_opt.step()

        # ---- policy ----
        logits = self.m_pi(s).view(BATCH, N_RB, N_BS)
        logp   = F.log_softmax(logits, -1)
        p      = logp.exp()
        with torch.no_grad():
            q1d = self.m_q1(s).view(BATCH, N_RB, N_BS)
            q2d = self.m_q2(s).view(BATCH, N_RB, N_BS)
            qd  = torch.min(q1d, q2d)
        pi_loss = (p * (alpha * logp - qd)).sum(-1).mean()
        if not pi_loss.isnan():
            self.m_pi_opt.zero_grad()
            pi_loss.backward()
            nn.utils.clip_grad_norm_(self.m_pi.parameters(), 10.0)
            self.m_pi_opt.step()

        # ---- temperature ----
        with torch.no_grad():
            ent = -(p * logp).sum(-1)                          # [B, N_RB]
        a_loss = (self.m_log_alpha * (ent - self.m_target_ent).detach()).mean()
        self.m_alpha_opt.zero_grad()
        a_loss.backward()
        self.m_alpha_opt.step()

        for pq, tq in [(self.m_q1, self.m_q1t), (self.m_q2, self.m_q2t)]:
            for pp, tp in zip(pq.parameters(), tq.parameters()):
                tp.data.mul_(1 - POLYAK).add_(POLYAK * pp.data)

    # ── worker update (continuous SAC) ──────────────────────────────────────────

    def update_worker(self):
        kpm, mask, acts, rews, kpm_n, mask_n, done = self.w_buf.sample(BATCH)
        kpm_t  = torch.FloatTensor(kpm).to(DEVICE)
        mask_t = torch.FloatTensor(mask).to(DEVICE)
        acts_t = torch.FloatTensor(acts).to(DEVICE)
        rews_t = torch.FloatTensor(rews).to(DEVICE)
        kpm_n_t  = torch.FloatTensor(kpm_n).to(DEVICE)
        mask_n_t = torch.FloatTensor(mask_n).to(DEVICE)
        done_t = torch.FloatTensor(done).unsqueeze(1).to(DEVICE)
        ids_t  = torch.FloatTensor(AGENT_IDS).unsqueeze(0).expand(BATCH, -1, -1).to(DEVICE)
        alpha  = self.w_log_alpha.exp().detach()
        sum_r  = rews_t.sum(dim=1, keepdim=True)

        sobs   = self._share_obs(kpm_t, mask_t)
        sobs_n = self._share_obs(kpm_n_t, mask_n_t)

        with torch.no_grad():
            wobs_n  = torch.cat([kpm_n_t, mask_n_t, ids_t], -1)
            na, nlp = self._worker_forward(wobs_n)
            na      = na * mask_n_t                       # mask next action
            na_flat = na.reshape(BATCH, -1)
            tgt = torch.min(self.w_q1t(torch.cat([sobs_n, na_flat], -1)),
                            self.w_q2t(torch.cat([sobs_n, na_flat], -1))) \
                  - alpha * nlp.sum(dim=1)
            backup = sum_r + GAMMA_W * (1 - done_t) * tgt

        acts_flat = (acts_t * mask_t).reshape(BATCH, -1)
        q1_l = ((self.w_q1(torch.cat([sobs, acts_flat], -1)) - backup) ** 2).mean()
        q2_l = ((self.w_q2(torch.cat([sobs, acts_flat], -1)) - backup) ** 2).mean()
        if not (q1_l + q2_l).isnan():
            self.w_q_opt.zero_grad()
            (q1_l + q2_l).backward()
            nn.utils.clip_grad_norm_(
                list(self.w_q1.parameters()) + list(self.w_q2.parameters()), 10.0)
            self.w_q_opt.step()

        wobs   = torch.cat([kpm_t, mask_t, ids_t], -1)
        new_a, lp = self._worker_forward(wobs)
        new_a  = new_a * mask_t
        q_val  = torch.min(self.w_q1(torch.cat([sobs, new_a.reshape(BATCH, -1)], -1)),
                           self.w_q2(torch.cat([sobs, new_a.reshape(BATCH, -1)], -1)))
        a_loss = (alpha * lp.sum(dim=1) - q_val).mean()
        if not a_loss.isnan():
            self.w_pi_opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(self.w_pi.parameters(), 10.0)
            self.w_pi_opt.step()

        with torch.no_grad():
            _, lp_det = self._worker_forward(wobs)
        al = -(self.w_log_alpha * (lp_det.sum(dim=1)
                                   + self.w_target_ent * N_BS)).mean()
        self.w_alpha_opt.zero_grad()
        al.backward()
        self.w_alpha_opt.step()

        for pq, tq in [(self.w_q1, self.w_q1t), (self.w_q2, self.w_q2t)]:
            for pp, tp in zip(pq.parameters(), tq.parameters()):
                tp.data.mul_(1 - POLYAK).add_(POLYAK * pp.data)

    # ── checkpoint ──────────────────────────────────────────────────────────────

    def _save_ckpt(self, step):
        self.best_ckpt = {'m_pi': deepcopy(self.m_pi.state_dict()),
                          'w_pi': deepcopy(self.w_pi.state_dict())}
        torch.save({'step': step, 'best_gput': self.best_gput,
                    'm_pi': self.m_pi.state_dict(),
                    'w_pi': self.w_pi.state_dict()},
                   os.path.join(RESULTS_DIR, "hier_rb_bc_best.pt"))

    def _restore_ckpt(self):
        if self.best_ckpt is not None:
            self.m_pi.load_state_dict(self.best_ckpt['m_pi'])
            self.w_pi.load_state_dict(self.best_ckpt['w_pi'])
            print(f"  [ckpt] restored best (peak_gput={self.best_gput:.4f})", flush=True)

    # ── evaluation ──────────────────────────────────────────────────────────────

    def evaluate(self, n_episodes=N_EVAL_EPS, manager='learned'):
        """manager: 'learned' | 'random' | 'static' (fixed freq-reuse)."""
        gputs, p99s = [], []
        static_assign = np.array([0, 1, 2, 0], dtype=np.int64)
        for ep in range(n_episodes):
            self.eval_env.rng = np.random.default_rng(SEED + 9999 + ep)
            kpm = self.eval_env.reset()
            gobs = self.eval_env.get_global_kpm().reshape(-1)
            if manager == 'random':
                assign = np.random.randint(0, N_BS, size=N_RB)
            elif manager == 'static':
                assign = static_assign
            else:
                assign = self.manager_act(gobs, greedy=True)
            mask = self.assign_to_mask(assign)
            g_sum, p99_sum, steps = 0.0, 0.0, 0
            while True:
                acts = self.worker_act(kpm, mask, greedy=True)
                kpm, _, done, info = self.eval_env.step(acts)
                g_sum   += info['goodput']
                p99_sum += info['hol_p99']
                steps   += 1
                if steps % K_MGR == 0:
                    gobs = self.eval_env.get_global_kpm().reshape(-1)
                    if manager == 'random':
                        assign = np.random.randint(0, N_BS, size=N_RB)
                    elif manager == 'static':
                        assign = static_assign
                    else:
                        assign = self.manager_act(gobs, greedy=True)
                    mask = self.assign_to_mask(assign)
                if done:
                    break
            gputs.append(g_sum / steps)
            p99s.append(p99_sum / steps)
        return float(np.mean(gputs)), float(np.mean(p99s))

    def _baseline_eval(self):
        out = {}
        for name, policy in [
            ('full_power', np.ones((N_BS, N_RB), dtype=np.float32)),
            ('freq_reuse', np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=np.float32)),
        ]:
            g_list, p99_list = [], []
            for ep in range(N_EVAL_EPS):
                self.eval_env.rng = np.random.default_rng(SEED + 9999 + ep)
                self.eval_env.reset()
                g_ep, p99_ep, steps = 0.0, 0.0, 0
                while True:
                    _, _, done, info = self.eval_env.step(policy)
                    g_ep   += info['goodput']
                    p99_ep += info['hol_p99']
                    steps  += 1
                    if done:
                        break
                g_list.append(g_ep / steps)
                p99_list.append(p99_ep / steps)
            out[name] = (float(np.mean(g_list)), float(np.mean(p99_list)))
        return out

    # ── training loop ──────────────────────────────────────────────────────────

    # ── BC warm-start ───────────────────────────────────────────────────────────
    def bc_pretrain(self, n_data=4000, steps=3000, batch=256):
        """BC warm-start: manager -> static freq-reuse partition [0,1,2,0],
        worker -> full power on owned RBs. States collected by static rollout."""
        static_assign = np.array([0, 1, 2, 0], dtype=np.int64)
        static_mask   = self.assign_to_mask(static_assign)
        tgt_assign    = torch.LongTensor(static_assign).to(DEVICE)        # [N_RB]
        mask_t        = torch.FloatTensor(static_mask).to(DEVICE)         # [N_BS,N_RB]
        ids_t         = torch.FloatTensor(AGENT_IDS).to(DEVICE)           # [N_BS,N_BS]

        print(f"[BC] collecting {n_data} states (static manager rollout)...", flush=True)
        env = CCEnvGoodputV2(seed=SEED + 123)
        kpm = env.reset()
        G, Wk = [], []
        for _ in range(n_data):
            G.append(env.get_global_kpm().reshape(-1).astype(np.float32))
            Wk.append(kpm.astype(np.float32))
            acts = (0.7 + 0.3 * np.random.rand(N_BS, N_RB).astype(np.float32)) * static_mask
            kpm, _, done, _ = env.step(acts)
            if done:
                kpm = env.reset()
        G  = torch.FloatTensor(np.stack(G)).to(DEVICE)                    # [n,27]
        Wk = torch.FloatTensor(np.stack(Wk)).to(DEVICE)                   # [n,N_BS,9]
        n  = G.shape[0]

        print(f"[BC] pretrain manager(CE->static) + worker(full power) "
              f"| steps={steps} batch={batch}", flush=True)
        for it in range(1, steps + 1):
            idx = torch.randint(0, n, (batch,), device=DEVICE)
            # manager: cross-entropy to the static partition
            logits = self.m_pi(G[idx]).view(batch, N_RB, N_BS)
            m_loss = F.cross_entropy(logits.reshape(batch * N_RB, N_BS),
                                     tgt_assign.repeat(batch))
            self.m_pi_opt.zero_grad(); m_loss.backward()
            nn.utils.clip_grad_norm_(self.m_pi.parameters(), 10.0)
            self.m_pi_opt.step()
            # worker: full power on owned RBs
            wk    = Wk[idx]                                               # [b,N_BS,9]
            b     = wk.shape[0]
            m_exp = mask_t.unsqueeze(0).expand(b, -1, -1)                 # [b,N_BS,N_RB]
            wobs  = torch.cat([wk, m_exp, ids_t.unsqueeze(0).expand(b, -1, -1)], dim=-1)
            out   = self.w_pi(wobs.reshape(b * N_BS, WORKER_OBS))
            pwr   = ((torch.tanh(out[:, :N_RB]) + 1) / 2).reshape(b, N_BS, N_RB)
            w_loss = F.mse_loss(pwr * m_exp, m_exp)                       # owned->1
            self.w_pi_opt.zero_grad(); w_loss.backward()
            nn.utils.clip_grad_norm_(self.w_pi.parameters(), 10.0)
            self.w_pi_opt.step()
            if it == 1 or it % 500 == 0:
                print(f"[BC] it {it:>5}/{steps} | m_CE {m_loss.item():.4f} "
                      f"| w_MSE {w_loss.item():.5f}", flush=True)
        with torch.no_grad():
            env.reset()
            a0 = self.manager_act(env.get_global_kpm().reshape(-1), greedy=True)
        print(f"[BC] done. manager greedy = {a0.tolist()} "
              f"(target {static_assign.tolist()})", flush=True)

    def run(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        log_path = os.path.join(RESULTS_DIR, "hier_rb_bc_log.txt")
        print(f"H-RB+BC | device={DEVICE} | MGR_OBS={MGR_OBS} WORKER_OBS={WORKER_OBS} "
              f"SHARE_OBS={SHARE_OBS} | K_MGR={K_MGR} γ_w={GAMMA_W} γ_m={GAMMA_M}",
              flush=True)

        self.bc_pretrain()

        kpm  = self.env.reset()
        gobs = self.env.get_global_kpm().reshape(-1)
        assign = self.manager_act(gobs)
        mask   = self.assign_to_mask(assign)
        m_gobs, m_assign, m_racc = gobs.copy(), assign.copy(), 0.0

        log_lines = []
        print(f"\n{'Step':>8}  {'Goodput':>9}  {'P99HOL':>8}  {'αW':>6}  {'αM':>6}  "
              f"{'BestGput':>10}", flush=True)

        for step in range(1, NUM_STEPS + 1):
            explore = step <= WARMUP
            acts = self.worker_act(kpm, mask, greedy=False) if not explore \
                   else (np.random.rand(N_BS, N_RB).astype(np.float32) * mask)
            kpm_n, rews, done, info = self.env.step(acts)
            m_racc += float(rews.sum())

            new_assign = assign
            if step % K_MGR == 0 or done:
                gobs_n = self.env.get_global_kpm().reshape(-1)
                self.m_buf.push(m_gobs, m_assign, m_racc, gobs_n, float(done))
                if not done:
                    new_assign = (np.random.randint(0, N_BS, size=N_RB) if explore
                                  else self.manager_act(gobs_n))
                m_gobs, m_assign, m_racc = gobs_n.copy(), new_assign.copy(), 0.0
            new_mask = self.assign_to_mask(new_assign)

            self.w_buf.push(kpm, mask, acts, rews, kpm_n, new_mask, float(done))
            kpm, assign, mask = kpm_n, new_assign, new_mask

            if done:
                kpm  = self.env.reset()
                gobs = self.env.get_global_kpm().reshape(-1)
                assign = (np.random.randint(0, N_BS, size=N_RB) if explore
                          else self.manager_act(gobs))
                mask = self.assign_to_mask(assign)
                m_gobs, m_assign, m_racc = gobs.copy(), assign.copy(), 0.0

            if not explore and step % TRAIN_EVERY == 0 \
               and len(self.w_buf) >= BATCH and len(self.m_buf) >= BATCH:
                self.update_worker()
                self.update_manager()

            if step % LOG_EVERY == 0:
                g, p99 = self.evaluate()
                aw = self.w_log_alpha.exp().item()
                am = self.m_log_alpha.exp().item()
                if g > self.best_gput:
                    self.best_gput = g
                    self._save_ckpt(step)
                    tag = f"{self.best_gput:.4f}*"
                else:
                    tag = f"{self.best_gput:.4f}"
                line = (f"{step:8d}  {g:9.4f}  {p99:8.1f}  {aw:6.3f}  {am:6.3f}  {tag}")
                print(line, flush=True)
                log_lines.append(line)

        with open(log_path, "w") as f:
            f.write("\n".join(log_lines))

        print("\n  [ckpt] training done.", flush=True)
        self._restore_ckpt()

        print("\n── Manager ablation ──", flush=True)
        g_l, p_l = self.evaluate(manager='learned')
        g_r, p_r = self.evaluate(manager='random')
        g_s, p_s = self.evaluate(manager='static')
        print(f"  learned manager : goodput={g_l:.4f}  P99={p_l:.1f}", flush=True)
        print(f"  random manager  : goodput={g_r:.4f}  P99={p_r:.1f}  "
              f"Δ(learned-random)={g_l-g_r:+.4f}", flush=True)
        print(f"  static partition: goodput={g_s:.4f}  P99={p_s:.1f}", flush=True)

        print("\n── Static baselines ──", flush=True)
        for name, (g, p99) in self._baseline_eval().items():
            print(f"  {name:12s}: goodput={g:.4f}  P99={p99:.1f}", flush=True)
        print(f"  prior dynamic best (goodput v5): 26.15 / P99=99.5", flush=True)
        print(f"  H-RB (learned)  : goodput={g_l:.4f}  P99={p_l:.1f}  "
              f"(peak={self.best_gput:.4f})", flush=True)

        np.save(os.path.join(RESULTS_DIR, "hier_rb_bc_ablation.npy"),
                np.array([g_l, g_r, g_s, p_l, p_r, p_s]))
        return g_l, p_l


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    HierRB().run()
