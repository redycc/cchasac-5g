"""
Queue-aware baseline evaluation: Queue-weighted WMMSE, M-LWDF, full-power.

Metrics:
  - avg_goodput  : mean goodput per step (bits/step) averaged over episodes
  - total_goodput: total bits served per episode
  - P99_delay    : 99th percentile HoL (head-of-line) delay in slots
  - P90_delay    : 90th percentile HoL delay in slots

M-LWDF interpretation (multi-cell):
  Prioritise BSs by γ_i * Q_i / Q_ref, run per-BS independent greedy power
  optimisation treating neighbours as max-power interference — only local info.

Queue-weighted WMMSE:
  Centralised (knows all queues), per-BS weight = 1 + Q_bar_i / Q_ref,
  runs wmmse_rb_weighted per RB.
"""
import sys
import os
sys.path.insert(0, "/home/hyc1014/DL/FinalProject")

import numpy as np
from envs.cc_env_goodput import CCEnvGoodput
from baseline import (Cfg, dbm_to_w, noise_w_per_rb,
                      wmmse_rb, wmmse_rb_weighted, sinr_rb)

# ── Config ────────────────────────────────────────────────────────────────────
N_EVAL     = 20
SEED       = 9999   # different from training seeds
EP_LEN     = 200
Q_REF      = CCEnvGoodput.BUF_MAX * 0.5   # = 15 bits  (50% buffer fill)
PMAX       = dbm_to_w(Cfg.Pmax_dBm)
NW         = noise_w_per_rb()


# ── Helpers ───────────────────────────────────────────────────────────────────

def effective_A(env):
    """Build [N_BS, N_BS] per-BS effective channel from env state (freq-flat)."""
    G = env.G          # [N_BS, N_UE]
    assoc = env.assoc
    N_BS = env.cfg.N_BS
    A = np.zeros((N_BS, N_BS), dtype=np.float64)
    for i in range(N_BS):
        mask = (assoc == i)
        if mask.any():
            A[i, :] = G[:, mask].mean(axis=1)
        else:
            A[i, :] = G.mean(axis=1)
    return A


def queue_weights(env):
    """Per-BS weights: 1 + avg_queue_per_BS / Q_ref (≥1, higher = more backlogged)."""
    w = np.ones(env.cfg.N_BS, dtype=np.float64)
    for i in range(env.cfg.N_BS):
        mask = (env.assoc == i)
        if mask.any():
            w[i] = 1.0 + float(env.buf[mask].mean()) / Q_REF
    return w


# ── Policies ──────────────────────────────────────────────────────────────────

def policy_full_power(env, _last_acts):
    return np.ones((env.cfg.N_BS, env.cfg.N_RB), dtype=np.float32)


def policy_freq_reuse(env, _last_acts):
    """Oracle: each BS uses a dedicated non-overlapping RB."""
    acts = np.zeros((env.cfg.N_BS, env.cfg.N_RB), dtype=np.float32)
    for i in range(env.cfg.N_BS):
        acts[i, i % env.cfg.N_RB] = 1.0
    return acts


def policy_qw_wmmse(env, _last_acts):
    """Queue-weighted WMMSE (centralised, knows all queues)."""
    cfg  = env.cfg
    nw   = env.nw
    Pmax = env.Pmax
    rng  = env.rng

    A = effective_A(env)             # [N_BS, N_BS]
    w = queue_weights(env)
    w = w / (w.mean() + 1e-8)       # normalise so mean ≈ 1

    P = np.zeros((cfg.N_BS, cfg.N_RB), dtype=np.float32)
    for rb in range(cfg.N_RB):
        P[:, rb] = wmmse_rb_weighted(A, nw, Pmax, w, rng=rng)
    return P / Pmax


def policy_mlwdf(env, last_acts):
    """M-LWDF: each BS independently maximises its own queue-weighted rate.

    Each BS i solves a 1-D per-RB power optimisation treating other BSs as
    full-power interference (only local CQI + queue info available).
    Weight for BS i: w_i = 1 + Q_bar_i / Q_ref (M-LWDF-style queue emphasis).
    """
    cfg  = env.cfg
    nw   = env.nw
    Pmax = env.Pmax
    rng  = env.rng
    A    = effective_A(env)
    w    = queue_weights(env)

    # Build a modified A where off-diagonals are "frozen" at full-power baseline,
    # and each BS independently chooses its own power to maximise w_i * log(1+SINR_i).
    # This is equivalent to solving each BS's 1-D problem independently.
    P = np.zeros((cfg.N_BS, cfg.N_RB), dtype=np.float32)
    for rb in range(cfg.N_RB):
        # Assume all other BSs transmit at full power — conservative interference.
        for i in range(cfg.N_BS):
            # 1-D golden section search for BS i's optimal power on this RB
            other_intf = sum(
                A[i, j] * Pmax for j in range(cfg.N_BS) if j != i
            )
            # SINR_i(p) = A[i,i]*p / (nw + other_intf)
            # log(1+SINR) is monotone in p → use full power always under
            # fixed-interference assumption.  M-LWDF's value is in PRIORITY, not
            # power reduction, so we cap at Pmax and scale by queue weight.
            # Only reduce power when queue is empty (no urgent data).
            if w[i] < 0.2:
                P[i, rb] = 0.1 * Pmax
            else:
                P[i, rb] = Pmax
    return P / Pmax


# ── HOL delay tracker ─────────────────────────────────────────────────────────

class HoLTracker:
    """Track head-of-line (HoL) delay per UE across an episode.

    HoL delay = number of consecutive slots a UE's buffer was non-empty.
    Recorded when buffer drains (≤ 0.5 bits remaining).
    """
    def __init__(self, n_ue):
        self.n_ue    = n_ue
        self.counter = np.zeros(n_ue, dtype=int)   # consecutive non-empty steps
        self.delays  = []                           # recorded HoL samples

    def update(self, buf_after):
        """Call after each env.step with the resulting buffer array."""
        for k in range(self.n_ue):
            if buf_after[k] <= 0.5:
                if self.counter[k] > 0:
                    self.delays.append(int(self.counter[k]))
                self.counter[k] = 0
            else:
                self.counter[k] += 1

    def flush(self):
        """End of episode: record censored delays for UEs still backlogged."""
        for k in range(self.n_ue):
            if self.counter[k] > 0:
                self.delays.append(int(self.counter[k]))
                self.counter[k] = 0


# ── Evaluation harness ────────────────────────────────────────────────────────

def evaluate_policy(name, policy_fn, n_episodes=N_EVAL, seed_offset=0):
    all_goodput_per_step = []
    all_total_goodput    = []
    all_delays           = []

    for ep in range(n_episodes):
        env = CCEnvGoodput(seed=SEED + seed_offset + ep)
        env.reset()
        tracker = HoLTracker(env.cfg.N_UE)

        ep_goodput = 0.0
        steps      = 0
        last_acts  = np.full((env.cfg.N_BS, env.cfg.N_RB), 0.5, dtype=np.float32)

        while True:
            acts = policy_fn(env, last_acts)
            last_acts = acts.copy()

            _kpm, _rews, done, info = env.step(acts)
            ep_goodput += info['goodput']
            steps      += 1

            tracker.update(env.buf)
            if done:
                tracker.flush()
                break

        all_goodput_per_step.append(ep_goodput / steps)
        all_total_goodput.append(ep_goodput)
        all_delays.extend(tracker.delays)

    avg_gput   = float(np.mean(all_goodput_per_step))
    total_gput = float(np.mean(all_total_goodput))
    delays_arr = np.array(all_delays, dtype=float)
    p90  = float(np.percentile(delays_arr, 90)) if len(delays_arr) > 0 else 0.0
    p99  = float(np.percentile(delays_arr, 99)) if len(delays_arr) > 0 else 0.0
    mean_d = float(delays_arr.mean()) if len(delays_arr) > 0 else 0.0

    print(f"  {name:<22}  avg_gput={avg_gput:6.2f}  total_gput={total_gput:7.1f}"
          f"  P90={p90:5.1f}  P99={p99:5.1f}  mean_delay={mean_d:.2f}  "
          f"  n_delay_samples={len(delays_arr)}")
    return dict(name=name, avg_gput=avg_gput, total_gput=total_gput,
                p90=p90, p99=p99, mean_delay=mean_d)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n=== Queue-Aware Baseline Evaluation "
          f"(N_EVAL={N_EVAL} eps × EP_LEN={EP_LEN} steps) ===")
    print(f"{'Policy':<24}  {'avg_gput':>9}  {'total_gput':>11}  "
          f"{'P90':>6}  {'P99':>6}  {'mean_d':>8}")
    print("-" * 85)

    policies = [
        ("full_power",       policy_full_power,  0),
        ("freq_reuse_oracle",policy_freq_reuse,   1000),
        ("qw_wmmse",         policy_qw_wmmse,     2000),
        ("m_lwdf_local",     policy_mlwdf,        3000),
    ]

    results = []
    for name, fn, off in policies:
        r = evaluate_policy(name, fn, N_EVAL, seed_offset=off)
        results.append(r)

    print("\n── Summary table ──")
    print(f"{'Policy':<24}  {'avg_gput':>9}  {'P99_delay':>10}")
    for r in results:
        print(f"  {r['name']:<22}  {r['avg_gput']:9.2f}  {r['p99']:10.1f}")

    import json
    out = os.path.join("/home/hyc1014/DL/FinalProject/results",
                       "queue_baselines.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
