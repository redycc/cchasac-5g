"""
Proper M-LWDF baseline with per-RB UE scheduling.

In the original baseline_eval_qos.py, M-LWDF degenerates to full_power because
the env uses equal RB sharing and M-LWDF is a SCHEDULING algorithm (which UE
gets each RB), not a power allocation algorithm.

This script implements M-LWDF correctly:
  - Each BS allocates full power per RB
  - For each RB: assign it to the UE in the cell with highest M-LWDF weight
  - Served rate for UE j = Σ_{rb: assigned to j} log2(1 + SINR_j_rb)
  This overrides the equal-RB-sharing assumption.

Also implements proportional-fair (PF) and max-rate scheduling for comparison.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from envs.cc_env_goodput import CCEnvGoodput
from baseline import Cfg, path_gain, associate, dbm_to_w, noise_w_per_rb, gen_topology

N_BS       = 3
N_RB       = 4
N_EVAL_EPS = 20
EP_LEN     = 200
SEED       = 9999
EMA_ALPHA  = 0.1
GAMMA_QOS  = 1.0
ARRIVE_RATE = CCEnvGoodput.ARRIVE_RATE  # 3.0
BUF_MAX     = CCEnvGoodput.BUF_MAX      # 30.0
WALK_SPEED  = CCEnvGoodput.WALK_SPEED   # 3.0


class ProperMLWDFEnv:
    """Env that computes per-UE rates using actual per-RB scheduling.

    Supports: M-LWDF, PF, max-rate, equal-sharing (reference), full-power.
    """
    def __init__(self, seed=SEED):
        self.cfg  = Cfg(freq_selective=True)
        self.nw   = noise_w_per_rb()
        self.Pmax = dbm_to_w(self.cfg.Pmax_dBm)
        self.rng  = np.random.default_rng(seed)
        self._step = 0

    def reset(self):
        cfg = self.cfg
        self.bs_pos, self.ue_pos = gen_topology(cfg, self.rng)
        self.G     = path_gain(cfg, self.bs_pos, self.ue_pos, self.rng)
        self.assoc = associate(self.G)
        self.buf   = np.zeros(cfg.N_UE, dtype=np.float32)
        self.age   = np.zeros(cfg.N_UE, dtype=np.float32)
        self.rate_avg = np.ones(cfg.N_UE, dtype=np.float32) * ARRIVE_RATE
        self._step = 0

    def _compute_sinr(self, P):
        """P: [N_BS, N_RB]  →  sinr: [N_RB, N_UE]"""
        received = P[:, :, None] * self.G[:, None, :]    # [N_BS, N_RB, N_UE]
        total    = received.sum(axis=0)                    # [N_RB, N_UE]
        desired  = received[self.assoc, :, np.arange(self.cfg.N_UE)].T   # [N_RB, N_UE]
        intf     = total - desired
        return desired / (self.nw + intf + 1e-20)          # [N_RB, N_UE]

    def step(self, scheduler_name="mlwdf"):
        """Run one step with the specified scheduler."""
        cfg  = self.cfg
        P    = np.ones((cfg.N_BS, cfg.N_RB), dtype=np.float32) * self.Pmax  # full power
        sinr = self._compute_sinr(P)   # [N_RB, N_UE]
        rate_per_rb = np.log2(1.0 + np.maximum(sinr, 0.0))   # [N_RB, N_UE]

        # Per-UE rate using chosen scheduler
        rate_ue = self._schedule(rate_per_rb, scheduler_name)  # [N_UE]

        # Serve from buffer
        served = np.minimum(rate_ue, self.buf)

        # Update HOL age
        self.age = np.where(self.buf > 0, self.age + 1, 0.0).astype(np.float32)

        # Update rate EMA
        self.rate_avg = ((1 - EMA_ALPHA) * self.rate_avg + EMA_ALPHA * served).astype(np.float32)

        # Buffer update
        arrivals = self.rng.poisson(ARRIVE_RATE, size=cfg.N_UE).astype(np.float32)
        self.buf = np.clip(self.buf - served + arrivals, 0.0, BUF_MAX)

        # Per-BS goodput
        goodput_per_bs = np.zeros(cfg.N_BS, dtype=np.float32)
        for i in range(cfg.N_BS):
            mask = (self.assoc == i)
            if mask.any():
                goodput_per_bs[i] = served[mask].sum()

        # UE walk + channel update
        self.ue_pos = self.ue_pos + self.rng.normal(0.0, WALK_SPEED,
                                   self.ue_pos.shape).astype(np.float32)
        self.ue_pos = np.clip(self.ue_pos, 0.0, cfg.area)
        self.G     = path_gain(cfg, self.bs_pos, self.ue_pos, self.rng)
        self.assoc = associate(self.G)
        self._step += 1

        return {
            'goodput':   float(goodput_per_bs.sum()),
            'hol_delay': self.age.copy(),
            'served_ue': served,
        }

    def _schedule(self, rate_per_rb, scheduler_name):
        """Assign RBs to UEs using scheduler_name, return per-UE rate."""
        cfg = self.cfg
        rate_ue = np.zeros(cfg.N_UE, dtype=np.float64)

        if scheduler_name == "equal":
            # Equal RB sharing (reference)
            for i in range(cfg.N_BS):
                mask = (self.assoc == i)
                n_i  = mask.sum()
                if n_i == 0: continue
                rb_frac = cfg.N_RB / float(n_i)
                rate_ue[mask] = rb_frac * rate_per_rb[:, mask].mean(axis=0)
            return rate_ue.astype(np.float32)

        # Compute per-UE scheduling weights
        if scheduler_name == "mlwdf":
            # M-LWDF: w_i = γ_i * W_i / R̄_i (HOL_delay / avg_rate)
            w = GAMMA_QOS * (self.age + 0.1) / np.maximum(self.rate_avg, 0.1)
        elif scheduler_name == "pf":
            # Proportional Fair: w_i = 1 / R̄_i
            w = 1.0 / np.maximum(self.rate_avg, 0.1)
        elif scheduler_name == "maxrate":
            # Max rate: w_i = 1 (no fairness, raw SINR)
            w = np.ones(cfg.N_UE, dtype=np.float32)
        elif scheduler_name == "queue_prop":
            # Queue-proportional: w_i = Q_i + 1
            w = self.buf + 1.0
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        # Per-cell: assign each RB to the UE with highest weight
        for i in range(cfg.N_BS):
            mask   = (self.assoc == i)
            if not mask.any(): continue
            ue_idx = np.where(mask)[0]
            w_cell = w[ue_idx]
            for rb in range(cfg.N_RB):
                # Could also do max-rate-weighted: w_cell * rate_per_rb[rb, ue_idx]
                if scheduler_name == "maxrate":
                    # For max-rate: use actual rate as weight (not uniform)
                    winner_local = np.argmax(rate_per_rb[rb, ue_idx])
                else:
                    winner_local = np.argmax(w_cell)
                winner_global = ue_idx[winner_local]
                rate_ue[winner_global] += rate_per_rb[rb, winner_global]

        return rate_ue.astype(np.float32)


def evaluate_scheduler(name, scheduler_key, n_eps=N_EVAL_EPS):
    all_gput, all_hol = [], []
    for ep in range(n_eps):
        env = ProperMLWDFEnv(seed=SEED + ep)
        env.reset()
        gput_ep, hol_ep = 0.0, []
        for _ in range(EP_LEN):
            info = env.step(scheduler_key)
            gput_ep += info['goodput']
            hol_ep.extend(info['hol_delay'].tolist())
        all_gput.append(gput_ep / EP_LEN)
        all_hol.extend(hol_ep)

    all_hol = np.array(all_hol)
    return {
        'name':      name,
        'gput_avg':  float(np.mean(all_gput)),
        'gput_std':  float(np.std(all_gput)),
        'p50_delay': float(np.percentile(all_hol, 50)),
        'p99_delay': float(np.percentile(all_hol, 99)),
        'frac_zero': float((all_hol == 0).mean()),
    }


if __name__ == "__main__":
    print("=" * 70)
    print(f"Proper Scheduler Baseline  ({N_EVAL_EPS} eps × {EP_LEN} steps, full power)")
    print("=" * 70)
    print(f"{'Scheduler':20s}  {'AvgGoodput':10s}  {'Std':6s}  "
          f"{'P50delay':9s}  {'P99delay':9s}  {'FracZero':9s}")
    print("-" * 70)

    schedulers = [
        ("equal_sharing",  "equal"),
        ("mlwdf",          "mlwdf"),
        ("prop_fair (PF)", "pf"),
        ("queue_prop",     "queue_prop"),
        ("max_rate",       "maxrate"),
    ]

    results = {}
    for name, key in schedulers:
        r = evaluate_scheduler(name, key)
        results[key] = r
        print(f"{name:20s}  {r['gput_avg']:10.4f}  {r['gput_std']:6.3f}  "
              f"{r['p50_delay']:9.2f}  {r['p99_delay']:9.2f}  {r['frac_zero']:9.3f}")

    print("=" * 70)
    print(f"\nKey takeaways:")
    print(f"  Arrival rate ceiling  = {ARRIVE_RATE * Cfg().N_UE:.1f} bits/step")
    print(f"  Freq-reuse oracle     = ~29.9 bits/step (from prev eval)")
    print(f"  M-LWDF vs equal      : {results['mlwdf']['gput_avg'] - results['equal']['gput_avg']:+.4f} goodput, "
          f"{results['mlwdf']['p99_delay'] - results['equal']['p99_delay']:+.1f} P99 delay")
    print(f"  PF vs equal          : {results['pf']['gput_avg'] - results['equal']['gput_avg']:+.4f} goodput, "
          f"{results['pf']['p99_delay'] - results['equal']['p99_delay']:+.1f} P99 delay")

    np.save("results/mlwdf_scheduler_baseline.npy", results)
    print("\nSaved to results/mlwdf_scheduler_baseline.npy")
