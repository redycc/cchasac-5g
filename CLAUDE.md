# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DL Final Project: **H-HASAC** — Hierarchical Heterogeneous-Agent Soft Actor-Critic for 5G multi-cell resource allocation. The system has a two-level hierarchy: a **Manager SAC** (slow time-scale, every K=10 steps) that issues sub-goal budgets, and **HASAC Workers** (fast time-scale, every step) that allocate per-RB power.

## Key Commands

```bash
# Train flat HASAC baseline (formula channel)
python3 scripts/train_flat_hasac.py

# Train H-HASAC (formula channel)
python3 scripts/train_h_hasac.py

# Train H-HASAC on DeepMIMO O1 3.5GHz (standalone, after flat is done)
python3 scripts/train_h_hasac_deepmimo.py

# Run flat + H-HASAC comparison in one shot
python3 scripts/train_deepmimo_hasac.py

# Send progress.md to Telegram DL group
python3 scripts/telegram_report.py

# Install HARL package (required once)
cd HARL && pip install -e .
```

## Architecture

```
HHASACRunner (scripts/train_h_hasac.py)
  └─ extends OffPolicyHARunner (HARL/harl/runners/off_policy_ha_runner.py)
       ├─ ManagerSAC             — SAC agent operating at K=10 step intervals
       │    obs: shared_obs (33-dim), act: n_agents×3 sub-goals [0,1]
       │    reward: Σ raw_env_reward over K steps
       ├─ HASAC Workers (×3)     — standard HARL HASAC agents
       │    obs: local (11-dim) + injected sub_goal (3-dim)
       │    act: power per RB (4-dim, clipped to p_max_frac)
       │    reward: raw_reward − β×(actual_p − target_p)²
       └─ FiveGEnv               — 3BS × 10UE × 4RB 5G simulator
```

### Key Files

| File | Role |
|------|------|
| `scripts/train_h_hasac.py` | **Main implementation**: ManagerSAC, HHASACRunner, training loop |
| `envs/fiveg_env.py` | 5G environment: SINR computation, reward, sub_goal injection |
| `envs/deepmimo_channel.py` | DeepMIMO O1 3.5GHz channel wrapper with caching |
| `HARL/harl/runners/off_policy_ha_runner.py` | Base runner (extended by HHASACRunner) |
| `HARL/harl/algorithms/critics/soft_twin_continuous_q_critic.py` | Shared Q-critic |
| `HARL/harl/configs/envs_cfgs/fiveg.yaml` | Default env config (n_bs, n_ue, n_rb, episode_length) |
| `HARL/harl/configs/algos_cfgs/hasac.yaml` | Default HASAC hyperparams |
| `progress.md` | **Always update this** when an experiment or change is completed |

### Observation / Action Spaces

- **Per-agent obs** (11-dim): `[sinr_rb×4, load, interference, throughput, n_ue, sub_goal×3]`
- **Shared obs** (33-dim): concatenation of all 3 agents' obs
- **Worker action** (4-dim): power fractions per RB in `[0, p_max_frac]`
- **Manager action** (9-dim): `[p_max_frac, i_thresh_frac, rb_share]` × 3 agents

## Critical Gotchas

### NaN Prevention (Bug #6 — solved)
Manager SAC can produce NaN sub-goals without gradient clipping. Four simultaneous fixes are required:
1. `ManagerSAC.update()`: skip NaN loss + `clip_grad_norm_(max_norm=10)`
2. `HHASACRunner._inject_subgoal()`: `np.nan_to_num(subgoal, nan=0.5)` + `isnan()` guard
3. `fiveg_env.set_sub_goals()`: `np.nan_to_num` + `np.clip`
4. `OffPolicyHARunner.train()`: skip NaN actor/critic loss + `clip_grad_norm_`

### n_step Must Be 1
`algo_args["algo"]["n_step"] = 1` is mandatory. Multi-step TD targets spanning Manager update boundaries propagate stale sub-goals and cause NaN chains.

### np.int Deprecation (Python ≥ 3.12)
Fixed in `HARL/harl/common/buffers/off_policy_buffer_ep.py`. Use `np.int64` or `int`.

### DeepMIMO Cache
First run takes ~30s to build `deepmimo_cache/`. Do not delete it. Cache key is `(tx_ids, pool_size)`.

### Aug EP ≠ Raw EP
`avg ep reward` in logs = augmented reward (includes goal penalty ≈ −40). Raw EP ≈ `mgr_reward × 20`. Always compare Raw EP against Flat HASAC baseline.

## Progress Tracking

**Always update `progress.md`** at the end of every experiment, bug fix, or architecture change. It is the single source of truth for current status and is sent to the DL group via `/sendTG`.

## Workflow Rules

- After any non-trivial code change, update `progress.md` with what changed and what the result was.
- Every change is reviewed by Codex before being considered complete.
- Results are saved to `results/` as `.npy` files. Training logs go to `results/*.txt`.
- Report experimental results in terms of **Raw EP** (not Aug EP) when comparing with Flat HASAC.
