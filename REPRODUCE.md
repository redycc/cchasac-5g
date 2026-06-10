# Reproduction Specification — Deployable Multi-Cell Power Allocation: HASAC vs C-HASAC

This document specifies **methods only** (no code) so the system can be re-implemented from scratch.
Primary goal: **compare HASAC (no `z`) vs C-HASAC (with a learned cross-cell context `z`)** on a
deployment-observable downlink power-control problem, and measure how close a *deployable* policy
gets to the *full-CSI optimum*. It also specifies the follow-on architecture (spatial gate × base)
and the supervised/RL training protocols needed to reproduce all conclusions.

---

## 1. Target / what to measure (be explicit)

Reproduce these comparisons, all measured as **mean delivered goodput** on held-out evaluation seeds,
reported as **% of the floor→ceiling gap** = `(policy − equal_floor) / (oracle_ceiling − equal_floor)`:

1. **HASAC vs C-HASAC** — identical setup, the *only* difference being whether the actor receives the
   manager context `z`. Expected: **C-HASAC ≈ HASAC** (z as an input is null).
2. **Pure (canonical) HASAC on a single fixed topology** — separate per-cell actors, common (team)
   reward, shared centralized critic, sequential updates. Expected: ~48% of the gap (learns a
   spatial-reuse policy), far above random-topology training (~0%).
3. **`z` as a multiplicative gate** vs `z` as an input — `power = gate × base`. Expected: gate
   architecture reaches ~80% (supervised), input-`z` stays ~0%.
4. **Supervised (BC) vs RL** for the same architecture — BC reaches the deployable ceiling (~85%);
   RL cannot (caps ~28–48%), and **RL un-learns any *learnable* combine** (degrades below plain HASAC).

The headline scientific claims to verify:
- A learned context `z` **as a policy input does not help** (null in every regime).
- The deployable coordination is **recoverable by supervision + a fixed multiplicative gate**, not by RL.
- The optimal per-cell power **decomposes as `gate(slow spatial) × base(fast own-CQI)`**; only the
  *fixed* multiply survives RL (a *learnable* combine, even one verified to be a multiplier, is un-learned).

---

## 2. Dataset / environment creation method

There is **no static dataset**; data is generated on-the-fly by a NumPy simulator that is the single
source of truth for channel → SINR → rate → reward. A "sample/scenario" = one BS/UE placement (a
"trial"); within a trial, time advances in slots with fading and/or traffic dynamics.

### 2.1 Scenario generation (per `reset`)
- Parameters (defaults): `N_BS = 4` cells, `N_UE = 8` users, square `area = 1200 m`,
  carrier `fc = 3.5 GHz`, max power `Pmax = −15 dBm`, shadowing std `σ_shadow = 4 dB`.
- Draw BS positions and UE positions **uniformly at random** in `[0, area]²`.
- Distance `d[j,u] = ‖bs_j − ue_u‖`, clipped to ≥ 1 m.
- Path loss (dB): `PL[j,u] = 32.4 + 21·log10(d[j,u]) + 20·log10(fc) + N(0, σ_shadow)`
  (one shadow draw per link, fixed for the trial).
- Large-scale gain: `g_ls[j,u] = 10^(−PL[j,u]/10)`.
- Association: `serv[u] = argmax_j g_ls[j,u]` (best-signal).
- Guarantee ≥1 UE per cell: if a cell is empty, reassign to it the best-gain UE taken **only** from a
  donor cell that has >1 UE (never empty a cell).

### 2.2 Within-trial dynamics
- **Fading (optional):** small-scale `h[j,u]` complex, `E|h|²=1`, evolves Gauss–Markov (Jakes-like):
  `h_t = ρ·h_{t−1} + sqrt(1−ρ²)·CN(0,1)`, `ρ = 0.9`. Effective gain `g = g_ls · |h|²`. (`g = g_ls`
  if fading off.)
- **Traffic / queues (for the goodput regime):** each UE has backlog `Q_u` (bits/s/Hz units),
  `q_max = 80`. Each cell is "hot" or "cold"; hot cells inject arrivals `arr_bits = 8` to their UEs
  each slot. Hot/cold toggles as a 2-state Markov chain with `p_on→off`, `p_off→on`
  (e.g. full-buffer: hot_init=1, p_on→off=0, p_off→on=1; bursty/dynamic-load: hot_init=0.5,
  p_on→off=p_off→on=0.05). Delivered goodput per UE = `min(Q_u, rate_u)`; then
  `Q_u ← min(Q_u − delivered_u + arrivals_u, q_max)`.
- **Mobility (optional):** UEs random-walk (`ue ← clip(ue + N(0, walk_std), 0, area)`) and `g_ls` is
  recomputed via the distance-only update `g_ls ← g_ls0 · (d0/d)^2.1` (keeps the reset-time shadow).

### 2.3 Two scenario regimes (both must be reproduced)
- **Random topology** (default): fresh placement every `reset`. This is the *hard generalization*
  setting; RL fails here (~0%).
- **Fixed single topology** (`fixed_topology`): generate ONE placement from a fixed `topo_seed`
  (e.g. 12345) and reuse it on every `reset` (only fading/queues vary). Train/eval share the same
  geometry. **The canonical HASAC-vs-C-HASAC comparison and all gate results use this.**
  ("Topology-dependent, train a separate policy per topology" is an accepted framing.)

### 2.4 Train/eval split
- Training uses env seed = run `--seed`; evaluation uses a **disjoint** seed block (e.g. `10000+seed …`
  for in-loop eval; `20000…` for final/oracle eval). Same distribution, no overlap → held-out.
- Each eval "trial" is rolled out for `eval_T = 150` slots; report the **steady-state** mean
  (discard the first ~T/3 slots as warm-up so PF averages settle).

---

## 3. System architecture

Three-tier information model (do not violate):
- **(A) BS-local, deployable** → the **actor/worker** input `o_i`.
- **(B) RIC-observable KPM** → the **manager/encoder** input → context `z` (C-HASAC only).
- **(C) sim-only privileged** (full CSI, counterfactuals) → the **critic** and the **reward** only,
  at training time. The critic is NEVER given `z`.

### 3.1 Worker / actor
- Input `o_i` = per-UE local features for the cell's own UEs, each UE:
  `[ achievable_rate, log(PF_weight), prev_power/Pmax, log(1+Q) ]` (+ optional neighbour RSRP, see 5).
  Features are normalized to comparable scales.
- The actor embeds each UE feature (MLP), **pools per cell** by segment-mean over the serving-BS index
  (permutation-equivariant; handles variable UEs/cell), then a head outputs **one scalar per cell**.
- Output = per-cell **squashed-Gaussian** action `a_i ∈ [−1,1]` (mean `μ`, state-dependent `log_std`,
  `tanh` squash with the standard log-prob correction). Map to power level `P_i = (a_i+1)/2 · Pmax`.
- **Two actor variants to reproduce:**
  - **Shared (homogeneous):** one network used by all cells (parameter sharing).
  - **Separate (heterogeneous, "pure HASAC"):** an independent network per cell index (no sharing).
    Required for the canonical comparison on a fixed topology.
- **C-HASAC only:** concatenate the per-cell context `z_i` to the pooled cell embedding before the head.
  **HASAC vs C-HASAC differ ONLY by this concatenation.**

### 3.2 Manager / encoder (produces `z`)
- Input = per-cell **KPM** = `[ load (#UEs), mean R̄ (PF starvation), mean backlog (mean Q) ]`. (No CSI.)
- Two encoder variants:
  - **Mean-pool Encoder (permutation-invariant):** per-cell MLP, then for each cell `i` pool the
    *other* cells (exclude self) → `z_i` (summarizes neighbours' load/starvation).
  - **Transformer Encoder (ID-aware):** per-cell KPM tokens + cell-ID positional embedding →
    self-attention across cells → per-cell `z_i` (can carry an asymmetric role signal).
- `z` dimension `z_dim = 16`. Trained **end-to-end with the actor** (its only gradient is the actor loss).

### 3.3 Critic
- **Centralized twin-Q** (two Q-heads, take the min), parameter-shared across agents, evaluated
  per-agent. Input = `share_obs` ⊕ `joint_action` ⊕ `agent_one-hot`, where
  `share_obs` = flattened full state `[ all g, all p, serv, w, rate (+Q) ]` (privileged). NOT given `z`.

### 3.4 Gate architecture (the constructive deployable result)
- `power_i = gate_i × base_i` (multiplicative).
  - `base_i ∈ [0,1]` from the worker (own CQI) — the *fast* level.
  - `gate_i ∈ [0,1]` — the *slow spatial* who-defers signal.
- The multiply is **fixed (non-trainable)** — this is essential (see §6, "RL un-learns a learnable combine").
- Deployable variants of the gate (all reach ~80–85%):
  - per-cell gate net trained (BC) to the **spatial oracle** (see §5.4); each cell's own net carries
    its fixed geometric role.
  - (optional, for cross-topology generalization) a manager fed **RSRP** so the gate is computed from
    the deployable coupling rather than per-cell-index memorization.

---

## 4. Rewards and required equations

Notation: per-UE power `p_u`; per-BS total power `P_bs[j] = Σ_{u: serv[u]=j} p_u`; effective gain `g[j,u]`.

**SINR & rate (per UE `u`, served by `s = serv[u]`):**
```
signal_u        = p_u · g[s,u]
total_rx_u      = Σ_j  P_bs[j] · g[j,u]
interference_u  = total_rx_u − P_bs[s] · g[s,u]            # inter-cell only (intra-cell orthogonal)
SINR_u          = signal_u / (interference_u + N0)         # N0 = fixed noise floor
rate_u          = log2(1 + SINR_u)                          # bits/s/Hz
```

**Delivered goodput (traffic regime):** `delivered_u = min(Q_u, rate_u)`.

**PF weight (proportional-fair):** `w_u = 1 / (R̄_u + ε)`, with running average updated each slot
`R̄_u ← (1−β)·R̄_u + β·delivered_u`, `β = 0.05` (use `rate_u` instead of `delivered_u` in full-buffer mode).

**Global objective (PF-weighted goodput):** `G = Σ_u w_u · delivered_u`.

**Per-cell action → per-UE power:** `P_i = (a_i+1)/2 · Pmax`; split equally among the cell's
**backlogged** UEs (`Q_u > 0`); if none backlogged, split among all (delivers 0 harmlessly).

**Reward (two options; reproduce both):**
- **Difference reward (per-agent, Nasir–Guo / Wolpert–Tumer):**
  ```
  r_i = G  −  G(cell i muted)
      = Σ_{u∈i} w_u·delivered_u  −  Σ_{u∉i} w_u·( delivered_u^{i muted} − delivered_u^{actual} )
      = own_i − harm_i      (harm_i ≥ 0; computed by a full-CSI counterfactual that sets p=0 in cell i)
  ```
  Key property: `∂r_i/∂a_i = ∂G/∂a_i` (each agent's marginal reward = its marginal effect on `G`).
- **Team (common) reward:** `r_i = G` for all `i` (cooperative; this is the canonical HASAC reward).

**Reward shaping (optional symmetry-breaker):** add `+λ·(P_i_frac − mean_frac)²` (rewards across-cell
power dispersion). This is the only thing that lifts plain RL off the floor (~+28% at `λ≈10`); it is a
*symmetry-breaking* term, not an information term.

---

## 5. Oracle, baselines, and supervised targets

**Full-CSI ORACLE (the "ceiling"):** at each slot, grid-search the per-cell power level over
`{0, 1/(K−1), …, 1}` (e.g. `K=5`) for all cells (cost `K^N_BS`), pick the combination maximizing the
per-slot delivered goodput `Σ_u min(Q_u, rate_u)` given the **current full channel** and queues.
This is greedy/per-slot and uses the per-cell + equal-split action class (same as the policy), so it
is the right ceiling *for this action class* (not the absolute physical optimum). Execute it as a policy
to get the ceiling goodput.

**SPATIAL oracle (slow gate target):** same grid-search but on the **large-scale gain `g_ls`** (i.e.
ignore instantaneous fading). This is the deployable, slow "who-defers" target used to supervise the gate.

**Baselines:** `equal` = all cells at `Pmax` (the floor); `round-robin` = one cell on per slot in
rotation. Report all of {equal, round-robin, oracle} on the same eval seeds for context.

**Supervised targets (for BC / distillation):**
- worker/agent or gate-architecture → the **full oracle** per-cell levels (MSE).
- gate net → the **spatial oracle** (MSE).
- base net → residual `base_gt = clip(optimal_level / spatial_level, 0, 1)` (MSE).
- combine NN (if testing a *learnable* combine) → train on **uniformly sampled** `(z,b) ∈ [0,1]² → z·b`
  to make it a true multiplier; verify on a grid that `combine(z,b) ≈ z·b` (MAE→0, corr→1).
  NOTE: training the combine on *real rollout `(z,b)` pairs* does NOT yield a multiplier (the pairs are
  correlated / don't cover the input space) and the wired system collapses — must use uniform pairs.

---

## 6. Training procedures

### 6.1 RL: HASAC / C-HASAC (HARL soft actor-critic, CTDE, off-policy)
- Max-entropy SAC objective per agent, **centralized critic, decentralized actors** (CTDE).
- **Sequential (HAML) actor update:** update agents one at a time in a random per-step order; when
  updating agent `i`, only agent `i`'s action carries gradient while the others' (already-updated)
  actions are held fixed (detached). This is HASAC's monotonic-improvement / anti-bad-Nash mechanism.
- **Auto-tuned temperature `α`** (entropy target = −1 per action dim, one scalar action/agent).
  (We added an optional `α` floor and a tiny action-regularizer for stability; for *pure* HASAC set both
  to 0.)
- Discount `γ = 0` (the per-slot problem is effectively a contextual bandit; with fading/queues it is
  near-myopic). Target-net soft update `τ = 0.005`.
- Hyperparameters (defaults): actor/critic/α learning rate `3e-4`, batch `256`, replay `1e6`,
  warm-up `5000` random-action steps, hidden width `256`, `z_dim = 16`, episode length `20`,
  total steps `40000`, eval every `2500`.
- **HASAC vs C-HASAC:** identical config; C-HASAC additionally builds the encoder and concatenates `z`
  to the actor. The critic and reward are unchanged. This isolates `z`'s contribution.
- Reward normalization by a running std is allowed (gradient scaling; does not change the optimum).

### 6.2 Supervised: behavior cloning / distillation
- Roll out the **oracle** to collect `(local obs, serv, KPM, oracle per-cell level)` tuples
  (e.g. 60 trials × 150 slots).
- Train the chosen network by **MSE** between its deterministic per-cell output `(a+1)/2` and the
  oracle level (or the spatial/residual targets in §5 for the modular gate).
- This is CTDE-legitimate: the oracle is a **training target**; at run time the policy uses only
  deployable inputs (local obs, KPM/RSRP) — **no oracle at inference**.

### 6.3 Modular gate pretraining (the constructive recipe)
1. Train the **gate** net → spatial oracle (MSE). 2. Train the **base** net → residual (MSE).
3. (If a learnable combine is used) train it → multiply on uniform `(z,b)`; else use a **fixed multiply**.
4. Wire `power = gate × base`. (Optional) RL-refine.
Expected: wired BC reaches ~80–85% of the gap. RL-refining a **learnable** combine **degrades** it
(verified-multiplier → ~6.9, below plain HASAC); only the **fixed** multiply holds under RL (~80%).

---

## 7. How to obtain the final answer (evaluation protocol)

1. Fix the topology (`topo_seed`) for the canonical comparison. Compute the references once on the eval
   seeds: `equal` (floor), `round-robin`, `oracle` (ceiling).
2. Train each method to `total_steps` (RL) or to convergence (BC), with **≥3 seeds**.
3. Evaluate the final (deterministic) policy on the **held-out eval seeds**, `eval_T = 150` slots,
   discard the first ~T/3, take mean delivered goodput.
4. Report per method: mean goodput, and **% of gap** `= (goodput − floor)/(ceiling − floor)`,
   with per-seed spread (the null/positive claims must be robust across seeds, not single-seed).
5. The comparison table to fill (single fixed topology, deployable inputs unless noted):

   | method | training | `z`? | expected % of gap |
   |--------|----------|------|-------------------|
   | equal (floor) | — | — | 0% |
   | round-robin | — | — | (≤ floor in interference-limited regime) |
   | HASAC (separate actors, team reward) | RL | no | ~48% |
   | C-HASAC (= HASAC + `z` input) | RL | yes | ~48% (≈ HASAC → `z` null) |
   | HASAC + dispersion shaping | RL | no | ~28% (random-topo) / lifts floor |
   | gate × base, fixed multiply | BC | gate=spatial | ~80–85% |
   | gate × base, learnable combine | BC→RL | — | degrades to ~6.9 goodput (RL un-learns it) |
   | oracle (full CSI) | — | — | 100% (ceiling, not deployable) |

6. **Conclusion test:** C-HASAC must NOT significantly beat HASAC (z-as-input null). The deployable
   gain must come from **supervision + the fixed multiplicative spatial gate**, and RL must be shown to
   un-learn any learnable combine. Probe `z` with a linear readout: high R²(`z`→neighbour-backlog) but
   ~0 R²(`z`→oracle-switch) and R²(neighbour-backlog→oracle-switch)≈0 confirm `z` carries *load* while
   the switch is set by *interference coupling* — explaining why load-based `z` is the wrong quantity.

---

## 8. Key numbers observed (single fixed topology, floor ≈ 5.3, oracle ≈ 9.1)
- pure HASAC ≈ 7.2 (~48%); C-HASAC ≈ pure HASAC (z null).
- `z` as input (concat / oracle-switch / frozen): ≈ floor under RL.
- spatial-gate × RL-worker (fixed multiply, oracle gate): ≈ 8.3 (~80%).
- 3-part BC (gate→spatial, base→residual, fixed multiply): ≈ 8.5 (~85%), fully deployable.
- learnable combine, BC then RL: BC ≈ 7.5 (combine verified multiplier, corr ≈ 0.998) → RL ≈ 6.9 (un-learned).

Reproduce these qualitatively (exact values depend on `topo_seed` and the noise floor `N0`).
