"""
Flat HASAC vs H-HASAC using DeepMIMO O1 3.5GHz channel.
Runs both experiments sequentially and saves results.
"""
import sys
import os
import numpy as np

sys.path.insert(0, "/home/hyc1014/DL/FinalProject/HARL")
sys.path.insert(0, "/home/hyc1014/DL/FinalProject")

from harl.utils.configs_tools import get_defaults_yaml_args
from harl.runners import RUNNER_REGISTRY
from scripts.train_h_hasac import build_runner as build_h_runner


def make_accumulating_runner(runner_cls):
    """Subclass runner to accumulate all episode rewards (not just last interval)."""
    class AccRunner(runner_cls):
        def run(self):
            self._all_ep_rewards_acc = []
            super().run()
            self.done_episodes_rewards = self._all_ep_rewards_acc

        def insert(self, data):
            before = len(self.done_episodes_rewards)
            super().insert(data)
            after = len(self.done_episodes_rewards)
            if after > before:
                self._all_ep_rewards_acc.extend(self.done_episodes_rewards[before:after])
    return AccRunner

NUM_STEPS = 300_000
SEED = 42
LOG_EVERY = 10_000
RESULTS = "/home/hyc1014/DL/FinalProject/results"
CACHE = "/home/hyc1014/DL/FinalProject/deepmimo_cache"
os.makedirs(RESULTS, exist_ok=True)

DEEPMIMO_OVERRIDES = {
    "channel_source": "deepmimo",
    "deepmimo_cache": CACHE,
    "ue_pool_size": 10_000,
}


def make_flat_args(seed=SEED, channel_source="formula"):
    algo_args, env_args = get_defaults_yaml_args("hasac", "fiveg")
    algo_args["train"]["n_rollout_threads"] = 1
    algo_args["train"]["num_env_steps"] = NUM_STEPS
    algo_args["train"]["warmup_steps"] = 2_000
    algo_args["train"]["train_interval"] = 10
    algo_args["train"]["eval_interval"] = LOG_EVERY
    algo_args["train"]["use_valuenorm"] = True
    algo_args["eval"]["use_eval"] = False
    algo_args["algo"]["batch_size"] = 256
    algo_args["algo"]["buffer_size"] = 100_000
    algo_args["algo"]["n_step"] = 3
    algo_args["algo"]["auto_alpha"] = True
    algo_args["algo"]["alpha"] = 0.01
    algo_args["algo"]["share_param"] = False
    algo_args["model"]["hidden_sizes"] = [128, 128]
    algo_args["seed"]["seed"] = seed
    algo_args["device"]["cuda"] = True
    algo_args["logger"]["log_dir"] = RESULTS

    env_args["n_bs"] = 3
    env_args["n_ue"] = 10
    env_args["n_rb"] = 4
    env_args["episode_length"] = 200
    env_args["hierarchical"] = False
    env_args["channel_source"] = channel_source
    if channel_source == "deepmimo":
        env_args["deepmimo_cache"] = CACHE
        env_args["ue_pool_size"] = 10_000
    return algo_args, env_args


# ── 1. Flat HASAC + DeepMIMO ─────────────────────────────────────────────────
print("=" * 60)
print("Flat HASAC  |  DeepMIMO O1 3.5GHz channel")
print("=" * 60)

algo_args, env_args = make_flat_args(channel_source="deepmimo")
main_args = {"algo": "hasac", "env": "fiveg", "exp_name": "flat_deepmimo"}
AccHASAC = make_accumulating_runner(RUNNER_REGISTRY["hasac"])
runner = AccHASAC(main_args, algo_args, env_args)
runner.run()
flat_dm_rewards = np.array(runner.done_episodes_rewards)
runner.close()
np.save(f"{RESULTS}/flat_deepmimo_ep_rewards.npy", flat_dm_rewards)
flat_final = np.mean(flat_dm_rewards[-20:]) if len(flat_dm_rewards) >= 20 else np.mean(flat_dm_rewards)
print(f"Flat HASAC + DeepMIMO: last-20 avg = {flat_final:.4f}  (n_episodes={len(flat_dm_rewards)})\n")


# ── 2. H-HASAC + DeepMIMO ────────────────────────────────────────────────────
print("=" * 60)
print("H-HASAC  |  DeepMIMO O1 3.5GHz channel")
print("=" * 60)

h_runner = build_h_runner(
    num_env_steps=NUM_STEPS, K=10, beta=0.3, seed=SEED,
    exp_name="h_hasac_deepmimo",
    env_overrides=DEEPMIMO_OVERRIDES,
)
h_runner.run()
# After run(), done_episodes_rewards holds ALL episodes (via _all_ep_rewards)
h_dm_rewards = np.array(h_runner.done_episodes_rewards)
h_runner.close()
np.save(f"{RESULTS}/h_hasac_deepmimo_ep_rewards.npy", h_dm_rewards)
np.save(f"{RESULTS}/h_hasac_deepmimo_mgr_rewards.npy",
        np.array(h_runner.mgr_rewards_log))
h_final = np.mean(h_dm_rewards[-20:]) if len(h_dm_rewards) >= 20 else (np.mean(h_dm_rewards) if len(h_dm_rewards) > 0 else 0.0)
print(f"H-HASAC + DeepMIMO: last-20 avg = {h_final:.4f}  (n_episodes={len(h_dm_rewards)})\n")


# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("FINAL RESULTS  (DeepMIMO O1 3.5GHz)")
print("=" * 60)
print(f"Flat HASAC  : {flat_final:.4f}")
print(f"H-HASAC     : {h_final:.4f}")
pct = (h_final - flat_final) / (abs(flat_final) + 1e-8) * 100
print(f"Improvement : {pct:+.1f}%")
