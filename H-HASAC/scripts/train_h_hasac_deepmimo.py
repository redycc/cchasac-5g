"""Run H-HASAC + DeepMIMO only (Flat DeepMIMO baseline already done)."""
import sys
import os
import numpy as np

sys.path.insert(0, "/home/hyc1014/DL/FinalProject/HARL")
sys.path.insert(0, "/home/hyc1014/DL/FinalProject")

from scripts.train_h_hasac import build_runner

RESULTS = "/home/hyc1014/DL/FinalProject/results"
CACHE   = "/home/hyc1014/DL/FinalProject/deepmimo_cache"
os.makedirs(RESULTS, exist_ok=True)

print("=" * 60)
print("H-HASAC  |  DeepMIMO O1 3.5GHz  |  n_step=1")
print("=" * 60)

h_runner = build_runner(
    num_env_steps=300_000, K=10, beta=0.3, seed=42,
    exp_name="h_hasac_deepmimo",
    env_overrides={
        "channel_source": "deepmimo",
        "deepmimo_cache": CACHE,
        "ue_pool_size": 10_000,
    },
)
h_runner.run()

h_rewards = np.array(h_runner.done_episodes_rewards)
np.save(f"{RESULTS}/h_hasac_deepmimo_ep_rewards.npy",  h_rewards)
np.save(f"{RESULTS}/h_hasac_deepmimo_mgr_rewards.npy", np.array(h_runner.mgr_rewards_log))
h_runner.close()

h_final = np.mean(h_rewards[-20:]) if len(h_rewards) >= 20 else (np.mean(h_rewards) if len(h_rewards) > 0 else 0.0)
flat_final = 173.19  # flat DeepMIMO baseline (already done)

print("=" * 60)
print("RESULT")
print("=" * 60)
print(f"H-HASAC + DeepMIMO : {h_final:.4f}  (n_episodes={len(h_rewards)})")
print(f"Flat    + DeepMIMO : {flat_final:.4f}  (baseline)")
pct = (h_final - flat_final) / flat_final * 100
print(f"Improvement        : {pct:+.1f}%")
