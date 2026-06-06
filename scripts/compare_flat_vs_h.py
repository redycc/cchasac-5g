"""Compare flat HASAC vs H-HASAC on 5G env. Saves reward curves for both."""
import sys
import os
import numpy as np

sys.path.insert(0, "/home/hyc1014/DL/FinalProject/HARL")
sys.path.insert(0, "/home/hyc1014/DL/FinalProject")

from harl.utils.configs_tools import get_defaults_yaml_args
from harl.runners import RUNNER_REGISTRY
from scripts.train_h_hasac import build_runner

NUM_STEPS = 300_000
SEED = 42
LOG_EVERY = 10_000
os.makedirs("/home/hyc1014/DL/FinalProject/results", exist_ok=True)


# ─── Flat HASAC ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Training FLAT HASAC ...")
print("=" * 60)

args = {"algo": "hasac", "env": "fiveg", "exp_name": "flat_hasac"}
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
algo_args["seed"]["seed"] = SEED
algo_args["device"]["cuda"] = True
algo_args["logger"]["log_dir"] = "/home/hyc1014/DL/FinalProject/results"
env_args["n_bs"] = 3
env_args["n_ue"] = 10
env_args["n_rb"] = 4
env_args["episode_length"] = 200
env_args["hierarchical"] = False

flat_runner = RUNNER_REGISTRY["hasac"](args, algo_args, env_args)
flat_runner.run()
flat_ep_rewards = np.array(flat_runner.done_episodes_rewards)
flat_runner.close()

np.save("/home/hyc1014/DL/FinalProject/results/flat_hasac_ep_rewards.npy", flat_ep_rewards)
print(f"Flat HASAC done. Episodes: {len(flat_ep_rewards)}, "
      f"Last-20 avg: {np.mean(flat_ep_rewards[-20:]):.3f}")


# ─── H-HASAC ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("Training H-HASAC ...")
print("=" * 60)

h_runner = build_runner(num_env_steps=NUM_STEPS, K=10, beta=0.3, seed=SEED, exp_name="h_hasac")
h_runner.run()
h_ep_rewards = np.array(h_runner.done_episodes_rewards)
h_runner.close()

np.save("/home/hyc1014/DL/FinalProject/results/h_hasac_ep_rewards.npy", h_ep_rewards)
print(f"H-HASAC done. Episodes: {len(h_ep_rewards)}, "
      f"Last-20 avg: {np.mean(h_ep_rewards[-20:]):.3f}")


# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
flat_final = np.mean(flat_ep_rewards[-20:]) if len(flat_ep_rewards) >= 20 else np.mean(flat_ep_rewards)
h_final = np.mean(h_ep_rewards[-20:]) if len(h_ep_rewards) >= 20 else np.mean(h_ep_rewards)
print(f"Flat HASAC   last-20 avg ep reward: {flat_final:.4f}")
print(f"H-HASAC      last-20 avg ep reward: {h_final:.4f}")
print(f"Improvement: {(h_final - flat_final) / (abs(flat_final) + 1e-8) * 100:+.1f}%")
print("\nResults saved to /home/hyc1014/DL/FinalProject/results/")
