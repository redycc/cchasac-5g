"""
R2-Flat HASAC baseline: flat HASAC with partial (R2) observations.
Each BS only sees its own local KPM (sinr×4, load, intf, tpt, n_ue = 8-dim).
No neighbor info, no sub-goal. Used as direct comparison for cc-HASAC.
"""
import sys

sys.path.insert(0, "/home/hyc1014/DL/FinalProject/HARL")
sys.path.insert(0, "/home/hyc1014/DL/FinalProject")

from harl.utils.configs_tools import get_defaults_yaml_args
from harl.runners import RUNNER_REGISTRY

args = {
    "algo": "hasac",
    "env": "fiveg",
    "exp_name": "r2_flat_hasac",
}

algo_args, env_args = get_defaults_yaml_args("hasac", "fiveg")

algo_args["train"]["n_rollout_threads"] = 1
algo_args["train"]["num_env_steps"] = 300_000
algo_args["train"]["warmup_steps"] = 2_000
algo_args["train"]["train_interval"] = 10
algo_args["train"]["eval_interval"] = 10_000
algo_args["train"]["use_valuenorm"] = True
algo_args["eval"]["use_eval"] = True
algo_args["eval"]["n_eval_rollout_threads"] = 1
algo_args["eval"]["eval_episodes"] = 5
algo_args["algo"]["batch_size"] = 256
algo_args["algo"]["buffer_size"] = 100_000
algo_args["algo"]["n_step"] = 1
algo_args["algo"]["gamma"] = 0.99
algo_args["algo"]["auto_alpha"] = True
algo_args["algo"]["alpha"] = 0.01
algo_args["algo"]["share_param"] = False
algo_args["algo"]["fixed_order"] = False
algo_args["model"]["hidden_sizes"] = [128, 128]
algo_args["seed"]["seed"] = 42
algo_args["device"]["cuda"] = True
algo_args["logger"]["log_dir"] = "/home/hyc1014/DL/FinalProject/results"

env_args["n_bs"] = 3
env_args["n_ue"] = 10
env_args["n_rb"] = 4
env_args["episode_length"] = 200
env_args["hierarchical"] = False
env_args["channel_source"] = "deepmimo"
env_args["obs_mode"] = "r2"          # partial obs: local KPM only, no neighbor
env_args["deepmimo_cache"] = "/home/hyc1014/DL/FinalProject/deepmimo_cache"

runner = RUNNER_REGISTRY["hasac"](args, algo_args, env_args)
runner.run()
all_eps = runner.done_episodes_rewards
final_raw_ep = float(np.mean(all_eps[-50:])) if len(all_eps) >= 50 else float(np.mean(all_eps)) if all_eps else float("nan")
runner.close()
import os as _os; sys.path.insert(0, _os.dirname(_os.path.abspath(__file__)))
from log_experiment import log_experiment
import numpy as np
log_experiment("R2-Flat HASAC", final_raw_ep, note="partial obs, DeepMIMO")
