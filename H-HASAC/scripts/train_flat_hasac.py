"""Train flat HASAC on 5G multi-cell env (quick validation script)."""
import sys
import os

sys.path.insert(0, "/home/hyc1014/DL/FinalProject/HARL")
sys.path.insert(0, "/home/hyc1014/DL/FinalProject")

from harl.utils.configs_tools import get_defaults_yaml_args
from harl.runners import RUNNER_REGISTRY

args = {
    "algo": "hasac",
    "env": "fiveg",
    "exp_name": "flat_hasac_v0",
}

algo_args, env_args = get_defaults_yaml_args("hasac", "fiveg")

# Override for quick validation (3-hour run)
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
algo_args["algo"]["n_step"] = 5
algo_args["algo"]["gamma"] = 0.99
algo_args["algo"]["auto_alpha"] = True
algo_args["algo"]["alpha"] = 0.01
algo_args["algo"]["share_param"] = False
algo_args["algo"]["fixed_order"] = False
algo_args["model"]["hidden_sizes"] = [128, 128]
algo_args["seed"]["seed"] = 42
algo_args["device"]["cuda"] = True
algo_args["logger"]["log_dir"] = "/home/hyc1014/DL/FinalProject/results"

# Env config
env_args["n_bs"] = 3
env_args["n_ue"] = 10
env_args["n_rb"] = 4
env_args["episode_length"] = 200
env_args["hierarchical"] = False

runner = RUNNER_REGISTRY["hasac"](args, algo_args, env_args)
runner.run()
runner.close()
