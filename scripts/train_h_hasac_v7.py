"""
H-HASAC v7: Asymmetric soft penalty — remove hard clip, soft budget enforcement.

Root problem in v1-v6: hard clip (ceiling) is only binding when budget < natural
power (~0.5). Manager either gives all max (inactive) or throttles one agent to
p_min (degenerate). No design led to meaningful multi-agent coordination.

New approach:
  - Remove hard clip entirely — Workers choose power freely in [0,1]
  - Asymmetric soft penalty: -beta * relu(actual_p - target_p)^2
    Workers are penalized for EXCEEDING the budget, not for being below it.
    Workers may use less than budget freely; penalty deters going over.
  - beta=5.0: large enough that exceeding budget is unprofitable
  - Manager reward = pure raw K-step sum (no shaping); learns budgets organically
  - p_min removed (full [0,1] for Manager); p_min constraint was causing
    the degenerate [p_min, max, max] strategy in v4-v6
  - K=20 unchanged
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from collections import deque

sys.path.insert(0, "/home/hyc1014/DL/FinalProject/HARL")
sys.path.insert(0, "/home/hyc1014/DL/FinalProject")

from harl.utils.configs_tools import get_defaults_yaml_args
from harl.runners.off_policy_ha_runner import OffPolicyHARunner


# ─── Minimal Manager SAC ──────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(128, 128)):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ManagerSAC:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.95, polyak=0.005, device="cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.act_dim = act_dim
        self.gamma = gamma
        self.polyak = polyak

        self.actor = MLP(obs_dim, act_dim * 2).to(self.device)
        self.q1 = MLP(obs_dim + act_dim, 1).to(self.device)
        self.q2 = MLP(obs_dim + act_dim, 1).to(self.device)
        self.q1_tgt = deepcopy(self.q1)
        self.q2_tgt = deepcopy(self.q2)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.log_alpha = torch.tensor(np.log(0.01), requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -act_dim

    @torch.no_grad()
    def get_action(self, obs_np):
        obs = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        out = self.actor(obs)
        mean, log_std = out[:, :self.act_dim], out[:, self.act_dim:]
        log_std = torch.clamp(log_std, -5, 2)
        std = log_std.exp()
        action = torch.tanh(mean + std * torch.randn_like(mean))
        return ((action + 1) / 2).squeeze(0).cpu().numpy()  # [0,1]

    def _policy(self, obs):
        out = self.actor(obs)
        mean, log_std = out[:, :self.act_dim], out[:, self.act_dim:]
        log_std = torch.clamp(log_std, -5, 2)
        std = log_std.exp()
        z = mean + std * torch.randn_like(mean)
        action = torch.tanh(z)
        log_prob = (
            -((z - mean) ** 2) / (2 * std ** 2 + 1e-8)
            - log_std - 0.5 * np.log(2 * np.pi)
            - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)
        return (action + 1) / 2, log_prob

    def update(self, batch):
        obs, act, rew, nobs, done = [torch.FloatTensor(x).to(self.device) for x in batch]
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            na, nlp = self._policy(nobs)
            tgt = torch.min(
                self.q1_tgt(torch.cat([nobs, na], -1)),
                self.q2_tgt(torch.cat([nobs, na], -1)),
            ) - alpha * nlp
            backup = rew + self.gamma * (1 - done) * tgt
        q1_l = ((self.q1(torch.cat([obs, act], -1)) - backup) ** 2).mean()
        q2_l = ((self.q2(torch.cat([obs, act], -1)) - backup) ** 2).mean()
        q_loss = q1_l + q2_l
        if not q_loss.isnan():
            self.q_opt.zero_grad(); q_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
            self.q_opt.step()
        new_a, lp = self._policy(obs)
        a_l = (alpha * lp - torch.min(
            self.q1(torch.cat([obs, new_a], -1)),
            self.q2(torch.cat([obs, new_a], -1))
        )).mean()
        if not a_l.isnan():
            self.actor_opt.zero_grad(); a_l.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_opt.step()
        al = -(self.log_alpha * (lp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(); al.backward(); self.alpha_opt.step()
        for p, tp in zip(self.q1.parameters(), self.q1_tgt.parameters()):
            tp.data.mul_(1 - self.polyak).add_(self.polyak * p.data)
        for p, tp in zip(self.q2.parameters(), self.q2_tgt.parameters()):
            tp.data.mul_(1 - self.polyak).add_(self.polyak * p.data)
        return {"q_loss": (q1_l + q2_l).item() / 2, "a_loss": a_l.item()}


class ReplayBuffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)

    def push(self, *args):
        self.buf.append(tuple(np.array(a) for a in args))

    def sample(self, n):
        idx = np.random.choice(len(self.buf), n, replace=False)
        batch = [self.buf[i] for i in idx]
        return [np.stack([b[j] for b in batch]) for j in range(len(batch[0]))]

    def __len__(self):
        return len(self.buf)


# ─── Hierarchical runner ──────────────────────────────────────────────────────

class HHASACRunner(OffPolicyHARunner):
    """Extends OffPolicyHARunner with a diversity-incentivized manager SAC layer."""

    def __init__(self, args, algo_args, env_args, manager_args=None):
        super().__init__(args, algo_args, env_args)

        m = manager_args or {}
        self.K = m.get("K", 20)
        self.beta = m.get("beta", 5.0)  # asymmetric penalty: only penalize exceeding budget
        self.mgr_batch = m.get("batch_size", 128)
        self.mgr_warmup = m.get("warmup_steps", 500)

        mgr_obs_dim = self.envs.share_observation_space[0].shape[0]
        n_workers = self.num_agents
        mgr_act_dim = n_workers * 3  # (p_max, i_thresh, rb_share) per worker

        self.manager = ManagerSAC(
            mgr_obs_dim, mgr_act_dim,
            lr=m.get("lr", 3e-4),
            gamma=m.get("gamma", 0.95),
            device="cuda" if algo_args["device"]["cuda"] else "cpu",
        )
        self.mgr_buf = ReplayBuffer(m.get("buffer_size", 30_000))

        self._mgr_obs_start = None
        self._mgr_act = None
        self._k_rewards = []
        self._mgr_step = 0

        self.mgr_rewards_log = []
        self.subgoal_log = []

    def _get_inner_env(self):
        return self.envs.envs[0]

    def _inject_subgoal(self, share_obs_np):
        """Generate new sub-goal from manager and inject into env."""
        mgr_obs = share_obs_np[0, 0]
        if np.isnan(mgr_obs).any():
            subgoal_flat = np.full(self.num_agents * 3, 0.5, dtype=np.float32)
        else:
            subgoal_flat = self.manager.get_action(mgr_obs)
            subgoal_flat = np.nan_to_num(subgoal_flat, nan=0.5)
        subgoals = subgoal_flat.reshape(self.num_agents, 3)
        # No p_min: full [0,1] range — Manager decides freely
        self._get_inner_env().set_sub_goals(subgoals)
        self._mgr_obs_start = mgr_obs.copy()
        self._mgr_act = subgoal_flat.copy()
        self._k_rewards = []
        self.subgoal_log.append(subgoals.copy())
        return subgoals

    def run(self):
        if self.algo_args["render"]["use_render"]:
            self.render()
            return

        self.train_episode_rewards = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.done_episodes_rewards = []
        self._all_ep_rewards = []

        print("start warmup")
        obs, share_obs, available_actions = self.warmup()
        print("finish warmup, start H-HASAC v7 training")

        subgoals = self._inject_subgoal(share_obs)

        steps = self.algo_args["train"]["num_env_steps"] // self.algo_args["train"]["n_rollout_threads"]
        train_interval = self.algo_args["train"]["train_interval"]
        update_num = int(self.algo_args["train"]["update_per_train"] * train_interval)
        log_interval = self.algo_args["train"].get("log_interval") or 10_000
        cur_step = 0

        for step in range(1, steps + 1):
            cur_step = step * self.algo_args["train"]["n_rollout_threads"]

            actions = self.get_actions(obs, available_actions=available_actions, add_random=False)

            # No hard clip — Workers choose freely; budget enforced via soft penalty below
            new_obs, new_share_obs, rewards, dones, infos, new_avail = self.envs.step(actions)

            # Asymmetric soft penalty: only penalise exceeding budget
            p_targets = subgoals[:, 0]  # (n_agents,) budget in [0,1]
            goal_penalty = np.zeros_like(rewards)
            for ag in range(self.num_agents):
                actual_p = float(actions[0, ag].mean())
                overage = max(0.0, actual_p - float(p_targets[ag]))
                goal_penalty[:, ag, :] = -self.beta * overage ** 2
            aug_rewards = rewards + goal_penalty

            self._k_rewards.append(float(np.mean(rewards)))

            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                available_actions.transpose(1, 0, 2) if available_actions is not None and len(np.array(available_actions).shape) == 3 else available_actions,
                aug_rewards,
                dones,
                infos,
                new_share_obs,
                new_obs,
                new_avail.transpose(1, 0, 2) if new_avail is not None and len(np.array(new_avail).shape) == 3 else new_avail,
            )
            self.insert(data)

            if len(self.done_episodes_rewards) > len(self._all_ep_rewards):
                self._all_ep_rewards.extend(
                    self.done_episodes_rewards[len(self._all_ep_rewards):]
                )

            obs = new_obs.copy()
            share_obs = new_share_obs.copy()
            available_actions = new_avail

            if step > self.algo_args["train"]["warmup_steps"] and step % train_interval == 0:
                for _ in range(update_num):
                    self.train()

            self._mgr_step += 1
            if self._mgr_step % self.K == 0 and self._mgr_obs_start is not None:
                mgr_obs_now = share_obs[0, 0]

                # Manager reward: pure raw K-step sum (no shaping)
                mgr_rew = float(np.mean(self._k_rewards)) * self.K

                self.mgr_rewards_log.append(mgr_rew)

                self.mgr_buf.push(
                    self._mgr_obs_start,
                    self._mgr_act,
                    np.array([mgr_rew]),
                    mgr_obs_now,
                    np.array([0.0]),
                )

                if len(self.mgr_buf) >= self.mgr_batch:
                    batch = self.mgr_buf.sample(self.mgr_batch)
                    self.manager.update(batch)

                subgoals = self._inject_subgoal(share_obs)

            if cur_step % log_interval == 0:
                avg_mgr_r = np.mean(self.mgr_rewards_log[-20:]) if self.mgr_rewards_log else 0.0
                p_vals = subgoals[:, 0]
                p_str = ",".join(f"{p:.2f}" for p in p_vals)
                p_std = float(np.std(p_vals))
                if len(self.done_episodes_rewards) > 0:
                    avg_ep_r = np.mean(self.done_episodes_rewards)
                    print(
                        f"Env fiveg Task bs3-ue10 Algo hasac Exp {self.args['exp_name']} "
                        f"Step {cur_step} / {self.algo_args['train']['num_env_steps']}, "
                        f"avg ep reward: {avg_ep_r:.3f}, mgr reward: {avg_mgr_r:.3f}, "
                        f"p=[{p_str}] std={p_std:.3f}\n"
                    )
                    self.log_file.write(f"{cur_step},{avg_ep_r}\n")
                    self.log_file.flush()
                    self.done_episodes_rewards = []
                else:
                    print(
                        f"Env fiveg Task bs3-ue10 Algo hasac Exp {self.args['exp_name']} "
                        f"Step {cur_step} / {self.algo_args['train']['num_env_steps']}, "
                        f"mgr reward: {avg_mgr_r:.3f} p=[{p_str}] std={p_std:.3f} (no ep done yet)\n"
                    )
                self.save()

        print("\nH-HASAC v7 training done.")
        self._all_ep_rewards.extend(self.done_episodes_rewards)
        self.done_episodes_rewards = self._all_ep_rewards


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_runner(num_env_steps=300_000, K=20, seed=42,
                 exp_name="h_hasac_v7", env_overrides=None):
    args = {"algo": "hasac", "env": "fiveg", "exp_name": exp_name}
    algo_args, env_args = get_defaults_yaml_args("hasac", "fiveg")

    algo_args["train"]["n_rollout_threads"] = 1
    algo_args["train"]["num_env_steps"] = num_env_steps
    algo_args["train"]["warmup_steps"] = 2_000
    algo_args["train"]["train_interval"] = 10
    algo_args["train"]["eval_interval"] = 10_000
    algo_args["train"]["use_valuenorm"] = True
    algo_args["eval"]["use_eval"] = False
    algo_args["algo"]["batch_size"] = 256
    algo_args["algo"]["buffer_size"] = 100_000
    algo_args["algo"]["n_step"] = 1
    algo_args["algo"]["auto_alpha"] = True
    algo_args["algo"]["alpha"] = 0.01
    algo_args["algo"]["share_param"] = False
    algo_args["model"]["hidden_sizes"] = [128, 128]
    algo_args["seed"]["seed"] = seed
    algo_args["device"]["cuda"] = True
    algo_args["logger"]["log_dir"] = "/home/hyc1014/DL/FinalProject/results"

    env_args["n_bs"] = 3
    env_args["n_ue"] = 10
    env_args["n_rb"] = 4
    env_args["episode_length"] = 200
    env_args["hierarchical"] = True

    if env_overrides:
        env_args.update(env_overrides)

    manager_args = {
        "K": K,
        "beta": 5.0,          # asymmetric soft penalty weight
        "batch_size": 128,
        "buffer_size": 30_000,
        "lr": 3e-4,
        "gamma": 0.95,
    }

    return HHASACRunner(args, algo_args, env_args, manager_args)


if __name__ == "__main__":
    runner = build_runner(
        num_env_steps=300_000,
        K=20,
        seed=42,
        exp_name="h_hasac_v7",
        env_overrides={"channel_source": "deepmimo"},
    )
    runner.run()
    runner.close()
