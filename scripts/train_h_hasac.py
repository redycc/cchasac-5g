"""
H-HASAC: Hierarchical HASAC for 5G multi-cell resource allocation.

Strategy: subclass OffPolicyHARunner and override run() to inject manager.
Manager updates sub-goals on runner.envs.envs[0] every K steps.
Workers (HASAC) see sub-goals concatenated in their observation automatically.
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
    """Extends OffPolicyHARunner with a manager SAC layer."""

    def __init__(self, args, algo_args, env_args, manager_args=None):
        super().__init__(args, algo_args, env_args)

        # Manager hyperparams
        m = manager_args or {}
        self.K = m.get("K", 10)
        self.beta = m.get("beta", 0.3)
        self.mgr_batch = m.get("batch_size", 128)
        self.mgr_warmup = m.get("warmup_steps", 500)

        # Manager input = share_obs (flattened state visible to all workers)
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

        # Manager state tracking
        self._mgr_obs_start = None
        self._mgr_act = None
        self._k_rewards = []
        self._mgr_step = 0

        # Stats
        self.mgr_rewards_log = []
        self.subgoal_log = []

    def _get_inner_env(self):
        return self.envs.envs[0]

    def _inject_subgoal(self, share_obs_np):
        """Generate new sub-goal from manager and inject into env."""
        mgr_obs = share_obs_np[0, 0]  # (share_dim,) - first thread, first agent
        # Guard: if mgr_obs contains NaN, use neutral sub-goals
        if np.isnan(mgr_obs).any():
            subgoal_flat = np.full(self.num_agents * 3, 0.5, dtype=np.float32)
        else:
            subgoal_flat = self.manager.get_action(mgr_obs)  # (n_workers*3,)
            # Guard: replace any NaN in manager output
            subgoal_flat = np.nan_to_num(subgoal_flat, nan=0.5)
        subgoals = subgoal_flat.reshape(self.num_agents, 3)
        self._get_inner_env().set_sub_goals(subgoals)
        self._mgr_obs_start = mgr_obs.copy()
        self._mgr_act = subgoal_flat.copy()
        self._k_rewards = []
        self.subgoal_log.append(subgoals.copy())
        return subgoals

    def run(self):
        """Override run() to inject manager into the training loop."""
        if self.algo_args["render"]["use_render"]:
            self.render()
            return

        self.train_episode_rewards = np.zeros(self.algo_args["train"]["n_rollout_threads"])
        self.done_episodes_rewards = []
        self._all_ep_rewards = []  # persistent accumulator (not cleared during logging)

        # Warmup
        print("start warmup")
        obs, share_obs, available_actions = self.warmup()
        print("finish warmup, start H-HASAC training")

        # Initial sub-goal
        subgoals = self._inject_subgoal(share_obs)

        steps = self.algo_args["train"]["num_env_steps"] // self.algo_args["train"]["n_rollout_threads"]
        train_interval = self.algo_args["train"]["train_interval"]
        update_num = int(self.algo_args["train"]["update_per_train"] * train_interval)
        log_interval = self.algo_args["train"].get("log_interval") or 10_000
        cur_step = 0

        for step in range(1, steps + 1):
            cur_step = step * self.algo_args["train"]["n_rollout_threads"]

            # ── Worker actions ─────────────────────────────────────
            actions = self.get_actions(obs, available_actions=available_actions, add_random=False)

            # Hard constraint: clip power to P_max from sub-goal
            p_max_fracs = np.clip(subgoals[:, 0], 0.05, 1.0)  # (n_agents,)
            for ag in range(self.num_agents):
                actions[:, ag, :] = np.clip(actions[:, ag, :], 0, p_max_fracs[ag])

            # ── Env step ───────────────────────────────────────────
            new_obs, new_share_obs, rewards, dones, infos, new_avail = self.envs.step(actions)

            # Augment worker reward with goal achievement
            goal_penalty = np.zeros_like(rewards)
            for ag in range(self.num_agents):
                target_p = p_max_fracs[ag]
                actual_p = actions[0, ag].mean()
                goal_penalty[:, ag, :] = -self.beta * (actual_p - target_p) ** 2
            aug_rewards = rewards + goal_penalty

            # Track for manager (use raw rewards, not augmented)
            self._k_rewards.append(float(np.mean(rewards)))

            # ── Insert into worker buffer ──────────────────────────
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
            self.insert(data)  # also updates self.done_episodes_rewards

            # Mirror newly added episodes to permanent list
            if len(self.done_episodes_rewards) > len(self._all_ep_rewards):
                self._all_ep_rewards.extend(
                    self.done_episodes_rewards[len(self._all_ep_rewards):]
                )

            obs = new_obs.copy()
            share_obs = new_share_obs.copy()
            available_actions = new_avail

            # ── Worker update ──────────────────────────────────────
            if step > self.algo_args["train"]["warmup_steps"] and step % train_interval == 0:
                for _ in range(update_num):
                    self.train()

            # ── Manager update every K steps ───────────────────────
            self._mgr_step += 1
            if self._mgr_step % self.K == 0 and self._mgr_obs_start is not None:
                mgr_obs_now = share_obs[0, 0]
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

                # Issue new sub-goal
                subgoals = self._inject_subgoal(share_obs)

            # ── Log every log_interval steps ──────────────────────
            if cur_step % log_interval == 0:
                avg_mgr_r = np.mean(self.mgr_rewards_log[-20:]) if self.mgr_rewards_log else 0.0
                if len(self.done_episodes_rewards) > 0:
                    avg_ep_r = np.mean(self.done_episodes_rewards)
                    print(
                        f"Env fiveg Task bs3-ue10 Algo hasac Exp {self.args['exp_name']} "
                        f"Step {cur_step} / {self.algo_args['train']['num_env_steps']}, "
                        f"avg ep reward: {avg_ep_r:.3f}, mgr reward: {avg_mgr_r:.3f}, "
                        f"subgoal[0]=[p={subgoals[0,0]:.2f},i={subgoals[0,1]:.2f},rb={subgoals[0,2]:.2f}]\n"
                    )
                    self.log_file.write(f"{cur_step},{avg_ep_r}\n")
                    self.log_file.flush()
                    self.done_episodes_rewards = []
                else:
                    print(
                        f"Env fiveg Task bs3-ue10 Algo hasac Exp {self.args['exp_name']} "
                        f"Step {cur_step} / {self.algo_args['train']['num_env_steps']}, "
                        f"mgr reward: {avg_mgr_r:.3f} (no ep done yet)\n"
                    )
                self.save()

        print("\nH-HASAC training done.")
        # Flush any remaining episodes
        self._all_ep_rewards.extend(self.done_episodes_rewards)
        self.done_episodes_rewards = self._all_ep_rewards


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_runner(num_env_steps=300_000, K=10, beta=0.3, seed=42,
                 exp_name="h_hasac_v0", env_overrides=None):
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
    algo_args["algo"]["n_step"] = 1  # must be 1: n-step returns spanning sub-goal boundaries cause NaN
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

    # Allow caller to override env settings (e.g., channel_source="deepmimo")
    if env_overrides:
        env_args.update(env_overrides)

    manager_args = {"K": K, "beta": beta, "batch_size": 128, "buffer_size": 30_000,
                    "lr": 3e-4, "gamma": 0.95}

    return HHASACRunner(args, algo_args, env_args, manager_args)


if __name__ == "__main__":
    runner = build_runner(num_env_steps=300_000, K=10, beta=0.3, seed=42)
    runner.run()
    all_eps = runner.done_episodes_rewards
    final_raw_ep = float(np.mean(all_eps[-50:])) if len(all_eps) >= 50 else float(np.mean(all_eps)) if all_eps else float("nan")
    runner.close()
    import sys, os; sys.path.insert(0, os.path.dirname(__file__))
    from log_experiment import log_experiment
    log_experiment("H-HASAC (formula)", final_raw_ep, note=f"K=10, beta=0.3, seed=42")
