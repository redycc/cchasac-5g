"""
z representation analysis for C-HASAC.
Loads best checkpoint, runs encoder on multiple scenarios,
analyzes correlation between z dimensions and power outputs,
and runs PCA on z vectors.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import env_chasac as E
from env_chasac import Env
from scripts.train_chasac import SetActor, Encoder, encode_kpm, build_obs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SCENARIOS = 200
SEED = 42

def load_checkpoint(path):
    state = torch.load(path, map_location=DEVICE)
    return state["actor"], state["encoder"]

def main():
    ckpt_path = "results/chasac_z_analysis_best.pt"
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    actor_state, encoder_state = load_checkpoint(ckpt_path)

    # ---- rebuild models ----
    cfg = {"n_bs": 3, "n_ue": 12, "z_dim": 16, "kpm_dim": 5,
           "ue_feat": 3, "hidden": 256}
    actor = SetActor(ue_feat=cfg["ue_feat"], z_dim=cfg["z_dim"], hidden=cfg["hidden"]).to(DEVICE)
    encoder = Encoder(kpm_dim=cfg["kpm_dim"], z_dim=cfg["z_dim"], hidden=128).to(DEVICE)
    actor.load_state_dict(actor_state)
    encoder.load_state_dict(encoder_state)
    actor.mu_bound = 5.0  # not saved in state_dict, must restore manually
    actor.eval(); encoder.eval()

    env = Env(E.Cfg(), seed=SEED)

    z_list, power_list, kpm_list = [], [], []
    env_cfg = E.Cfg()
    pmax = E.dbm_to_w(env_cfg.Pmax_dBm)

    with torch.no_grad():
        for i in range(N_SCENARIOS):
            env = Env(env_cfg, seed=SEED + i)
            env.reset()
            # run T steps; collect z and power at each step (skip t=0, start from t=1+)
            for t in range(10):
                obs_local, mask_np, obs_kpm, _ = build_obs(env)

                kpm = torch.as_tensor(obs_kpm[None], device=DEVICE)
                z = encode_kpm(encoder, kpm, cfg["n_bs"], remove_own=False)  # [1, z_dim]

                obs_l = torch.as_tensor(obs_local[None], device=DEVICE)
                mask = torch.as_tensor(mask_np[None], device=DEVICE)
                a = actor.act(obs_l, mask, z, deterministic=True)[0].cpu().numpy()  # [N_UE]

                # per-BS mean power fraction
                serv = env.serv
                bs_power = np.array([((a[serv == b] + 1) / 2).mean() if (serv == b).any() else 0.0
                                     for b in range(cfg["n_bs"])])

                if t >= 2:  # skip first 2 steps (Rbar still near initial value)
                    z_list.append(z.squeeze(0).cpu().numpy())
                    power_list.append(bs_power)
                    kpm_list.append(obs_kpm)

                # step environment with actor's action
                from scripts.train_chasac import action_to_powerlist
                pl = action_to_powerlist(a, serv, cfg["n_bs"], pmax)
                env.step(pl)

    z_arr = np.array(z_list)       # [N, 16]
    pwr_arr = np.array(power_list) # [N, 3]
    kpm_arr = np.array(kpm_list)   # [N, 3, 5]

    print(f"\n=== z statistics ===")
    print(f"z mean: {z_arr.mean(0).round(3)}")
    print(f"z std:  {z_arr.std(0).round(3)}")
    print(f"z range: [{z_arr.min():.3f}, {z_arr.max():.3f}]")

    print(f"\n=== z vs BS power correlation ===")
    print(f"{'z_dim':>6}  {'corr_BS0':>9}  {'corr_BS1':>9}  {'corr_BS2':>9}")
    max_corr = []
    for d in range(cfg["z_dim"]):
        corrs = [np.corrcoef(z_arr[:, d], pwr_arr[:, b])[0, 1] for b in range(cfg["n_bs"])]
        max_corr.append(max(abs(c) for c in corrs))
        print(f"  z[{d:02d}]  {corrs[0]:+9.3f}  {corrs[1]:+9.3f}  {corrs[2]:+9.3f}")

    print(f"\nTop-5 z dims by max |corr| with any BS power:")
    top5 = np.argsort(max_corr)[::-1][:5]
    for d in top5:
        corrs = [np.corrcoef(z_arr[:, d], pwr_arr[:, b])[0, 1] for b in range(cfg["n_bs"])]
        print(f"  z[{d:02d}] max|corr|={max_corr[d]:.3f}  per BS: {[f'{c:+.3f}' for c in corrs]}")

    # ---- PCA ----
    z_centered = z_arr - z_arr.mean(0)
    U, S, Vt = np.linalg.svd(z_centered, full_matrices=False)
    var_explained = (S**2) / (S**2).sum()
    print(f"\n=== PCA variance explained ===")
    cumvar = 0
    for i, v in enumerate(var_explained[:8]):
        cumvar += v
        print(f"  PC{i+1}: {v*100:.1f}%  (cumulative: {cumvar*100:.1f}%)")

    # ---- KPM correlation ----
    kpm_names = ["load", "throughput", "P_bs", "dist_j", "dist_k"]
    print(f"\n=== z[top5] vs KPM correlation ===")
    for d in top5:
        print(f"  z[{d:02d}]:")
        for bs in range(cfg["n_bs"]):
            for ki, kname in enumerate(kpm_names):
                corr = np.corrcoef(z_arr[:, d], kpm_arr[:, bs, ki])[0, 1]
                if abs(corr) > 0.3:
                    print(f"    BS{bs} {kname}: {corr:+.3f}")

    # ---- on/off switch check ----
    print(f"\n=== On/Off switch check ===")
    print(f"Fraction of scenarios where any BS has mean power < 0.1: "
          f"{(pwr_arr.min(1) < 0.1).mean():.2%}")
    print(f"Fraction of scenarios where any BS has mean power > 0.9: "
          f"{(pwr_arr.max(1) > 0.9).mean():.2%}")
    print(f"Mean power per BS: {pwr_arr.mean(0).round(3)}")
    print(f"Std power per BS:  {pwr_arr.std(0).round(3)}")

    np.save("results/chasac_z_analysis_z_vectors.npy", z_arr)
    np.save("results/chasac_z_analysis_power.npy", pwr_arr)
    print(f"\nSaved z_vectors and power to results/")

if __name__ == "__main__":
    main()
