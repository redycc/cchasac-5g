"""
Generate publication-quality figures from z analysis data.
Saves all figures to results/z_figs/
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SAVE_DIR = "results/z_figs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── load pre-computed arrays ──────────────────────────────────────────────────
z_arr   = np.load("results/chasac_z_analysis_z_vectors.npy")   # [N, 16]
pwr_arr = np.load("results/chasac_z_analysis_power.npy")        # [N, 3]

# also need kpm — recompute from raw data
import torch
import env_chasac as E
from scripts.train_chasac import SetActor, Encoder, encode_kpm, build_obs, action_to_powerlist

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED, N_SCENARIOS = 42, 200

ckpt = torch.load("results/chasac_z_analysis_best.pt", map_location=DEVICE)
cfg_env = E.Cfg()
actor   = SetActor(ue_feat=3, z_dim=16, hidden=256).to(DEVICE)
encoder = Encoder(kpm_dim=5, z_dim=16, hidden=128).to(DEVICE)
actor.load_state_dict(ckpt["actor"]); encoder.load_state_dict(ckpt["encoder"])
actor.mu_bound = 5.0; actor.eval(); encoder.eval()

kpm_list = []
pmax = E.dbm_to_w(cfg_env.Pmax_dBm)
with torch.no_grad():
    for i in range(N_SCENARIOS):
        env = E.Env(cfg_env, seed=SEED + i)
        env.reset()
        for t in range(10):
            obs_local, mask_np, obs_kpm, _ = build_obs(env)
            if t >= 2:
                kpm_list.append(obs_kpm)
            a = actor.act(torch.as_tensor(obs_local[None], device=DEVICE),
                          torch.as_tensor(mask_np[None], device=DEVICE),
                          encode_kpm(encoder, torch.as_tensor(obs_kpm[None], device=DEVICE), 3, False),
                          deterministic=True)[0].cpu().numpy()
            env.step(action_to_powerlist(a, env.serv, 3, pmax))

kpm_arr = np.array(kpm_list)  # [N, 3, 5]
KPM_NAMES = ["Load", "Throughput", "P_bs", "Dist_j", "Dist_k"]
N = len(z_arr)

# ── PCA ───────────────────────────────────────────────────────────────────────
z_c = z_arr - z_arr.mean(0)
_, S, Vt = np.linalg.svd(z_c, full_matrices=False)
var_exp = (S**2) / (S**2).sum()
pcs = z_c @ Vt.T   # [N, 16] principal component scores

# ── FIGURE 1: PCA Variance Explained ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.5))
bars = ax.bar(range(1, 17), var_exp * 100, color="#2e86ab", alpha=0.85, edgecolor="white")
ax.plot(range(1, 17), np.cumsum(var_exp) * 100, "o-", color="#dc3545", lw=1.5,
        ms=4, label="Cumulative")
ax.axhline(95, ls="--", color="gray", lw=1, alpha=0.7, label="95% threshold")
bars[0].set_color("#1a3a5c")   # highlight PC1
ax.set_xlabel("Principal Component", fontsize=11)
ax.set_ylabel("Variance Explained (%)", fontsize=11)
ax.set_title("PCA of Learned Context z  (16-dim → effectively 1-dim)", fontsize=11)
ax.set_xticks(range(1, 17))
ax.legend(fontsize=9)
ax.annotate(f"PC1 = {var_exp[0]*100:.1f}%", xy=(1, var_exp[0]*100),
            xytext=(3, var_exp[0]*100 + 2), fontsize=10, color="#1a3a5c", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1a3a5c", lw=1.2))
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig1_pca_variance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1_pca_variance.png")

# ── FIGURE 2: z vs BS Power Correlation Heatmap ───────────────────────────────
corr_mat = np.array([[np.corrcoef(z_arr[:, d], pwr_arr[:, b])[0, 1]
                      for b in range(3)] for d in range(16)])  # [16, 3]

fig, ax = plt.subplots(figsize=(4, 6))
im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["BS 0", "BS 1", "BS 2"], fontsize=10)
ax.set_yticks(range(16)); ax.set_yticklabels([f"z[{d:02d}]" for d in range(16)], fontsize=8)
ax.set_title("z dim vs BS Power\nCorrelation", fontsize=11)
for d in range(16):
    for b in range(3):
        v = corr_mat[d, b]
        if abs(v) > 0.18:
            ax.text(b, d, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.28 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig2_z_power_corr.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_z_power_corr.png")

# ── FIGURE 3: z vs KPM Correlation Heatmap ───────────────────────────────────
# average over BSes for each KPM
corr_kpm = np.zeros((16, 5))
for d in range(16):
    for ki in range(5):
        vals = kpm_arr[:, :, ki].mean(1)   # mean across BS for this KPM
        corr_kpm[d, ki] = np.corrcoef(z_arr[:, d], vals)[0, 1]

fig, ax = plt.subplots(figsize=(5, 6))
im = ax.imshow(corr_kpm, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
ax.set_xticks(range(5)); ax.set_xticklabels(KPM_NAMES, fontsize=9, rotation=20, ha="right")
ax.set_yticks(range(16)); ax.set_yticklabels([f"z[{d:02d}]" for d in range(16)], fontsize=8)
ax.set_title("z dim vs KPM\nCorrelation (mean over BSes)", fontsize=11)
for d in range(16):
    for ki in range(5):
        v = corr_kpm[d, ki]
        if abs(v) > 0.2:
            ax.text(ki, d, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.35 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig3_z_kpm_corr.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_z_kpm_corr.png")

# ── FIGURE 4: PC1 vs Throughput scatter ───────────────────────────────────────
tp_mean = kpm_arr[:, :, 1].mean(1)   # mean throughput across BSes
fig, ax = plt.subplots(figsize=(5, 4))
sc = ax.scatter(tp_mean, pcs[:, 0], c=pwr_arr.mean(1), cmap="viridis",
                s=8, alpha=0.5)
plt.colorbar(sc, ax=ax, label="Mean BS power frac")
r = np.corrcoef(tp_mean, pcs[:, 0])[0, 1]
ax.set_xlabel("Mean Throughput (KPM)", fontsize=11)
ax.set_ylabel("z PC1 Score", fontsize=11)
ax.set_title(f"PC1 vs System Throughput  (r = {r:+.3f})", fontsize=11)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig4_pc1_vs_throughput.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig4_pc1_vs_throughput.png")

# ── FIGURE 5: Power Distribution per BS ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
colors = ["#2e86ab", "#e07a5f", "#3d405b"]
for b, (ax, c) in enumerate(zip(axes, colors)):
    ax.hist(pwr_arr[:, b], bins=30, color=c, alpha=0.8, edgecolor="white")
    ax.axvline(pwr_arr[:, b].mean(), color="black", ls="--", lw=1.5,
               label=f"mean={pwr_arr[:,b].mean():.2f}")
    ax.set_xlabel("Mean Power Fraction", fontsize=10)
    ax.set_title(f"BS {b}", fontsize=11)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Count", fontsize=10)
fig.suptitle("Per-BS Power Distribution  (bimodal → soft on/off behavior)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/fig5_power_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig5_power_distribution.png")

# ── FIGURE 6: Summary Panel ───────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(1, 3, wspace=0.35)

# Left: PCA bars
ax1 = fig.add_subplot(gs[0])
ax1.bar(range(1, 9), var_exp[:8] * 100, color="#2e86ab", alpha=0.85, edgecolor="white")
ax1.bar(1, var_exp[0] * 100, color="#1a3a5c", alpha=0.95, edgecolor="white",
        label=f"PC1={var_exp[0]*100:.1f}%")
ax1.set_title("(a) z PCA", fontsize=11)
ax1.set_xlabel("PC index"); ax1.set_ylabel("Variance (%)")
ax1.legend(fontsize=9)

# Middle: correlation heatmap (top 8 z dims × 3 BSes)
ax2 = fig.add_subplot(gs[1])
top8 = np.argsort([max(abs(corr_mat[d])) for d in range(16)])[::-1][:8]
sub = corr_mat[top8]
im2 = ax2.imshow(sub, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
ax2.set_xticks([0,1,2]); ax2.set_xticklabels(["BS0","BS1","BS2"], fontsize=9)
ax2.set_yticks(range(8)); ax2.set_yticklabels([f"z[{top8[i]:02d}]" for i in range(8)], fontsize=8)
ax2.set_title("(b) z–Power Corr (top-8 dims)", fontsize=11)
plt.colorbar(im2, ax=ax2, fraction=0.05)

# Right: power histogram combined
ax3 = fig.add_subplot(gs[2])
for b, c in zip(range(3), ["#2e86ab","#e07a5f","#3d405b"]):
    ax3.hist(pwr_arr[:, b], bins=25, color=c, alpha=0.55, label=f"BS{b}", edgecolor="none")
ax3.set_title("(c) BS Power Distribution", fontsize=11)
ax3.set_xlabel("Mean Power Fraction"); ax3.set_ylabel("Count")
ax3.legend(fontsize=9)
ax3.set_xlim(0, 1)

plt.savefig(f"{SAVE_DIR}/fig6_summary_panel.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig6_summary_panel.png")

print(f"\nAll figures saved to {SAVE_DIR}/")
