"""Regenerate every new figure in the ORIGINAL manuscript's plotting style.

Style extracted from the code that produced the submitted figures:
  * DL figures  -- train_unified_5fold.py: plot_roc_curve / plot_roc_overlay /
    plot_confusion_matrix. dpi=150, grid(alpha=0.3), bold titles fontsize 14,
    axis labels 13, legend frameon+fancybox, mean curve #B2182B lw=2.5 with a
    +/-1 SD band at alpha 0.15, per-fold curves from viridis, chance dashed.
  * SPP figures -- spp_analysis.py: HC #4C72B0, PD #DD8452, outlier #C44E52,
    grid(alpha=0.3), titles 13, labels 11, bold suptitle 14, DejaVu Sans.
"""

import os

# Directory layout. Override with environment variables to point at a
# different location; defaults assume the results were downloaded into
# ./results next to this repository.
RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./results/analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import os, glob, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

warnings.filterwarnings("ignore")
R = RESULTS_DIR
D = R + "/dl"
FIG = OUTPUT_DIR + "/figures"
os.makedirs(FIG, exist_ok=True)
SEEDS, MODELS = [42, 43, 44], ["A", "B", "C"]
LBL = {"A": "Model A (IBA1)", "B": "Model B (pSyn)", "C": "Model C (IBA1+pSyn)"}

# ---- original rcParams --------------------------------------------------
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
C_HC, C_PD, C_OUT = "#4C72B0", "#DD8452", "#C44E52"
C_MEAN, C_MAIN = "#B2182B", "#2166AC"
C_MODEL = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}


def load(seed, region, m):
    return pd.read_csv(f"{D}/seed{seed}_{region}_patient_predictions_{m}.csv")


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


per_seed = pd.read_csv(OUTPUT_DIR + "/v2_pooled_oof_per_seed.csv")

# =========================================================================
# 1. ROC overlay, one panel per model -- plot_roc_overlay style
# =========================================================================
for region, rlab in (("substantiaNigra", "Substantia Nigra"), ("putamen", "Putamen")):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=150)
    for ax, m in zip(axes, MODELS):
        base_fpr = np.linspace(0, 1, 101)
        tprs, aucs = [], []
        cmap = plt.cm.viridis
        for i, seed in enumerate(SEEDS):
            d = load(seed, region, m)
            fpr, tpr, _ = roc_curve(d.True_Label, d.Pred_Prob_PD)
            a = roc_auc_score(d.True_Label, d.Pred_Prob_PD)
            aucs.append(a)
            ti = np.interp(base_fpr, fpr, tpr); ti[0] = 0.0
            tprs.append(ti)
            ax.plot(fpr, tpr, lw=1.2, alpha=0.55, color=cmap(i / max(len(SEEDS) - 1, 1)),
                    label=f"Repetition {i + 1} (AUC={a:.2f})")
        arr = np.array(tprs)
        mean_tpr, std_tpr = arr.mean(axis=0), arr.std(axis=0)
        ax.plot(base_fpr, mean_tpr, color=C_MEAN, lw=2.5,
                label=f"Mean (AUC={np.mean(aucs):.3f} +/- {np.std(aucs):.3f})")
        ax.fill_between(base_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        color=C_MEAN, alpha=0.15, label="+/- 1 SD")
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, alpha=0.6)
        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate", fontsize=13)
        ax.set_title(LBL[m], fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9, frameon=True, fancybox=True)
        ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Donor-level ROC, pooled out-of-fold -- {rlab}",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    save(fig, f"fig_roc_{region}")

# =========================================================================
# 2. Precision-recall, same style
# =========================================================================
for region, rlab in (("substantiaNigra", "Substantia Nigra"), ("putamen", "Putamen")):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=150)
    for ax, m in zip(axes, MODELS):
        cmap = plt.cm.viridis
        for i, seed in enumerate(SEEDS):
            d = load(seed, region, m)
            pr, rc, _ = precision_recall_curve(d.True_Label, d.Pred_Prob_PD)
            ax.plot(rc, pr, lw=1.2, alpha=0.55, color=cmap(i / max(len(SEEDS) - 1, 1)),
                    label=f"Repetition {i + 1}")
        ax.axhline(0.5, ls="--", color="grey", lw=1, alpha=0.6, label="Chance (balanced)")
        ax.set_xlabel("Recall", fontsize=13)
        ax.set_ylabel("Precision", fontsize=13)
        ax.set_title(LBL[m], fontsize=14, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9, frameon=True, fancybox=True)
        ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Donor-level precision-recall -- {rlab}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save(fig, f"fig_pr_{region}")

# =========================================================================
# 3. AUROC with bootstrap CI, per repetition
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=150, sharey=True)
for ax, (region, rlab) in zip(axes, (("substantiaNigra", "Substantia Nigra"),
                                     ("putamen", "Putamen"))):
    sub = per_seed[per_seed.Region == region]
    for i, m in enumerate(MODELS):
        s = sub[sub.Model == m]
        for j, (_, r) in enumerate(s.iterrows()):
            x = i + (j - 1) * 0.22
            ax.plot([x, x], [r.lo, r.hi], color=C_MODEL[m], lw=2.2, alpha=0.85,
                    solid_capstyle="round")
            ax.plot(x, r.AUROC, "o", ms=9, color=C_MODEL[m],
                    markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.axhline(0.5, ls="--", color="grey", lw=1.2, alpha=0.7)
    ax.set_xticks(range(3))
    ax.set_xticklabels([LBL[m].replace("Model ", "") for m in MODELS], fontsize=11)
    ax.set_title(rlab, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3, axis="y")
axes[0].set_ylabel("Donor-level AUROC (95% CI)", fontsize=13)
fig.suptitle("Pooled out-of-fold AUROC across three cross-validation repetitions",
             fontsize=16, fontweight="bold")
plt.tight_layout()
save(fig, "fig_auroc_ci_by_seed")

# =========================================================================
# 4. CNN vs interpretable baselines -- forest style
# =========================================================================
base = pd.read_csv(OUTPUT_DIR + "/dl_burden_baselines.csv")
base = base[base.Region == "SN"]
sn = per_seed[per_seed.Region == "substantiaNigra"]
items = []
for m in MODELS:
    s = sn[sn.Model == m]
    items.append((f"CNN -- {LBL[m].replace('Model ', '')}",
                  s.AUROC.mean(), s.lo.mean(), s.hi.mean(), True))
for _, r in base.iterrows():
    items.append((r.Baseline, r.AUROC, r.lo, r.hi, False))
items.sort(key=lambda t: t[1])

fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
for i, (name, a, lo, hi, is_cnn) in enumerate(items):
    c = C_OUT if is_cnn else C_HC
    ax.plot([lo, hi], [i, i], color=c, lw=2.4, alpha=0.85, solid_capstyle="round")
    ax.plot(a, i, "o", ms=10, color=c, markeredgecolor="white",
            markeredgewidth=1.3, zorder=5)
    ax.text(hi + 0.012, i, f"{a:.3f}", va="center", fontsize=10, color=c,
            fontweight="bold")
ax.axvline(0.5, ls="--", color="grey", lw=1.2, alpha=0.7, label="Chance")
ax.set_yticks(range(len(items)))
ax.set_yticklabels([t[0] for t in items], fontsize=11)
ax.set_xlabel("Donor-level AUROC (95% CI)", fontsize=13)
ax.set_title("Learned representation vs interpretable scalars -- Substantia Nigra",
             fontsize=14, fontweight="bold")
ax.set_xlim(0, 1.08)
ax.grid(True, alpha=0.3, axis="x")
ax.legend(loc="lower right", fontsize=11, frameon=True, fancybox=True)
plt.tight_layout()
save(fig, "fig_cnn_vs_baselines")

# =========================================================================
# 5. XAI validity
# =========================================================================
val = pd.read_csv(OUTPUT_DIR + "/xai_validity.csv")
avg = pd.read_csv(OUTPUT_DIR + "/xai_donor_level_seedavg.csv")

fig, axes = plt.subplots(1, 3, figsize=(19, 6), dpi=150)

ax = axes[0]
for i, m in enumerate(MODELS):
    v = val[(val.model == m) & (val.metric == "iba1_enrichment")].iloc[0]
    ax.plot([v.ci_lo, v.ci_hi], [i, i], color=C_MODEL[m], lw=2.4, solid_capstyle="round")
    ax.plot(v["median"], i, "o", ms=10, color=C_MODEL[m],
            markeredgecolor="white", markeredgewidth=1.3, zorder=5)
ax.axvline(1.0, ls="--", color="grey", lw=1.2, alpha=0.7, label="No preference")
ax.set_yticks(range(3))
ax.set_yticklabels([LBL[m].replace("Model ", "") for m in MODELS], fontsize=11)
ax.set_xlabel("IBA1 enrichment within AOR", fontsize=13)
ax.set_title("Relevance is anchored on marker-positive tissue",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")
ax.legend(loc="lower right", fontsize=10, frameon=True, fancybox=True)

ax = axes[1]
for i, m in enumerate(MODELS):
    v = val[(val.model == m) & (val.metric == "z_psyn_spearman")].iloc[0]
    ax.plot([v.ci_lo, v.ci_hi], [i, i], color=C_MODEL[m], lw=2.4, solid_capstyle="round")
    ax.plot(v["median"], i, "o", ms=10, color=C_MODEL[m],
            markeredgecolor="white", markeredgewidth=1.3, zorder=5)
ax.axvline(0.0, ls="--", color="grey", lw=1.2, alpha=0.7)
ax.set_yticks(range(3)); ax.set_yticklabels([])
ax.set_xlabel("Axial relevance vs pSyn depth profile (Spearman $\\rho$)", fontsize=13)
ax.set_title("Axial profile tracks tissue signal", fontsize=13, fontweight="bold")
ax.set_xlim(0, 1); ax.grid(True, alpha=0.3, axis="x")

ax = axes[2]
sub = avg[avg.model == "A"]
rng = np.random.default_rng(42)
data = [sub.loc[sub.group == g, "iba1_enrichment"].dropna().to_numpy()
        for g in ("HC", "PD")]
bp = ax.boxplot(data, labels=["HC", "PD"], patch_artist=True, widths=0.5)
for patch, color in zip(bp["boxes"], [C_HC, C_PD]):
    patch.set_facecolor(color); patch.set_alpha(0.4)
for i, (v, c) in enumerate(zip(data, (C_HC, C_PD))):
    ax.scatter(i + 1 + rng.uniform(-0.1, 0.1, len(v)), v, color="black",
               alpha=0.6, s=32, zorder=5)
ax.set_ylabel("IBA1 enrichment within AOR", fontsize=13)
ax.set_title("No group difference (BH $p$ = 0.95)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
fig.suptitle("Held-out relevance: validity checks and group comparison",
             fontsize=16, fontweight="bold")
plt.tight_layout()
save(fig, "fig_xai_validity")

# =========================================================================
# 6. Composite grids of existing original-style panels
# =========================================================================
def grid(paths, titles, ncols, name, per_w, per_h, suptitle):
    paths = [p for p in paths if p and os.path.exists(p)]
    if not paths:
        print("  SKIP", name); return
    nrows = (len(paths) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(per_w * ncols, per_h * nrows), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    for ax, p, t in zip(axes, paths, titles):
        ax.imshow(mpimg.imread(p))
        ax.set_title(t, fontsize=12, fontweight="bold")
        ax.axis("off")
    for ax in axes[len(paths):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=16, fontweight="bold")
    plt.tight_layout()
    save(fig, name)


L = RESULTS_DIR + "/spp-results/L_plots"
picks, titles = [], []
for grp, reg, lab in (("HC", "substantiaNigra", "Control -- Substantia Nigra"),
                      ("PD", "substantiaNigra", "Parkinson's -- Substantia Nigra"),
                      ("HC", "putamen", "Control -- Putamen"),
                      ("PD", "putamen", "Parkinson's -- Putamen")):
    cand = sorted(glob.glob(f"{L}/L_{grp}_{reg}_*.png"))
    if cand:
        picks.append(cand[0]); titles.append(lab)
grid(picks, titles, 2, "fig_L_function_envelopes", 8, 6,
     "Bivariate $L$-function with Monte Carlo random-labelling envelopes")

DF = RESULTS_DIR + "/dl_figs"
tc, tt = [], []
for m in MODELS:
    for fold in (1, 3):
        p = f"{DF}/substantiaNigra_training_curves_{m}_Fold_{fold:02d}.png"
        if os.path.exists(p):
            tc.append(p); tt.append(f"{LBL[m].replace('Model ', '')} -- fold {fold}")
grid(tc, tt, 2, "fig_training_curves_sn", 9, 4.5,
     "Training and validation curves, leakage-free pipeline -- Substantia Nigra")

cm, ct = [], []
for m in MODELS:
    p = f"{DF}/substantiaNigra_confusion_matrix_{m}_Fold_04.png"
    if os.path.exists(p):
        cm.append(p); ct.append(LBL[m].replace("Model ", ""))
grid(cm, ct, 3, "fig_confusion_sn", 6, 5.5,
     "Patch-level confusion matrices, held-out fold -- Substantia Nigra")

print("\nAll figures regenerated in the original style ->", FIG)
