"""Pooled out-of-fold analysis of the leakage-free repeated CV (v4 runs)."""

import os

# Directory layout. Override with environment variables to point at a
# different location; defaults assume the results were downloaded into
# ./results next to this repository.
RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./results/analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import glob, os, warnings
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             matthews_corrcoef, balanced_accuracy_score)

warnings.filterwarnings("ignore")
D = RESULTS_DIR + "/dl"
OUT = OUTPUT_DIR
SEEDS, REGIONS, MODELS = [42, 43, 44], ["substantiaNigra", "putamen"], ["A", "B", "C"]
NB = 10000


def load(seed, region, m):
    return pd.read_csv(f"{D}/seed{seed}_{region}_patient_predictions_{m}.csv")


def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(AUROC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p),
                Accuracy=(tp + tn) / len(y),
                Balanced_accuracy=balanced_accuracy_score(y, yh),
                Sensitivity=tp / (tp + fn) if tp + fn else np.nan,
                Specificity=tn / (tn + fp) if tn + fp else np.nan,
                Precision=tp / (tp + fp) if tp + fp else np.nan,
                MCC=matthews_corrcoef(y, yh))


def boot_ci(y, p, nb=NB, seed=42):
    rng = np.random.default_rng(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    out = []
    for _ in range(nb):
        s = np.concatenate([rng.choice(i0, len(i0), True), rng.choice(i1, len(i1), True)])
        if len(np.unique(y[s])) > 1:
            out.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def perm_p(y, p, nb=NB, seed=42):
    rng = np.random.default_rng(seed)
    obs = abs(roc_auc_score(y, p) - 0.5)
    yy = y.copy(); c = 0
    for _ in range(nb):
        rng.shuffle(yy)
        if abs(roc_auc_score(yy, p) - 0.5) >= obs:
            c += 1
    return (c + 1) / (nb + 1)


# ---------------- per-seed pooled OOF -------------------------------------
rows = []
for seed in SEEDS:
    for region in REGIONS:
        for m in MODELS:
            df = load(seed, region, m)
            y, p = df.True_Label.to_numpy(), df.Pred_Prob_PD.to_numpy()
            lo, hi = boot_ci(y, p)
            rows.append(dict(seed=seed, Region=region, Model=m, n=len(df),
                             **metrics(y, p), lo=lo, hi=hi, perm_p=perm_p(y, p)))
per_seed = pd.DataFrame(rows)
per_seed.to_csv(f"{OUT}/v2_pooled_oof_per_seed.csv", index=False)

# ---------------- across seeds --------------------------------------------
agg = (per_seed.groupby(["Region", "Model"])
       .agg(AUROC_mean=("AUROC", "mean"), AUROC_sd=("AUROC", "std"),
            AUROC_min=("AUROC", "min"), AUROC_max=("AUROC", "max"),
            PR_mean=("PR_AUC", "mean"), BA_mean=("Balanced_accuracy", "mean"),
            Sens=("Sensitivity", "mean"), Spec=("Specificity", "mean"),
            Prec=("Precision", "mean"), MCC=("MCC", "mean"),
            perm_p_median=("perm_p", "median"))
       .reset_index())

# seed-pooled: average the donor probability across seeds, then one AUROC
pool_rows = []
for region in REGIONS:
    for m in MODELS:
        d = pd.concat([load(s, region, m).assign(seed=s) for s in SEEDS])
        g = d.groupby(["Patient_ID", "True_Label"]).Pred_Prob_PD.mean().reset_index()
        y, p = g.True_Label.to_numpy(), g.Pred_Prob_PD.to_numpy()
        lo, hi = boot_ci(y, p)
        pool_rows.append(dict(Region=region, Model=m, n=len(g), **metrics(y, p),
                              lo=lo, hi=hi, perm_p=perm_p(y, p)))
seed_pooled = pd.DataFrame(pool_rows)
seed_pooled.to_csv(f"{OUT}/v2_seed_pooled_oof.csv", index=False)
agg.to_csv(f"{OUT}/v2_across_seeds.csv", index=False)

# ---------------- paired model comparisons (seed-averaged donor scores) ----
def paired(region, a, b, nb=NB, seed=42):
    da = pd.concat([load(s, region, a) for s in SEEDS]).groupby(
        ["Patient_ID", "True_Label"]).Pred_Prob_PD.mean().reset_index()
    db = pd.concat([load(s, region, b) for s in SEEDS]).groupby(
        ["Patient_ID", "True_Label"]).Pred_Prob_PD.mean().reset_index()
    mm = da.merge(db, on=["Patient_ID", "True_Label"], suffixes=("_x", "_y"))
    y = mm.True_Label.to_numpy()
    px, py = mm.Pred_Prob_PD_x.to_numpy(), mm.Pred_Prob_PD_y.to_numpy()
    obs = roc_auc_score(y, px) - roc_auc_score(y, py)
    rng = np.random.default_rng(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    d = []
    for _ in range(nb):
        s = np.concatenate([rng.choice(i0, len(i0), True), rng.choice(i1, len(i1), True)])
        if len(np.unique(y[s])) > 1:
            d.append(roc_auc_score(y[s], px[s]) - roc_auc_score(y[s], py[s]))
    d = np.array(d)
    return dict(Region=region, Comparison=f"{a} - {b}", delta=float(obs),
                lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                p=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))

comp = pd.DataFrame([paired(r, a, b) for r in REGIONS
                     for a, b in [("C", "B"), ("C", "A"), ("B", "A")]])
comp.to_csv(f"{OUT}/v2_model_comparisons.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("=== PER-SEED POOLED OOF (donor level) ===")
print(per_seed[["seed", "Region", "Model", "n", "AUROC", "lo", "hi", "perm_p"]]
      .round(3).to_string(index=False))
print("\n=== ACROSS SEEDS (mean of the three pooled OOF estimates) ===")
print(agg.round(3).to_string(index=False))
print("\n=== SEED-POOLED (donor probability averaged over seeds, then one AUROC) ===")
print(seed_pooled[["Region", "Model", "n", "AUROC", "lo", "hi", "perm_p", "PR_AUC",
                   "Balanced_accuracy", "Sensitivity", "Specificity", "Precision", "MCC"]]
      .round(3).to_string(index=False))
print("\n=== PAIRED MODEL COMPARISONS ===")
print(comp.round(3).to_string(index=False))
