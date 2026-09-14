"""XAI v2: (a) PD vs HC group differences, (b) validity of the explanations.

(a) is what the submitted manuscript claimed. (b) is what the data actually
support: whether relevance is anchored on marker-positive tissue at all.
"""

import os

# Directory layout. Override with environment variables to point at a
# different location; defaults assume the results were downloaded into
# ./results next to this repository.
RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./results/analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import glob, warnings
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
D = RESULTS_DIR + "/xai"
OUT = OUTPUT_DIR
SEEDS, MODELS = [42, 43, 44], ["A", "B", "C"]

donor = []
for s in SEEDS:
    for m in MODELS:
        d = pd.read_csv(f"{D}/seed{s}_{m}_per_donor_metrics.csv")
        d["seed"], d["model"] = s, m
        donor.append(d)
donor = pd.concat(donor, ignore_index=True)
print("donor rows:", len(donor), "| donors:", donor.patient_id.nunique())

METRICS = ["psyn_enrichment", "iba1_enrichment", "z_psyn_spearman",
           "z_concentration", "aor_area_frac", "cam_mean", "peak_z"]

# ---- (a) group differences, averaged over seeds ---------------------------
avg = (donor.groupby(["model", "patient_id", "group"])[METRICS]
       .mean().reset_index())
grp_rows = []
for m in MODELS:
    sub = avg[avg.model == m]
    for k in METRICS:
        a = sub.loc[sub.group == "PD", k].dropna().to_numpy()
        b = sub.loc[sub.group == "HC", k].dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        sd = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
                     / (len(a)+len(b)-2))
        grp_rows.append(dict(model=m, metric=k, n_PD=len(a), n_HC=len(b),
                             mean_PD=a.mean(), mean_HC=b.mean(),
                             p=float(u.pvalue),
                             d=float((a.mean()-b.mean())/sd) if sd > 0 else np.nan))
grp = pd.DataFrame(grp_rows)
grp["p_BH"] = multipletests(grp.p, method="fdr_bh")[1]

# ---- (b) validity: is relevance anchored on marker-positive tissue? -------
# enrichment = P(marker-positive | inside AOR) / P(marker-positive).
# 1.0 means the explanation ignores the marker; > 1 means it concentrates on it.
val_rows = []
for m in MODELS:
    sub = avg[avg.model == m]
    for k, null, label in [
            ("iba1_enrichment", 1.0, "relevance concentrated on IBA1-positive tissue"),
            ("psyn_enrichment", 1.0, "relevance concentrated on pSyn-positive tissue"),
            ("z_psyn_spearman", 0.0, "axial relevance tracks the pSyn depth profile")]:
        v = sub[k].dropna().to_numpy()
        if len(v) < 3:
            continue
        t = stats.wilcoxon(v - null)
        boot = [np.median(np.random.default_rng(i).choice(v, len(v), True))
                for i in range(10000)]
        val_rows.append(dict(model=m, metric=k, claim=label, n=len(v),
                             median=float(np.median(v)),
                             ci_lo=float(np.percentile(boot, 2.5)),
                             ci_hi=float(np.percentile(boot, 97.5)),
                             null=null, p=float(t.pvalue)))
val = pd.DataFrame(val_rows)
val["p_BH"] = multipletests(val.p, method="fdr_bh")[1]

# ---- seed-to-seed stability of the explanations --------------------------
stab = []
for m in MODELS:
    sub = donor[donor.model == m]
    for k in ["iba1_enrichment", "psyn_enrichment", "z_psyn_spearman"]:
        w = sub.pivot_table(index="patient_id", columns="seed", values=k)
        if w.shape[1] < 2:
            continue
        cors = [stats.spearmanr(w[a], w[b])[0]
                for i, a in enumerate(w.columns) for b in w.columns[i+1:]]
        stab.append(dict(model=m, metric=k, mean_seed_rho=float(np.nanmean(cors))))
stab = pd.DataFrame(stab)

grp.to_csv(f"{OUT}/xai_group_differences.csv", index=False)
val.to_csv(f"{OUT}/xai_validity.csv", index=False)
stab.to_csv(f"{OUT}/xai_seed_stability.csv", index=False)
avg.to_csv(f"{OUT}/xai_donor_level_seedavg.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("\n=== (a) PD vs HC DIFFERENCES IN RELEVANCE (seed-averaged, donor level) ===")
print(grp.round(4).to_string(index=False))
print("\n=== (b) VALIDITY OF THE EXPLANATIONS (one-sample vs null) ===")
print(val.round(4).to_string(index=False))
print("\n=== SEED-TO-SEED STABILITY (Spearman across donors) ===")
print(stab.round(3).to_string(index=False))
