"""Recompute spatial-statistics analyses requested by reviewers
(R1.4/R1.5/R1.9/R1.10/R1.11, R2.5).

Inputs : <RESULTS_DIR>/spp-results/spp_fov_results_filtered.csv  (789 FOVs, 27 donors)
Outputs: <OUTPUT_DIR>/spp_*.csv
"""

import os

# Directory layout. Override with environment variables to point at a
# different location; defaults assume the results were downloaded into
# ./results next to this repository.
RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./results/analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import warnings, json
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = RESULTS_DIR
OUT = OUTPUT_DIR
OUTLIERS = ["PD8", "PD10", "HC9"]

df = pd.read_csv(RESULTS_DIR + "/spp-results/spp_fov_results_filtered.csv")
df["region"] = df["region"].replace({"substantiaNigra": "SN"})

# ---------- first-order (density / burden) outcomes, previously never tested -
# The acquisition window is identical for every FOV, so per-FOV object counts
# are proportional to first-order intensity (objects per unit volume).
df["pSyn_burden_um3"] = df["n_proteins"] * df["prot_vol_mean"]      # total pSyn volume
for c, new in [("n_cells", "log_n_cells"), ("n_proteins", "log_n_proteins"),
               ("pSyn_burden_um3", "log_pSyn_burden"), ("prot_vol_mean", "log_prot_vol_mean"),
               ("mean_nnd", "log_mean_nnd")]:
    df[new] = np.log(df[c].clip(lower=1e-9))

print("FOVs=%d  donors=%d  PD=%d  HC=%d" % (
    len(df), df.patient_id.nunique(),
    df[df.is_pd == 1].patient_id.nunique(), df[df.is_pd == 0].patient_id.nunique()))
print(df.groupby(["region", "group"]).agg(FOV=("fov_id", "size"),
                                          donors=("patient_id", "nunique")).to_string())


def lmm(data, outcome, formula="~ is_pd"):
    """Random-intercept LMM. No random slope: diagnosis is constant within a
    donor, so a random slope for is_pd is not identifiable."""
    return smf.mixedlm(outcome + " " + formula, data, groups=data["patient_id"]).fit(reml=True)


def row(label, m, term="is_pd"):
    ci = m.conf_int().loc[term]
    return dict(Test=label, coef=float(m.params[term]), se=float(m.bse[term]),
                ci_lo=float(ci.iloc[0]), ci_hi=float(ci.iloc[1]), p=float(m.pvalues[term]))


def cohens_d_boot(data, outcome, nboot=10000, seed=42):
    pm = data.groupby(["patient_id", "is_pd"])[outcome].mean().reset_index()
    a = pm.loc[pm.is_pd == 1, outcome].to_numpy()
    b = pm.loc[pm.is_pd == 0, outcome].to_numpy()

    def d(a, b):
        s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                    / (len(a) + len(b) - 2))
        return (a.mean() - b.mean()) / s
    rng = np.random.default_rng(seed)
    bs = [d(rng.choice(a, len(a), True), rng.choice(b, len(b), True)) for _ in range(nboot)]
    return d(a, b), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# 1. CONFIRMATORY FAMILY: first-order density + second-order + co-occupancy
res = []
for reg in ["SN", "putamen"]:
    d = df[df.region == reg]
    for out, lab in [("log_n_cells", "microglial density (log n cells/FOV)"),
                     ("log_n_proteins", "pSyn object density (log n objects/FOV)"),
                     ("log_pSyn_burden", "pSyn burden (log total pSyn volume/FOV)"),
                     ("log_prot_vol_mean", "mean pSyn aggregate volume (log)"),
                     ("log_mean_nnd", "mean NND (log)"),
                     ("log_spp_score", "SPP score (log)"),
                     ("log_spp_score_marked", "mark-weighted SPP score (log)"),
                     ("log_overlap_index", "Overlap Index (log)")]:
        r = row(reg + " | " + lab, lmm(d, out))
        dd, lo, hi = cohens_d_boot(d, out)
        r.update(Region=reg, Outcome=out, d=dd, d_lo=lo, d_hi=hi, n_FOV=len(d),
                 n_donors=d.patient_id.nunique())
        res.append(r)
first_order = pd.DataFrame(res)

# 2. UNIFIED DIAGNOSIS x REGION INTERACTION (R2.5)
inter = []
for out in ["log_overlap_index", "log_spp_score", "log_spp_score_marked",
            "log_n_cells", "log_n_proteins", "log_pSyn_burden"]:
    m = smf.mixedlm(out + " ~ is_pd * C(region, Treatment(reference='putamen'))",
                    df, groups=df["patient_id"]).fit(reml=True)
    term = [t for t in m.params.index if t.startswith("is_pd:")][0]
    ci = m.conf_int().loc[term]
    inter.append(dict(Outcome=out, Term="diagnosis x region (SN vs putamen)",
                      coef=float(m.params[term]), ci_lo=float(ci.iloc[0]),
                      ci_hi=float(ci.iloc[1]), p=float(m.pvalues[term])))
interaction = pd.DataFrame(inter)

# 3. SN Overlap Index across model specifications
sn = df[df.region == "SN"]
rob = []
m = lmm(sn, "log_overlap_index")
r = row("", m); r.pop("Test")
rob.append(dict(Model="LMM random intercept (log OI)", **r))
g = smf.gee("overlap_index ~ is_pd", "patient_id", sn,
            family=sm.families.Gamma(sm.families.links.Log())).fit()
rob.append(dict(Model="GEE-Gamma log link (OI)", coef=float(g.params["is_pd"]),
                se=float(g.bse["is_pd"]), ci_lo=float(g.conf_int().loc["is_pd"].iloc[0]),
                ci_hi=float(g.conf_int().loc["is_pd"].iloc[1]), p=float(g.pvalues["is_pd"])))
g2 = smf.gee("log_overlap_index ~ is_pd", "patient_id", sn,
             family=sm.families.Gaussian()).fit()
rob.append(dict(Model="GEE-Gaussian (log OI)", coef=float(g2.params["is_pd"]),
                se=float(g2.bse["is_pd"]), ci_lo=float(g2.conf_int().loc["is_pd"].iloc[0]),
                ci_hi=float(g2.conf_int().loc["is_pd"].iloc[1]), p=float(g2.pvalues["is_pd"])))
pm = sn.groupby(["patient_id", "is_pd"])["log_overlap_index"].mean().reset_index()
obs = (pm.loc[pm.is_pd == 1, "log_overlap_index"].mean()
       - pm.loc[pm.is_pd == 0, "log_overlap_index"].mean())
rng = np.random.default_rng(42)
lab = pm.is_pd.to_numpy().copy()
v = pm.log_overlap_index.to_numpy()
cnt = 0
for _ in range(10000):
    p = rng.permutation(lab)
    if abs(v[p == 1].mean() - v[p == 0].mean()) >= abs(obs):
        cnt += 1
rob.append(dict(Model="donor-level permutation (10,000)", coef=float(obs), se=np.nan,
                ci_lo=np.nan, ci_hi=np.nan, p=float((cnt + 1) / 10001)))
robust = pd.DataFrame(rob)

# 4. MULTIPLICITY (R1.9 / R2.5)
fam = first_order.copy()
fam["p_holm"] = multipletests(fam.p, method="holm")[1]
fam["p_BH"] = multipletests(fam.p, method="fdr_bh")[1]

# 5. OUTLIER EXCLUSION (R1.5): objective rule + sensitivity
pat = df.groupby(["patient_id", "group"]).agg(
    n_fov=("fov_id", "size"), n_cells=("n_cells", "mean"), n_prot=("n_proteins", "mean"),
    spp=("log_spp_score", "mean"), ovl=("log_overlap_index", "mean")).reset_index()
flag = {}
for col in ["n_fov", "n_cells", "n_prot", "spp", "ovl"]:
    x = pat[col].to_numpy(float)
    mad = stats.median_abs_deviation(x, scale="normal")
    z = np.abs(x - np.median(x)) / mad
    flag[col] = sorted(pat.patient_id[z > 3])
    pat["robZ_" + col] = np.round(z, 2)

sens = []
for lbl, d in [("full cohort (primary)", df),
               ("excl. PD8/PD10/HC9 (sensitivity)", df[~df.patient_id.isin(OUTLIERS)])]:
    for reg in ["SN", "putamen"]:
        sub = d[d.region == reg]
        m = lmm(sub, "log_overlap_index")
        dd, lo, hi = cohens_d_boot(sub, "log_overlap_index")
        sens.append(dict(Cohort=lbl, Region=reg, n_donors=sub.patient_id.nunique(),
                         n_FOV=len(sub), coef=float(m.params["is_pd"]),
                         p=float(m.pvalues["is_pd"]), d=dd, d_lo=lo, d_hi=hi))
sensitivity = pd.DataFrame(sens)

# 6. PATHOLOGY-BURDEN BASELINES for the CNN (R1.8 / R2.4)
base = []
for reg in ["SN", "putamen"]:
    sub = df[df.region == reg]
    pmn = sub.groupby(["patient_id", "is_pd"]).mean(numeric_only=True).reset_index()
    y = pmn.is_pd.to_numpy()
    for feat, lab2 in [("n_proteins", "pSyn object count / FOV"),
                       ("pSyn_burden_um3", "total pSyn volume / FOV"),
                       ("prot_vol_mean", "mean pSyn aggregate volume"),
                       ("n_cells", "microglial object count / FOV"),
                       ("overlap_index", "Overlap Index"),
                       ("mean_nnd", "mean NND"),
                       ("spp_score", "SPP score")]:
        x = pmn[feat].to_numpy()
        a = roc_auc_score(y, x)
        rng = np.random.default_rng(42)
        i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
        bs = []
        for _ in range(10000):
            s = np.concatenate([rng.choice(i0, len(i0), True), rng.choice(i1, len(i1), True)])
            if len(np.unique(y[s])) > 1:
                bs.append(roc_auc_score(y[s], x[s]))
        base.append(dict(Region=reg, Baseline=lab2, n_donors=len(y), AUROC=a,
                         lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
baselines = pd.DataFrame(base)

for name, t in [("spp_first_order_and_family", fam), ("spp_interaction", interaction),
                ("spp_sn_overlap_robustness", robust), ("spp_outlier_sensitivity", sensitivity),
                ("spp_patient_robustZ", pat), ("dl_burden_baselines", baselines)]:
    t.to_csv(OUT + "/" + name + ".csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
print("\n=== 1+4. CONFIRMATORY FAMILY (random-intercept LMM) + multiplicity control ===")
print(fam[["Test", "coef", "ci_lo", "ci_hi", "p", "p_holm", "p_BH", "d", "d_lo", "d_hi"]]
      .round(4).to_string(index=False))
print("\n=== 2. UNIFIED DIAGNOSIS x REGION INTERACTION ===")
print(interaction.round(4).to_string(index=False))
print("\n=== 3. SN OVERLAP INDEX ACROSS SPECIFICATIONS ===")
print(robust.round(4).to_string(index=False))
print("\n=== 5a. OBJECTIVE OUTLIER RULE (robust z > 3, MAD-based, donor level) ===")
print(json.dumps(flag, indent=1))
print(pat.round(2).to_string(index=False))
print("\n=== 5b. SENSITIVITY TO DONOR EXCLUSION ===")
print(sensitivity.round(4).to_string(index=False))
print("\n=== 6. PATHOLOGY-BURDEN BASELINES (donor-level AUROC) ===")
print(baselines.round(3).to_string(index=False))
