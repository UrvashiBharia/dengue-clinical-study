"""Profile of the WHO-2009 classes (DWOS / DWS / DS):
  1. ICU admission / transfusion rate within each class.
  2. % of patients in each class with every marker that differed significantly between classes.

Run:  python analysis/dx_profile.py   ->  analysis/output/dx_profile.xlsx
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from common import DX_ORDER, OUT, autosize, fmt_p, load_data, med_iqr

warnings.filterwarnings("ignore")
OUT.mkdir(parents=True, exist_ok=True)
df = load_data()


def cochran_armitage(flag, order):
    """Cochran-Armitage test for trend in a proportion across ordered groups (scores 0,1,2)."""
    tab = pd.crosstab(order, flag).reindex(columns=[0, 1], fill_value=0)
    n_i = tab.sum(axis=1).values.astype(float)
    x_i = tab[1].values.astype(float)
    s = np.asarray(tab.index, dtype=float)
    N, X = n_i.sum(), x_i.sum()
    p_bar = X / N
    t = np.sum(s * (x_i - n_i * p_bar))
    var = p_bar * (1 - p_bar) * (np.sum(n_i * s ** 2) - np.sum(n_i * s) ** 2 / N)
    if var == 0:
        return np.nan
    return 2 * stats.norm.sf(abs(t / np.sqrt(var)))


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "-"


# --------------------------------------------------------------------------- 1. outcomes by class
outcome_rows = []
for oc, nm in [("icu", "ICU admission"), ("transfusion", "Transfusion"), ("icu_or_tx", "ICU or transfusion"),
               ("prolonged_los", "LOS > 3 days")]:
    row = {"Outcome": nm}
    for k in DX_ORDER:
        s = df.loc[df.dx == k, oc].dropna()
        row[k] = pct(int(s.sum()), len(s))
    s = df[oc].dropna()
    row["All"] = pct(int(s.sum()), len(s))
    ct = pd.crosstab(df.dx, df[oc])
    row["p (chi-square)"] = stats.chi2_contingency(ct)[1]
    row["p (trend)"] = cochran_armitage(df[oc].fillna(0).astype(int), df.dx_ord)
    outcome_rows.append(row)
row = {"Outcome": "LOS, median (IQR) days"}
for k in DX_ORDER:
    row[k] = med_iqr(df.loc[df.dx == k, "los"])
row["All"] = med_iqr(df.los)
row["p (chi-square)"] = stats.kruskal(*[df.loc[df.dx == k, "los"] for k in DX_ORDER]).pvalue
row["p (trend)"] = stats.spearmanr(df.los, df.dx_ord)[1]
outcome_rows.append(row)
row = {"Outcome": "ICU stay among ICU patients, median (IQR) days"}
for k in DX_ORDER:
    row[k] = med_iqr(df.loc[(df.dx == k) & (df.icu == 1), "icu_los"])
row["All"] = med_iqr(df.loc[df.icu == 1, "icu_los"])
outcome_rows.append(row)
by_class = pd.DataFrame(outcome_rows)

# reverse view: of patients needing ICU/transfusion, which class were they?
rev = []
for oc, nm in [("icu", "ICU admission"), ("transfusion", "Transfusion"), ("icu_or_tx", "ICU or transfusion")]:
    s = df[df[oc] == 1]
    rev.append({"Among patients with": nm, "n": len(s),
                **{k: pct(int((s.dx == k).sum()), len(s)) for k in DX_ORDER}})
reverse = pd.DataFrame(rev)

# --------------------------------------------------------------------------- 2. marker prevalence by class
# Markers that differed significantly across DWOS/DWS/DS in dengue_analysis.py / ratios_analysis.py.
# Continuous markers are dichotomised at the cut-off shown (source in "Cut-off basis").
MARKERS = [
    ("Oliguria", lambda d: d.oliguria == 1, "oliguria", "Clinical", "present"),
    ("Bleeding (any)", lambda d: d.bleeding == 1, "bleeding", "Clinical", "present"),
    ("Rash", lambda d: d.rash == 1, "rash", "Clinical", "present"),
    ("USG abnormal (ascites / HSM / effusion)", lambda d: d.usg_binary == 1, "usg_binary", "Clinical", "present"),
    ("Fever >= 4 days at admission", lambda d: d.days_fever >= 4, "days_fever", "Clinical", "cohort 75th centile"),
    ("Platelets < 100 x10^3/uL", lambda d: d.plt < 100, "plt", "Lab", "WHO thrombocytopenia threshold"),
    ("Platelets < 50 x10^3/uL", lambda d: d.plt < 50, "plt", "Lab", "severe thrombocytopenia"),
    ("Platelets < 20 x10^3/uL", lambda d: d.plt < 20, "plt", "Lab", "critical thrombocytopenia"),
    ("PT >= 14 s", lambda d: d.pt >= 14, "pt", "Lab", "ROC-optimal cut-off (ICU/transfusion)"),
    ("aPTT >= 38 s", lambda d: d.ptt >= 38, "ptt", "Lab", "ROC-optimal cut-off (ICU/transfusion)"),
    ("AST > 120 U/L (>3x ULN)", lambda d: d.ast > 120, "ast", "Lab", "3x upper limit of normal (40)"),
    ("ALT > 120 U/L (>3x ULN)", lambda d: d.alt > 120, "alt", "Lab", "3x upper limit of normal (40)"),
    ("Haematocrit/platelet ratio > 0.5", lambda d: d.hct_plt > 0.5, "hct_plt", "Ratio", "platelets < 2 x Hct"),
    ("APRI > 1", lambda d: d.apri > 1, "apri", "Ratio", "conventional APRI cut-off"),
    ("PT x aPTT index >= 1.28", lambda d: d.pt_x_ptt >= 1.28, "pt_x_ptt", "Ratio", "ROC-optimal cut-off (ICU/transfusion)"),
    ("Platelet/WBC ratio <= 16", lambda d: d.plt_wbc <= 16, "plt_wbc", "Ratio", "ROC-optimal cut-off (ICU/transfusion)"),
]
prev_rows = []
for nm, cond, base, typ, basis in MARKERS:
    avail = df[base].notna()
    sub = df[avail]
    flag = cond(sub).astype(int)
    row = {"Marker": nm, "Type": typ, "Cut-off basis": basis, "n with data": int(avail.sum())}
    for k in DX_ORDER:
        f = flag[sub.dx == k]
        row[k] = pct(int(f.sum()), len(f))
    row["All"] = pct(int(flag.sum()), len(flag))
    ct = pd.crosstab(sub.dx, flag)
    row["p (chi-square 3x2)"] = stats.chi2_contingency(ct)[1] if ct.shape[1] == 2 else np.nan
    row["p (trend)"] = cochran_armitage(flag, sub.dx_ord)
    # pairwise
    for a, b in [("DWOS", "DWS"), ("DWS", "DS"), ("DWOS", "DS")]:
        fa, fb = flag[sub.dx == a], flag[sub.dx == b]
        row[f"p {a} vs {b}"] = stats.fisher_exact([[fa.sum(), len(fa) - fa.sum()], [fb.sum(), len(fb) - fb.sum()]])[1]
    prev_rows.append(row)
prevalence = pd.DataFrame(prev_rows)

# continuous medians for the same markers
CONT = [("days_fever", "Days of fever at admission"), ("plt", "Platelets (x10^3/uL)"), ("pt", "PT (s)"),
        ("ptt", "aPTT (s)"), ("ast", "AST (U/L)"), ("alt", "ALT (U/L)"), ("br_indir", "Indirect bilirubin"),
        ("hct_plt", "Haematocrit/platelet ratio"), ("apri", "APRI"), ("pt_x_ptt", "PT x aPTT index"),
        ("plt_wbc", "Platelet/WBC ratio")]
cont_rows = []
for v, nm in CONT:
    m = df[v].notna()
    groups = [df.loc[m & (df.dx == k), v] for k in DX_ORDER]
    cont_rows.append({"Marker": nm, "n with data": int(m.sum()),
                      **{f"{k} median (IQR)": med_iqr(g) for k, g in zip(DX_ORDER, groups)},
                      "p (Kruskal-Wallis)": stats.kruskal(*groups).pvalue,
                      "p (trend, Spearman)": stats.spearmanr(df.loc[m, v], df.loc[m, "dx_ord"])[1]})
continuous = pd.DataFrame(cont_rows)

# how many markers does each patient carry?
flags = pd.DataFrame({nm: cond(df).astype(int) for nm, cond, *_ in MARKERS[:6]})   # 5 clinical + platelets < 100
df["n_markers"] = flags.sum(axis=1)
burden = pd.crosstab(df.n_markers.clip(upper=4).map(lambda v: f"{v}" if v < 4 else ">=4"), df.dx,
                     margins=True, margins_name="Total").reindex(columns=DX_ORDER + ["Total"])
burden_pct = (pd.crosstab(df.n_markers.clip(upper=4).map(lambda v: f"{v}" if v < 4 else ">=4"), df.dx,
                          normalize="columns") * 100).round(1).reindex(columns=DX_ORDER)


def pfmt(t):
    t = t.copy()
    for c in t.columns:
        if str(c).startswith("p ") or str(c).startswith("p("):
            t[c] = t[c].map(fmt_p)
    return t


with pd.ExcelWriter(OUT / "dx_profile.xlsx", engine="openpyxl") as xw:
    pfmt(by_class).to_excel(xw, sheet_name="Outcomes_by_class", index=False)
    reverse.to_excel(xw, sheet_name="Outcomes_by_class", index=False, startrow=len(by_class) + 3)
    pfmt(prevalence).to_excel(xw, sheet_name="Marker_prevalence", index=False)
    pfmt(continuous).to_excel(xw, sheet_name="Marker_medians", index=False)
    burden.to_excel(xw, sheet_name="Marker_burden")
    burden_pct.to_excel(xw, sheet_name="Marker_burden", startrow=len(burden) + 3)
    autosize(xw)

pd.set_option("display.width", 300, "display.max_columns", 30, "display.max_colwidth", 45)
print(pfmt(by_class).to_string(index=False)); print(); print(reverse.to_string(index=False)); print()
print(pfmt(prevalence).drop(columns=["Cut-off basis"]).to_string(index=False)); print()
print(pfmt(continuous).to_string(index=False)); print()
print("Number of markers carried (oliguria, bleeding, rash, USG, fever >=4 d, platelets <100):")
print(burden); print(burden_pct)
