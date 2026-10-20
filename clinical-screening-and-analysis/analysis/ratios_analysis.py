"""Add-on: WBC, absolute lymphocyte count and derived lab ratios vs LOS, ICU/transfusion and WHO class.

Run:  python analysis/ratios_analysis.py   ->  analysis/output/ratios_results.xlsx
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests

from common import OUT, autosize, fmt_p, load_data, med_iqr

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)
OUT.mkdir(parents=True, exist_ok=True)
df = load_data()
# Derived ratios are built in common.load_data(); no neutrophil column exists, so
# "non-lymphocyte count" (WBC - ALC; mostly neutrophils) stands in for ANC.
df["bili_total"] = df["bilirubin_total"]

MARKERS = {
    "wbc": "WBC (x10^3/uL)",
    "alc": "Absolute lymphocyte count (x10^3/uL)",
    "nonlymph": "Non-lymphocyte count, WBC-ALC (~neutrophils)",
    "nlr_proxy": "Non-lymphocyte/lymphocyte ratio (proxy NLR)",
    "plr": "Platelet-to-lymphocyte ratio (PLR)",
    "plt_wbc": "Platelet-to-WBC ratio",
    "hct_plt": "Haematocrit-to-platelet ratio",
    "ast_alt": "AST/ALT (De Ritis) ratio",
    "apri": "APRI (AST-to-platelet ratio index)",
    "bili_total": "Total bilirubin",
    "bili_dir_frac": "Direct/total bilirubin fraction",
    "pt_x_ptt": "PT x aPTT coagulopathy index",
    # single markers for comparison
    "plt": "Platelets (reference)", "pt": "PT (reference)", "lymph_pct": "Lymphocyte % (reference)",
}

# clinically used cut-offs
CATS = [
    ("Leucopenia WBC < 4", lambda d: d.wbc < 4, "wbc"),
    ("Severe leucopenia WBC < 2", lambda d: d.wbc < 2, "wbc"),
    ("Leucocytosis WBC > 11", lambda d: d.wbc > 11, "wbc"),
    ("Lymphopenia ALC < 1.0", lambda d: d.alc < 1.0, "alc"),
    ("Severe lymphopenia ALC < 0.5", lambda d: d.alc < 0.5, "alc"),
    ("Proxy NLR > 3", lambda d: d.nlr_proxy > 3, "nlr_proxy"),
    ("AST/ALT > 2", lambda d: d.ast_alt > 2, "ast_alt"),
    ("APRI > 1", lambda d: d.apri > 1, "apri"),
    ("Hct/plt > 0.5 (i.e. plt < 2x Hct)", lambda d: d.hct_plt > 0.5, "hct_plt"),
]

OUTCOMES = [("icu_or_tx", "ICU or transfusion"), ("severe_ds", "Severe dengue (DS)"),
            ("dws_or_ds", "DWS or DS vs DWOS"), ("prolonged_los", "LOS > 3 days")]


def auc_ci(y, s, n_boot=2000):
    y, s = np.asarray(y), np.asarray(s)
    auc = roc_auc_score(y, s)
    boots = [roc_auc_score(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))
             if y[i].min() != y[i].max()]
    return auc, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


# ---------------------------------------------------------------- continuous tests
rows = []
for v, lab in MARKERS.items():
    m = df[v].notna() & np.isfinite(df[v])
    x = df.loc[m, v]
    r = {"Marker": lab, "n": int(m.sum()), "Median (IQR), all": med_iqr(x)}
    rho, p = stats.spearmanr(x, df.loc[m, "los"])
    r["LOS Spearman rho"], r["LOS p"] = rho, p
    g = [x[df.loc[m, "dx_ord"] == k] for k in (0, 1, 2)]
    r["DWOS median"], r["DWS median"], r["DS median"] = (med_iqr(gg) for gg in g)
    r["Dx Kruskal p"] = stats.kruskal(*[gg for gg in g if len(gg)]).pvalue
    r["Dx trend p"] = stats.spearmanr(x, df.loc[m, "dx_ord"])[1]
    for oc, nm in OUTCOMES[:3]:
        yy = df.loc[m, oc]
        if yy.nunique() < 2:
            continue
        auc = roc_auc_score(yy, x)
        sign = 1 if auc >= 0.5 else -1
        a, lo, hi = auc_ci(yy, sign * x)
        fpr, tpr, thr = roc_curve(yy, sign * x)
        j = np.argmax(tpr - fpr)
        r[f"{nm}: event median"] = med_iqr(x[yy == 1])
        r[f"{nm}: no-event median"] = med_iqr(x[yy == 0])
        r[f"{nm}: p"] = stats.mannwhitneyu(x[yy == 1], x[yy == 0]).pvalue
        r[f"{nm}: AUC (95% CI)"] = f"{a:.2f} ({lo:.2f}-{hi:.2f})" + (" [low=risk]" if sign < 0 else "")
        r[f"{nm}: cut-off sens/spec"] = f"{'>=' if sign > 0 else '<='} {sign * thr[j]:.3g}: {100 * tpr[j]:.0f}% / {100 * (1 - fpr[j]):.0f}%"
    rows.append(r)
cont = pd.DataFrame(rows)
for pc in [c for c in cont.columns if c.endswith(" p") or c.endswith(": p")]:
    ratio_rows = ~cont["Marker"].str.contains("reference")
    q = np.full(len(cont), np.nan)
    q[ratio_rows.values] = multipletests(cont.loc[ratio_rows, pc].fillna(1), method="fdr_bh")[1]
    cont.insert(cont.columns.get_loc(pc) + 1, pc.replace(" p", " q").replace(": p", ": q"), q)

# ---------------------------------------------------------------- categorical cut-offs
crow = []
for nm, cond, base in CATS:
    avail = df[base].notna()
    flag = cond(df)[avail].astype(int)
    for oc, onm in OUTCOMES:
        y = df.loc[avail, oc]
        a = ((flag == 1) & (y == 1)).sum(); b = ((flag == 1) & (y == 0)).sum()
        c = ((flag == 0) & (y == 1)).sum(); d = ((flag == 0) & (y == 0)).sum()
        p = stats.fisher_exact([[a, b], [c, d]])[1]
        aa, bb, cc, dd = (a, b, c, d) if min(a, b, c, d) else (a + .5, b + .5, c + .5, d + .5)
        orr = aa * dd / (bb * cc); se = np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
        crow.append({"Cut-off": nm, "n tested": int(avail.sum()), "n positive": int(flag.sum()), "Outcome": onm,
                     "Event if positive": f"{a}/{a + b} ({100 * a / max(a + b, 1):.0f}%)",
                     "Event if negative": f"{c}/{c + d} ({100 * c / max(c + d, 1):.0f}%)",
                     "OR (95% CI)": f"{orr:.2f} ({np.exp(np.log(orr) - 1.96 * se):.2f}-{np.exp(np.log(orr) + 1.96 * se):.2f})",
                     "p (Fisher)": p,
                     "Sensitivity": a / (a + c) if a + c else np.nan, "Specificity": d / (b + d) if b + d else np.nan,
                     "PPV": a / (a + b) if a + b else np.nan, "NPV": d / (c + d) if c + d else np.nan})
    # LOS
    y = df.loc[avail, "los"]
    crow.append({"Cut-off": nm, "n tested": int(avail.sum()), "n positive": int(flag.sum()), "Outcome": "LOS (days)",
                 "Event if positive": f"median {med_iqr(y[flag == 1])}, mean {y[flag == 1].mean():.2f}",
                 "Event if negative": f"median {med_iqr(y[flag == 0])}, mean {y[flag == 0].mean():.2f}",
                 "OR (95% CI)": "", "p (Fisher)": stats.mannwhitneyu(y[flag == 1], y[flag == 0]).pvalue if flag.sum() and (flag == 0).sum() else np.nan,
                 "Sensitivity": np.nan, "Specificity": np.nan, "PPV": np.nan, "NPV": np.nan})
cat = pd.DataFrame(crow)

# ---------------------------------------------------------------- ALC / WBC by adult vs pediatric (age-specific norms)
age_rows = []
for v in ["wbc", "alc", "lymph_pct", "nlr_proxy", "plr"]:
    for grp in ["Adult", "Pediatric"]:
        s = df[df.age_cat == grp]
        m = s[v].notna()
        if s.loc[m, "icu_or_tx"].nunique() < 2:
            p_icu = np.nan
        else:
            p_icu = stats.mannwhitneyu(s.loc[m & (s.icu_or_tx == 1), v], s.loc[m & (s.icu_or_tx == 0), v]).pvalue
        rho, p_los = stats.spearmanr(s.loc[m, v], s.loc[m, "los"])
        age_rows.append({"Marker": MARKERS[v], "Group": grp, "n": int(m.sum()), "Median (IQR)": med_iqr(s.loc[m, v]),
                         "ICU/tx event median": med_iqr(s.loc[m & (s.icu_or_tx == 1), v]),
                         "ICU/tx no-event median": med_iqr(s.loc[m & (s.icu_or_tx == 0), v]),
                         "ICU/tx p": p_icu, "LOS rho": rho, "LOS p": p_los})
by_age = pd.DataFrame(age_rows)


def pfmt(t):
    t = t.copy()
    for c in t.columns:
        if (c.endswith(" p") or c.endswith(": p") or c.endswith(" q") or c.endswith(": q") or c.startswith("p ")
                or c.endswith(" trend p") or c.endswith(" Kruskal p")) and t[c].dtype.kind == "f":
            t[c] = t[c].map(fmt_p)
        if c.endswith("rho") and t[c].dtype.kind == "f":
            t[c] = t[c].round(3)
    for c in ["Sensitivity", "Specificity", "PPV", "NPV"]:
        if c in t:
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{100 * v:.0f}%")
    return t


defs = pd.DataFrame({"Marker": list(MARKERS.values())[:12], "Formula": [
    "as recorded", "WBC x lymphocyte% / 100", "WBC - ALC (neutrophils + monocytes + eos/basos; no differential in sheet)",
    "(WBC - ALC) / ALC  = (100 - L%) / L%  -> mathematically equivalent to lymphocyte %, same AUC",
    "Platelets / ALC", "Platelets / WBC", "Hct / Platelets", "AST / ALT", "(AST / 40) / Platelets x 100",
    "Direct + indirect bilirubin", "Direct / total bilirubin", "(PT/13) x (aPTT/32), cohort normal medians"]})

with pd.ExcelWriter(OUT / "ratios_results.xlsx", engine="openpyxl") as xw:
    defs.to_excel(xw, sheet_name="Definitions", index=False)
    pfmt(cont).to_excel(xw, sheet_name="Continuous_markers", index=False)
    pfmt(cat).to_excel(xw, sheet_name="Clinical_cutoffs", index=False)
    pfmt(by_age).to_excel(xw, sheet_name="Adult_vs_Ped", index=False)
    autosize(xw)

pd.set_option("display.width", 300, "display.max_columns", 40, "display.max_colwidth", 45)
show = pfmt(cont)
print(show[["Marker", "n", "LOS Spearman rho", "LOS p", "LOS q", "DWOS median", "DWS median", "DS median", "Dx Kruskal p", "Dx Kruskal q"]].to_string(index=False))
for _, nm in OUTCOMES[:3]:
    print("\n--", nm)
    print(show[["Marker", f"{nm}: event median", f"{nm}: no-event median", f"{nm}: p", f"{nm}: q", f"{nm}: AUC (95% CI)", f"{nm}: cut-off sens/spec"]].to_string(index=False))
print(); print(pfmt(cat).to_string(index=False))
print(); print(pfmt(by_age).to_string(index=False))
