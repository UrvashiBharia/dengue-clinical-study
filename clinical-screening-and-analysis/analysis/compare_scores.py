"""Compare the Bleeding-USG-Fever score with published dengue / ICU severity scores on the same patients.

Each score lives in its own module under analysis/scores/ (items, points, citation, data mapping).
Outcomes: ICU admission or transfusion, severe dengue (DS), DWS or DS.
Subgroups: all patients, adults (>= 18 y), children (< 18 y).
AUCs of each score vs ours are compared with DeLong's test for correlated ROC curves.

Run:  python analysis/compare_scores.py   ->  analysis/output/score_comparison.xlsx
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from common import OUT, autosize, delong, fmt_p, load_data
from scores import ALL_SCORES, bleeding_usg_fever

warnings.filterwarnings("ignore")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
df = load_data()

OUTCOMES = [("icu_or_tx", "ICU or transfusion"), ("severe_ds", "Severe dengue (DS)"), ("dws_or_ds", "DWS or DS")]
SUBGROUPS = [("All patients", df.index), ("Adults (>=18 y)", df.index[df.age >= 18]),
             ("Children (<18 y)", df.index[df.age < 18])]

for mod in ALL_SCORES:
    df[mod.INFO["short"]] = mod.score(df).astype(float)
OURS = bleeding_usg_fever.INFO["short"]


def boot_ci(y, s, n=2000):
    b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], s[i]))
    return np.percentile(b, [2.5, 97.5])


def test_metrics(y, s, cutoff):
    if cutoff is None:                         # no published cut-off -> Youden in this cohort
        fpr, tpr, thr = roc_curve(y, s)
        cutoff = thr[np.argmax(tpr - fpr)]
        label = f"Youden {cutoff:.3g}"
    else:
        label = f">= {cutoff:g}"
    pos = s >= cutoff
    tp = (pos & (y == 1)).sum(); fp = (pos & (y == 0)).sum()
    fn = (~pos & (y == 1)).sum(); tn = (~pos & (y == 0)).sum()
    return dict(cut=label, sens=tp / (tp + fn) if tp + fn else np.nan, spec=tn / (tn + fp) if tn + fp else np.nan,
                ppv=tp / (tp + fp) if tp + fp else np.nan, npv=tn / (tn + fn) if tn + fn else np.nan,
                flagged=int(pos.sum()), missed=int(fn))


rows = []
for sub_name, idx in SUBGROUPS:
    d = df.loc[idx]
    for oc, oc_name in OUTCOMES:
        y = d[oc].values.astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        ours = d[OURS].values
        for mod in ALL_SCORES:
            col = mod.INFO["short"]
            s = d[col].values
            auc = roc_auc_score(y, s)
            lo, hi = boot_ci(y, s)
            m = test_metrics(y, s, mod.INFO["cutoff"])
            rows.append({"Subgroup": sub_name, "Outcome": oc_name, "Events / N": f"{y.sum()}/{len(y)}",
                         "Score": mod.INFO["name"], "AUC": auc, "AUC 95% CI": f"{lo:.2f}-{hi:.2f}",
                         "p vs our score (DeLong)": np.nan if col == OURS else delong(y, ours, s),
                         "Cut-off": m["cut"], "Sensitivity": m["sens"], "Specificity": m["spec"],
                         "PPV": m["ppv"], "NPV": m["npv"], "Flagged": m["flagged"], "Missed events": m["missed"]})
res = pd.DataFrame(rows)

items = pd.DataFrame([{"Score": mod.INFO["name"], "Citation": mod.INFO["citation"],
                       "Developed in": mod.INFO["population"], "Original outcome": mod.INFO["original_outcome"],
                       "Item": it[0], "Points": it[1], "Available here?": it[2], "Mapped from": it[3]}
                      for mod in ALL_SCORES for it in mod.INFO["items"]])
availability = (items.assign(ok=~items["Available here?"].str.startswith("NOT"))
                .groupby("Score", sort=False)["ok"].agg(lambda s: f"{s.sum()} of {len(s)}")
                .rename("Items measurable").reset_index())


def pfmt(t):
    t = t.copy()
    t["AUC"] = t["AUC"].round(2)
    t["p vs our score (DeLong)"] = t["p vs our score (DeLong)"].map(fmt_p)
    for c in ["Sensitivity", "Specificity", "PPV", "NPV"]:
        t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{100 * v:.0f}%")
    return t


with pd.ExcelWriter(OUT / "score_comparison.xlsx", engine="openpyxl") as xw:
    pd.DataFrame({"Note": [
        "Each score is implemented in its own file in analysis/scores/ with full citation and point table.",
        "Items not recorded in this dataset are scored 0. SOFA, mSOFA and APACHE II are therefore PARTIAL",
        "scores (organ systems / physiology missing) and must not be reported as the validated scales.",
        "AUC comparison vs the Bleeding-USG-Fever score uses DeLong's test on the same patients.",
        "Our score was derived in this cohort, so its AUC here is optimistic (realistic ~0.84).",
    ]}).to_excel(xw, sheet_name="README", index=False)
    pfmt(res).to_excel(xw, sheet_name="Comparison", index=False)
    availability.to_excel(xw, sheet_name="Item_availability", index=False)
    items.to_excel(xw, sheet_name="Score_items", index=False)
    autosize(xw)

pd.set_option("display.width", 320, "display.max_columns", 20, "display.max_colwidth", 52)
print(availability.to_string(index=False)); print()
show = pfmt(res)
for (sub, oc), g in show.groupby(["Subgroup", "Outcome"], sort=False):
    print(f"\n## {sub} | {oc} | events {g['Events / N'].iloc[0]}")
    print(g.sort_values("AUC", ascending=False)[["Score", "AUC", "AUC 95% CI", "p vs our score (DeLong)", "Cut-off",
                                                  "Sensitivity", "Specificity", "PPV", "NPV", "Missed events"]].to_string(index=False))
