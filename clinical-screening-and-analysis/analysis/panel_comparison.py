"""Head-to-head comparison of the derived panel against established triage approaches,
evaluated on the same patients for ICU/transfusion and severe dengue.

Fixed rules (no fitting) are scored directly. Models that need coefficients are refitted on this
cohort and scored by repeated 5-fold cross-validation so they are on equal footing with our panel.

Run:  python analysis/panel_comparison.py   ->  analysis/output/panel_comparison.xlsx
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUT, autosize, load_data

warnings.filterwarnings("ignore")
OUT.mkdir(parents=True, exist_ok=True)
df = load_data()
rng = np.random.default_rng(42)



def recommended_score(d):
    """Final proposed score (0-5): bleeding 2 points, plasma leakage on USG 2 points, fever >= 4 days 1 point."""
    return 2 * d.bleeding.fillna(0) + 2 * d.usg_binary.fillna(0) + (d.days_fever >= 4).astype(int)


# ------------------------------------------------------------------ fixed rules (binary flag per patient)
RULES = {
    "WHO-2009 class DWS or DS (warning-sign triage)": lambda d: d.dx_ord >= 1,
    "WHO-1997 DHF proxy: platelets <=100 + bleeding + plasma leakage on USG":
        lambda d: (d.plt <= 100) & (d.bleeding == 1) & (d.usg_binary == 1),
    "Thrombocytopenia + plasma leakage: platelets <=100 + USG abnormal":
        lambda d: (d.plt <= 100) & (d.usg_binary == 1),
    "Platelets < 50 alone": lambda d: d.plt < 50,
    "Platelets < 20 alone (transfusion trigger)": lambda d: d.plt < 20,
    "Recommended score >= 2 (bleeding 2, USG 2, fever >=4 d 1) [this study]":
        lambda d: recommended_score(d) >= 2,
    "Bedside score >= 2 (plt<50, bleeding, USG, oliguria) [this study]":
        lambda d: ((d.plt < 50).astype(int) + d.bleeding.fillna(0) + d.usg_binary.fillna(0) + d.oliguria.fillna(0)) >= 2,
}
# ------------------------------------------------------------------ refitted models
MODELS = {
    "Derived panel: bleeding + USG + days of fever [this study]": ["bleeding", "usg_binary", "days_fever"],
    "Alternative panel: bleeding + aPTT + days of fever [this study]": ["bleeding", "ptt", "days_fever"],
    "Routine blood-count model (WBC + Hct + platelets; Potts/Tanner-type)": ["wbc", "hct", "plt"],
    "Blood count + AST (Potts-type, AST available in ~50%)": ["wbc", "hct", "plt", "ast"],
    "Coagulation: PT + aPTT": ["pt", "ptt"],
    "PT alone": ["pt"],
    "Platelets alone": ["plt"],
}
# continuous "score" versions of ordinal rules for AUC
ORDINAL = {"WHO-2009 class DWS or DS (warning-sign triage)": "dx_ord"}
df["recommended_score"] = recommended_score(df)
ORDINAL["Recommended score >= 2 (bleeding 2, USG 2, fever >=4 d 1) [this study]"] = "recommended_score"


def cv_predictions(cols, y, n_rep=40):
    X = df[cols].values.astype(float)
    preds = np.zeros((n_rep, len(y)))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=1)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    for i, (tr, te) in enumerate(cv.split(X, y)):
        model.fit(X[tr], y[tr])
        preds[i // 5, te] = model.predict_proba(X[te])[:, 1]
    return preds.mean(axis=0)        # out-of-fold probability averaged across repeats


def boot_ci(y, s, n=2000):
    b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], s[i]))
    return np.percentile(b, [2.5, 97.5])


rows = []
for outcome, oname in [("icu_or_tx", "ICU or transfusion"), ("severe_ds", "Severe dengue (DS)")]:
    y = df[outcome].values
    for name, rule in RULES.items():
        if outcome == "severe_ds" and name.startswith("WHO-2009"):
            continue                                    # DS is part of the WHO class itself
        flag = rule(df).fillna(False).astype(int).values
        tp = ((flag == 1) & (y == 1)).sum(); fp = ((flag == 1) & (y == 0)).sum()
        fn = ((flag == 0) & (y == 1)).sum(); tn = ((flag == 0) & (y == 0)).sum()
        score = df[ORDINAL[name]].values if name in ORDINAL else flag
        auc = roc_auc_score(y, score); lo, hi = boot_ci(y, score)
        rows.append({"Outcome": oname, "Approach": name, "Type": "Fixed rule (no fitting)",
                     "AUC (95% CI)": f"{auc:.2f} ({lo:.2f}-{hi:.2f})", "auc": auc,
                     "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp),
                     "PPV": tp / (tp + fp) if tp + fp else np.nan, "NPV": tn / (tn + fn) if tn + fn else np.nan,
                     "Flagged": f"{tp + fp}/{len(y)}", "Missed events": int(fn)})
    for name, cols in MODELS.items():
        p = cv_predictions(cols, y)
        auc = roc_auc_score(y, p); lo, hi = boot_ci(y, p)
        fpr, tpr, thr = roc_curve(y, p)
        # report at the threshold giving sensitivity >= 90% (rule-out use), and at Youden
        j90 = np.argmax(tpr >= 0.90)
        jy = np.argmax(tpr - fpr)
        rows.append({"Outcome": oname, "Approach": name, "Type": "Logistic, cross-validated",
                     "AUC (95% CI)": f"{auc:.2f} ({lo:.2f}-{hi:.2f})", "auc": auc,
                     "Sensitivity": tpr[jy], "Specificity": 1 - fpr[jy], "PPV": np.nan, "NPV": np.nan,
                     "Flagged": f"spec at sens>=90%: {100 * (1 - fpr[j90]):.0f}%",
                     "Missed events": int(round((1 - tpr[jy]) * y.sum()))})
res = pd.DataFrame(rows).sort_values(["Outcome", "auc"], ascending=[True, False]).drop(columns="auc")

out = res.copy()
for c in ["Sensitivity", "Specificity", "PPV", "NPV"]:
    out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{100 * v:.0f}%")
# risk tiers of the recommended score
TIERS = [("Low", [0]), ("Intermediate", [1]), ("High", [2, 3]), ("Very high", [4, 5])]
tier_rows = []
for tier, pts in TIERS:
    s_ = df[df.recommended_score.isin(pts)]
    tier_rows.append({"Risk tier": tier, "Score": "-".join(map(str, [pts[0], pts[-1]])) if len(pts) > 1 else str(pts[0]),
                      "Patients": len(s_),
                      "ICU or transfusion": f"{int(s_.icu_or_tx.sum())} ({100 * s_.icu_or_tx.mean():.0f}%)",
                      "Severe dengue": f"{int(s_.severe_ds.sum())} ({100 * s_.severe_ds.mean():.0f}%)",
                      "DWS or DS": f"{int(s_.dws_or_ds.sum())} ({100 * s_.dws_or_ds.mean():.0f}%)"})
tiers = pd.DataFrame(tier_rows)

with pd.ExcelWriter(OUT / "panel_comparison.xlsx", engine="openpyxl") as xw:
    out.to_excel(xw, sheet_name="Comparison", index=False)
    tiers.to_excel(xw, sheet_name="Recommended_score_tiers", index=False)
    autosize(xw)
pd.set_option("display.width", 300, "display.max_columns", 20, "display.max_colwidth", 75)
print(out.to_string(index=False))
print(); print(tiers.to_string(index=False))
