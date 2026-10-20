"""Secondary / exploratory analyses that followed the main score development.

  A. What 'abnormal USG' contains: fluid (ascites / effusion) vs organomegaly only, and outcomes of each.
  B. Patients with NEITHER bleeding NOR abnormal USG: how many events, who they are, can labs find them?
  C. Lab-only search: best 1-3 predictor panels with bleeding and USG excluded (repeated 5-fold CV AUC).
  D. Added value of clotting tests / platelets on top of bleeding + USG + fever (likelihood-ratio test),
     and of an aPTT point added to the 2/2/1 score.
  E. Head-to-head on the same patients (PT and aPTT available): clinical version vs lab version vs
     WHO-2009 class, for ICU or transfusion, ICU alone, transfusion alone and severe dengue (DeLong test).
  F. Bootstrap optimism of the SECOND search (C), repeating screening + search in every resample, so the
     lab version gets a realistic AUC like the clinical version did (internal_validation.py).

These analyses were run after the main results were seen; report them as secondary / exploratory.

Run:  python analysis/secondary_analyses.py   ->  analysis/output/secondary_analyses.xlsx
"""
import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUT, autosize, delong, fmt_p, load_data

warnings.filterwarnings("ignore")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
df = load_data()
df["bedside_score"] = (2 * df["bleeding"].fillna(0) + 2 * df["usg_binary"].fillna(0)
                       + (df["days_fever"] >= 4).astype(int))
df["lab_score"] = ((df["pt"] >= 14).astype(int) + (df["ptt"] >= 36).astype(int)
                   + (df["days_fever"] >= 4).astype(int))

# Candidates for the lab-only search: everything except bleeding and abnormal USG
LAB_CANDIDATES = ["oliguria", "rash", "arthralgia", "lethargy", "nausea_vomiting", "syncope", "bodyache",
                  "headache", "comorbidity", "hct", "plt", "wbc", "lymph_pct", "alc", "pt", "ptt",
                  "days_fever", "hct_plt", "pt_x_ptt", "plt_wbc", "age"]
# composites must not be combined with their own components
REDUNDANT = [({"pt_x_ptt"}, {"pt", "ptt"}), ({"hct_plt"}, {"plt", "hct"})]


def boot_auc_ci(y, s, n=1000):
    b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], s[i]))
    return np.percentile(b, [2.5, 97.5])


def rate(s):
    return f"{int(s.sum())}/{len(s)} ({100 * s.mean():.0f}%)"


def sens_spec(y, s, t):
    pos = s >= t
    tp = (pos & (y == 1)).sum(); fp = (pos & (y == 0)).sum()
    fn = (~pos & (y == 1)).sum(); tn = (~pos & (y == 0)).sum()
    return tp / (tp + fn), tn / (tn + fp), tn / (tn + fn)


# =====================================================================================================
# A. abnormal USG: fluid vs organomegaly only
usg_text = df["usg"].astype("string").str.lower().fillna("")
df["usg_fluid"] = usg_text.str.contains("ascit|effusion").astype(int)
df["usg_organ_only"] = (usg_text.str.contains("megaly") & (df["usg_fluid"] == 0)).astype(int)
rows = []
for name, mask in [("Abnormal USG (any)", df.usg_binary == 1), ("Fluid: ascites / pleural effusion", df.usg_fluid == 1),
                   ("Organomegaly only (hepato/splenomegaly)", df.usg_organ_only == 1), ("Normal USG", df.usg_binary == 0)]:
    s = df[mask]
    rows.append({"USG finding": name, "n": len(s), "ICU": rate(s.icu), "Transfusion": rate(s.transfusion.dropna()),
                 "ICU or transfusion": rate(s.icu_or_tx), "Severe dengue": rate(s.severe_ds),
                 "DWOS / DWS / DS": "/".join(str((s.dx == k).sum()) for k in ["DWOS", "DWS", "DS"])})
usg_breakdown = pd.DataFrame(rows)
usg_findings = df.loc[df.usg_binary == 1, "usg"].astype("string").str.lower().str.strip().value_counts().rename_axis(
    "USG text").reset_index(name="Patients")

# =====================================================================================================
# B. patients with neither bleeding nor abnormal USG
neither = (df.bleeding.fillna(0) == 0) & (df.usg_binary.fillna(0) == 0)
hidden_summary = pd.DataFrame([{
    "Group": "Neither bleeding nor abnormal USG", "Patients": int(neither.sum()),
    "ICU or transfusion": int(df.loc[neither, "icu_or_tx"].sum()), "ICU": int(df.loc[neither, "icu"].sum()),
    "Transfusion": int(df.loc[neither, "transfusion"].sum()), "Severe dengue": int(df.loc[neither, "severe_ds"].sum())}, {
    "Group": "Bleeding or abnormal USG", "Patients": int((~neither).sum()),
    "ICU or transfusion": int(df.loc[~neither, "icu_or_tx"].sum()), "ICU": int(df.loc[~neither, "icu"].sum()),
    "Transfusion": int(df.loc[~neither, "transfusion"].sum()), "Severe dengue": int(df.loc[~neither, "severe_ds"].sum())}])
hidden_cases = df.loc[neither & (df.icu_or_tx == 1),
                      ["population", "age", "icu", "transfusion", "dx", "plt", "pt", "ptt", "days_fever", "oliguria", "rash"]]
d = df[neither]
hidden_auc = []
for v, sign in [("pt", 1), ("ptt", 1), ("pt_x_ptt", 1), ("plt", -1), ("hct_plt", 1), ("days_fever", 1)]:
    m = d[v].notna()
    yy = d.loc[m, "icu_or_tx"].values
    if len(np.unique(yy)) == 2:
        hidden_auc.append({"Marker (within 'neither' group)": v, "n": int(m.sum()), "Events": int(yy.sum()),
                           "AUC": round(roc_auc_score(yy, sign * d.loc[m, v].values), 2)})
hidden_auc = pd.DataFrame(hidden_auc)

# =====================================================================================================
# C. lab-only search (bleeding and USG excluded)


def screen(data, y, candidates):
    ps = {}
    for v in candidates:
        x = data[v]; m = x.notna()
        if x[m].nunique() <= 2:
            t = pd.crosstab(x[m], y[m])
            if t.shape != (2, 2):
                continue
            ps[v] = stats.fisher_exact(t.values)[1]
        else:
            ps[v] = stats.mannwhitneyu(x[m][y[m] == 1], x[m][y[m] == 0]).pvalue
    return [v for v, p in sorted(ps.items(), key=lambda t: t[1]) if p < 0.20][:10]


def combos(pool, kmax=3):
    for k in range(1, kmax + 1):
        for c in itertools.combinations(pool, k):
            if any(a & set(c) and b & set(c) for a, b in REDUNDANT):
                continue
            yield list(c)


def cv_auc(data, cols, y, reps):
    X = data[cols].values.astype(float)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    aucs = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=1).split(X, y):
        model.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
    return np.mean(aucs)


y_all = df["icu_or_tx"].values
pool = screen(df, df["icu_or_tx"], LAB_CANDIDATES)
first_pass = sorted(((cv_auc(df, c, y_all, 5), c) for c in combos(pool)), key=lambda t: -t[0])
refined = sorted(((cv_auc(df, c, y_all, 40), c) for _, c in first_pass[:10]), key=lambda t: -t[0])
lab_search = pd.DataFrame([{"Rank": i + 1, "Panel (bleeding and USG excluded)": " + ".join(c),
                            "CV AUC (5-fold x40)": round(a, 3)} for i, (a, c) in enumerate(refined)])
print("Lab-only pool:", pool)
print(lab_search.head(6).to_string(index=False))

# =====================================================================================================
# D. added value on top of bleeding + USG + fever
BASE = ["bleeding", "usg_binary", "days_fever"]
added = []
for add in ["pt", "ptt", "pt_x_ptt", "plt"]:
    sub = df.dropna(subset=[add])
    yy = sub["icu_or_tx"].values.astype(float)
    X0 = sub[BASE].fillna(sub[BASE].median())
    m0 = sm.Logit(yy, sm.add_constant(X0)).fit(disp=0)
    m1 = sm.Logit(yy, sm.add_constant(X0.assign(**{add: sub[add]}))).fit(disp=0)
    lr = 2 * (m1.llf - m0.llf)
    added.append({"Added to bleeding + USG + fever": add, "n": len(sub), "LR test p": stats.chi2.sf(lr, 1),
                  "AUC before": round(roc_auc_score(yy, m0.predict()), 3), "AUC after": round(roc_auc_score(yy, m1.predict()), 3)})
added = pd.DataFrame(added)
point_rows = []
for name, s in [("Bedside score 2/2/1", df.bedside_score),
                ("Bedside + aPTT>=36 (1 pt)", df.bedside_score + (df.ptt >= 36).astype(int)),
                ("Bedside + aPTT>=36 (2 pt)", df.bedside_score + 2 * (df.ptt >= 36).astype(int))]:
    row = {"Score": name, "AUC": round(roc_auc_score(y_all, s), 3)}
    for t in (2, 3):
        se, sp, _ = sens_spec(y_all, s.values, t)
        row[f">= {t}: sens / spec"] = f"{100 * se:.0f}% / {100 * sp:.0f}%"
    point_rows.append(row)
aptt_point = pd.DataFrame(point_rows)

# =====================================================================================================
# E. head-to-head on identical patients (PT and aPTT available), out-of-fold predictions for fitted models
h = df[df.pt.notna() & df.ptt.notna()].reset_index(drop=True)


def out_of_fold(cols, y, reps=40):
    X = h[cols].values.astype(float)
    P = np.zeros(len(y)); n = np.zeros(len(y))
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=7).split(X, y):
        model.fit(X[tr], y[tr])
        P[te] += model.predict_proba(X[te])[:, 1]; n[te] += 1
    return P / n


h2h = []
for oc, oname in [("icu_or_tx", "ICU or transfusion"), ("icu", "ICU alone"), ("transfusion", "Transfusion alone"),
                  ("severe_ds", "Severe dengue")]:
    hh = h.dropna(subset=[oc])
    y = hh[oc].values.astype(int)
    keep = hh.index.values
    tools = {"Clinical model: bleeding + USG + fever (CV)": out_of_fold(BASE, h[oc].fillna(0).values.astype(int))[keep],
             "Lab model: PT + aPTT + fever (CV)": out_of_fold(["pt", "ptt", "days_fever"], h[oc].fillna(0).values.astype(int))[keep],
             "Clinical points score 2/2/1": hh.bedside_score.values,
             "Lab points score PT>=14 + aPTT>=36 + fever>=4": hh.lab_score.values}
    if oc != "severe_ds":                       # DS is part of the WHO classification itself
        tools["WHO-2009 class (DWOS < DWS < DS)"] = hh.dx_ord.values
    clin, lab = tools["Clinical model: bleeding + USG + fever (CV)"], tools["Lab model: PT + aPTT + fever (CV)"]
    for name, s in tools.items():
        lo, hi = boot_auc_ci(y, s)
        h2h.append({"Outcome": oname, "Events / N": f"{y.sum()}/{len(y)}", "Tool": name,
                    "AUC": round(roc_auc_score(y, s), 2), "95% CI": f"{lo:.2f}-{hi:.2f}",
                    "p vs clinical model (DeLong)": np.nan if name.startswith("Clinical model") else delong(y, clin, s),
                    "p vs lab model (DeLong)": np.nan if name.startswith("Lab model") else delong(y, lab, s)})
h2h = pd.DataFrame(h2h)

# =====================================================================================================
# F. bootstrap optimism of the second (lab-only) search: screening + search repeated in each resample
Xall = df[LAB_CANDIDATES].astype(float)
Xall = Xall.fillna(Xall.median())


def develop(X, y):
    best = (-1.0, None, None)
    for cols in combos(screen(X, pd.Series(y, index=X.index), LAB_CANDIDATES)):
        try:
            m = sm.Logit(y, sm.add_constant(X[cols].values, has_constant="add")).fit(disp=0, method="bfgs", maxiter=300)
        except Exception:
            continue
        a = roc_auc_score(y, m.predict(sm.add_constant(X[cols].values, has_constant="add")))
        if a > best[0]:
            best = (a, cols, m)
    return best


app_auc, app_cols, _ = develop(Xall, y_all.astype(float))
optimism, slopes, chosen = [], [], []
for _ in range(200):
    i = rng.integers(0, len(y_all), len(y_all))
    Xb, yb = Xall.iloc[i].reset_index(drop=True), y_all[i].astype(float)
    if yb.sum() < 5:
        continue
    ab, cb, mb = develop(Xb, yb)
    lp = sm.add_constant(Xall[cb].values, has_constant="add") @ mb.params
    optimism.append(ab - roc_auc_score(y_all, lp))
    slopes.append(sm.Logit(y_all.astype(float), sm.add_constant(lp)).fit(disp=0).params[1])
    chosen.append(" + ".join(sorted(cb)))
boot = pd.DataFrame([{"Second search (bleeding and USG excluded)": "selected model", "Predictors": " + ".join(app_cols),
                      "Apparent AUC": round(app_auc, 3), "Mean optimism": round(np.mean(optimism), 3),
                      "Optimism-corrected AUC": round(app_auc - np.mean(optimism), 3),
                      "Calibration slope (shrinkage)": round(np.mean(slopes), 2), "Bootstrap samples": len(optimism)}])
selection_freq = pd.Series(chosen).value_counts().head(10).rename_axis("Panel chosen in bootstrap sample").reset_index(
    name="Times chosen (of 200)")


# =====================================================================================================
def pfmt(t):
    t = t.copy()
    for c in t.columns:
        if "p " in c or c.startswith("p") or c.endswith(" p"):
            if t[c].dtype.kind == "f":
                t[c] = t[c].map(fmt_p)
    return t


with pd.ExcelWriter(OUT / "secondary_analyses.xlsx", engine="openpyxl") as xw:
    pd.DataFrame({"Note": [
        "Exploratory analyses run after the main results were seen; report as secondary.",
        "A: abnormal USG split into fluid vs organomegaly only.",
        "B: patients with neither bleeding nor abnormal USG.",
        "C: best panels with bleeding and USG excluded (lab-only search).",
        "D: added value of PT / aPTT / platelets on top of bleeding + USG + fever.",
        "E: head-to-head on the same patients (PT and aPTT available), DeLong test.",
        "F: bootstrap optimism of the second search (lab version realistic AUC).",
    ]}).to_excel(xw, sheet_name="README", index=False)
    usg_breakdown.to_excel(xw, sheet_name="A_USG_breakdown", index=False)
    usg_findings.to_excel(xw, sheet_name="A_USG_breakdown", index=False, startrow=len(usg_breakdown) + 3)
    hidden_summary.to_excel(xw, sheet_name="B_Neither_sign", index=False)
    hidden_auc.to_excel(xw, sheet_name="B_Neither_sign", index=False, startrow=5)
    lab_search.to_excel(xw, sheet_name="C_Lab_only_search", index=False)
    pfmt(added).to_excel(xw, sheet_name="D_Added_value", index=False)
    aptt_point.to_excel(xw, sheet_name="D_Added_value", index=False, startrow=len(added) + 3)
    pfmt(h2h).to_excel(xw, sheet_name="E_Head_to_head", index=False)
    boot.to_excel(xw, sheet_name="F_Bootstrap_lab_search", index=False)
    selection_freq.to_excel(xw, sheet_name="F_Bootstrap_lab_search", index=False, startrow=4)
    autosize(xw)

# individual case rows are patient-level data: kept in output/private/ (excluded from version control)
PRIVATE = OUT / "private"
PRIVATE.mkdir(parents=True, exist_ok=True)
hidden_cases.to_excel(PRIVATE / "events_without_bleeding_or_usg.xlsx", index=False)

pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 50)
for title, t in [("A. USG breakdown", usg_breakdown), ("B. Neither sign", hidden_summary), ("B. AUC within group", hidden_auc),
                 ("D. Added value", pfmt(added)), ("D. aPTT point", aptt_point), ("E. Head-to-head", pfmt(h2h)),
                 ("F. Bootstrap of second search", boot), ("F. Selection frequency", selection_freq)]:
    print(f"\n== {title}"); print(t.to_string(index=False))
