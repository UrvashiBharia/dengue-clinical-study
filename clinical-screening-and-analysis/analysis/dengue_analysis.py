"""Dengue serobook analysis: predictors of LOS, ICU/transfusion, and WHO-2009 class (DWOS/DWS/DS).

Run:  python analysis/dengue_analysis.py
Outputs go to analysis/output/ (results workbook + figures).
"""
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from common import (AGE_LABELS, CLINICAL, DEFINITIONAL, LABEL, LABS, OUT, SHEETS, autosize,
                    fmt_p, load_data, med_iqr)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

OUT.mkdir(parents=True, exist_ok=True)
df = load_data()


def bin_or(x, y):
    """2x2 OR with Haldane correction when a cell is zero, Woolf CI, Fisher exact p."""
    m = x.notna() & y.notna()
    x, y = x[m].astype(int), y[m].astype(int)
    a = ((x == 1) & (y == 1)).sum(); b = ((x == 1) & (y == 0)).sum()
    c = ((x == 0) & (y == 1)).sum(); d = ((x == 0) & (y == 0)).sum()
    p = stats.fisher_exact([[a, b], [c, d]])[1]
    aa, bb, cc, dd = (a, b, c, d) if min(a, b, c, d) > 0 else (a + .5, b + .5, c + .5, d + .5)
    orr = aa * dd / (bb * cc)
    se = np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    lo, hi = np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se)
    sens = a / (a + c) if a + c else np.nan
    spec = d / (b + d) if b + d else np.nan
    ppv = a / (a + b) if a + b else np.nan
    npv = d / (c + d) if c + d else np.nan
    return dict(n=int(m.sum()), a=a, b=b, c=c, d=d, OR=orr, lo=lo, hi=hi, p=p,
                sens=sens, spec=spec, ppv=ppv, npv=npv)


def auc_ci(y, s, n_boot=2000):
    y, s = np.asarray(y), np.asarray(s)
    auc = roc_auc_score(y, s)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        boots.append(roc_auc_score(y[idx], s[idx]))
    return auc, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    j = np.argmax(tpr - fpr)
    return thr[j], tpr[j], 1 - fpr[j]


def cont_logit_or(x, y):
    """Univariable logistic OR per 1 SD."""
    m = x.notna() & y.notna()
    if y[m].nunique() < 2:
        return np.nan, np.nan, np.nan
    z = (x[m] - x[m].mean()) / x[m].std()
    try:
        r = sm.Logit(y[m].astype(float), sm.add_constant(z)).fit(disp=0)
        b, se = r.params.iloc[1], r.bse.iloc[1]
        return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)
    except Exception:
        return np.nan, np.nan, np.nan


# --------------------------------------------------------------------------- 1. univariable
def univariable_binary_outcome(outcome, label):
    rows = []
    y = df[outcome]
    for v in CLINICAL:
        r = bin_or(df[v], y)
        rows.append({"Outcome": label, "Parameter": LABEL[v], "Type": "Clinical (1/0)", "n": r["n"],
                     "Event rate if present": f"{r['a']}/{r['a'] + r['b']} ({100 * r['a'] / max(r['a'] + r['b'], 1):.0f}%)",
                     "Event rate if absent": f"{r['c']}/{r['c'] + r['d']} ({100 * r['c'] / max(r['c'] + r['d'], 1):.0f}%)",
                     "Effect": f"OR {r['OR']:.2f} ({r['lo']:.2f}-{r['hi']:.2f})", "Test": "Fisher exact",
                     "p": r["p"], "AUC (95% CI)": "",
                     "Sensitivity": r["sens"], "Specificity": r["spec"], "PPV": r["ppv"], "NPV": r["npv"],
                     "Cut-off": "present", "In WHO definition": "yes" if v in DEFINITIONAL else ""})
    for v in LABS:
        m = df[v].notna()
        x, yy = df.loc[m, v], y[m]
        if yy.nunique() < 2 or m.sum() < 15:
            continue
        p = stats.mannwhitneyu(x[yy == 1], x[yy == 0]).pvalue
        auc = roc_auc_score(yy, x)
        direction = 1 if auc >= 0.5 else -1          # low values predictive -> flip for AUC/cutoff
        a, lo, hi = auc_ci(yy, direction * x)
        thr, se_, sp_ = youden(yy, direction * x)
        cut = f"{'>=' if direction == 1 else '<='} {direction * thr:.3g}"
        orr, olo, ohi = cont_logit_or(x, yy)
        rows.append({"Outcome": label, "Parameter": LABEL[v], "Type": "Lab / continuous", "n": int(m.sum()),
                     "Event rate if present": f"event: {med_iqr(x[yy == 1])}",
                     "Event rate if absent": f"no event: {med_iqr(x[yy == 0])}",
                     "Effect": f"OR/SD {orr:.2f} ({olo:.2f}-{ohi:.2f})", "Test": "Mann-Whitney U",
                     "p": p, "AUC (95% CI)": f"{a:.2f} ({lo:.2f}-{hi:.2f})" + (" [low=risk]" if direction == -1 else ""),
                     "Sensitivity": se_, "Specificity": sp_, "PPV": np.nan, "NPV": np.nan,
                     "Cut-off": cut, "In WHO definition": "yes" if v in DEFINITIONAL else ""})
    t = pd.DataFrame(rows)
    t["q (BH-FDR)"] = multipletests(t["p"], method="fdr_bh")[1]
    return t


def univariable_los():
    rows = []
    for v in CLINICAL:
        m = df[v].notna()
        g1, g0 = df.loc[m & (df[v] == 1), "los"], df.loc[m & (df[v] == 0), "los"]
        p = stats.mannwhitneyu(g1, g0).pvalue if len(g1) and len(g0) else np.nan
        rows.append({"Parameter": LABEL[v], "Type": "Clinical (1/0)", "n": int(m.sum()),
                     "LOS if present, median (IQR)": f"{med_iqr(g1)}  [n={len(g1)}]",
                     "LOS if absent, median (IQR)": f"{med_iqr(g0)}  [n={len(g0)}]",
                     "Mean diff (days)": g1.mean() - g0.mean(), "Spearman rho": np.nan,
                     "Test": "Mann-Whitney U", "p": p})
    for v in LABS:
        m = df[v].notna() & df["los"].notna()
        rho, p = stats.spearmanr(df.loc[m, v], df.loc[m, "los"])
        rows.append({"Parameter": LABEL[v], "Type": "Lab / continuous", "n": int(m.sum()),
                     "LOS if present, median (IQR)": "", "LOS if absent, median (IQR)": "",
                     "Mean diff (days)": np.nan, "Spearman rho": rho, "Test": "Spearman", "p": p})
    t = pd.DataFrame(rows)
    t["q (BH-FDR)"] = multipletests(t["p"], method="fdr_bh")[1]
    return t


def univariable_dx():
    rows = []
    for v in CLINICAL:
        m = df[v].notna()
        ct = pd.crosstab(df.loc[m, v], df.loc[m, "dx"]).reindex(columns=["DWOS", "DWS", "DS"], fill_value=0)
        p = stats.chi2_contingency(ct)[1] if ct.shape[0] == 2 else np.nan
        # monotone trend across DWOS<DWS<DS
        rho, pt = stats.spearmanr(df.loc[m, v], df.loc[m, "dx_ord"])
        pct = {k: f"{ct.loc[1, k] if 1 in ct.index else 0}/{ct[k].sum()} ({100 * (ct.loc[1, k] if 1 in ct.index else 0) / max(ct[k].sum(), 1):.0f}%)"
               for k in ["DWOS", "DWS", "DS"]}
        rows.append({"Parameter": LABEL[v], "Type": "Clinical (1/0)", "n": int(m.sum()),
                     "DWOS": pct["DWOS"], "DWS": pct["DWS"], "DS": pct["DS"],
                     "Test": "Chi-square (2x3)", "p (overall)": p, "p (trend, Spearman)": pt,
                     "In WHO definition": "yes" if v in DEFINITIONAL else ""})
    for v in LABS:
        m = df[v].notna()
        groups = [df.loc[m & (df.dx == k), v] for k in ["DWOS", "DWS", "DS"]]
        p = stats.kruskal(*[g for g in groups if len(g)]).pvalue
        rho, pt = stats.spearmanr(df.loc[m, v], df.loc[m, "dx_ord"])
        rows.append({"Parameter": LABEL[v], "Type": "Lab / continuous", "n": int(m.sum()),
                     "DWOS": med_iqr(groups[0]), "DWS": med_iqr(groups[1]), "DS": med_iqr(groups[2]),
                     "Test": "Kruskal-Wallis", "p (overall)": p, "p (trend, Spearman)": pt,
                     "In WHO definition": "yes" if v in DEFINITIONAL else ""})
    t = pd.DataFrame(rows)
    t["q (BH-FDR)"] = multipletests(t["p (overall)"].fillna(1), method="fdr_bh")[1]
    return t


uni_los = univariable_los()
uni_icu = univariable_binary_outcome("icu_or_tx", "ICU admission or transfusion")
uni_dx = univariable_dx()
uni_ds = univariable_binary_outcome("severe_ds", "Severe dengue (DS) vs DWOS+DWS")
uni_dws = univariable_binary_outcome("dws_or_ds", "DWS or DS vs DWOS")
uni_plos = univariable_binary_outcome("prolonged_los", "Prolonged LOS (>3 days)")

# --------------------------------------------------------------------------- 2. combined panel
# Candidates: markers with <=25% missing (AST/ALT/bilirubin are ~50% missing -> excluded from panel).
PANEL_CANDIDATES = [v for v in CLINICAL + LABS if df[v].isna().mean() <= 0.25 and v not in ("age",)]


def cv_auc(X, y, n_rep=20):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=1)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=2000))
    aucs = []
    for tr, te in cv.split(X, y):
        model.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
    return np.mean(aucs)


def search_panels(outcome, max_size, top_k=10):
    y = df[outcome].values
    # pre-screen to univariable p<0.20 to keep the search tractable and clinically sensible
    uni = {}
    for v in PANEL_CANDIDATES:
        m = df[v].notna()
        if df.loc[m, v].nunique() <= 2:
            uni[v] = bin_or(df[v], df[outcome])["p"]
        else:
            uni[v] = stats.mannwhitneyu(df.loc[m & (df[outcome] == 1), v], df.loc[m & (df[outcome] == 0), v]).pvalue
    pool = [v for v, p in sorted(uni.items(), key=lambda kv: kv[1]) if p < 0.20][:10]
    res = []
    for k in range(1, max_size + 1):
        for combo in itertools.combinations(pool, k):
            X = df[list(combo)].values.astype(float)
            res.append((combo, cv_auc(X, y, n_rep=5)))
    res.sort(key=lambda r: -r[1])
    # refine the top combos with more CV repeats
    refined = [(c, cv_auc(df[list(c)].values.astype(float), y, n_rep=40)) for c, _ in res[:top_k]]
    refined.sort(key=lambda r: -r[1])
    return pool, refined


def nested_cv_auc(outcome, pool, max_size):
    """Honest AUC of the whole 'pick the best panel' procedure (selection inside each outer fold)."""
    y = df[outcome].values
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    preds = np.zeros(len(y))
    for tr, te in outer.split(df, y):
        best, best_auc = None, -1
        for k in range(1, max_size + 1):
            for combo in itertools.combinations(pool, k):
                X = df[list(combo)].values.astype(float)
                inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=3)
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      LogisticRegression(max_iter=2000))
                aucs = []
                for itr, ite in inner.split(X[tr], y[tr]):
                    model.fit(X[tr][itr], y[tr][itr])
                    aucs.append(roc_auc_score(y[tr][ite], model.predict_proba(X[tr][ite])[:, 1]))
                if np.mean(aucs) > best_auc:
                    best, best_auc = combo, np.mean(aucs)
        X = df[list(best)].values.astype(float)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
        preds[te] = model.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, preds)


def fit_panel(outcome, combo):
    """Fit final panel; return coefficients, apparent + bootstrap-optimism-corrected AUC, Youden sens/spec."""
    y = df[outcome].values
    X = df[list(combo)].copy()
    X = X.fillna(X.median())
    r = sm.Logit(y.astype(float), sm.add_constant(X.astype(float))).fit(disp=0, method="bfgs", maxiter=500)
    prob = r.predict(sm.add_constant(X.astype(float)))
    app_auc, lo, hi = auc_ci(y, prob)
    optimism = []
    for _ in range(300):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        try:
            rb = sm.Logit(y[idx].astype(float), sm.add_constant(X.iloc[idx].astype(float))).fit(disp=0, method="bfgs", maxiter=500)
            optimism.append(roc_auc_score(y[idx], rb.predict(sm.add_constant(X.iloc[idx].astype(float))))
                            - roc_auc_score(y, rb.predict(sm.add_constant(X.astype(float)))))
        except Exception:
            pass
    thr, se_, sp_ = youden(y, prob)
    pred = (prob >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    coef = pd.DataFrame({"Term": ["Intercept"] + [LABEL[c] for c in combo],
                         "Coef (log-odds)": r.params.values,
                         "Adjusted OR": np.exp(r.params.values),
                         "OR 95% CI": [f"{np.exp(a):.2f}-{np.exp(b):.2f}" for a, b in r.conf_int().values],
                         "p": r.pvalues.values})
    summ = dict(apparent_auc=app_auc, auc_lo=lo, auc_hi=hi,
                optimism_corrected_auc=app_auc - np.mean(optimism), prob_cutoff=thr,
                sensitivity=se_, specificity=sp_, ppv=tp / (tp + fp), npv=tn / (tn + fn),
                tp=tp, fp=fp, fn=fn, tn=tn, events=int(y.sum()), n=len(y))
    return coef, summ, prob


PANEL_TARGETS = [("icu_or_tx", "ICU admission or transfusion", 3),
                 ("severe_ds", "Severe dengue (DS)", 3),
                 ("dws_or_ds", "DWS or DS (vs DWOS)", 4),
                 ("prolonged_los", "Prolonged LOS (>3 days)", 4)]
panel_rows, panel_coefs, panel_probs, panel_search = [], {}, {}, []
for outcome, label, kmax in PANEL_TARGETS:
    pool, ranked = search_panels(outcome, kmax)
    nested = nested_cv_auc(outcome, pool, min(kmax, 3))
    best = ranked[0][0]
    coef, summ, prob = fit_panel(outcome, best)
    panel_coefs[label] = coef
    panel_probs[label] = (df[outcome].values, prob, best)
    for c, a in ranked:
        panel_search.append({"Outcome": label, "Panel": " + ".join(LABEL[x] for x in c),
                             "Markers": len(c), "Cross-validated AUC (5-fold x40)": a})
    panel_rows.append({"Outcome": label, "Events / N": f"{summ['events']}/{summ['n']}",
                       "Screened pool (univariable p<0.20)": ", ".join(LABEL[x] for x in pool),
                       "Best panel": " + ".join(LABEL[x] for x in best),
                       "Apparent AUC (95% CI)": f"{summ['apparent_auc']:.2f} ({summ['auc_lo']:.2f}-{summ['auc_hi']:.2f})",
                       "Cross-validated AUC": ranked[0][1],
                       "Bootstrap optimism-corrected AUC": summ["optimism_corrected_auc"],
                       "Nested-CV AUC (selection included)": nested,
                       "Sensitivity": summ["sensitivity"], "Specificity": summ["specificity"],
                       "PPV": summ["ppv"], "NPV": summ["npv"],
                       "Predicted-probability cut-off (Youden)": summ["prob_cutoff"],
                       "TP/FP/FN/TN": f"{summ['tp']}/{summ['fp']}/{summ['fn']}/{summ['tn']}"})
panel_summary = pd.DataFrame(panel_rows)
panel_search = pd.DataFrame(panel_search)

# Simple bedside score (count of dichotomised markers) for ICU/transfusion and DS
def bedside_score(outcome, items):
    s = sum(cond(df).astype(int) for _, cond in items)
    y = df[outcome]
    rows = []
    for k in range(1, len(items) + 1):
        pos = s >= k
        tp = (pos & (y == 1)).sum(); fp = (pos & (y == 0)).sum()
        fn = (~pos & (y == 1)).sum(); tn = (~pos & (y == 0)).sum()
        rows.append({"Score >=": k, "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp),
                     "PPV": tp / (tp + fp) if tp + fp else np.nan, "NPV": tn / (tn + fn) if tn + fn else np.nan,
                     "Patients flagged": int(pos.sum())})
    auc = roc_auc_score(y, s)
    dist = pd.crosstab(s, y).rename(columns={0: "No event", 1: "Event"})
    return pd.DataFrame(rows), auc, dist


# --------------------------------------------------------------------------- 3. lymphocyte % categories
def lymph_analysis():
    d = df[df["lymph_cat"].notna()].copy()
    out = []
    for cat in ["Low (<20%)", "Normal (20-40%)", "High (>40%)"]:
        g = d[d.lymph_cat == cat]
        out.append({"Lymphocyte category": cat, "n": len(g),
                    "LOS median (IQR)": med_iqr(g.los), "LOS mean": g.los.mean(),
                    "ICU/transfusion": f"{g.icu_or_tx.sum()} ({100 * g.icu_or_tx.mean():.0f}%)",
                    "DWOS": f"{(g.dx == 'DWOS').sum()} ({100 * (g.dx == 'DWOS').mean():.0f}%)",
                    "DWS": f"{(g.dx == 'DWS').sum()} ({100 * (g.dx == 'DWS').mean():.0f}%)",
                    "DS": f"{(g.dx == 'DS').sum()} ({100 * (g.dx == 'DS').mean():.0f}%)",
                    "Platelets median": med_iqr(g.plt), "WBC median": med_iqr(g.wbc),
                    "Days of fever median": med_iqr(g.days_fever),
                    "Pediatric %": f"{100 * g.pediatric.mean():.0f}%"})
    desc = pd.DataFrame(out)
    tests = []
    tests.append({"Comparison": "LOS across 3 categories", "Test": "Kruskal-Wallis",
                  "p": stats.kruskal(*[d.loc[d.lymph_cat == c, "los"] for c in d.lymph_cat.cat.categories]).pvalue})
    for oc, nm in [("icu_or_tx", "ICU/transfusion"), ("severe_ds", "DS"), ("dws_or_ds", "DWS or DS")]:
        ct = pd.crosstab(d.lymph_cat, d[oc])
        tests.append({"Comparison": f"{nm} across 3 categories", "Test": "Chi-square (3x2)",
                      "p": stats.chi2_contingency(ct)[1]})
    ct = pd.crosstab(d.lymph_cat, d.dx)
    tests.append({"Comparison": "Dx distribution (DWOS/DWS/DS) across 3 categories", "Test": "Chi-square (3x3)",
                  "p": stats.chi2_contingency(ct)[1]})
    # each abnormal direction vs normal, and as a diagnostic test
    pv = []
    for cat, flag in [("Low (<20%)", d.lymph_pct < 20), ("High (>40%)", d.lymph_pct > 40)]:
        sub = d[(d.lymph_cat == cat) | (d.lymph_cat == "Normal (20-40%)")]
        for oc, nm in [("icu_or_tx", "ICU/transfusion"), ("severe_ds", "Severe dengue (DS)"),
                       ("dws_or_ds", "DWS or DS"), ("prolonged_los", "Prolonged LOS >3 d")]:
            r = bin_or((sub.lymph_cat == cat).astype(int), sub[oc])
            whole = bin_or(flag.astype(int), d[oc])      # as a test in the full cohort
            pv.append({"Category": cat, "Outcome": nm,
                       "OR vs normal (95% CI)": f"{r['OR']:.2f} ({r['lo']:.2f}-{r['hi']:.2f})", "p (Fisher)": r["p"],
                       "Sensitivity": whole["sens"], "Specificity": whole["spec"],
                       "PPV": whole["ppv"], "NPV": whole["npv"]})
        g1 = d.loc[d.lymph_cat == cat, "los"]; g0 = d.loc[d.lymph_cat == "Normal (20-40%)", "los"]
        pv.append({"Category": cat, "Outcome": "LOS (vs normal)",
                   "OR vs normal (95% CI)": f"mean diff {g1.mean() - g0.mean():+.2f} d", "p (Fisher)": stats.mannwhitneyu(g1, g0).pvalue,
                   "Sensitivity": np.nan, "Specificity": np.nan, "PPV": np.nan, "NPV": np.nan})
    # high vs low directly
    hl = d[d.lymph_cat != "Normal (20-40%)"]
    for oc, nm in [("icu_or_tx", "ICU/transfusion"), ("severe_ds", "Severe dengue (DS)"), ("dws_or_ds", "DWS or DS")]:
        r = bin_or((hl.lymph_cat == "High (>40%)").astype(int), hl[oc])
        pv.append({"Category": "High vs Low (direct)", "Outcome": nm,
                   "OR vs normal (95% CI)": f"{r['OR']:.2f} ({r['lo']:.2f}-{r['hi']:.2f})", "p (Fisher)": r["p"],
                   "Sensitivity": np.nan, "Specificity": np.nan, "PPV": np.nan, "NPV": np.nan})
    pv.append({"Category": "High vs Low (direct)", "Outcome": "LOS",
               "OR vs normal (95% CI)": f"mean diff {hl.loc[hl.lymph_cat == 'High (>40%)', 'los'].mean() - hl.loc[hl.lymph_cat == 'Low (<20%)', 'los'].mean():+.2f} d",
               "p (Fisher)": stats.mannwhitneyu(hl.loc[hl.lymph_cat == 'High (>40%)', 'los'], hl.loc[hl.lymph_cat == 'Low (<20%)', 'los']).pvalue,
               "Sensitivity": np.nan, "Specificity": np.nan, "PPV": np.nan, "NPV": np.nan})
    # adjusted for platelets, age group (adult/ped) and days of fever
    adj = []
    for oc, nm in [("icu_or_tx", "ICU/transfusion"), ("dws_or_ds", "DWS or DS"), ("prolonged_los", "Prolonged LOS >3 d")]:
        sub = d.dropna(subset=["plt"]).copy()
        X = pd.DataFrame({"Low (<20%)": (sub.lymph_cat == "Low (<20%)").astype(float),
                          "High (>40%)": (sub.lymph_cat == "High (>40%)").astype(float),
                          "Platelets (per 10)": sub.plt / 10, "Pediatric": sub.pediatric.astype(float)})
        try:
            r = sm.Logit(sub[oc].astype(float), sm.add_constant(X)).fit(disp=0)
            for t in ["Low (<20%)", "High (>40%)"]:
                ci = r.conf_int().loc[t]
                adj.append({"Outcome": nm, "Category (ref = normal)": t,
                            "Adjusted OR (platelets, pediatric)": f"{np.exp(r.params[t]):.2f} ({np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f})",
                            "p": r.pvalues[t]})
        except Exception as e:
            adj.append({"Outcome": nm, "Category (ref = normal)": "model failed", "Adjusted OR (platelets, pediatric)": str(e), "p": np.nan})
    return desc, pd.DataFrame(tests), pd.DataFrame(pv), pd.DataFrame(adj)


lym_desc, lym_tests, lym_pv, lym_adj = lymph_analysis()

# --------------------------------------------------------------------------- 4. demographics
def compare_groups(col, a_val, b_val, a_name, b_name):
    A, B = df[df[col] == a_val], df[df[col] == b_val]
    rows = [{"Variable": "N", a_name: len(A), b_name: len(B), "p": np.nan, "Test": ""}]
    rows.append({"Variable": "Age, median (IQR)", a_name: med_iqr(A.age), b_name: med_iqr(B.age),
                 "p": stats.mannwhitneyu(A.age, B.age).pvalue, "Test": "Mann-Whitney"})
    rows.append({"Variable": "LOS (days), median (IQR)", a_name: f"{med_iqr(A.los)}; mean {A.los.mean():.2f}",
                 b_name: f"{med_iqr(B.los)}; mean {B.los.mean():.2f}",
                 "p": stats.mannwhitneyu(A.los, B.los).pvalue, "Test": "Mann-Whitney"})
    for oc, nm in [("icu", "ICU admission"), ("transfusion", "Transfusion"), ("icu_or_tx", "ICU or transfusion"),
                   ("severe_ds", "Severe dengue (DS)"), ("dws_or_ds", "DWS or DS"), ("prolonged_los", "LOS >3 days")]:
        a, b = A[oc].dropna(), B[oc].dropna()
        p = stats.fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1]
        rows.append({"Variable": nm, a_name: f"{int(a.sum())} ({100 * a.mean():.1f}%)",
                     b_name: f"{int(b.sum())} ({100 * b.mean():.1f}%)", "p": p, "Test": "Fisher exact"})
    ct = pd.crosstab(df[df[col].isin([a_val, b_val])][col], df.dx)
    rows.append({"Variable": "Dx DWOS/DWS/DS",
                 a_name: "/".join(str((A.dx == k).sum()) for k in ["DWOS", "DWS", "DS"]),
                 b_name: "/".join(str((B.dx == k).sum()) for k in ["DWOS", "DWS", "DS"]),
                 "p": stats.chi2_contingency(ct)[1], "Test": "Chi-square"})
    for v in CLINICAL:
        a, b = A[v].dropna(), B[v].dropna()
        p = stats.fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1]
        rows.append({"Variable": LABEL[v], a_name: f"{int(a.sum())}/{len(a)} ({100 * a.mean():.0f}%)",
                     b_name: f"{int(b.sum())}/{len(b)} ({100 * b.mean():.0f}%)", "p": p, "Test": "Fisher exact"})
    for v in LABS[:-1]:
        a, b = A[v].dropna(), B[v].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        rows.append({"Variable": LABEL[v], a_name: f"{med_iqr(a)} [n={len(a)}]", b_name: f"{med_iqr(b)} [n={len(b)}]",
                     "p": stats.mannwhitneyu(a, b).pvalue, "Test": "Mann-Whitney"})
    return pd.DataFrame(rows)


sex_cmp = compare_groups("sex", "Male", "Female", "Male", "Female")
age_cmp = compare_groups("age_cat", "Adult", "Pediatric", "Adult", "Pediatric")
# sex within strata
sex_strata = []
for stratum in ["Adult", "Pediatric"]:
    s = df[df.age_cat == stratum]
    for oc, nm in [("icu_or_tx", "ICU or transfusion"), ("dws_or_ds", "DWS or DS"), ("severe_ds", "DS"), ("prolonged_los", "LOS >3 d")]:
        m, f = s[s.sex == "Male"][oc], s[s.sex == "Female"][oc]
        sex_strata.append({"Stratum": stratum, "Outcome": nm,
                           "Male": f"{int(m.sum())}/{len(m)} ({100 * m.mean():.0f}%)",
                           "Female": f"{int(f.sum())}/{len(f)} ({100 * f.mean():.0f}%)",
                           "p (Fisher)": stats.fisher_exact([[m.sum(), len(m) - m.sum()], [f.sum(), len(f) - f.sum()]])[1]})
    m, f = s[s.sex == "Male"].los, s[s.sex == "Female"].los
    sex_strata.append({"Stratum": stratum, "Outcome": "LOS median (IQR)", "Male": med_iqr(m), "Female": med_iqr(f),
                       "p (Fisher)": stats.mannwhitneyu(m, f).pvalue})
sex_strata = pd.DataFrame(sex_strata)

# adjusted model: sex + pediatric + comorbidity
adj_demo = []
for oc, nm in [("icu_or_tx", "ICU or transfusion"), ("dws_or_ds", "DWS or DS"), ("prolonged_los", "LOS >3 d")]:
    X = df[["male", "pediatric", "comorbidity"]].astype(float)
    r = sm.Logit(df[oc].astype(float), sm.add_constant(X)).fit(disp=0)
    for t, tn in [("male", "Male (ref female)"), ("pediatric", "Pediatric (ref adult)"), ("comorbidity", "Any comorbidity")]:
        ci = r.conf_int().loc[t]
        adj_demo.append({"Outcome": nm, "Term": tn,
                         "Adjusted OR (95% CI)": f"{np.exp(r.params[t]):.2f} ({np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f})",
                         "p": r.pvalues[t]})
# LOS: negative-binomial-free -> linear model on LOS (robust SE)
X = sm.add_constant(df[["male", "pediatric", "comorbidity"]].astype(float))
r = sm.OLS(df.los.astype(float), X).fit(cov_type="HC3")
for t, tn in [("male", "Male (ref female)"), ("pediatric", "Pediatric (ref adult)"), ("comorbidity", "Any comorbidity")]:
    ci = r.conf_int().loc[t]
    adj_demo.append({"Outcome": "LOS (days), linear", "Term": tn,
                     "Adjusted OR (95% CI)": f"beta {r.params[t]:+.2f} d ({ci[0]:+.2f} to {ci[1]:+.2f})", "p": r.pvalues[t]})
adj_demo = pd.DataFrame(adj_demo)

# age groups
ag_rows = []
for g in AGE_LABELS:
    s = df[df.age_group == g]
    if len(s) == 0:
        continue
    ag_rows.append({"Age group": g, "n": len(s), "Male %": f"{100 * s.male.mean():.0f}%",
                    "Comorbidity %": f"{100 * s.comorbidity.mean():.0f}%",
                    "LOS median (IQR)": med_iqr(s.los), "LOS mean": round(s.los.mean(), 2),
                    "LOS >3 d %": round(100 * s.prolonged_los.mean(), 1),
                    "ICU or transfusion %": round(100 * s.icu_or_tx.mean(), 1),
                    "DWS or DS %": round(100 * s.dws_or_ds.mean(), 1),
                    "DS %": round(100 * s.severe_ds.mean(), 1),
                    "Platelets median": med_iqr(s.plt), "Lymph % median": med_iqr(s.lymph_pct),
                    "Bleeding %": round(100 * s.bleeding.mean(), 1),
                    "USG abnormal %": round(100 * s.usg_binary.mean(), 1),
                    "Days fever median": med_iqr(s.days_fever)})
age_groups = pd.DataFrame(ag_rows)
# composite "poor outcome" = any of ICU/transfusion, DS, or LOS > 3
df["poor_outcome"] = ((df.icu_or_tx == 1) | (df.severe_ds == 1) | (df.prolonged_los == 1)).astype(int)
age_groups["Any poor outcome %"] = [round(100 * df[df.age_group == g].poor_outcome.mean(), 1) for g in age_groups["Age group"]]
age_tests = []
for oc, nm in [("icu_or_tx", "ICU or transfusion"), ("dws_or_ds", "DWS or DS"), ("severe_ds", "DS"),
               ("prolonged_los", "LOS >3 d"), ("poor_outcome", "Any poor outcome")]:
    ct = pd.crosstab(df.age_group, df[oc])
    rho, pt = stats.spearmanr(df.age, df[oc])
    age_tests.append({"Outcome": nm, "Chi-square across groups p": stats.chi2_contingency(ct)[1],
                      "Age (continuous) Spearman rho": rho, "trend p": pt})
age_tests.append({"Outcome": "LOS", "Chi-square across groups p": stats.kruskal(*[df[df.age_group == g].los for g in AGE_LABELS if (df.age_group == g).any()]).pvalue,
                  "Age (continuous) Spearman rho": stats.spearmanr(df.age, df.los)[0], "trend p": stats.spearmanr(df.age, df.los)[1]})
age_tests = pd.DataFrame(age_tests)

# --------------------------------------------------------------------------- descriptive table 1
t1 = []
for nm, s in [("All", df)] + [(k, df[df.population == k]) for k in SHEETS]:
    t1.append({"Group": nm, "N": len(s), "Age median (IQR)": med_iqr(s.age),
               "DWOS/DWS/DS": "/".join(str((s.dx == k).sum()) for k in ["DWOS", "DWS", "DS"]),
               "ICU": int(s.icu.sum()), "Transfusion": int(s.transfusion.sum()),
               "ICU or transfusion": int(s.icu_or_tx.sum()), "LOS median (IQR)": med_iqr(s.los),
               "Comorbidity": int(s.comorbidity.sum())})
table1 = pd.DataFrame(t1)
missing = pd.DataFrame({"Variable": [LABEL.get(v, v) for v in CLINICAL + LABS],
                        "n available": [int(df[v].notna().sum()) for v in CLINICAL + LABS],
                        "% missing": [round(100 * df[v].isna().mean(), 1) for v in CLINICAL + LABS]})

# --------------------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (label, (y, prob, combo)) in zip(axes, panel_probs.items()):
    fpr, tpr, _ = roc_curve(y, prob)
    ax.plot(fpr, tpr, color="#1f5fa8", lw=2, label=f"Panel AUC {roc_auc_score(y, prob):.2f}")
    # best single marker overlay
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--")
    ax.set_title(label + "\n" + " + ".join(LABEL[c] for c in combo), fontsize=8)
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity"); ax.legend(loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig1_panel_roc.png", dpi=200); plt.close(fig)

# forest plot ICU/transfusion clinical markers
f = uni_icu[uni_icu.Type == "Clinical (1/0)"].copy()
vals = f["Effect"].str.extract(r"OR ([\d.]+) \(([\d.]+)-([\d.]+)\)").astype(float)
fig, ax = plt.subplots(figsize=(6, 4.5))
yy = np.arange(len(f))[::-1]
ax.errorbar(vals[0], yy, xerr=[vals[0] - vals[1], vals[2] - vals[0]], fmt="o", color="#1f5fa8", ecolor="#7fa6d6", capsize=2)
ax.axvline(1, color="#888", lw=0.8); ax.set_xscale("log"); ax.set_yticks(yy); ax.set_yticklabels(f.Parameter)
for yi, p in zip(yy, f.p):
    ax.text(ax.get_xlim()[1] if False else vals[2].max() * 1.3, yi, f"p={fmt_p(p)}", va="center", fontsize=7)
ax.set_xlabel("Odds ratio (log scale)"); ax.set_title("Clinical features vs ICU admission or transfusion")
fig.tight_layout(); fig.savefig(OUT / "fig2_forest_icu_tx.png", dpi=200); plt.close(fig)

# age-group outcome profile
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(age_groups))
w = 0.2
for i, (col, colr) in enumerate([("ICU or transfusion %", "#c0392b"), ("DS %", "#e67e22"),
                                  ("DWS or DS %", "#1f5fa8"), ("LOS >3 d %", "#7f8c8d")]):
    ax.bar(x + (i - 1.5) * w, age_groups[col], w, label=col.replace(" %", ""), color=colr)
ax.set_xticks(x); ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(age_groups["Age group"], age_groups.n)])
ax.set_ylabel("% of patients"); ax.legend(frameon=False, fontsize=7); ax.set_title("Outcomes by age group")
fig.tight_layout(); fig.savefig(OUT / "fig3_age_groups.png", dpi=200); plt.close(fig)

# lymphocyte categories
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
d = df[df.lymph_cat.notna()]
cats = ["Low (<20%)", "Normal (20-40%)", "High (>40%)"]
axes[0].boxplot([d[d.lymph_cat == c].los for c in cats], tick_labels=cats)
axes[0].set_ylabel("Length of stay (days)"); axes[0].set_title("LOS by lymphocyte %")
bottom = np.zeros(3)
for k, colr in [("DWOS", "#9ecae1"), ("DWS", "#3182bd"), ("DS", "#08306b")]:
    v = np.array([100 * (d[d.lymph_cat == c].dx == k).mean() for c in cats])
    axes[1].bar(cats, v, bottom=bottom, color=colr, label=k); bottom += v
axes[1].set_ylabel("% of category"); axes[1].legend(frameon=False, fontsize=7); axes[1].set_title("WHO class by lymphocyte %")
fig.tight_layout(); fig.savefig(OUT / "fig4_lymphocyte.png", dpi=200); plt.close(fig)

# --------------------------------------------------------------------------- bedside scores
icu_score_items = [("Platelets < 50", lambda d: d.plt < 50), ("Bleeding", lambda d: d.bleeding == 1),
                   ("USG abnormal", lambda d: d.usg_binary == 1), ("Oliguria", lambda d: d.oliguria == 1)]
bs_icu, bs_icu_auc, bs_icu_dist = bedside_score("icu_or_tx", icu_score_items)

# --------------------------------------------------------------------------- write workbook
def pfmt(t):
    t = t.copy()
    for c in t.columns:
        if c.startswith("p") or c.startswith("q") or c.endswith(" p") or "trend p" in c:
            if t[c].dtype.kind == "f":
                t[c] = t[c].map(fmt_p)
    for c in ["Sensitivity", "Specificity", "PPV", "NPV"]:
        if c in t.columns:
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{100 * v:.0f}%")
    return t


with pd.ExcelWriter(OUT / "dengue_results.xlsx", engine="openpyxl") as xw:
    readme = pd.DataFrame({"Sheet": [
        "1_Cohort", "1b_Missingness", "2_LOS_univariable", "3_ICU_TX_univariable", "4_Dx_3group",
        "4b_DS_vs_rest", "4c_DWS+DS_vs_DWOS", "4d_ProlongedLOS", "5_Panel_summary", "5b_Panel_search",
        "5c_Panel_coefs", "5d_Bedside_score", "6_Lymph_desc", "6b_Lymph_tests", "6c_Lymph_predictive",
        "6d_Lymph_adjusted", "7_Male_vs_Female", "7b_Sex_by_stratum", "8_Adult_vs_Ped", "8b_Adjusted_demo",
        "9_Age_groups", "9b_Age_tests"],
        "Contents": [
            "Cohort description per sheet", "Data availability for each parameter",
            "Each parameter vs length of stay (Mann-Whitney / Spearman), BH-FDR q",
            "Each parameter vs ICU admission or transfusion: OR, p, AUC, optimal cut-off, sens/spec",
            "Each parameter across DWOS/DWS/DS (chi-square / Kruskal-Wallis + trend)",
            "Each parameter vs severe dengue (binary)", "Each parameter vs DWS or DS (binary)",
            "Each parameter vs LOS > 3 days (binary)",
            "Best combined panel per outcome with cross-validated, optimism-corrected and nested-CV AUC, sens/spec",
            "Top 10 candidate panels per outcome ranked by cross-validated AUC",
            "Logistic regression coefficients of each final panel",
            "Simple additive bedside score for ICU/transfusion: sens/spec per threshold",
            "Outcomes by lymphocyte % category", "Global tests across lymphocyte categories",
            "Low and high lymphocyte % vs normal: OR, sens, spec, PPV, NPV",
            "Low/high lymphocyte % adjusted for platelets and pediatric status",
            "Male vs female: outcomes, symptoms, labs", "Male vs female within adult and pediatric strata",
            "Adult vs pediatric: outcomes, symptoms, labs", "Multivariable: sex, pediatric, comorbidity",
            "Outcomes by age group", "Tests across age groups / age trend"]})
    readme.to_excel(xw, sheet_name="README", index=False)
    table1.to_excel(xw, sheet_name="1_Cohort", index=False)
    missing.to_excel(xw, sheet_name="1b_Missingness", index=False)
    pfmt(uni_los).to_excel(xw, sheet_name="2_LOS_univariable", index=False)
    pfmt(uni_icu).to_excel(xw, sheet_name="3_ICU_TX_univariable", index=False)
    pfmt(uni_dx).to_excel(xw, sheet_name="4_Dx_3group", index=False)
    pfmt(uni_ds).to_excel(xw, sheet_name="4b_DS_vs_rest", index=False)
    pfmt(uni_dws).to_excel(xw, sheet_name="4c_DWS+DS_vs_DWOS", index=False)
    pfmt(uni_plos).to_excel(xw, sheet_name="4d_ProlongedLOS", index=False)
    pfmt(panel_summary).to_excel(xw, sheet_name="5_Panel_summary", index=False)
    panel_search.to_excel(xw, sheet_name="5b_Panel_search", index=False)
    r0 = 0
    for k, v in panel_coefs.items():
        pd.DataFrame({"Term": [k]}).to_excel(xw, sheet_name="5c_Panel_coefs", index=False, header=False, startrow=r0)
        pfmt(v).to_excel(xw, sheet_name="5c_Panel_coefs", index=False, startrow=r0 + 1)
        r0 += len(v) + 4
    pfmt(bs_icu).to_excel(xw, sheet_name="5d_Bedside_score", index=False)
    bs_icu_dist.to_excel(xw, sheet_name="5d_Bedside_score", startrow=len(bs_icu) + 3)
    pd.DataFrame({"Item": [n for n, _ in icu_score_items] + ["Score AUC"],
                  "Points": [1] * len(icu_score_items) + [round(bs_icu_auc, 3)]}).to_excel(
        xw, sheet_name="5d_Bedside_score", index=False, startcol=8)
    lym_desc.to_excel(xw, sheet_name="6_Lymph_desc", index=False)
    pfmt(lym_tests).to_excel(xw, sheet_name="6b_Lymph_tests", index=False)
    pfmt(lym_pv).to_excel(xw, sheet_name="6c_Lymph_predictive", index=False)
    pfmt(lym_adj).to_excel(xw, sheet_name="6d_Lymph_adjusted", index=False)
    pfmt(sex_cmp).to_excel(xw, sheet_name="7_Male_vs_Female", index=False)
    pfmt(sex_strata).to_excel(xw, sheet_name="7b_Sex_by_stratum", index=False)
    pfmt(age_cmp).to_excel(xw, sheet_name="8_Adult_vs_Ped", index=False)
    pfmt(adj_demo).to_excel(xw, sheet_name="8b_Adjusted_demo", index=False)
    age_groups.to_excel(xw, sheet_name="9_Age_groups", index=False)
    pfmt(age_tests).to_excel(xw, sheet_name="9b_Age_tests", index=False)
    autosize(xw)

# --------------------------------------------------------------------------- console digest
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
print(table1.to_string(index=False))
for nm, t in [("LOS", uni_los), ("ICU/TX", uni_icu), ("DX", uni_dx), ("DS", uni_ds), ("DWS+DS", uni_dws), ("PLOS", uni_plos)]:
    print(f"\n==== {nm}")
    cols = [c for c in t.columns if c not in ("Outcome", "Type", "Test", "PPV", "NPV")]
    print(pfmt(t)[cols].to_string(index=False))
print("\n==== PANELS"); print(pfmt(panel_summary).T.to_string())
print(panel_search.groupby("Outcome").head(4).to_string(index=False))
for k, v in panel_coefs.items():
    print(k); print(pfmt(v).to_string(index=False))
print("\n==== BEDSIDE ICU score AUC", round(bs_icu_auc, 3)); print(pfmt(bs_icu).to_string(index=False)); print(bs_icu_dist)
print("\n==== LYMPH"); print(lym_desc.to_string(index=False)); print(pfmt(lym_tests).to_string(index=False))
print(pfmt(lym_pv).to_string(index=False)); print(pfmt(lym_adj).to_string(index=False))
print("\n==== SEX"); print(pfmt(sex_cmp).to_string(index=False)); print(pfmt(sex_strata).to_string(index=False))
print("\n==== AGE CAT"); print(pfmt(age_cmp).to_string(index=False)); print(pfmt(adj_demo).to_string(index=False))
print("\n==== AGE GROUPS"); print(age_groups.to_string(index=False)); print(pfmt(age_tests).to_string(index=False))
