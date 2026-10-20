"""Shared data loading, derived variables and formatting helpers for all analysis scripts."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "fin dengue serobook - without missingness.xlsx"
OUT = ROOT / "analysis" / "output"

SHEETS = {"adult_male": ("Adult", "Male"), "adult_female": ("Adult", "Female"),
          "ped_male": ("Pediatric", "Male"), "ped_female": ("Pediatric", "Female")}
DX_ORDER = ["DWOS", "DWS", "DS"]
AGE_BINS = [0, 12, 17, 24, 34, 44, 59, 200]
AGE_LABELS = ["5-12", "13-17", "18-24", "25-34", "35-44", "45-59", ">=60"]

CLINICAL = ["oliguria", "arthralgia", "bleeding", "rash", "lethargy", "nausea_vomiting",
            "syncope", "bodyache", "headache", "usg_binary", "comorbidity"]
LABS = ["hct", "plt", "wbc", "lymph_pct", "alc", "pt", "ptt", "ast", "alt", "br_dir",
        "br_indir", "days_fever", "age"]
LABEL = {"oliguria": "Oliguria", "arthralgia": "Arthralgia", "bleeding": "Bleeding",
         "rash": "Rash", "lethargy": "Lethargy", "nausea_vomiting": "Nausea/vomiting",
         "syncope": "Syncope", "bodyache": "Body ache", "headache": "Headache",
         "usg_binary": "USG abnormal (ascites/HSM/effusion)", "comorbidity": "Any comorbidity",
         "hct": "Haematocrit (%)", "plt": "Platelets (x10^3/uL)", "wbc": "WBC (x10^3/uL)",
         "lymph_pct": "Lymphocyte %", "alc": "Abs. lymphocyte count (x10^3/uL)", "pt": "PT (s)",
         "ptt": "aPTT (s)", "ast": "AST (U/L)", "alt": "ALT (U/L)", "br_dir": "Direct bilirubin",
         "br_indir": "Indirect bilirubin", "days_fever": "Days of fever at admission",
         "age": "Age (years)"}
# Markers that are themselves part of the WHO-2009 DWS/DS definition (index sheet)
DEFINITIONAL = {"bleeding", "lethargy", "usg_binary", "nausea_vomiting", "rash", "hct", "plt",
                "wbc", "ast", "alt"}

NUM_COLS = ["age", "days_fever", "oliguria", "hct", "plt", "wbc", "lymph_pct", "alc", "pt", "ptt",
            "ast", "alt", "br_dir", "br_indir", "arthralgia", "bleeding", "rash", "lethargy",
            "nausea_vomiting", "syncope", "bodyache", "headache", "usg_binary", "los", "icu_los",
            "transfusion"]


def load_data(src=SRC):
    """Read the four population sheets, drop empty template rows and add derived variables."""
    frames = []
    for sh, (ag, sx) in SHEETS.items():
        d = pd.read_excel(src, sheet_name=sh)
        d = d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")]
        d = d[d["dx"].notna()].copy()                      # drop empty template rows
        d["population"], d["age_cat"], d["sex"] = sh, ag, sx
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"lymphocyte_percent (20-40)": "lymph_pct",
                            "absolute lymphocyte count (1-5)": "alc"})
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # ALC is a sheet formula (wbc*lymph%/100) that writes 0 when inputs are blank
    df.loc[df["lymph_pct"].isna() | df["wbc"].isna(), "alc"] = np.nan

    df["comorbidity"] = df["comorbs"].map(lambda v: int(pd.notna(v) and str(v).strip() not in ("", "0")))
    df["icu"] = (df["icu_los"] > 0).astype(int)
    df["icu_or_tx"] = ((df["icu_los"] > 0) | (df["transfusion"] == 1)).astype(int)
    df["dx"] = df["dx"].str.strip().str.upper()
    df["dx_ord"] = df["dx"].map({k: i for i, k in enumerate(DX_ORDER)})
    df["dws_or_ds"] = (df["dx_ord"] >= 1).astype(int)
    df["severe_ds"] = (df["dx_ord"] == 2).astype(int)
    df["prolonged_los"] = (df["los"] > 3).astype(int)     # above cohort median (3 d)
    df["pediatric"] = (df["age_cat"] == "Pediatric").astype(int)
    df["male"] = (df["sex"] == "Male").astype(int)
    df["lymph_cat"] = pd.cut(df["lymph_pct"], [-np.inf, 19.999, 40, np.inf],
                             labels=["Low (<20%)", "Normal (20-40%)", "High (>40%)"])
    df["age_group"] = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS)

    # derived lab ratios (no neutrophil column exists; WBC-ALC stands in for ANC)
    df["bilirubin_total"] = df["br_dir"] + df["br_indir"]
    df["nonlymph"] = df["wbc"] - df["alc"]
    df["nlr_proxy"] = df["nonlymph"] / df["alc"].replace(0, np.nan)
    df["plr"] = df["plt"] / df["alc"].replace(0, np.nan)
    df["plt_wbc"] = df["plt"] / df["wbc"]
    df["hct_plt"] = df["hct"] / df["plt"]
    df["ast_alt"] = df["ast"] / df["alt"]
    df["apri"] = (df["ast"] / 40) / df["plt"] * 100
    df["bili_dir_frac"] = df["br_dir"] / df["bilirubin_total"].replace(0, np.nan)
    df["pt_x_ptt"] = (df["pt"] / 13) * (df["ptt"] / 32)
    return df


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def med_iqr(s):
    s = s.dropna()
    if len(s) == 0:
        return ""
    q1, q2, q3 = np.percentile(s, [25, 50, 75])
    return f"{q2:.3g} ({q1:.3g}-{q3:.3g})"


def autosize(xw):
    """Widen columns and freeze the header row of every sheet in an openpyxl ExcelWriter."""
    for ws in xw.book.worksheets:
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
        ws.freeze_panes = "B2"


# DeLong test for two correlated ROC curves (Sun & Xu 2014 fast algorithm)
def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong(y, s1, s2):
    """Two-sided p-value for H0: AUC(s1) == AUC(s2) on the same patients."""
    y = np.asarray(y).astype(bool)
    preds = np.vstack([s1, s2]).astype(float)
    pos, neg = preds[:, y], preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(p) for p in pos])
    ty = np.array([_midrank(q) for q in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return np.nan
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return 2 * stats.norm.sf(abs(z))
