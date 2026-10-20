"""APACHE II  -- PARTIAL (only 3 of 12 physiology variables + age + chronic health measurable)
Knaus WA, Draper EA, Wagner DP, Zimmerman JE. APACHE II: a severity of disease classification system.
Crit Care Med 1985;13:818-829.

Full score = Acute Physiology Score (12 variables, 0-4 each: temperature, MAP, heart rate,
respiratory rate, oxygenation, arterial pH, sodium, potassium, creatinine, haematocrit, WBC, GCS)
+ age points + chronic health points. Range 0-71.

Variables scored here:
  Haematocrit (%)     >=60: 4 | 50-59.9: 2 | 46-49.9: 1 | 30-45.9: 0 | 20-29.9: 2 | <20: 4
  WBC (x10^3/uL)      >=40: 4 | 20-39.9: 2 | 15-19.9: 1 | 3-14.9: 0  | 1-2.9: 2   | <1: 4
  Age (years)         <=44: 0 | 45-54: 2 | 55-64: 3 | 65-74: 5 | >=75: 6
  Chronic health      5 points for severe organ insufficiency / immunocompromise
                      (here: AML, CKD, cirrhosis, if listed in comorbidities)

NOT AVAILABLE (scored 0): temperature, MAP, heart rate, respiratory rate, oxygenation, pH, sodium,
potassium, creatinine, GCS. This partial score is NOT APACHE II; it shows only how much of the
APACHE II signal is recoverable from routine dengue ward data.
"""
import numpy as np
import pandas as pd

from ._helpers import comorbidity_list

INFO = {
    "name": "APACHE II (Knaus 1985) - PARTIAL",
    "short": "APACHE II (partial)",
    "citation": "Knaus WA et al. Crit Care Med 1985;13:818-829",
    "population": "General ICU",
    "original_outcome": "Hospital mortality",
    "cutoff": None,
    "cutoff_label": "no dengue-specific cut-off; Youden used",
    "items": [
        ("Haematocrit", "0-4", "available", "hct"),
        ("WBC", "0-4", "available", "wbc"),
        ("Age", "0-6", "available", "age"),
        ("Chronic health", "0 or 5", "proxy", "AML / CKD / cirrhosis in comorbidity text"),
        ("Temp, MAP, HR, RR, oxygenation, pH, Na, K, creatinine, GCS", "0-4 each",
         "NOT AVAILABLE -> 0", "-"),
    ],
}
SEVERE_CHRONIC = ("aml", "ckd", "cirrhosis", "dialysis", "immunocompromised")


def score(df: pd.DataFrame) -> pd.Series:
    h, w, a = df["hct"], df["wbc"], df["age"]
    hct_pts = np.select([h >= 60, h >= 50, h >= 46, h >= 30, h >= 20, h < 20], [4, 2, 1, 0, 2, 4], 0)
    wbc_pts = np.select([w >= 40, w >= 20, w >= 15, w >= 3, w >= 1, w < 1], [4, 2, 1, 0, 2, 4], 0)
    age_pts = np.select([a >= 75, a >= 65, a >= 55, a >= 45], [6, 5, 3, 2], 0)
    chronic = comorbidity_list(df["comorbs"]).apply(
        lambda items: 5 if any(k in i for i in items for k in SEVERE_CHRONIC) else 0)
    return pd.Series(hct_pts + wbc_pts + age_pts, index=df.index) + chronic
