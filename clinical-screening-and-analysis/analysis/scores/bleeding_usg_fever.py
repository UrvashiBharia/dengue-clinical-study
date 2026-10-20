"""BLEEDING-USG-FEVER SCORE  (this study, derived in this cohort)

Item                                                         Points
Any bleeding at admission                                       2
Plasma leakage on USG (ascites / effusion / HSM)                2
Fever for >= 4 days                                             1
Range 0-5. Proposed cut-off: >= 2 (high risk).
Risk tiers: 0 low, 1 intermediate, 2-3 high, 4-5 very high.
"""
import pandas as pd

INFO = {
    "name": "Bleeding-USG-Fever score (this study)",
    "short": "BUF (this study)",
    "citation": "Derived in this cohort (see panel_comparison.py)",
    "population": "Children and adults, this cohort",
    "original_outcome": "ICU admission or transfusion",
    "cutoff": 2,
    "cutoff_label": ">= 2",
    "items": [
        ("Bleeding", "2", "available", "bleeding column"),
        ("Plasma leakage on USG", "2", "available", "usg_binary column"),
        ("Fever >= 4 days", "1", "available (missing in 14 -> 0)", "days_fever column"),
    ],
}


def score(df: pd.DataFrame) -> pd.Series:
    return (2 * df["bleeding"].fillna(0)
            + 2 * df["usg_binary"].fillna(0)
            + (df["days_fever"] >= 4).astype(int))
