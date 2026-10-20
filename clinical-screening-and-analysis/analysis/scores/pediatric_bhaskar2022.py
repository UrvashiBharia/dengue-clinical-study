"""PEDIATRIC COMPLICATED DENGUE SCORE
Bhaskar M, Mahalingam S, Harish BN, Achappa B. Predictive scoring system for risk of complications in
pediatric dengue infection. F1000Research 2022;11:446.
Population: 303 Indian children 0-18 y (63 complicated). Outcome: complicated dengue (WHO).
Published: cut-off >= 2 -> sensitivity 84.1%, specificity 72.5%, AUC 0.86 (0.80-0.92).

Item                                     Points
Hypotension (age-specific charts)          3
PCV (haematocrit) >= 42%                   1
Platelets <= 75,000 /uL                    1
WBC >= 7,000 /uL                           1
ALT >= 70 U/L                              1
Range 0-7; cut-off >= 2.

Mapping in this dataset:
  - Hypotension NOT recorded -> 0 (so max achievable 4 of 7).
  - ALT available in ~50% (only 13 of 36 children); missing -> 0.
  - Developed for children; applied to everyone here and reported separately for children.
"""
import pandas as pd

INFO = {
    "name": "Pediatric Complicated Dengue Score (Bhaskar 2022)",
    "short": "Pediatric (Bhaskar)",
    "citation": "Bhaskar M et al. F1000Research 2022;11:446",
    "population": "Indian children 0-18 y",
    "original_outcome": "Complicated dengue (WHO)",
    "cutoff": 2,
    "cutoff_label": ">= 2",
    "items": [
        ("Hypotension", "3", "NOT AVAILABLE -> 0", "-"),
        ("PCV >= 42%", "1", "available", "hct"),
        ("Platelets <= 75,000", "1", "available", "plt"),
        ("WBC >= 7,000", "1", "available", "wbc"),
        ("ALT >= 70 U/L", "1", "partial (n=85; 13 children)", "alt"),
    ],
}


def score(df: pd.DataFrame) -> pd.Series:
    return (0                                      # hypotension not recorded
            + (df["hct"] >= 42).astype(int)
            + (df["plt"] <= 75).astype(int)
            + (df["wbc"] >= 7).astype(int)
            + (df["alt"] >= 70).astype(int))
