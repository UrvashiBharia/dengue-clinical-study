"""DENGUE INFECTION SEVERITY SCORE (DISS)
Pongpan S, Wisitwong A, Tawichasri C, Patumanond J. Int J Clin Pediatr 2013;2(1):12-18 (derivation);
Pongpan S et al. Validation of dengue infection severity score. Risk Manag Healthc Policy 2014;7:45-49.
Population: Thai children aged 1-15 y (derivation n=777, validation n=400).
Outcome: DF / DHF / DSS (WHO 1997).

Item (published)                     Criterion            Points
Age                                  > 6 years              1
Hepatomegaly                         yes                    8.5
Systolic blood pressure              < 90 mmHg              2
White cell count                     > 5,000 /uL            1
Platelets                            <= 50,000 /uL          4.5
Classification: < 2.5 DF; 2.5-11.5 DHF; > 11.5 DSS.
Published AUC (validation): DHF+DSS vs DF 0.71; DSS vs rest 0.76.

Mapping in this dataset:
  - Hepatomegaly taken from ultrasound text (the original used clinical examination).
  - Systolic BP is NOT recorded -> scored 0 (max achievable 15 of 17).
"""
import pandas as pd

from ._helpers import hepatomegaly

INFO = {
    "name": "Dengue Infection Severity Score (DISS)",
    "short": "DISS",
    "citation": "Pongpan et al. Int J Clin Pediatr 2013; validation Risk Manag Healthc Policy 2014;7:45-49",
    "population": "Thai children 1-15 y",
    "original_outcome": "DHF / DSS (WHO 1997)",
    "cutoff": 2.5,
    "cutoff_label": ">= 2.5 (DHF or worse)",
    "items": [
        ("Age > 6 y", "1", "available", "age"),
        ("Hepatomegaly", "8.5", "proxy", "ultrasound text 'hepatomegaly'/'hepatosplenomegaly'"),
        ("Systolic BP < 90 mmHg", "2", "NOT AVAILABLE -> 0", "-"),
        ("WBC > 5,000/uL", "1", "available", "wbc > 5"),
        ("Platelets <= 50,000/uL", "4.5", "available", "plt <= 50"),
    ],
}


def score(df: pd.DataFrame) -> pd.Series:
    return ((df["age"] > 6).astype(int) * 1
            + hepatomegaly(df) * 8.5
            + 0                                   # systolic BP < 90 not recorded
            + (df["wbc"] > 5).astype(int) * 1
            + (df["plt"] <= 50).astype(int) * 4.5)
