"""ADULT SEVERE DENGUE (SD) RISK SCORE
Lee IK, Liu JW, Chen YH, et al. Development of a simple clinical risk score for early prediction of
severe dengue in adult patients. PLoS One 2016;11(5):e0154772.
Population: adults (>= 18 y) with lab-confirmed dengue, southern Taiwan.
Outcome: WHO-2009 severe dengue.

Model 1 (illness <= 4 days)                     Points
  Leukocytosis (WBC > 10 x10^9/L)                  3
  Minor gastrointestinal bleeding (melena)         2
  Age >= 65 years                                  1
  Platelets >= 100 x10^9/L                        -2
  Range -2 to +6; cut-off >= 1 (sens 70.3%, spec 90.6%, AUC 0.848; validation AUC 0.904)

Model 2 (illness > 4 days)                      Points
  Leukocytosis (WBC > 10 x10^9/L)                  2
  Age >= 65 years                                  1
  Range 0 to 3; cut-off >= 1 (sens 80.3%, spec 85.8%, AUC 0.859; validation AUC 0.917)

Mapping in this dataset:
  - Model chosen by days_fever; if days_fever is missing, model 1 is used.
  - Minor GI bleeding = melena / PR bleed / haematemesis in bleeding-type text.
  - Developed for adults; applied to everyone here and reported separately for adults.
"""
import pandas as pd

from ._helpers import gi_bleeding

INFO = {
    "name": "Adult Severe Dengue Risk Score (Lee 2016)",
    "short": "Adult SD (Lee)",
    "citation": "Lee IK et al. PLoS One 2016;11:e0154772",
    "population": "Taiwanese adults >= 18 y",
    "original_outcome": "WHO-2009 severe dengue",
    "cutoff": 1,
    "cutoff_label": ">= 1",
    "items": [
        ("Leukocytosis WBC > 10", "3 (model 1) / 2 (model 2)", "available", "wbc > 10"),
        ("Minor GI bleeding", "2 (model 1)", "proxy", "bleeding-type text: melena, PR bleed, haematemesis"),
        ("Age >= 65", "1", "available", "age"),
        ("Platelets >= 100", "-2 (model 1)", "available", "plt >= 100"),
        ("Day of illness (selects model)", "-", "available (missing in 14 -> model 1)", "days_fever"),
    ],
}


def score(df: pd.DataFrame) -> pd.Series:
    leuko = (df["wbc"] > 10).astype(int)
    old = (df["age"] >= 65).astype(int)
    model1 = 3 * leuko + 2 * gi_bleeding(df) + old - 2 * (df["plt"] >= 100).astype(int)
    model2 = 2 * leuko + old
    late = df["days_fever"] > 4
    return model1.where(~late, model2)
