"""DENGUE ICU COMPLICATION RISK MODEL
Peres IT, Ranzani OT, Bastos LSL, et al. Clinical characteristics, complications and outcomes of
critically ill patients with dengue in Brazil, 2012-2024. Int J Infect Dis 2025;159:108023.
Population: 11,047 adult (>= 16 y) dengue ICU admissions, 253 Brazilian ICUs.
Outcome: new complication during ICU stay (non-invasive / invasive ventilation, vasopressor,
blood transfusion, renal replacement therapy). Published test-set AUC ~0.71.

Published logistic equation (Equation 1):
logit(p) = -3.35461
         + age 20-39 x 0.31409 + age 40-49 x 0.19208 + age 50-59 x 0.30071
         + age 60-69 x 0.51040 + age 70-79 x 0.55154 + age >= 80 x 0.97917      (ref < 20)
         + platelets 100-150k x 0.04042 + platelets 50-100k x (-0.29755)
         + platelets < 50k x 0.44430                                              (ref > 150k)
         + leukocytes 2.5-5.0k x 0.05018 + 5.0-7.0k x 0.52003 + > 7.0k x 0.88603  (ref < 2.5k)
         + comorbidities 1-2 x 0.53722 + >= 3 x 1.23915                           (ref 0)

Mapping in this dataset: all four inputs available. Comorbidity count = number of chronic conditions
in the free-text comorbidity column (acute/non-chronic entries excluded, see _helpers.NON_CHRONIC).
Note: derived in ICU patients for complications *within* ICU; here applied at ward admission.
"""
import numpy as np
import pandas as pd

from ._helpers import comorbidity_list

INFO = {
    "name": "Dengue ICU Complication Risk Model (Peres 2025)",
    "short": "ICU model (Peres)",
    "citation": "Peres IT et al. Int J Infect Dis 2025;159:108023",
    "population": "Brazilian adults >= 16 y already in ICU",
    "original_outcome": "Complication during ICU stay (ventilation, vasopressor, transfusion, RRT)",
    "cutoff": None,
    "cutoff_label": "probability (no published cut-off); Youden used",
    "items": [
        ("Age class", "coefficient", "available", "age"),
        ("Platelet class", "coefficient", "available", "plt"),
        ("Leukocyte class", "coefficient", "available", "wbc"),
        ("Number of comorbidities", "coefficient", "available", "comorbs free text"),
    ],
}


def score(df: pd.DataFrame) -> pd.Series:
    age, plt, wbc = df["age"], df["plt"], df["wbc"]
    n_com = comorbidity_list(df["comorbs"]).apply(len)
    lp = (-3.35461
          + np.select([age.between(20, 39.999), age.between(40, 49.999), age.between(50, 59.999),
                       age.between(60, 69.999), age.between(70, 79.999), age >= 80],
                      [0.31409, 0.19208, 0.30071, 0.51040, 0.55154, 0.97917], 0)
          + np.select([plt < 50, plt < 100, plt <= 150], [0.44430, -0.29755, 0.04042], 0)
          + np.select([wbc > 7, wbc > 5, wbc >= 2.5], [0.88603, 0.52003, 0.05018], 0)
          + np.select([n_com >= 3, n_com >= 1], [1.23915, 0.53722], 0))
    return pd.Series(1 / (1 + np.exp(-lp)), index=df.index)
