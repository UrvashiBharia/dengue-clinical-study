"""MODIFIED SOFA (mSOFA) SCORE FOR DENGUE  -- PARTIAL (only 3 of 6 organ systems measurable)
McBride A, Vuong NL, Van Hao N, et al. A modified Sequential Organ Failure Assessment score for
dengue: development, evaluation and proposal for use in clinical trials. BMC Infect Dis 2022;22:722.
Population: 124 adults with dengue shock, Vietnam. Outcome: ICU admission, organ support, death.
Published baseline-mSOFA AUC for ICU admission: 0.71 (0.59-0.83).

System          0              1             2              3                      4
Respiratory     SpO2/FiO2>=400 <400          <315           <235 and/or resp supp  <150 with resp support
Coagulation     plt >=150k     <150k         <100k          <50k                   <20k
Liver           bili <1.2      1.2-1.9       2.0-5.9        6.0-11.9               >12.0 mg/dL
Cardiovascular  MAP>=70 & PP>=20  MAP<70 or PP<20  PP<10 or dopa<=5/dobutamine  dopa 5.1-15 / epi or NE <=0.1  dopa>15 / epi or NE >0.1
CNS             GCS 15         13-14         10-12          6-9                    <6
Renal           creat <1.2     1.2-1.9       2.0-3.4        3.5-4.9 or UO<500      >5.0 or UO<200
(Differs from classic SOFA by: SpO2/FiO2 instead of PaO2/FiO2, pulse pressure in cardiovascular.)

Mapping in this dataset:
  - Coagulation: available.
  - Liver: total bilirubin = direct + indirect (available in ~47%; missing -> 0).
  - Renal: creatinine NOT recorded; oliguria flag used as a proxy for UO < 500 mL/d -> 3.
  - Respiratory, cardiovascular (MAP / pulse pressure), CNS: NOT recorded -> 0.
  So this is a coagulation + liver + renal sub-score, NOT the validated mSOFA.
"""
import numpy as np
import pandas as pd

INFO = {
    "name": "Modified SOFA for dengue (McBride 2022) - PARTIAL",
    "short": "mSOFA (partial)",
    "citation": "McBride A et al. BMC Infect Dis 2022;22:722",
    "population": "Vietnamese adults with dengue shock",
    "original_outcome": "ICU admission, organ support, death",
    "cutoff": 2,
    "cutoff_label": ">= 2 (conventional SOFA threshold)",
    "items": [
        ("Respiratory SpO2/FiO2", "0-4", "NOT AVAILABLE -> 0", "-"),
        ("Coagulation (platelets)", "0-4", "available", "plt"),
        ("Liver (bilirubin)", "0-4", "partial (n~80)", "br_dir + br_indir"),
        ("Cardiovascular MAP / pulse pressure", "0-4", "NOT AVAILABLE -> 0", "-"),
        ("CNS (GCS)", "0-4", "NOT AVAILABLE -> 0", "-"),
        ("Renal (creatinine / urine output)", "0-4", "proxy: oliguria -> 3", "oliguria"),
    ],
}


def coagulation_points(plt):
    return pd.Series(np.select([plt < 20, plt < 50, plt < 100, plt < 150], [4, 3, 2, 1], 0), index=plt.index)


def liver_points(bili):
    return pd.Series(np.select([bili >= 12, bili >= 6, bili >= 2, bili >= 1.2], [4, 3, 2, 1], 0), index=bili.index)


def renal_points(oliguria):
    return 3 * oliguria.fillna(0)


def score(df: pd.DataFrame) -> pd.Series:
    bili = df["br_dir"] + df["br_indir"]
    return coagulation_points(df["plt"]) + liver_points(bili) + renal_points(df["oliguria"])
