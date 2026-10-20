"""CLASSIC SOFA SCORE  -- PARTIAL (only 3 of 6 organ systems measurable)
Vincent JL, Moreno R, Takala J, et al. The SOFA (Sepsis-related Organ Failure Assessment) score to
describe organ dysfunction/failure. Intensive Care Med 1996;22:707-710.

System          0              1             2              3                      4
Respiratory     PaO2/FiO2>=400 <400          <300           <200 with resp support <100 with resp support
Coagulation     plt >=150k     <150k         <100k          <50k                   <20k
Liver           bili <1.2      1.2-1.9       2.0-5.9        6.0-11.9               >=12.0 mg/dL
Cardiovascular  MAP>=70        MAP<70        dopa<=5/dobutamine  dopa>5 / epi or NE <=0.1  dopa>15 / epi or NE >0.1
CNS             GCS 15         13-14         10-12          6-9                    <6
Renal           creat <1.2     1.2-1.9       2.0-3.4        3.5-4.9 or UO<500      >=5.0 or UO<200

Mapping in this dataset: identical to the partial mSOFA because the components that differ between
SOFA and mSOFA (respiratory ratio, pulse pressure) are both unavailable. Coagulation + liver + renal
(oliguria proxy) only. Kept as a separate module so each scale is labelled and auditable on its own.
"""
import pandas as pd

from .msofa_dengue_partial import coagulation_points, liver_points, renal_points

INFO = {
    "name": "Classic SOFA (Vincent 1996) - PARTIAL",
    "short": "SOFA (partial)",
    "citation": "Vincent JL et al. Intensive Care Med 1996;22:707-710",
    "population": "General ICU",
    "original_outcome": "Organ dysfunction / mortality",
    "cutoff": 2,
    "cutoff_label": ">= 2 (Sepsis-3 threshold)",
    "items": [
        ("Respiratory PaO2/FiO2", "0-4", "NOT AVAILABLE -> 0", "-"),
        ("Coagulation (platelets)", "0-4", "available", "plt"),
        ("Liver (bilirubin)", "0-4", "partial (n~80)", "br_dir + br_indir"),
        ("Cardiovascular MAP / vasopressors", "0-4", "NOT AVAILABLE -> 0", "-"),
        ("CNS (GCS)", "0-4", "NOT AVAILABLE -> 0", "-"),
        ("Renal (creatinine / urine output)", "0-4", "proxy: oliguria -> 3", "oliguria"),
    ],
}


def score(df: pd.DataFrame) -> pd.Series:
    bili = df["br_dir"] + df["br_indir"]
    return coagulation_points(df["plt"]) + liver_points(bili) + renal_points(df["oliguria"])
