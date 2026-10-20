"""One module per severity score. Each module exposes:

    INFO   : dict with name, citation, original population/outcome, published cut-off and a
             per-item table showing what was available in this dataset and how it was mapped.
    score(df) -> pd.Series   : the score for every patient (higher = more severe).

Items that cannot be measured in this dataset are scored 0 ("assumed normal"), which is the usual
convention for missing physiology in SOFA/APACHE. INFO["items"] states which items those are, so a
partial score is never mistaken for the validated full score.
"""
from . import (adult_sd_lee2016, apache2_partial, bleeding_usg_fever, diss_pongpan, icu_peres2025,
               msofa_dengue_partial, pediatric_bhaskar2022, sofa_classic_partial)

ALL_SCORES = [
    bleeding_usg_fever,
    diss_pongpan,
    adult_sd_lee2016,
    icu_peres2025,
    msofa_dengue_partial,
    sofa_classic_partial,
    apache2_partial,
    pediatric_bhaskar2022,
]
