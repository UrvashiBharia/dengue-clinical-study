"""Small helpers shared by the score modules (text parsing of free-text columns)."""
import pandas as pd

# Free-text comorbidity entries that are acute events or non-chronic findings, not comorbidities
NON_CHRONIC = {"repeat admit", "enteric fever", "uti", "scabies", "lactational amenorrhea", "abortion",
               "op poisoning", "endocervical polyp", "0", ""}


def text(series):
    return series.astype("string").str.lower().fillna("")


def has_any(series, keywords):
    t = text(series)
    return t.apply(lambda s: any(k in s for k in keywords)).astype(int)


def comorbidity_list(series):
    """Split the free-text comorbidity cell into a list of chronic conditions."""
    def parse(v):
        if pd.isna(v):
            return []
        items = [x.strip().lower() for x in str(v).split(",")]
        return [x for x in items if x not in NON_CHRONIC]
    return series.apply(parse)


def hepatomegaly(df):
    """Hepatomegaly from the ultrasound free-text column (includes 'hepatosplenomegaly')."""
    return has_any(df["usg"], ["hepatomegaly", "hepatosplenomegaly"])


def gi_bleeding(df):
    """Gastrointestinal bleeding from the bleeding-type free text (melena, PR bleed, haematemesis)."""
    return has_any(df["bleeding_type"], ["melena", "pr bleed", "pr  bleed", "per rectum", "hematemesis"])
