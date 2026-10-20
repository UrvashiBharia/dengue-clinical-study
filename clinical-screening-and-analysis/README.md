# Clinical and laboratory predictors of severity in dengue

Statistical analysis code for a hospital cohort of 170 serologically confirmed dengue patients
(adult and paediatric, male and female). The analysis tests how admission clinical features and
laboratory markers relate to:

- length of hospital stay (LOS),
- ICU admission or blood-product transfusion,
- WHO-2009 class: dengue without warning signs (DWOS), with warning signs (DWS), severe dengue (DS).

It also builds combined predictive panels, compares lymphocyte % above and below normal, and
compares sex, paediatric vs adult, and age groups.

## Folder layout

```
clinical-screening-and-analysis/
├── README.md
├── requirements.txt
├── run_all.py                    # runs every analysis and figure script in order
└── analysis/
    ├── common.py                 # data loading, cleaning, derived variables, shared helpers
    ├── dengue_analysis.py        # main analysis (univariable tests, panels, lymphocyte, demographics, figures)
    ├── ratios_analysis.py        # WBC, absolute lymphocyte count and derived lab ratios
    ├── dx_profile.py             # outcomes and marker prevalence within DWOS / DWS / DS
    ├── panel_comparison.py       # derived panel vs WHO triage, DHF criteria and blood-count models
    ├── compare_scores.py         # our score vs published severity scores (DeLong AUC comparison)
    ├── internal_validation.py    # bootstrap of the whole score-development procedure (realistic AUC)
    ├── secondary_analyses.py     # USG breakdown, lab-only version, added value, head-to-head vs WHO
    ├── figures.py                # all publication figures (fig01-fig17) + normality table
    ├── clustering.py             # consensus clustering + t-SNE (figC1-C3, figT1-T3)
    ├── scores/                   # ONE FILE PER SCORE: items, points, citation, data mapping
    │   ├── bleeding_usg_fever.py     # Bleeding-USG-Fever score (this study)
    │   ├── diss_pongpan.py           # Dengue Infection Severity Score (Pongpan 2013/2014)
    │   ├── adult_sd_lee2016.py       # Adult Severe Dengue Risk Score (Lee 2016)
    │   ├── icu_peres2025.py          # Dengue ICU complication risk model (Peres 2025)
    │   ├── msofa_dengue_partial.py   # Modified SOFA for dengue (McBride 2022) - partial
    │   ├── sofa_classic_partial.py   # Classic SOFA (Vincent 1996) - partial
    │   ├── apache2_partial.py        # APACHE II (Knaus 1985) - partial
    │   └── pediatric_bhaskar2022.py  # Pediatric complicated dengue score (Bhaskar 2022)
    ├── RESULTS_SUMMARY.md        # methods text, results and interpretation
    └── output/                   # generated tables (.xlsx) and figures (.png)
```

## Data

The scripts expect the source workbook `fin dengue serobook - without missingness.xlsx` in this
folder (`clinical-screening-and-analysis/`). It has one sheet per population (`adult_male`, `adult_female`, `ped_male`,
`ped_female`) and an `index` sheet with code lists and the WHO-2009 class definitions.
Expected columns: `age, comorbs, days_fever, oliguria, hct, plt, wbc,
lymphocyte_percent (20-40), absolute lymphocyte count (1-5), pt, ptt, ast, alt, br_dir, br_indir,
arthralgia, bleeding, rash, lethargy, nausea_vomiting, syncope, bodyache, headache, usg_binary,
los, icu_los, transfusion, dx`.

The data file is excluded by the repository `.gitignore`, as are per-patient outputs
(cluster assignments and individual case rows), which the scripts write to `analysis/output/private/`. Patient-level data should only be shared publicly if it is
fully de-identified and sharing is permitted by your ethics approval.

Cleaning rules (in `common.py`):
- Rows with no `dx` are empty template rows and are dropped.
- Absolute lymphocyte count is set to missing where WBC or lymphocyte % is blank (the sheet formula
  returns 0 there).
- ICU admission = `icu_los > 0`; composite outcome = ICU admission or transfusion.
- Prolonged LOS = more than 3 days (cohort median).

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python run_all.py
```

`dengue_analysis.py` takes about 10–15 minutes because of the cross-validated panel search; the other
scripts finish in under a minute. All random steps use fixed seeds, so results are reproducible.

## Outputs

| File | Contents |
|---|---|
| `output/dengue_results.xlsx` | Cohort description, every parameter vs each outcome (p, FDR q, OR, AUC, cut-off, sensitivity/specificity), combined panels, lymphocyte categories, sex / age comparisons. A README sheet lists every tab. |
| `output/ratios_results.xlsx` | WBC, absolute lymphocyte count, NLR proxy, PLR, platelet/WBC, Hct/platelet, AST/ALT, APRI, bilirubin, PT × aPTT. |
| `output/dx_profile.xlsx` | ICU/transfusion rate within each WHO class; % of patients with each significant marker in each class. |
| `output/panel_comparison.xlsx` | Derived panel vs established triage approaches on the same patients (AUC, sensitivity, specificity, missed events). |
| `output/score_comparison.xlsx` | Every published score vs ours: AUC, DeLong p, sensitivity/specificity at published cut-offs, by adults / children; item-availability table. |
| `output/secondary_analyses.xlsx` | USG fluid vs organomegaly, patients with neither sign, lab-only search, added value of aPTT, head-to-head vs WHO (DeLong), bootstrap of the lab-only search. |
| `output/figures/` | 300-dpi publication figures fig01–fig17, clustering figC1–C3, t-SNE figT1–T3, `normality_tests.csv`. |
| `output/clustering.xlsx` | Consensus-clustering model selection (PAC, silhouette), cluster profiles, outcomes by cluster, per-patient assignments and t-SNE coordinates. |
| `output/fig1–fig4*.png` | Panel ROC curves, forest plot, outcomes by age group, lymphocyte categories. |

## Statistical methods (summary)

Fisher's exact test, χ², Cochran–Armitage trend test, Mann–Whitney U, Kruskal–Wallis and Spearman
correlation, with Benjamini–Hochberg FDR correction. ROC AUCs have bootstrap 95% CIs and Youden-optimal
cut-offs. Combined panels are logistic regression models selected by repeated 5-fold cross-validated AUC.
They are reported with bootstrap optimism-corrected AUC and nested cross-validated AUC (selection
repeated inside every fold). Full details are in `analysis/RESULTS_SUMMARY.md`.
