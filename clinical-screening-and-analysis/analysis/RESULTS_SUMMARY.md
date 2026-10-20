# Dengue serobook: statistical analysis summary

Source: `fin dengue serobook - without missingness.xlsx` (4 population sheets + index).
Full tables: `analysis/output/dengue_results.xlsx` (one sheet per analysis, README sheet lists them).
Figures: `analysis/output/fig1_panel_roc.png`, `fig2_forest_icu_tx.png`, `fig3_age_groups.png`, `fig4_lymphocyte.png`.
Code (re-runnable): `analysis/dengue_analysis.py`.

---

## 1. Methods (draft text for the paper)

**Cohort.** 170 patients with serologically confirmed dengue: 77 adult males, 57 adult females, 22 paediatric males and 14 paediatric females (age 5–17 = paediatric; 18–81 = adult). Empty template rows were excluded. Absolute lymphocyte count (ALC) was recomputed as missing where WBC or lymphocyte % was blank, because the sheet formula returned 0 in those rows.

**Outcomes.** (i) Length of stay (LOS, days); (ii) composite of ICU admission (ICU-LOS > 0) or blood-product transfusion; (iii) WHO-2009 class: DWOS / DWS / DS, analysed as 3 groups and as binary DWS+DS vs DWOS and DS vs non-DS. For panel building LOS was also dichotomised at the cohort median (prolonged LOS > 3 days).

**Statistics.** Binary clinical features: Fisher's exact test with odds ratios (Woolf 95% CI; Haldane correction for zero cells), plus sensitivity, specificity, PPV and NPV. Continuous laboratory markers: Mann–Whitney U test, univariable logistic regression (OR per 1 SD), ROC AUC with bootstrap 95% CI (2,000 resamples) and Youden-optimal cut-off. LOS: Mann–Whitney U (binary) and Spearman's ρ (continuous). DWOS/DWS/DS: χ² (binary) or Kruskal–Wallis (continuous), with Spearman trend test across the ordered classes. P-values are also reported with Benjamini–Hochberg FDR correction (q).

**Combined panel.** Candidates were markers with ≤ 25% missing data (AST, ALT and bilirubin were excluded because about 50% of values are missing) and univariable p < 0.20. All 1–3 marker combinations (1–4 for DWS+DS and prolonged LOS, which have more events) were fitted by logistic regression and ranked by repeated 5-fold cross-validated AUC (40 repeats). Remaining missing values were median-imputed. The best panel is reported with three AUCs: apparent, bootstrap optimism-corrected, and nested cross-validated. Nested CV repeats the panel selection inside every fold, so it is the most honest estimate of how the panel would perform in new patients. A 4-item additive bedside score was also tested.

---

## 2. Key results

### 2a. Individual parameters (significant at p < 0.05; ✓ = survives FDR q < 0.05)

| Parameter | LOS | ICU / transfusion | DWOS→DWS→DS |
|---|---|---|---|
| **Bleeding** | +0.5 d, p = 0.068 | OR 21.4 (7.5–60.8), p < 0.001 ✓ | 1% → 39% → 61%, p < 0.001 ✓ |
| **USG abnormal** (ascites/HSM/effusion) | median 3.5 vs 3 d, p = 0.058 | OR 17.2 (6.2–48.0), p < 0.001 ✓ | 2% → 30% → 61%, p < 0.001 ✓ |
| **Oliguria** | ns | OR 3.0 (1.1–7.9), p = 0.037 | 3% → 50% → 22%, p < 0.001 ✓ |
| **Rash** | ns | OR 3.4 (1.3–8.8), p = 0.018 | 12% → 26% → 39%, p = 0.011 ✓ |
| Nausea/vomiting | ns | ns | less common in DS (OR 0.32 for DS, p = 0.035) |
| **Platelets** | ns | AUC 0.71, cut-off ≤ 68, p = 0.002 ✓ | 100 → 67 → 56, p < 0.001 ✓ |
| **PT** | ρ 0.14, p = 0.085 | AUC 0.84, cut-off ≥ 14 s, p < 0.001 ✓ | 13.2 → 13.8 → 14.2, p < 0.001 ✓ |
| **aPTT** | ns | AUC 0.80, cut-off ≥ 38 s, p < 0.001 ✓ | 32 → 34 → 38, p = 0.001 ✓ |
| **AST** (n = 85) | ns | AUC 0.73, ≥ 79 U/L, p = 0.006 ✓ | trend p = 0.037 |
| ALT (n = 85) | ns | AUC 0.70, ≥ 87 U/L, p = 0.017 | DS 88.5 vs 43.5, p = 0.026 |
| Indirect bilirubin (n = 80) | ns | p = 0.028 | trend p = 0.048 |
| **Days of fever at admission** | ns | p = 0.025 | 3 → 3 → 4 d, p < 0.001 ✓ |
| ALC | **ρ −0.17, p = 0.033** | ns | ns |
| Hct, WBC, lymphocyte %, age, arthralgia, lethargy, syncope, body ache, headache | ns | ns | ns |

**LOS** was weakly predicted by everything. Only lower ALC reached p < 0.05 (ρ = −0.17), and it did not survive FDR correction. Bleeding, abnormal USG and haematocrit were borderline (p 0.06–0.07). LOS in this cohort is tightly clustered (median 3 days, IQR 2–4), which limits power.

### 2b. Best combined panels (Fig 1; sheet 5_Panel_summary)

| Outcome (events/N) | Best panel | Cross-validated AUC | Nested-CV AUC | Sens / Spec at Youden cut-off | NPV |
|---|---|---|---|---|---|
| **ICU or transfusion** (23/170) | Bleeding + abnormal USG + days of fever | 0.91 | **0.84** | 91% / 80% | 98% |
| **Severe dengue** (18/170) | Abnormal USG + PT + days of fever | 0.90 | **0.77** | 94% / 76% | 99% |
| **DWS or DS** (64/170) | Bleeding + oliguria + abnormal USG + days of fever | 0.92 | **0.85** | 89% / 91% | 93% |
| Prolonged LOS > 3 d (55/170) | ALC + abnormal USG | 0.64 | 0.60 | 65% / 64% | 80% |

Adjusted ORs for the ICU/transfusion panel: bleeding 17.3 (5.0–59.9), abnormal USG 10.8 (3.1–37.6), each extra day of fever 1.58 (1.13–2.19).

Alternative panels perform almost identically (sheet 5b). For example, bleeding + aPTT + days of fever gives a cross-validated AUC of 0.91 for ICU/transfusion. A lab-heavy alternative is therefore available if USG is not done routinely.

**Combined panel vs best single marker.** For ICU/transfusion, PT alone has AUC 0.84 and bleeding alone has sensitivity 70% and specificity 90%. The panel raises sensitivity to 91% at 80% specificity. Its value is mainly as a rule-out tool (NPV 98–99%), not a rule-in tool (PPV 31–42%).

**Bedside score** (1 point each for platelets < 50, bleeding, abnormal USG, oliguria; AUC 0.85 for ICU/transfusion):

| Score | ≥ 1 | ≥ 2 | ≥ 3 |
|---|---|---|---|
| Sensitivity | 87% | **74%** | 35% |
| Specificity | 63% | **87%** | 98% |
| NPV | 97% | 96% | 91% |

### 2c. Lymphocyte % above vs below normal (20–40%) (Fig 4; sheets 6–6d)

| | Low < 20% (n = 49) | Normal (n = 78) | High > 40% (n = 34) |
|---|---|---|---|
| ICU/transfusion | **20%** | 12% | 12% |
| Severe dengue (DS) | **16%** | 9% | 6% |
| DWS or DS | 37% | 40% | 38% |
| LOS, mean | 3.35 d | 3.15 d | **2.79 d** |
| Platelets, median | 95 | 92.5 | **70** |

- **Low lymphocyte % vs normal:** ICU/transfusion OR 1.97 (0.74–5.25), p = 0.21. After adjusting for platelets and paediatric status: aOR 2.29 (0.83–6.29), p = 0.11. Severe dengue OR 1.98, p = 0.26.
- **High lymphocyte % vs normal:** ICU/transfusion OR 1.02, p = 1.0. Severe dengue OR 0.63, p = 0.72. LOS 0.36 days shorter, p = 0.23.
- **High vs low directly:** severe dengue OR 0.32 (0.06–1.61), p = 0.19. LOS 0.55 days shorter, p = 0.13.
- **Predictive value:** both directions have poor sensitivity (12–47%). Specificity is about 72–79% and NPV about 85–92%.

**Interpretation.** The two directions point in opposite ways but neither reaches significance. Lymphopenia (< 20%) trends towards more ICU/transfusion and severe dengue. Relative lymphocytosis (> 40%) trends towards milder disease and shorter stay, even though platelets are lower in that group. This fits the known course of dengue: lymphopenia reflects the febrile or early critical phase, while relative lymphocytosis with atypical lymphocytes typically appears in late critical or recovery phase. Neither is a usable stand-alone predictor in this sample. Roughly 3–4 times more patients would be needed to confirm the lymphopenia signal.

### 2d. Male vs female (sheets 7, 7b)

- **Outcomes did not differ:**
  - ICU/transfusion: 14.1% vs 12.7%, p = 0.82
  - DWS/DS: 37% vs 38%, p = 1.0
  - DS: 11% vs 10%
  - LOS: mean 3.03 vs 3.35 days, p = 0.074
  - Adjusted OR for male sex (ICU/transfusion): 1.18 (0.48–2.94)
- **Males differed on some measures:** higher haematocrit (39.3 vs 34%, p < 0.001, a physiological difference), higher AST (96 vs 61.5, p = 0.009) and ALT (50 vs 41.5, p = 0.023), and they presented slightly later (days of fever 3 [IQR 3–4] vs 3 [IQR 2–3.5], p = 0.017).
- **In children only, boys had more DWS/DS:** 50% vs 7%, p = 0.011. However, the paediatric-female sheet has only 14 completed rows out of 74 (many labs blank) and no ICU or transfusion events, so this should be reported cautiously or confirmed after data completion.

### 2e. Paediatric vs adult (sheets 8, 8b)

- **Outcomes were the same:** ICU/transfusion 13.9% vs 13.4%, DS 11% vs 10%, DWS/DS 33% vs 39%, LOS median 3 days in both. Adjusted OR for paediatric status (ICU/transfusion): 1.18 (0.40–3.50).
- **Children had a milder lab profile:**
  - Higher platelets: 116 vs 84, p = 0.017
  - Lower haematocrit: 33.9 vs 37.7, p = 0.024
  - Lower direct and indirect bilirubin: p = 0.001 and p = 0.009
  - Admitted earlier in the illness: fever 2 vs 3 days, p = 0.025

### 2f. Age groups (Fig 3; sheet 9)

| Age | n | ICU/transfusion | DS | DWS/DS | LOS > 3 d | Comorbidity |
|---|---|---|---|---|---|---|
| **5–12** | 12 | **0%** | **0%** | **17%** | 42% | 0% |
| 13–17 | 24 | 21% | 17% | 42% | 21% | 21% |
| 18–24 | 55 | 11% | 15% | 40% | 27% | 15% |
| **25–34** | 42 | **9.5%** | **4.8%** | 38% | 38% | 24% |
| 35–44 | 19 | 16% | 5% | 37% | 42% | 47% |
| 45–59 | 10 | **30%** | **20%** | 30% | 30% | 30% |
| ≥ 60 | 8 | 25% | 13% | **50%** | 38% | **75%** |

The best outcomes were seen at ages **5–12** and **25–34**. Severe outcomes were most frequent at **45–59 and ≥ 60** and, unexpectedly, in **adolescents (13–17)**. None of these differences is statistically significant (χ² p = 0.31–0.83), because the groups are small.

**Possible explanations to discuss:**
1. **5–12 years.** These children are most likely to have a primary infection, with no heterotypic antibodies and therefore little antibody-dependent enhancement. They were admitted earliest (median fever 2 days), had the highest platelets (123k) and had no comorbidities. Earlier admission also explains why LOS was not shorter: they are observed through the whole critical phase.
2. **25–34 years.** This group has peak immune competence and few comorbidities, and is likely to present early because of work and health-seeking behaviour.
3. **13–17 years (adolescents).** In endemic settings adolescents have had enough time to acquire a first infection, so a second infection with a different serotype (and ADE) becomes likely. Adolescent capillaries are also more permeable than adult capillaries. In this group, 25% had an abnormal USG (plasma leakage) and 25% had bleeding. The group is dominated by boys from the paediatric-male sheet.
4. **≥ 45 years.** Comorbidity rises sharply (30–75%), including diabetes, hypertension and CKD. Older patients also have immunosenescence and endothelial dysfunction, may be taking antiplatelet drugs, and presented later (median fever 4 days at 45–59). The trend towards more ICU/transfusion with comorbidity (aOR 2.38, p = 0.07) supports this.

Serostatus (primary vs secondary infection, NS1/IgM/IgG pattern) is not recorded in the sheet. If it is available, adding it would let you test explanations 1 and 3 directly.

---

## 3. Limitations to state in the paper

1. **Incorporation bias.** The WHO-2009 DWS/DS definitions (see the index sheet) include bleeding, ascites on USG, lethargy, platelet fall and liver enzymes. Their strong association with DWS/DS is therefore partly built into the definitions. The ICU/transfusion composite is the outcome least affected by this, so it should be the primary prognostic outcome. Even here, transfusion decisions are partly triggered by bleeding and low platelets.
2. **Few events.** There were 23 ICU/transfusion events and 18 DS cases. The panels are limited to 3 predictors (about 6–8 events per variable), so they are exploratory. Nested-CV AUCs (0.77–0.85) are the realistic estimates and need external validation.
3. **Missing data.**
   - AST, ALT and bilirubin are missing in about 50% of patients, and PT/aPTT in about 20%.
   - In the paediatric-male sheet, days of fever is missing for 14 of 22 patients.
   - The paediatric-female sheet is mostly incomplete.
   - Panel models used median imputation; multiple imputation is recommended for the final paper.
4. **LOS is clustered (2–4 days)**, which limits its ability to discriminate between patients.
5. **Multiple testing.** About 24 parameters were tested for each outcome, so report FDR q-values alongside raw p-values.

---

## 4. Add-on: WBC, absolute lymphocyte count and derived ratios

Full tables: `analysis/output/ratios_results.xlsx`. Code: `analysis/ratios_analysis.py`.

The sheet has no neutrophil count, so a neutrophil-to-lymphocyte ratio (NLR) cannot be calculated directly. The proxy used here is (WBC − ALC) / ALC. This equals (100 − L%) / L%, so it carries exactly the same information as lymphocyte % and does not count as an independent marker.

**Continuous markers (AUC; q = FDR-corrected across the ratio markers)**

| Marker | ICU / transfusion | Severe dengue | DWOS → DWS → DS | LOS |
|---|---|---|---|---|
| WBC | AUC 0.50, p = 1.0 | 0.52 | p = 0.41 | ρ −0.07 |
| ALC | 0.54, p = 0.59 | 0.55 | p = 0.80 | ρ −0.17, p = 0.033 (q 0.40) |
| Proxy NLR | 0.58 | 0.62, p = 0.09 | p = 0.10 | ρ 0.13 |
| Platelet / lymphocyte (PLR) | 0.56 | 0.59 | p = 0.11 | ns |
| Platelet / WBC | **0.65**, p = 0.018 (q 0.044) | **0.70**, p = 0.008 | p = 0.018 | ns |
| **Haematocrit / platelet** | **0.70**, p = 0.002 (q 0.008) | **0.73**, p = 0.002 | 0.36 → 0.60 → 0.69, p < 0.001 | ns |
| **APRI** (n = 83) | **0.81**, p < 0.001 (q 0.001) | 0.71, p = 0.019 | 2.1 → 5.1 → 7.9, p = 0.008 | ns |
| AST / ALT (De Ritis) | 0.53 | 0.57 | p = 0.26 | ns |
| Total bilirubin | 0.71, p = 0.016 | 0.63 | p = 0.15 | ns |
| **PT × aPTT index** | **0.85**, p < 0.001 (cut-off ≥ 1.28: 68% / 90%) | **0.80** | 1.00 → 1.13 → 1.28, p < 0.001 | ns |

**Clinical cut-offs**
- **ALC < 1.0 (lymphopenia)**
  - LOS > 3 days: 40% vs 21%, OR 2.40 (1.18–4.88), p = 0.017
  - Mean LOS: 3.34 vs 2.87 days, p = 0.019
  - No link with ICU/transfusion or WHO class.
- **ALC < 0.5:** severe dengue 20% vs 8%, OR 2.73, p = 0.093 (trend only).
- **WBC < 4 (leucopenia, 56% of patients)**
  - No association with any outcome.
  - Severe leucopenia (WBC < 2): LOS > 3 days 48% vs 28%, p = 0.063.
- **Hct / platelet > 0.5 (platelets below twice the haematocrit)**

  | Outcome | OR | p | NPV |
  |---|---|---|---|
  | ICU / transfusion | 3.47 | 0.011 | 92% |
  | Severe dengue | 3.48 | 0.036 | 94% |
  | DWS / DS | 2.87 | 0.002 | — |

- **APRI > 1**
  - No ICU/transfusion or severe dengue events below this cut-off (0 of 17 patients), giving NPV 100% and sensitivity 100%.
  - Specificity is only about 25%, and this rests on the 83 patients who had AST measured.

**Interpretation**
- WBC, ALC and the lymphocyte-based ratios (NLR, PLR) do not predict severity in this cohort.
- Low ALC is linked to a slightly longer stay. The clearest form of this is ALC < 1.0, which adds about 0.5 days.
- The ratios that work are the ones built on the strong single markers:
  - platelets (Hct/platelet, platelet/WBC);
  - coagulation (PT × aPTT);
  - liver injury relative to platelets (APRI).
- PT × aPTT (AUC 0.85) and APRI (AUC 0.81) match or slightly beat the best single lab marker, PT (0.84), for ICU/transfusion. The Hct/platelet ratio captures the WHO "rising haematocrit with falling platelets" warning sign in one number.

---

## 5. Profile of each WHO class (`analysis/output/dx_profile.xlsx`, code: `analysis/dx_profile.py`)

### 5a. ICU admission and transfusion within each class

| | DWOS (n = 106) | DWS (n = 46) | DS (n = 18) | p (trend) |
|---|---|---|---|---|
| ICU admission | 0% | 6.5% (3) | **38.9% (7)** | < 0.001 |
| Transfusion | 0.9% (1) | 17.8% (8) | **61.1% (11)** | < 0.001 |
| **ICU or transfusion** | 0.9% (1) | 19.6% (9) | **72.2% (13)** | < 0.001 |
| LOS > 3 days | 28.3% | 28.3% | 66.7% | 0.012 |
| LOS, median (IQR) | 3 (2–4) | 3 (3–4) | 4 (3–5) | 0.009 |

- **13 of the 18 severe dengue patients (72%) needed ICU care or transfusion.** The other 5 (28%) were managed on the ward without either.
- Seen the other way round, among the 23 patients who needed ICU or transfusion: 57% were DS, 39% were DWS and 4% (1 patient) were DWOS.

### 5b. % of patients with each significant marker, by class

| Marker | DWOS | DWS | DS | p (trend) |
|---|---|---|---|---|
| Oliguria | 2.9% | **50.0%** | 22.2% | < 0.001 |
| Bleeding (any) | 1.0% | 39.1% | **61.1%** | < 0.001 |
| Rash | 12.5% | 26.1% | 38.9% | 0.003 |
| USG abnormal (ascites / HSM / effusion) | 1.9% | 29.5% | **61.1%** | < 0.001 |
| Fever ≥ 4 days at admission | 26.3% | 36.6% | 62.5% | 0.005 |
| Platelets < 100 ×10³/µL | 49.0% | 66.7% | **94.1%** | < 0.001 |
| Platelets < 50 ×10³/µL | 18.0% | 28.9% | 41.2% | 0.021 |
| Platelets < 20 ×10³/µL | 1.0% | 8.9% | 11.8% | 0.009 |
| PT ≥ 14 s | 7.7% | 35.3% | **64.7%** | < 0.001 |
| aPTT ≥ 38 s | 18.4% | 33.3% | 53.3% | 0.002 |
| AST > 120 U/L (n = 85) | 25.0% | 44.4% | 50.0% | 0.045 |
| ALT > 120 U/L (n = 85) | 13.6% | 11.1% | 28.6% | 0.30 (ns) |
| Hct / platelet ratio > 0.5 | 34.0% | 55.6% | 70.6% | < 0.001 |
| APRI > 1 (n = 83) | 70.5% | 84.6% | **100%** | 0.014 |
| PT × aPTT index ≥ 1.28 | 8.0% | 30.3% | 46.7% | < 0.001 |
| Platelet / WBC ratio ≤ 16 | 24.0% | 31.1% | 58.8% | 0.007 |

Cut-off sources:
- WHO and standard clinical thresholds: platelets, AST/ALT at 3 × the upper limit of normal, and APRI > 1.
- ROC-optimal cut-offs from the ICU/transfusion analysis: PT, aPTT, PT × aPTT and platelet/WBC.
- Fever ≥ 4 days is the cohort's 75th centile.

ALT was significant as a continuous value (Kruskal–Wallis p = 0.026, driven by DS) but not at the 3 × ULN cut-off.

Oliguria was most common in DWS (50%), not DS (22%). It separates warning-sign dengue from DWOS but does not rise further in severe dengue. Bleeding, abnormal USG, platelets < 100k, prolonged PT and APRI > 1 all rise steadily from DWOS to DS.

**Marker burden** (count of oliguria, bleeding, rash, abnormal USG, fever ≥ 4 days and platelets < 100k):

| Markers carried | DWOS | DWS | DS |
|---|---|---|---|
| 0 | 38.7% | 2.2% | 0% |
| 1 | 38.7% | 15.2% | 11.1% |
| 2 | 17.9% | 41.3% | 11.1% |
| 3 | 4.7% | 28.3% | 27.8% |
| ≥ 4 | 0% | 13.0% | **50.0%** |

- 77% of DWOS patients had at most 1 of these markers.
- 78% of DS patients had 3 or more.

---

## 6. Derived panel vs established approaches (`analysis/output/panel_comparison.xlsx`, code: `analysis/panel_comparison.py`)

All approaches were scored on the same 170 patients.
- **Fixed rules** are applied as written, with no fitting.
- **Models** were refitted on this cohort and scored by repeated 5-fold cross-validation.
- The "Potts / Tanner-type" rows use the *variables* of those published blood-count models, not their published coefficients. They measure how much information those variables carry here; they are not exact external validations.

**Outcome: ICU admission or transfusion (23 / 170)**

| Approach | AUC (95% CI) | Sensitivity | Specificity | Missed events |
|---|---|---|---|---|
| **Derived panel: bleeding + USG + days of fever** | **0.90 (0.82–0.96)** (nested CV 0.84) | 91% | 80% | 2 |
| WHO-2009 class DWS or DS | 0.90 (0.83–0.95) | 96% | 71% | 1 |
| Alternative panel: bleeding + aPTT + days of fever | 0.90 (0.83–0.95) | 83% | 84% | 4 |
| PT + aPTT | 0.81 | 65% | 88% | 8 |
| Bedside score ≥ 2 | 0.80 | 74% | 87% | 6 |
| PT alone | 0.79 | 61% | 89% | 9 |
| Platelets ≤ 100 + USG leakage | 0.73 | 52% | 95% | 11 |
| WHO-1997 DHF criteria (platelets ≤ 100 + bleeding + leakage) | 0.71 | 43% | 98% | 13 |
| Platelets alone | 0.69 | 74% | 69% | 6 |
| Blood count + AST (Potts-type) | 0.63 | 57% | 69% | 10 |
| WBC + Hct + platelets (Potts / Tanner-type) | 0.60 | 87% | 42% | 3 |
| Platelets < 20 (transfusion trigger) | 0.58 | 17% | 98% | 19 |

**Interpretation**
1. **The derived panel matches WHO-2009 warning-sign triage; it does not beat it.** Both have AUC 0.90. The panel trades a little sensitivity (91% vs 96%) for higher specificity (80% vs 71%). In this cohort that means 19 fewer false alarms at the cost of 1 more missed patient.
2. **The panel's realistic AUC is 0.84 (nested CV), not 0.90.** It was selected on this data, which gives it an advantage over the fixed comparators.
3. **Its advantages are practical, not statistical.** It uses 3 objective items available at admission, whereas the WHO classification needs a full warning-sign assessment. Depending on how the `dx` column was assigned, the WHO class may also reflect the whole admission rather than the admission state.
4. **Clinical signs carry the information.** Blood-count-only models (Potts / Tanner-type variables) perform poorly here (AUC 0.60–0.63), and platelets alone reach only 0.69. Bleeding and ultrasound evidence of plasma leakage add the predictive value.
5. **Classic DHF criteria are highly specific (98%) but miss more than half of events (sensitivity 43%).**
6. **Platelets < 20k identified only 17% of patients who were transfused or admitted to ICU.**

**Caveats**
- These comparisons do not reproduce the published weights of external scores.
- When citing published performance of other tools in the discussion, take the figures from the original papers.

---

## 7. Recommended panel: Bleeding–USG–Fever score (0–5)

| Item (at admission) | Points |
|---|---|
| Any bleeding | 2 |
| Plasma leakage on USG (ascites / pleural effusion / hepatosplenomegaly) | 2 |
| Fever for ≥ 4 days | 1 |

The weights follow the adjusted ORs of the fitted panel: bleeding 17.3, USG 10.8, about 1.6 per fever day.

| Risk tier | Score | Patients | ICU or transfusion | Severe dengue | DWS or DS |
|---|---|---|---|---|---|
| Low | 0 | 92 | **1 (1%)** | 1 (1%) | 15% |
| Intermediate | 1 | 36 | 3 (8%) | 3 (8%) | 31% |
| High | 2–3 | 28 | 8 (29%) | 6 (21%) | 89% |
| Very high | 4–5 | 14 | **11 (79%)** | 8 (57%) | 100% |

- **Discrimination:** AUC 0.90 for ICU/transfusion, the same as the full regression panel, and 0.88 for severe dengue.
- **At score ≥ 2:** sensitivity 83%, specificity 84%, NPV 97%.
- **Rule-out:** a score of 0 identifies over half the cohort (54%) at 1% risk.
- **Items tested and not added:**
  - PT ≥ 14 s (AUC 0.89 with it; no gain).
  - Platelets, which carry no information beyond bleeding and USG.
- **Validation still needed:** the score was derived and tested in the same cohort (realistic AUC about 0.84) and must be validated externally.

**Where ultrasound is unavailable:** give the 2 USG points for **aPTT ≥ 36 s** instead.

| Coagulation item (2 points) | ICU or transfusion AUC | Sensitivity / specificity at score ≥ 2 | Severe dengue AUC | Sensitivity / specificity at score ≥ 2 |
|---|---|---|---|---|
| USG leakage (reference) | 0.90 | 83% / 84% | 0.88 | 78% / 82% |
| **aPTT ≥ 36 s** | 0.86 | 83% / 71% | 0.87 | 89% / 70% |
| PT ≥ 14 s | 0.86 | 74% / 82% | 0.84 | 67% / 80% |
| PT ≥ 14 s or aPTT ≥ 36 s | 0.87 | 83% / 69% | 0.90 | 89% / 68% |
| PT ≥ 14 s and aPTT ≥ 36 s | 0.87 | 74% / 85% | 0.82 | 67% / 82% |

- **What the cut-offs mean:** they sit just above the 75th centile of DWOS patients in this cohort (aPTT 35 s, PT 13.6 s), i.e. at or just above the usual upper limit of normal. Each laboratory's own reference range should be checked.
- **Cost of the substitution:** aPTT keeps sensitivity at the USG level but loses about 13 points of specificity.
- **Missing values:** aPTT was missing in 35 patients, who were scored 0 for this item.

---

## 8. Comparison with published severity scores (`analysis/output/score_comparison.xlsx`; code: `analysis/compare_scores.py`; one file per score in `analysis/scores/`)

**Items available in this dataset:**
- **Computable exactly:** Adult SD score (Lee 2016) and ICU model (Peres 2025).
- **Missing blood pressure / hypotension, scored 0:** DISS and the Paediatric score.
- **Partial only:** SOFA and mSOFA have 3 of 6 organ systems (no SpO2/PaO2, MAP/pulse pressure, GCS or creatinine). APACHE II has 2 of 12 physiology variables plus age and chronic health.

**ICU or transfusion, all patients (23 / 170)**

| Score | AUC (95% CI) | p vs ours (DeLong) | Published cut-off | Sensitivity / specificity |
|---|---|---|---|---|
| **Bleeding-USG-Fever (this study)** | **0.90 (0.82–0.95)** (realistic ~0.84) | – | ≥ 2 | 83% / 84% |
| Paediatric complicated dengue (Bhaskar 2022) | 0.73 (0.61–0.83) | 0.002 | ≥ 2 | 52% / 85% |
| DISS (Pongpan) | 0.71 (0.58–0.84) | 0.002 | ≥ 2.5 | 57% / 80% |
| mSOFA, partial | 0.70 (0.58–0.82) | 0.002 | ≥ 2 | 87% / 43% |
| SOFA, partial | 0.70 (0.59–0.81) | 0.002 | ≥ 2 | 87% / 43% |
| Adult SD score (Lee 2016) | 0.67 (0.57–0.76) | < 0.001 | ≥ 1 | 17% / 95% |
| APACHE II, partial | 0.63 (0.52–0.74) | < 0.001 | – | – |
| ICU model (Peres 2025) | 0.57 (0.45–0.70) | < 0.001 | – | – |

**Severe dengue (DS):** ours 0.88. The others range from 0.55 to 0.73 (Paediatric 0.73, Lee 0.72, DISS 0.71, partial SOFA/mSOFA 0.68); all p < 0.05.

**DWS or DS:**
- Ours 0.83 vs partial SOFA/mSOFA 0.79 (p = 0.38, not different).
- All other scores ≤ 0.63.

**Subgroups:**
- **Adults:** same ranking.
- **Children (5 ICU/transfusion events):** ours 0.97 vs 0.60–0.75. Most differences are not significant because there are so few events.

---

## 9. Distributions, normality and bimodality (`figures/fig07`, `fig08`, `fig17`; `figures/normality_tests.csv`)

**Normality**
- Every continuous variable except haematocrit departs from a normal distribution (Shapiro–Wilk p < 0.001). Haematocrit is normal (p = 0.54).
- Most variables are right-skewed: WBC, APRI, AST/ALT, platelets and days of fever have skew 1.7–3.4.
- This justifies the median (IQR) summaries and non-parametric tests used throughout.

**Bimodality** (Hartigan dip test, repeated with jitter to break ties from rounded reporting)
- **aPTT is genuinely bimodal** (p = 0.005), with peaks at about 28 s and about 34–35 s.
- Sample order shows no drift, so there is no sign of a reagent change.
- The upper peak (> 30 s) rises with severity (trend p = 0.006):

  | Class | In upper peak |
  |---|---|
  | DWOS | 56% |
  | DWS | 76% |
  | DS | 87% |

- **Interpretation:** a normal-clotting group and a mildly prolonged-clotting group.
- PT, PT × aPTT and LOS looked bimodal only because of tied, rounded values; none is bimodal once ties are broken.

## 10. Consensus clustering and t-SNE (`analysis/clustering.py`; `output/clustering.xlsx`; `figures/figC1–C3`, `figT1–T3`)

**Method**
- **Inputs:** 20 admission features, 12 binary and 8 numeric. Outcomes and WHO class were excluded.
- **Resampling:** Monti consensus clustering, 500 resamples of 80% of patients, k = 2–6.
- **Algorithms:** k-means on standardised data, and Gower distance with hierarchical clustering.
- **Selection:** PAC plus a minimum cluster size of 10. Gower/hierarchical clustering only split off tiny outlier groups (2–11 patients), so it was not used.

**Selected solution:** k-means, k = 2 (PAC 0.23, silhouette 0.15).

| | Cluster 1 (n = 128) | Cluster 2 (n = 42) | p |
|---|---|---|---|
| Bleeding | 2% | 64% | < 0.001 |
| USG plasma leakage | 2% | 59% | < 0.001 |
| Oliguria | 10% | 43% | < 0.001 |
| Rash | 10% | 45% | < 0.001 |
| Platelets, median | 98k | 59k | < 0.001 |
| PT, median | 13.2 s | 14.2 s | < 0.001 |
| aPTT, median | 32 s | 38 s | < 0.001 |
| *Outcomes (not used for clustering)* | | | |
| **ICU or transfusion** | **3%** | **45%** | < 0.001 |
| Severe dengue | 3% | 33% | < 0.001 |
| DWS or DS | 21% | 88% | < 0.001 |
| LOS, median | 3 d | 3 d (IQR 3–4) | 0.011 |

Age, sex, days of fever, WBC, haematocrit and lymphocyte % do not differ between the clusters.

**Interpretation**
- Unsupervised clustering, with no outcome information, recovers a "leak-and-bleed / coagulopathy" phenotype that carries 83% of ICU/transfusion events (19 of 23).
- Its defining features are the same ones in the Bleeding–USG–Fever score plus PT/aPTT. This is independent, data-driven support for the score's choice of items.
- **Caveat:** stability is moderate (PAC 0.23; < 0.1 would be strong) and the silhouette is low. The t-SNE (figT1, no colouring) shows one continuous cloud with no visibly separate islands.
- The two clusters are best described as the two ends of a **severity continuum**, not as distinct disease subtypes.

---

## 11. Internal validation of the whole development procedure (`analysis/internal_validation.py`)

**Method:** 200 bootstrap samples. Each one repeats the full development procedure:
- screening (p < 0.20);
- exhaustive search of 1–3 predictor models;
- selection of the best model.

Each selected model is then tested on the original data (Harrell / Steyerberg method; recommended over split-sample or single-run cross-validation for small datasets).

| Measure | Value |
|---|---|
| Apparent AUC of the selected model | 0.92 |
| Optimism (bootstrap) | 0.043 |
| **Optimism-corrected AUC (primary estimate)** | **0.88** |
| Nested 5-fold CV AUC (sensitivity analysis; pessimistic because each fold trains on 80% of an already small sample) | 0.84 |
| Bootstrap calibration slope (shrinkage factor) | 0.80 |

**Interpretation**
- **Discrimination** is good and holds up after accounting for the selection procedure.
- **Calibration slope 0.80:** the logistic model's predicted *probabilities* are about 20% too extreme. If a probability-based model is reported, its coefficients should be multiplied by 0.80 (uniform shrinkage).
- **The points score is unaffected.** Shrinkage changes every coefficient by the same factor, so the ranking of patients, the AUC and the 2/2/1 risk tiers all stay the same. The tiers report observed risks, not model predictions.

---

## 12. Secondary / exploratory analyses (`analysis/secondary_analyses.py`; `output/secondary_analyses.xlsx`)

**Abnormal USG**
- Of the 26 abnormal scans, 13 showed fluid (ascites or pleural effusion) and 13 showed organomegaly only.
- Both groups had the same 54% rate of ICU or transfusion.
- The score item should therefore be named "abnormal USG (fluid or organomegaly)".

**Patients with neither bleeding nor abnormal USG:** 128 patients, of whom 4 (3%) needed ICU or transfusion.

**Lab-only version (bleeding and USG excluded)**

| Validation | AUC |
|---|---|
| Apparent (PT + aPTT + fever) | 0.87 |
| **Bootstrap optimism-corrected** | **0.80** |

- Calibration slope 0.73.
- The selected panel was chosen in only 14 of 200 bootstrap samples, so this version is less stable than the clinical one.
- Compare with the clinical version: corrected AUC 0.88, calibration slope 0.80.

**Added value**
- Adding aPTT to bleeding + USG + fever improves the model (likelihood-ratio test p = 0.008; apparent AUC 0.92 → 0.94).
- Platelets add nothing (p = 0.65).

**Head-to-head on the same 135 patients (DeLong test)**

| Outcome | Clinical model | Lab model | WHO-2009 |
|---|---|---|---|
| ICU or transfusion | 0.89 | 0.89 | 0.90 (no significant differences) |
| ICU alone | 0.69–0.81 | 0.80 | 0.95 (circular: WHO "severe" includes shock) |
| Transfusion alone | 0.91 | 0.88 | 0.88 |
| Severe dengue | 0.86–0.88 | 0.83–0.85 | – |
