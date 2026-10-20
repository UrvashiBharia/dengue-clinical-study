"""Publication figures for every relevant finding.

Run:  python analysis/figures.py   ->  analysis/output/figures/fig01...fig16 (.png, 300 dpi)

Colour conventions (reference palette, validated slots):
  blue  #2a78d6  = significant / primary series      grey #b9b8b2 = not significant
  WHO class (ordered): DWOS #86b6ef, DWS #2a78d6, DS #104281 (one-hue ordinal ramp)
  second / third series: orange #eb6834, aqua #1baf7a
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests

from common import CLINICAL, DX_ORDER, LABEL, OUT, load_data
from scores import ALL_SCORES

warnings.filterwarnings("ignore")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
df = load_data()
for mod in ALL_SCORES:
    df[mod.INFO["short"]] = mod.score(df).astype(float)

# --------------------------------------------------------------------- style
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, GREY_LIGHT = "#b9b8b2", "#e8e7e3"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
DX_COL = {"DWOS": "#86b6ef", "DWS": "#2a78d6", "DS": "#104281"}
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.labelcolor": INK2, "axes.edgecolor": GREY, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GREY_LIGHT,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False, "legend.fontsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight",
    "text.color": INK,
})

OUTCOMES = [("icu_or_tx", "ICU or transfusion"), ("severe_ds", "Severe dengue (DS)"), ("dws_or_ds", "DWS or DS")]
LABS_EXT = {"plt": "Platelets (×10³/µL)", "pt": "PT (s)", "ptt": "aPTT (s)", "days_fever": "Days of fever",
            "ast": "AST (U/L)", "alt": "ALT (U/L)", "hct": "Haematocrit (%)", "wbc": "WBC (×10³/µL)",
            "lymph_pct": "Lymphocyte %", "alc": "Abs. lymphocyte count", "br_dir": "Direct bilirubin",
            "br_indir": "Indirect bilirubin", "hct_plt": "Hct / platelet ratio", "apri": "APRI",
            "pt_x_ptt": "PT × aPTT index", "plt_wbc": "Platelet / WBC ratio", "nlr_proxy": "Proxy NLR",
            "plr": "Platelet / lymphocyte ratio", "ast_alt": "AST / ALT ratio", "age": "Age (years)"}
CLIN_LABEL = {**{c: LABEL[c] for c in CLINICAL}, "usg_binary": "USG plasma leakage"}


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def fmt_p(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, max(0, c - h), min(1, c + h)


def cochran_armitage(flag, order):
    tab = pd.crosstab(order, flag).reindex(columns=[0, 1], fill_value=0)
    n_i, x_i = tab.sum(axis=1).values.astype(float), tab[1].values.astype(float)
    s = np.asarray(tab.index, dtype=float)
    N, pb = n_i.sum(), x_i.sum() / n_i.sum()
    t = np.sum(s * (x_i - n_i * pb))
    var = pb * (1 - pb) * (np.sum(n_i * s ** 2) - np.sum(n_i * s) ** 2 / N)
    return 2 * stats.norm.sf(abs(t / np.sqrt(var))) if var > 0 else np.nan


def boot_auc_ci(y, s, n=1000):
    b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], s[i]))
    return np.percentile(b, [2.5, 97.5])


def save(fig, name, note=None):
    if note:
        fig.text(0.0, -0.045, note, fontsize=7, color=INK3, ha="left", va="top", wrap=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("saved", name)


# =====================================================================================================
# fig01 - clinical presentation by WHO class
clin = ["bleeding", "usg_binary", "oliguria", "rash", "arthralgia", "nausea_vomiting", "lethargy",
        "syncope", "bodyache", "headache"]
rows = []
for v in clin:
    m = df[v].notna()
    pct = [100 * df.loc[m & (df.dx == k), v].mean() for k in DX_ORDER]
    rows.append((v, pct, cochran_armitage(df.loc[m, v].astype(int), df.loc[m, "dx_ord"]),
                 stats.chi2_contingency(pd.crosstab(df.loc[m, v], df.loc[m, "dx"]))[1]))
rows.sort(key=lambda r: r[3])
fig, ax = plt.subplots(figsize=(7.2, 5.6))
h = 0.26
for i, (v, pct, pt, pc) in enumerate(rows[::-1]):
    for j, k in enumerate(DX_ORDER):
        ax.barh(i + (1 - j) * h, pct[j], height=h * 0.9, color=DX_COL[k], label=k if i == 0 else None)
    lab = f"{CLIN_LABEL[v]}  {star(pc)}"
    ax.text(-2, i, lab, ha="right", va="center", fontsize=8.5, color=INK if pc < 0.05 else INK3,
            fontweight="bold" if pc < 0.05 else "normal")
    ax.text(102, i, fmt_p(pc), va="center", fontsize=7.5, color=INK2 if pc < 0.05 else INK3)
ax.set_yticks([]); ax.set_xlim(0, 100); ax.set_xlabel("% of patients with the feature")
ax.grid(axis="y", visible=False)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ["DWOS (n=106)", "DWS (n=46)", "DS (n=18)"], loc="lower left", bbox_to_anchor=(0, 1.0),
          ncol=3, fontsize=8)
ax.set_title("Clinical presentation by WHO-2009 class", loc="left", pad=28)
save(fig, "fig01_clinical_by_who_class.png",
     "χ² test across the three classes; bold = p<0.05 (* p<0.05, ** p<0.01, *** p<0.001).")

# =====================================================================================================
# fig02 - lab markers by WHO class (box + points)
labs2 = [("plt", False), ("pt", False), ("ptt", False), ("days_fever", False), ("ast", True), ("alt", True),
         ("hct_plt", True), ("apri", True), ("pt_x_ptt", False)]
fig, axes = plt.subplots(3, 3, figsize=(9, 8.4))
for ax, (v, logy) in zip(axes.flat, labs2):
    groups = [df.loc[(df.dx == k) & df[v].notna(), v].values for k in DX_ORDER]
    p = stats.kruskal(*groups).pvalue
    bp = ax.boxplot(groups, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, linewidth=1.4), whiskerprops=dict(color=INK3, linewidth=0.8),
                    capprops=dict(color=INK3, linewidth=0.8), boxprops=dict(linewidth=0))
    for patch, k in zip(bp["boxes"], DX_ORDER):
        patch.set_facecolor(DX_COL[k]); patch.set_alpha(0.35)
    for j, (g, k) in enumerate(zip(groups, DX_ORDER)):
        ax.scatter(j + 1 + rng.uniform(-0.15, 0.15, len(g)), g, s=9, color=DX_COL[k], edgecolor="white",
                   linewidth=0.4, zorder=3)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels([f"{k}\nn={len(g)}" for k, g in zip(DX_ORDER, groups)], fontsize=7.5)
    ax.set_title(f"{LABS_EXT[v]}", loc="left", fontsize=9)
    ax.set_title(f"{fmt_p(p)} {star(p)}", loc="right", fontsize=8,
                 color=INK if p < 0.05 else INK3, fontweight="bold" if p < 0.05 else "normal")
    ax.grid(axis="x", visible=False)
fig.suptitle("Laboratory markers across WHO-2009 classes", x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "fig02_labs_by_who_class.png", "Box = median and IQR; points = individual patients. Kruskal–Wallis test. "
     "AST, ALT, Hct/platelet and APRI on log scale.")

# =====================================================================================================
# fig03 - correlation heatmap: every marker vs every outcome (Spearman)
markers = clin + ["comorbidity", "age", "days_fever", "hct", "plt", "wbc", "lymph_pct", "alc", "pt", "ptt", "ast",
                  "alt", "br_dir", "br_indir", "hct_plt", "apri", "pt_x_ptt", "plt_wbc", "nlr_proxy", "plr", "ast_alt"]
outs = [("los", "Length of stay"), ("icu_or_tx", "ICU or transfusion"), ("severe_ds", "Severe dengue"),
        ("dws_or_ds", "DWS or DS"), ("dx_ord", "WHO class (ordinal)")]
R = np.zeros((len(markers), len(outs))); P = np.ones_like(R)
for i, v in enumerate(markers):
    for j, (o, _) in enumerate(outs):
        m = df[v].notna() & df[o].notna()
        R[i, j], P[i, j] = stats.spearmanr(df.loc[m, v], df.loc[m, o])
Q = multipletests(P.ravel(), method="fdr_bh")[1].reshape(P.shape)
fig, ax = plt.subplots(figsize=(6.4, 10))
im = ax.imshow(R, cmap=DIVERGING, vmin=-0.6, vmax=0.6, aspect="auto")
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        mark = "●" if Q[i, j] < 0.05 else ("○" if P[i, j] < 0.05 else "")
        ax.text(j, i, f"{R[i, j]:.2f}{(' ' + mark) if mark else ''}", ha="center", va="center", fontsize=6.8,
                color=INK if abs(R[i, j]) < 0.4 else "white")
ax.set_xticks(range(len(outs))); ax.set_xticklabels([o[1] for o in outs], rotation=30, ha="right")
ax.set_yticks(range(len(markers)))
ax.set_yticklabels([CLIN_LABEL.get(v, LABS_EXT.get(v, LABEL.get(v, v))) for v in markers], fontsize=7.5)
ax.axhline(len(clin) + 0.5, color="white", linewidth=3)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.35, pad=0.02)
cb.set_label("Spearman ρ", color=INK2); cb.outline.set_visible(False)
ax.set_title("Correlation of each clinical feature and lab marker with each outcome", loc="left", pad=10)
save(fig, "fig03_correlation_heatmap.png",
     "Spearman rank correlation (for two binary variables this equals the phi coefficient). ● significant after "
     "Benjamini–Hochberg FDR (q<0.05); ○ p<0.05 but not after FDR. Red = higher value with worse outcome; "
     "blue = lower value with worse outcome. Rows above the white line are clinical features.")

# =====================================================================================================
# fig04 - forest plots: clinical features vs ICU/transfusion and severe dengue


def or_ci(x, y):
    m = x.notna() & y.notna()
    x, y = x[m].astype(int), y[m].astype(int)
    a = ((x == 1) & (y == 1)).sum(); b = ((x == 1) & (y == 0)).sum()
    c = ((x == 0) & (y == 1)).sum(); d = ((x == 0) & (y == 0)).sum()
    p = stats.fisher_exact([[a, b], [c, d]])[1]
    if min(a, b, c, d) == 0:
        a, b, c, d = a + .5, b + .5, c + .5, d + .5
    o = a * d / (b * c); se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return o, np.exp(np.log(o) - 1.96 * se), np.exp(np.log(o) + 1.96 * se), p


feat = clin + ["comorbidity"]
order = np.argsort([or_ci(df[v], df["icu_or_tx"])[0] for v in feat])   # sort both panels by ICU/tx OR
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), sharey=True)
for ax, (oc, name) in zip(axes, OUTCOMES[:2]):
    res = [or_ci(df[v], df[oc]) for v in feat]
    for yi, k in enumerate(order):
        o, lo, hi, p = res[k]
        col = BLUE if p < 0.05 else GREY
        ax.plot([lo, hi], [yi, yi], color=col, linewidth=2, solid_capstyle="round")
        ax.scatter(o, yi, s=40, color=col, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(160, yi, f"{o:.1f}  {fmt_p(p)}", va="center", fontsize=7.2, color=INK2 if p < 0.05 else INK3)
    ax.axvline(1, color=INK3, linewidth=0.8)
    ax.set_xscale("log"); ax.set_xlim(0.1, 120)
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 100])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(name, loc="left")
    ax.set_xlabel("Odds ratio (95% CI), log scale")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(feat))); axes[0].set_yticklabels([CLIN_LABEL.get(feat[k], LABEL.get(feat[k])) for k in order])
fig.suptitle("Clinical features as predictors of serious outcomes", x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout(w_pad=6)
save(fig, "fig04_forest_clinical.png", "Fisher's exact test. Blue = p<0.05; grey = not significant. "
     "Numbers at right: odds ratio and p-value.")

# =====================================================================================================
# fig05 - AUC of every continuous marker for each outcome (dot + CI)
cont = ["pt_x_ptt", "pt", "apri", "ptt", "plt", "ast", "hct_plt", "br_indir", "alt", "days_fever", "plt_wbc",
        "lymph_pct", "nlr_proxy", "plr", "alc", "hct", "wbc", "ast_alt", "age"]
fig, axes = plt.subplots(1, 3, figsize=(11, 6), sharey=True)
order = None
for ax, (oc, name) in zip(axes, OUTCOMES):
    res = []
    for v in cont:
        m = df[v].notna() & np.isfinite(df[v])
        y, x = df.loc[m, oc].values, df.loc[m, v].values
        a = roc_auc_score(y, x); sgn = 1 if a >= 0.5 else -1
        lo, hi = boot_auc_ci(y, sgn * x)
        p = stats.mannwhitneyu(x[y == 1], x[y == 0]).pvalue
        res.append((max(a, 1 - a), lo, hi, p, sgn))
    if order is None:
        order = np.argsort([r[0] for r in res])
    for yi, k in enumerate(order):
        a, lo, hi, p, sgn = res[k]
        col = BLUE if p < 0.05 else GREY
        ax.plot([lo, hi], [yi, yi], color=col, linewidth=2, solid_capstyle="round")
        ax.scatter(a, yi, s=36, color=col, edgecolor="white", linewidth=1.5, zorder=3,
                   marker="o" if sgn > 0 else "D")
    ax.axvline(0.5, color=INK3, linewidth=0.8)
    ax.set_xlim(0.3, 1.0); ax.set_title(name, loc="left"); ax.set_xlabel("AUC (95% CI)")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(cont))); axes[0].set_yticklabels([LABS_EXT[cont[k]] for k in order])
fig.suptitle("Discrimination (AUC) of each laboratory marker", x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "fig05_auc_lab_markers.png", "Blue = significant (Mann–Whitney p<0.05); grey = not significant. "
     "● higher value = higher risk; ◆ lower value = higher risk (AUC shown for the risk direction). "
     "Bootstrap 95% CI (1,000 resamples). 0.5 = no better than chance.")

# =====================================================================================================
# fig06 - ROC curves of the best lab markers for ICU/transfusion
top = [("pt_x_ptt", 1), ("pt", 1), ("apri", 1), ("ptt", 1), ("plt", -1), ("hct_plt", 1)]
fig, axes = plt.subplots(2, 3, figsize=(9, 6.2))
for ax, (v, sgn) in zip(axes.flat, top):
    m = df[v].notna()
    y, x = df.loc[m, "icu_or_tx"].values, sgn * df.loc[m, v].values
    fpr, tpr, thr = roc_curve(y, x); a = roc_auc_score(y, x); j = np.argmax(tpr - fpr)
    ax.plot([0, 1], [0, 1], color=GREY, linewidth=0.8)
    ax.plot(fpr, tpr, color=BLUE, linewidth=2)
    ax.scatter(fpr[j], tpr[j], s=40, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
    ax.annotate(f"cut-off {'≥' if sgn > 0 else '≤'} {sgn * thr[j]:.3g}\nsens {100 * tpr[j]:.0f}%, spec {100 * (1 - fpr[j]):.0f}%",
                (fpr[j], tpr[j]), xytext=(fpr[j] + 0.12, tpr[j] - 0.25), fontsize=7, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK3, linewidth=0.6))
    ax.set_title(f"{LABS_EXT[v]}   AUC {a:.2f}", loc="left", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("1 − specificity", fontsize=8); ax.set_ylabel("Sensitivity", fontsize=8)
fig.suptitle("ROC curves of the best laboratory predictors of ICU admission or transfusion", x=0.01, ha="left",
             fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "fig06_roc_lab_markers.png", "Dot = Youden-optimal cut-off. Grey diagonal = chance.")

# =====================================================================================================
# fig07 - distributions + normality (histogram + KDE, Shapiro-Wilk)
dist_vars = ["age", "days_fever", "los", "hct", "plt", "wbc", "lymph_pct", "alc", "pt", "ptt", "ast", "alt",
             "br_indir", "hct_plt", "apri", "pt_x_ptt"]
DIST_LABEL = {**LABS_EXT, "los": "Length of stay (days)"}
norm_rows = []
fig, axes = plt.subplots(4, 4, figsize=(11, 9.5))
for ax, v in zip(axes.flat, dist_vars):
    x = df[v].dropna().values
    W, p = stats.shapiro(x); sk = stats.skew(x)
    norm_rows.append({"Variable": DIST_LABEL[v], "n": len(x), "Mean": x.mean(), "SD": x.std(ddof=1),
                      "Median": np.median(x), "IQR": f"{np.percentile(x, 25):.3g}-{np.percentile(x, 75):.3g}",
                      "Skewness": sk, "Shapiro-Wilk W": W, "p": p, "Normal?": "yes" if p >= 0.05 else "no"})
    ax.hist(x, bins=18, color=BLUE, alpha=0.35, edgecolor="white", linewidth=1, density=True)
    if len(np.unique(x)) > 5:
        kde = stats.gaussian_kde(x); xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, kde(xs), color=BLUE, linewidth=1.8)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, x.mean(), x.std()), color=ORANGE, linewidth=1.2)
    ax.set_title(DIST_LABEL[v], loc="left", fontsize=8.5)
    ax.text(0.98, 0.96, f"n={len(x)}\nskew {sk:.1f}\nShapiro {fmt_p(p)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.8, color=INK if p >= 0.05 else INK2)
    ax.set_yticks([]); ax.grid(False)
fig.legend(handles=[plt.Line2D([], [], color=BLUE, lw=1.8, label="Observed distribution (KDE)"),
                    plt.Line2D([], [], color=ORANGE, lw=1.2, label="Normal curve with same mean and SD")],
           loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=2)
fig.suptitle("Distribution of continuous variables and test for normality", x=0.01, ha="left",
             fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "fig07_distributions_normality.png",
     "Shapiro–Wilk p<0.05 = distribution departs from normal; this is why median (IQR) and non-parametric tests "
     "(Mann–Whitney, Kruskal–Wallis, Spearman) are used throughout.")
normality = pd.DataFrame(norm_rows)

# fig08 - Q-Q plots of the same variables
fig, axes = plt.subplots(4, 4, figsize=(10, 9.5))
for ax, v in zip(axes.flat, dist_vars):
    x = np.sort(df[v].dropna().values)
    (osm, osr), (slope, inter, r) = stats.probplot(x, dist="norm")
    ax.scatter(osm, osr, s=8, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.plot(osm, slope * np.asarray(osm) + inter, color=ORANGE, linewidth=1.2)
    ax.set_title(DIST_LABEL[v], loc="left", fontsize=8.5)
    ax.tick_params(labelsize=6.5)
fig.supxlabel("Theoretical normal quantiles", color=INK2, fontsize=9)
fig.supylabel("Observed values", color=INK2, fontsize=9)
fig.suptitle("Normal Q–Q plots (points on the orange line = normally distributed)", x=0.01, ha="left",
             fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "fig08_qq_plots.png")

# =====================================================================================================
# fig09 - lymphocyte % categories: outcome rates with 95% CI
cats = ["Low (<20%)", "Normal (20-40%)", "High (>40%)"]
d = df[df.lymph_cat.notna()]
outs9 = [("icu_or_tx", "ICU or transfusion"), ("severe_ds", "Severe dengue"), ("dws_or_ds", "DWS or DS"),
         ("prolonged_los", "LOS > 3 days")]
fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
for ax, (oc, name) in zip(axes, outs9):
    p = stats.chi2_contingency(pd.crosstab(d.lymph_cat, d[oc]))[1]
    for j, c in enumerate(cats):
        g = d[d.lymph_cat == c][oc]
        r, lo, hi = wilson(g.sum(), len(g))
        col = BLUE if c != "Normal (20-40%)" else GREY
        ax.plot([j, j], [100 * lo, 100 * hi], color=col, linewidth=2, solid_capstyle="round")
        ax.scatter(j, 100 * r, s=50, color=col, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(j + 0.08, 100 * r, f"{100 * r:.0f}%", fontsize=7.5, color=INK2, va="center")
    ax.set_xticks(range(3)); ax.set_xticklabels(["<20%", "20–40%", ">40%"])
    ax.set_xlim(-0.5, 2.6); ax.set_title(name, loc="left", fontsize=9)
    ax.text(0.98, 0.97, fmt_p(p), transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK3)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("% of patients (95% CI)")
fig.suptitle("Outcomes by lymphocyte % (normal range 20–40%; n = 49 / 78 / 34)", x=0.01, ha="left",
             fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "fig09_lymphocyte_categories.png", "Wilson 95% CI; χ² test across the three categories. "
     "No difference reaches significance.")

# fig10 - absolute lymphocyte count < 1.0 and length of stay
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
d = df[df.alc.notna()]
grp = [("ALC ≥ 1.0", d[d.alc >= 1.0].los), ("ALC < 1.0", d[d.alc < 1.0].los)]
ax = axes[0]
for j, (nm, g) in enumerate(grp):
    vals, cnt = np.unique(g, return_counts=True)
    ax.barh(j + (vals - 4) * 0.0, 0, color="white")
    for v_, c_ in zip(vals, cnt):
        ax.scatter(v_, j, s=c_ * 9, color=BLUE if j == 1 else GREY, alpha=0.7, edgecolor="white", linewidth=1)
    ax.scatter(g.mean(), j, marker="|", s=400, color=INK, linewidth=2, zorder=4)
    ax.text(g.mean(), j + 0.28, f"mean {g.mean():.2f} d  (n={len(g)})", ha="center", fontsize=8, color=INK2)
ax.set_yticks([0, 1]); ax.set_yticklabels([g[0] for g in grp]); ax.set_xlim(0.5, 9.5); ax.set_ylim(-0.5, 1.6)
ax.set_xlabel("Length of stay (days); dot size = number of patients")
ax.set_title(f"Length of stay by lymphopenia  ({fmt_p(stats.mannwhitneyu(grp[1][1], grp[0][1]).pvalue)})",
             loc="left", fontsize=9)
ax.grid(axis="y", visible=False)
ax = axes[1]
for j, (nm, cond) in enumerate([("ALC ≥ 1.0", d.alc >= 1.0), ("ALC < 1.0", d.alc < 1.0)]):
    r, lo, hi = wilson(d[cond].prolonged_los.sum(), cond.sum())
    ax.bar(j, 100 * r, width=0.55, color=BLUE if j == 1 else GREY)
    ax.plot([j, j], [100 * lo, 100 * hi], color=INK, linewidth=1)
    ax.text(j, 100 * hi + 2, f"{100 * r:.0f}%", ha="center", fontsize=8, color=INK2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["ALC ≥ 1.0", "ALC < 1.0"]); ax.set_ylim(0, 60)
ax.set_ylabel("% with LOS > 3 days (95% CI)")
ax.set_title("Prolonged stay: OR 2.40, p=0.017", loc="left", fontsize=9); ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "fig10_alc_length_of_stay.png", "Not significant after correction for multiple testing; report as a signal.")

# =====================================================================================================
# fig11 - demographics: sex and adult vs paediatric (outcome rates with CI)
comparisons = [("Sex", [("Male", df.sex == "Male"), ("Female", df.sex == "Female")]),
               ("Age category", [("Adult", df.age_cat == "Adult"), ("Pediatric", df.age_cat == "Pediatric")])]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, (title, groups) in zip(axes, comparisons):
    for gi, (gname, cond) in enumerate(groups):
        col = BLUE if gi == 0 else ORANGE
        for j, (oc, oname) in enumerate(outs9):
            g = df[cond][oc]
            r, lo, hi = wilson(g.sum(), len(g))
            x = j + (gi - 0.5) * 0.28
            ax.plot([x, x], [100 * lo, 100 * hi], color=col, linewidth=2, solid_capstyle="round")
            ax.scatter(x, 100 * r, s=40, color=col, edgecolor="white", linewidth=1.5, zorder=3,
                       label=f"{gname} (n={cond.sum()})" if j == 0 else None)
    ticks = []
    for j, (oc, oname) in enumerate(outs9):
        a, b = df[groups[0][1]][oc], df[groups[1][1]][oc]
        p = stats.fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1]
        ticks.append(f"{oname}\n{fmt_p(p)}")
    ax.set_xticks(range(4)); ax.set_xticklabels(ticks, fontsize=8)
    ax.set_title(title, loc="left"); ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("% of patients (95% CI)")
fig.suptitle("Outcomes by sex and by adult vs paediatric (no significant differences)", x=0.01, ha="left",
             fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "fig11_sex_and_pediatric.png", "Wilson 95% CI; Fisher's exact test.")

# fig12 - age groups (small multiples, one colour)
AG = ["5-12", "13-17", "18-24", "25-34", "35-44", "45-59", ">=60"]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
for ax, (oc, name) in zip(axes, outs9):
    p = stats.chi2_contingency(pd.crosstab(df.age_group, df[oc]))[1]
    for j, g in enumerate(AG):
        s = df[df.age_group == g][oc]
        r, lo, hi = wilson(s.sum(), len(s))
        ax.plot([j, j], [100 * lo, 100 * hi], color=BLUE, linewidth=1.6, alpha=0.6, solid_capstyle="round")
        ax.scatter(j, 100 * r, s=34, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
    ax.set_xticks(range(len(AG))); ax.set_xticklabels([a.replace(">=", "≥") for a in AG], rotation=45, fontsize=7.5)
    ax.set_title(name, loc="left", fontsize=9)
    ax.text(0.98, 0.97, fmt_p(p), transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK3)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("% of patients (95% CI)")
counts = ", ".join(f"{g.replace('>=', '≥')}: {int((df.age_group == g).sum())}" for g in AG)
fig.suptitle("Outcomes by age group", x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "fig12_age_groups.png", f"Group sizes — {counts}. Wilson 95% CI; χ² test. Wide intervals: "
     "no age-group difference is significant.")

# =====================================================================================================
# fig13 - marker burden by WHO class (100% stacked)
flags = (df.oliguria.fillna(0) + df.bleeding.fillna(0) + df.rash.fillna(0) + df.usg_binary.fillna(0)
         + (df.days_fever >= 4).astype(int) + (df.plt < 100).astype(int)).clip(upper=4)
fig, ax = plt.subplots(figsize=(7, 2.8))
for i, k in enumerate(DX_ORDER):
    s = flags[df.dx == k]
    left = 0
    for lvl in range(5):
        w = 100 * (s == lvl).mean()
        ax.barh(i, w, left=left, color=RAMP5[lvl], edgecolor="white", linewidth=2, height=0.6,
                label=("≥4" if lvl == 4 else str(lvl)) if i == 0 else None)
        if w >= 8:
            ax.text(left + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=7.5,
                    color="white" if lvl >= 2 else INK)
        left += w
ax.set_yticks(range(3)); ax.set_yticklabels([f"{k} (n={(df.dx == k).sum()})" for k in DX_ORDER])
ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("% of patients"); ax.grid(False)
ax.legend(title="Markers carried:", ncol=6, loc="lower left", bbox_to_anchor=(0, 1.0), title_fontsize=8,
          alignment="left")
ax.set_title("Number of key markers carried, by WHO class", loc="left", pad=34)
save(fig, "fig13_marker_burden.png", "Markers: oliguria, bleeding, rash, USG plasma leakage, fever ≥4 days, "
     "platelets <100k.", )

# =====================================================================================================
# fig14 - Bleeding-USG-Fever score: risk tiers
tiers = [("0\nLow", [0]), ("1\nIntermediate", [1]), ("2–3\nHigh", [2, 3]), ("4–5\nVery high", [4, 5])]
sc = df["BUF (this study)"]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
for ax, (oc, name) in zip(axes, OUTCOMES[:2]):
    for j, (lab, pts) in enumerate(tiers):
        s = df[sc.isin(pts)][oc]
        r, lo, hi = wilson(s.sum(), len(s))
        ax.bar(j, 100 * r, width=0.6, color=RAMP5[[0, 1, 3, 4][j]])
        ax.plot([j, j], [100 * lo, 100 * hi], color=INK, linewidth=1)
        ax.text(j, 100 * hi + 3, f"{100 * r:.0f}%\n({int(s.sum())}/{len(s)})", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(4)); ax.set_xticklabels([t[0] for t in tiers], fontsize=8)
    ax.set_title(name, loc="left"); ax.set_ylim(0, 110); ax.grid(axis="x", visible=False)
    ax.set_xlabel("Bleeding–USG–Fever score")
axes[0].set_ylabel("% with outcome (95% CI)")
fig.suptitle("Bleeding–USG–Fever score: observed risk by tier", x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "fig14_score_risk_tiers.png", "Score: bleeding 2, USG plasma leakage 2, fever ≥4 days 1.")

# =====================================================================================================
# fig15 - our score vs published scores: AUC with CI
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
names = [m.INFO for m in ALL_SCORES]
order15 = None
for ax, (oc, name) in zip(axes, OUTCOMES[:2]):
    y = df[oc].values
    res = []
    for info in names:
        s = df[info["short"]].values
        a = roc_auc_score(y, s); lo, hi = boot_auc_ci(y, s)
        res.append((a, lo, hi))
    if order15 is None:
        order15 = np.argsort([r[0] for r in res])
    for yi, k in enumerate(order15):
        a, lo, hi = res[k]
        ours = names[k]["short"].startswith("BUF")
        partial = "PARTIAL" in names[k]["name"]
        col = BLUE if ours else (GREY if partial else INK3)
        ax.plot([lo, hi], [yi, yi], color=col, linewidth=2.2 if ours else 1.6, solid_capstyle="round")
        ax.scatter(a, yi, s=55 if ours else 38, color=col, edgecolor="white", linewidth=1.5, zorder=3,
                   marker="o" if not partial else "s")
        ax.text(1.005, yi, f"{a:.2f}", va="center", fontsize=7.5, color=INK if ours else INK2,
                fontweight="bold" if ours else "normal")
    ax.axvline(0.5, color=INK3, linewidth=0.8)
    ax.set_xlim(0.3, 1.05); ax.set_title(name, loc="left"); ax.set_xlabel("AUC (95% CI)")
    ax.grid(axis="y", visible=False)
labels15 = []
for k in order15:
    nm = names[k]["name"].replace(" - PARTIAL", " (partial)*")
    labels15.append(nm)
axes[0].set_yticks(range(len(names))); axes[0].set_yticklabels(labels15, fontsize=8)
fig.suptitle("Bleeding–USG–Fever score vs published severity scores (same 170 patients)", x=0.01, ha="left",
             fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "fig15_score_vs_published.png",
     "Blue = this study (derived in this cohort; realistic AUC ≈0.84). Squares / * = partial scores — organ systems "
     "or vital signs not recorded were scored 0, so these are not the validated scales. DISS and paediatric score "
     "lack blood pressure. Bootstrap 95% CI.")

# fig16 - ROC: our score vs WHO-2009 triage vs best published score
fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
series = [("BUF (this study)", "Bleeding–USG–Fever (this study)", BLUE),
          ("WHO", "WHO-2009 class (DWS/DS)", ORANGE),
          ("Pediatric (Bhaskar)", "Paediatric score (Bhaskar 2022)", AQUA)]
for ax, (oc, name) in zip(axes, OUTCOMES[:2]):
    y = df[oc].values
    ax.plot([0, 1], [0, 1], color=GREY, linewidth=0.8)
    for col_name, lab, colr in series:
        if col_name == "WHO" and oc == "severe_ds":
            continue
        s = df["dx_ord"].values if col_name == "WHO" else df[col_name].values
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=colr, linewidth=2, label=f"{lab}  AUC {roc_auc_score(y, s):.2f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity"); ax.set_title(name, loc="left")
    ax.legend(loc="lower right", fontsize=7)
fig.suptitle("ROC: Bleeding–USG–Fever score vs WHO-2009 triage and the best published score", x=0.01, ha="left",
             fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "fig16_roc_score_comparison.png", "WHO-2009 class is not shown for severe dengue because DS is part of that "
     "classification.")

# =====================================================================================================
normality.round(3).to_csv(FIG / "normality_tests.csv", index=False)
print(normality.round(3).to_string(index=False))

# =====================================================================================================
# fig17 - aPTT bimodality (Hartigan dip test with tie-breaking jitter) and its link to severity
import diptest  # noqa: E402

x = df.ptt.dropna().values
dip_p = np.median([diptest.diptest(x + rng.uniform(-0.5, 0.5, len(x)))[1] for _ in range(200)])
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]
ax.hist(x, bins=np.arange(23.5, 53.5, 1), color=BLUE, alpha=0.35, edgecolor="white", linewidth=1)
kde = stats.gaussian_kde(x, bw_method=0.25); xs = np.linspace(23, 53, 300)
ax.plot(xs, kde(xs) * len(x), color=BLUE, linewidth=2)
ax.axvline(30.5, color=INK3, linewidth=0.8)
ax.text(25, ax.get_ylim()[1] * 0.80, "low mode\n≤30 s", ha="center", fontsize=7.5, color=INK2)
ax.text(44, ax.get_ylim()[1] * 0.80, "high mode\n>30 s", ha="center", fontsize=7.5, color=INK2)
ax.set_xlabel("aPTT (s)"); ax.set_ylabel("Patients")
ax.set_title(f"aPTT is bimodal (Hartigan dip test p={dip_p:.3f})", loc="left", fontsize=9)
ax.grid(axis="x", visible=False)
ax = axes[1]
d = df[df.ptt.notna()]
for j, k in enumerate(DX_ORDER):
    g = d[d.dx == k]
    r, lo, hi = wilson((g.ptt > 30).sum(), len(g))
    ax.bar(j, 100 * r, width=0.6, color=DX_COL[k])
    ax.plot([j, j], [100 * lo, 100 * hi], color=INK, linewidth=1)
    ax.text(j, 100 * hi + 3, f"{100 * r:.0f}%\n({(g.ptt > 30).sum()}/{len(g)})", ha="center", fontsize=7.5, color=INK2)
p_tr = cochran_armitage((d.ptt > 30).astype(int), d.dx_ord)
ax.set_xticks(range(3)); ax.set_xticklabels(DX_ORDER); ax.set_ylim(0, 115)
ax.set_ylabel("% in high aPTT mode (95% CI)")
ax.set_title(f"High mode rises with severity ({fmt_p(p_tr)}, trend)", loc="left", fontsize=9)
ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "fig17_aptt_bimodality.png", "Dip test repeated 200 times with ±0.5 s jitter to break ties from "
     "whole-second reporting; median p shown. No drift in aPTT across sample order (no evidence of a reagent change). "
     "PT, PT×aPTT and LOS were not bimodal once ties were broken.")
