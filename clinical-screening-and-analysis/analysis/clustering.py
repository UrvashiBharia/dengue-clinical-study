"""Consensus clustering (Monti et al. 2003) and t-SNE of patients using admission features only.

Features (outcomes and WHO class deliberately EXCLUDED so clusters can be validated against them):
  binary : oliguria, arthralgia, bleeding, rash, lethargy, nausea/vomiting, syncope, body ache,
           headache, USG plasma leakage, any comorbidity, male sex
  numeric: age, days of fever, haematocrit, log platelets, log WBC, lymphocyte %, PT, aPTT
  (AST/ALT/bilirubin excluded: ~50% missing.) Remaining gaps median-imputed.

Consensus clustering: 500 resamples of 80% of patients, k = 2..6, two algorithms
  (a) k-means on standardised features, (b) average-linkage hierarchical clustering on Gower distance.
Model selection: PAC (proportion of ambiguous clustering, consensus in (0.1, 0.9); lower = more stable),
CDF delta-area, cluster sizes and silhouette. Final labels: average-linkage on 1 - consensus matrix.

Run:  python analysis/clustering.py  ->  analysis/output/clustering.xlsx + figures/figC*.png, figT*.png
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import OUT, autosize, fmt_p, load_data, med_iqr

warnings.filterwarnings("ignore")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
df = load_data()

BINARY = ["oliguria", "arthralgia", "bleeding", "rash", "lethargy", "nausea_vomiting", "syncope", "bodyache",
          "headache", "usg_binary", "comorbidity", "male"]
NUMERIC = ["age", "days_fever", "hct", "log_plt", "log_wbc", "lymph_pct", "pt", "ptt"]
NAMES = {"oliguria": "Oliguria", "arthralgia": "Arthralgia", "bleeding": "Bleeding", "rash": "Rash",
         "lethargy": "Lethargy", "nausea_vomiting": "Nausea/vomiting", "syncope": "Syncope", "bodyache": "Body ache",
         "headache": "Headache", "usg_binary": "USG plasma leakage", "comorbidity": "Any comorbidity",
         "male": "Male sex", "age": "Age", "days_fever": "Days of fever", "hct": "Haematocrit",
         "log_plt": "Platelets (log)", "log_wbc": "WBC (log)", "lymph_pct": "Lymphocyte %", "pt": "PT", "ptt": "aPTT"}
df["log_plt"] = np.log(df["plt"])
df["log_wbc"] = np.log(df["wbc"])
X = df[BINARY + NUMERIC].copy()
X = X.fillna(X.median())
Z = StandardScaler().fit_transform(X.values)

# --------------------------------------------------------------------- Gower distance (mixed data)
rngs = X[NUMERIC].max() - X[NUMERIC].min()
num = X[NUMERIC].values / rngs.values
binv = X[BINARY].values
D_num = np.abs(num[:, None, :] - num[None, :, :]).sum(axis=2)
D_bin = (binv[:, None, :] != binv[None, :, :]).sum(axis=2)
GOWER = (D_num + D_bin) / (len(NUMERIC) + len(BINARY))

N, K_RANGE, REPS, FRAC = len(df), range(2, 7), 500, 0.8


def cluster_once(idx, k, algo):
    if algo == "kmeans":
        return KMeans(n_clusters=k, n_init=5, random_state=int(rng.integers(1e9))).fit_predict(Z[idx])
    sub = GOWER[np.ix_(idx, idx)]
    return fcluster(linkage(squareform(sub, checks=False), "average"), k, "maxclust")


def consensus(k, algo):
    M = np.zeros((N, N)); I = np.zeros((N, N))
    for _ in range(REPS):
        idx = np.sort(rng.choice(N, int(FRAC * N), replace=False))
        lab = cluster_once(idx, k, algo)
        same = (lab[:, None] == lab[None, :]).astype(float)
        M[np.ix_(idx, idx)] += same
        I[np.ix_(idx, idx)] += 1
    C = np.divide(M, I, out=np.zeros_like(M), where=I > 0)
    np.fill_diagonal(C, 1)
    return C


def final_labels(C, k):
    L = linkage(squareform(1 - C, checks=False), "average")
    return fcluster(L, k, "maxclust"), leaves_list(L)


def cdf_area(C):
    v = np.sort(C[np.triu_indices(N, 1)])
    grid = np.linspace(0, 1, 101)
    cdf = np.searchsorted(v, grid, side="right") / len(v)
    return grid, cdf, np.trapezoid(cdf, grid)


results, cons = [], {}
for algo in ["kmeans", "gower_hclust"]:
    prev_area = None
    for k in K_RANGE:
        C = consensus(k, algo)
        lab, order = final_labels(C, k)
        grid, cdf, area = cdf_area(C)
        vals = C[np.triu_indices(N, 1)]
        pac = np.mean((vals > 0.1) & (vals < 0.9))
        sil = silhouette_score(GOWER, lab, metric="precomputed") if algo == "gower_hclust" \
            else silhouette_score(Z, lab)
        sizes = np.bincount(lab)[1:]
        delta = area if prev_area is None else (area - prev_area) / prev_area
        prev_area = area
        cons[(algo, k)] = dict(C=C, lab=lab, order=order, grid=grid, cdf=cdf)
        results.append({"Algorithm": algo, "k": k, "PAC (lower = more stable)": pac, "CDF area": area,
                        "Delta area": delta, "Silhouette": sil, "Cluster sizes": " / ".join(map(str, sorted(sizes, reverse=True))),
                        "Smallest cluster": int(sizes.min())})
        print(f"{algo:13} k={k}  PAC={pac:.3f}  silhouette={sil:.3f}  sizes={sorted(sizes, reverse=True)}")
sel = pd.DataFrame(results)

# choose: lowest PAC among solutions whose smallest cluster has >= 10 patients (avoid outlier clusters)
ok = sel[sel["Smallest cluster"] >= 10]
best = ok.sort_values("PAC (lower = more stable)").iloc[0]
ALGO, K = best["Algorithm"], int(best["k"])
print(f"\nSelected: {ALGO}, k={K} (PAC {best['PAC (lower = more stable)']:.3f})")
lab = cons[(ALGO, K)]["lab"]
# relabel clusters by size (1 = largest)
size_rank = {c: i + 1 for i, c in enumerate(pd.Series(lab).value_counts().index)}
df["cluster"] = pd.Series(lab).map(size_rank).values
CL = sorted(df.cluster.unique())

# --------------------------------------------------------------------- cluster profiles
prof = []
for v in BINARY:
    row = {"Feature": NAMES[v], "Type": "binary"}
    for c in CL:
        s = df.loc[df.cluster == c, v]
        row[f"Cluster {c}"] = f"{int(s.sum())}/{s.notna().sum()} ({100 * s.mean():.0f}%)"
    row["p"] = stats.chi2_contingency(pd.crosstab(df[v], df.cluster))[1] if df[v].nunique() > 1 else np.nan
    prof.append(row)
for v, raw in [("age", "age"), ("days_fever", "days_fever"), ("hct", "hct"), ("log_plt", "plt"),
               ("log_wbc", "wbc"), ("lymph_pct", "lymph_pct"), ("pt", "pt"), ("ptt", "ptt")]:
    row = {"Feature": NAMES[v].replace(" (log)", ""), "Type": "median (IQR)"}
    for c in CL:
        row[f"Cluster {c}"] = med_iqr(df.loc[df.cluster == c, raw])
    row["p"] = stats.kruskal(*[df.loc[(df.cluster == c) & df[raw].notna(), raw] for c in CL]).pvalue
    prof.append(row)
profile = pd.DataFrame(prof)

OUTS = [("icu_or_tx", "ICU or transfusion"), ("severe_ds", "Severe dengue"), ("dws_or_ds", "DWS or DS"),
        ("prolonged_los", "LOS > 3 days")]
orow = []
for oc, nm in OUTS:
    row = {"Outcome (NOT used for clustering)": nm}
    for c in CL:
        s = df.loc[df.cluster == c, oc]
        row[f"Cluster {c}"] = f"{int(s.sum())}/{len(s)} ({100 * s.mean():.0f}%)"
    row["p"] = stats.chi2_contingency(pd.crosstab(df[oc], df.cluster))[1]
    orow.append(row)
row = {"Outcome (NOT used for clustering)": "WHO class DWOS/DWS/DS"}
for c in CL:
    s = df.loc[df.cluster == c, "dx"]
    row[f"Cluster {c}"] = "/".join(str((s == k).sum()) for k in ["DWOS", "DWS", "DS"])
row["p"] = stats.chi2_contingency(pd.crosstab(df.dx, df.cluster))[1]
orow.append(row)
row = {"Outcome (NOT used for clustering)": "LOS, median (IQR)"}
for c in CL:
    row[f"Cluster {c}"] = med_iqr(df.loc[df.cluster == c, "los"])
row["p"] = stats.kruskal(*[df.loc[df.cluster == c, "los"] for c in CL]).pvalue
orow.append(row)
outcomes = pd.DataFrame(orow)

# --------------------------------------------------------------------- t-SNE
emb = {}
for perp in [15, 30, 50]:
    emb[perp] = TSNE(n_components=2, perplexity=perp, init="pca", random_state=42, max_iter=2000).fit_transform(Z)
df["tsne_1"], df["tsne_2"] = emb[30][:, 0], emb[30][:, 1]

# --------------------------------------------------------------------- figures
BLUE, INK, INK2, INK3, GREY, GREY_LIGHT = "#2a78d6", "#0b0b0b", "#52514e", "#8a8984", "#b9b8b2", "#e8e7e3"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]                 # validated all-pairs for <= 3 series
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelcolor": INK2, "axes.edgecolor": GREY, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GREY_LIGHT, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 8, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name, note=None):
    if note:
        fig.text(0.0, -0.045, note, fontsize=7, color=INK3, ha="left", va="top", wrap=True)
    fig.savefig(FIG / name); plt.close(fig); print("saved", name)


# figT1 - t-SNE, every patient, no colouring
fig, ax = plt.subplots(figsize=(5.6, 5.2))
ax.scatter(emb[30][:, 0], emb[30][:, 1], s=22, color="#6f6e69", edgecolor="white", linewidth=0.6)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"t-SNE of all {N} patients (admission features)", loc="left")
save(fig, "figT1_tsne_uncoloured.png", "20 admission features (12 binary, 8 numeric), standardised; perplexity 30. "
     "Axes have no units. Outcomes and WHO class not used.")

# figT2 - perplexity sensitivity (uncoloured)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, perp in zip(axes, [15, 30, 50]):
    ax.scatter(emb[perp][:, 0], emb[perp][:, 1], s=14, color="#6f6e69", edgecolor="white", linewidth=0.5)
    ax.set_title(f"Perplexity {perp}", loc="left"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("t-SNE robustness to perplexity (no colouring)", x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "figT2_tsne_perplexity.png", "A real structure should persist across perplexities; shapes that change "
     "between panels are artefacts of the embedding.")

# figC1 - consensus matrices for k = 2..5 (selected algorithm)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
seq = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#5598e7", "#104281"])
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
for ax, k in zip(axes, [2, 3, 4, 5]):
    d = cons[(ALGO, k)]
    o = d["order"]
    im = ax.imshow(d["C"][np.ix_(o, o)], cmap=seq, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"k = {k}", loc="left"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01); cb.set_label("Consensus (share of resamples clustered together)")
cb.outline.set_visible(False)
fig.suptitle(f"Consensus matrices ({'k-means' if ALGO == 'kmeans' else 'Gower + hierarchical'}, 500 × 80% resamples)",
             x=0.01, ha="left", fontweight="bold", fontsize=11)
save(fig, "figC1_consensus_matrices.png", "Clean dark blocks on a white background = stable clusters; "
     "intermediate blue = patients that switch cluster between resamples.")

# figC2 - model selection: CDF, PAC, silhouette (both algorithms)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
ax = axes[0]
for i, k in enumerate(K_RANGE):
    d = cons[(ALGO, k)]
    ax.plot(d["grid"], d["cdf"], color=RAMP[i], linewidth=1.8, label=f"k={k}")
ax.set_xlabel("Consensus index"); ax.set_ylabel("CDF"); ax.legend(loc="upper left")
ax.set_title("Consensus CDF (selected algorithm)", loc="left", fontsize=9)
for ax, col, title in [(axes[1], "PAC (lower = more stable)", "PAC — lower is more stable"),
                       (axes[2], "Silhouette", "Silhouette — higher is better separated")]:
    for i, (algo, nm) in enumerate([("kmeans", "k-means"), ("gower_hclust", "Gower + hierarchical")]):
        s = sel[sel.Algorithm == algo]
        ax.plot(s.k, s[col], color=CAT[i], linewidth=2, marker="o", markersize=6, markeredgecolor="white", label=nm)
    ax.set_xticks(list(K_RANGE)); ax.set_xlabel("Number of clusters (k)"); ax.set_title(title, loc="left", fontsize=9)
    ax.legend()
fig.suptitle("Choosing the number of clusters", x=0.01, ha="left", fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "figC2_cluster_selection.png", f"Selected: {'k-means' if ALGO == 'kmeans' else 'Gower + hierarchical'}, "
     f"k = {K} (lowest PAC among solutions with no cluster < 10 patients).")

# figC3 - cluster profile heatmap (standardised feature means) + outcomes
from matplotlib.colors import LinearSegmentedColormap as LSC  # noqa: E402
div = LSC.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
feat = BINARY + NUMERIC
means = pd.DataFrame(Z, columns=feat).groupby(df.cluster.values).mean().T
fig, axes = plt.subplots(1, 2, figsize=(10, 6.2), gridspec_kw={"width_ratios": [1, 1.1]})
ax = axes[0]
im = ax.imshow(means.values, cmap=div, vmin=-1.2, vmax=1.2, aspect="auto")
for i in range(means.shape[0]):
    for j in range(means.shape[1]):
        ax.text(j, i, f"{means.values[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                color="white" if abs(means.values[i, j]) > 0.8 else INK)
ax.set_xticks(range(len(CL))); ax.set_xticklabels([f"C{c}\nn={(df.cluster == c).sum()}" for c in CL])
ax.set_yticks(range(len(feat))); ax.set_yticklabels([NAMES[f] for f in feat], fontsize=8)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Features used for clustering\n(mean z-score per cluster)", loc="left", fontsize=9)
cb = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.02); cb.outline.set_visible(False)
ax = axes[1]
w = 0.8 / len(CL)
for i, c in enumerate(CL):
    for j, (oc, nm) in enumerate(OUTS):
        s = df.loc[df.cluster == c, oc]
        k_, n_ = s.sum(), len(s); p_ = k_ / n_
        z = 1.96; den = 1 + z * z / n_; cen = (p_ + z * z / (2 * n_)) / den
        h = z * np.sqrt(p_ * (1 - p_) / n_ + z * z / (4 * n_ * n_)) / den
        x = j + (i - (len(CL) - 1) / 2) * w
        ax.plot([x, x], [100 * max(0, cen - h), 100 * min(1, cen + h)], color=CAT[i % 3], linewidth=1.8)
        ax.scatter(x, 100 * p_, s=40, color=CAT[i % 3], edgecolor="white", linewidth=1.5, zorder=3,
                   label=f"Cluster {c}" if j == 0 else None)
ax.set_xticks(range(len(OUTS)))
ax.set_xticklabels([f"{nm}\np{'' if fmt_p(p).startswith('<') else '='}{fmt_p(p)}"
                    for (oc, nm), p in zip(OUTS, outcomes.p[:4])], fontsize=8)
ax.set_ylabel("% of patients (95% CI)"); ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
ax.set_title("Outcomes by cluster (outcomes NOT used for clustering)", loc="left", fontsize=9)
fig.tight_layout()
save(fig, "figC3_cluster_profiles.png", "Left: red = above cohort average, blue = below. Right: Wilson 95% CI; χ² test.")

# figT3 - t-SNE coloured by consensus cluster (supplementary)
if len(CL) <= 3:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    for i, c in enumerate(CL):
        m = df.cluster == c
        ax.scatter(df.loc[m, "tsne_1"], df.loc[m, "tsne_2"], s=22, color=CAT[i], edgecolor="white", linewidth=0.6,
                   label=f"Cluster {c} (n={m.sum()})")
    ax.legend(loc="best"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE coloured by consensus cluster (supplementary)", loc="left")
    save(fig, "figT3_tsne_by_cluster.png")

# --------------------------------------------------------------------- workbook
with pd.ExcelWriter(OUT / "clustering.xlsx", engine="openpyxl") as xw:
    pd.DataFrame({"Note": [
        "Consensus clustering (Monti 2003): 500 resamples of 80% of patients; k = 2..6; two algorithms.",
        "Features: 12 binary + 8 numeric admission variables (see clustering.py). Outcomes and WHO class excluded.",
        f"Selected solution: {ALGO}, k = {K} (lowest PAC with no cluster < 10 patients).",
        "Outcome sheet tests whether data-driven clusters carry prognostic information.",
    ]}).to_excel(xw, sheet_name="README", index=False)
    s = sel.copy()
    for c in ["PAC (lower = more stable)", "CDF area", "Delta area", "Silhouette"]:
        s[c] = s[c].round(3)
    s.to_excel(xw, sheet_name="Model_selection", index=False)
    p = profile.copy(); p["p"] = p["p"].map(fmt_p); p.to_excel(xw, sheet_name="Cluster_profiles", index=False)
    o = outcomes.copy(); o["p"] = o["p"].map(fmt_p); o.to_excel(xw, sheet_name="Cluster_outcomes", index=False)
    autosize(xw)

# per-patient assignments are patient-level data: kept in output/private/ (excluded from version control)
PRIVATE = OUT / "private"
PRIVATE.mkdir(parents=True, exist_ok=True)
df[["population", "serial_number", "cluster", "tsne_1", "tsne_2"]].to_excel(PRIVATE / "cluster_assignments.xlsx",
                                                                           index=False)

pd.set_option("display.width", 250, "display.max_columns", 20)
print(); print(profile.assign(p=profile.p.map(fmt_p)).to_string(index=False))
print(); print(outcomes.assign(p=outcomes.p.map(fmt_p)).to_string(index=False))
