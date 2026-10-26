"""Paso 2 — Análisis exploratorio y estadístico (Objetivos específicos 1 y 2).

* Descriptiva de la cohorte por desenlace (tabla 1) y valores faltantes.
* Asociación univariante de variables clínicas (Mann-Whitney U) y alteraciones
  citogenéticas/mutaciones conductoras (Fisher exacto, OR con IC 95 %), con corrección
  de Benjamini-Hochberg (FDR).
* Incidencia acumulada de transformación a LMA por categoría IPSS-M (Aalen-Johansen, con
  la muerte sin LMA como evento competitivo) y supervivencia libre de leucemia (Kaplan-Meier).

Uso:  python scripts/02_eda_estadistica.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import AalenJohansenFitter, KaplanMeierFitter  # noqa: E402
from lifelines.statistics import multivariate_logrank_test  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from tfm_mds import config, plots  # noqa: E402
from tfm_mds.features import predictor_columns  # noqa: E402

warnings.filterwarnings("ignore")
plots.set_style()
T = pd.read_csv(config.ANALYTIC_DATASET)
G = predictor_columns(T)
y = T["AML_T"]

# ---------------------------------------------------------------------------
# 1. Tabla descriptiva
# ---------------------------------------------------------------------------
NICE = {"AGE": "Edad (años)", "BM_BLAST": "Blastos MO (%)", "PB_BLAST": "Blastos SP (%)",
        "HB": "Hemoglobina (g/dL)", "PLT": "Plaquetas (×10⁹/L)", "WBC": "Leucocitos (×10⁹/L)",
        "ANC": "Neutrófilos (×10⁹/L)", "MONOCYTES": "Monocitos (×10⁹/L)",
        "TMB_NONSYNONYMOUS": "TMB (no sinónimas)", "N_DRIVER_GENES": "Nº genes conductores mutados"}
rows = []
for c, lab in NICE.items():
    a, b = T.loc[y == 0, c].dropna(), T.loc[y == 1, c].dropna()
    _, p = stats.mannwhitneyu(a, b)
    rows.append({"variable": lab, "Sin LMA (n=%d)" % (y == 0).sum(): f"{a.median():.1f} [{a.quantile(.25):.1f}–{a.quantile(.75):.1f}]",
                 "LMA (n=%d)" % (y == 1).sum(): f"{b.median():.1f} [{b.quantile(.25):.1f}–{b.quantile(.75):.1f}]",
                 "% faltantes": f"{T[c].isna().mean() * 100:.1f}", "p": p, "prueba": "Mann-Whitney U"})
for c, levels in {"SEX": None, "MDS_TYPE": None, "IPSSM": config.IPSSM_ORDER}.items():
    ct = pd.crosstab(T[c], y)
    if levels:
        ct = ct.reindex(levels)
    _, p, _, _ = stats.chi2_contingency(ct.dropna())
    for lev, r in ct.iterrows():
        rows.append({"variable": f"{c} = {lev}",
                     "Sin LMA (n=%d)" % (y == 0).sum(): f"{int(r[0])} ({r[0] / (y == 0).sum() * 100:.1f} %)",
                     "LMA (n=%d)" % (y == 1).sum(): f"{int(r[1])} ({r[1] / (y == 1).sum() * 100:.1f} %)",
                     "% faltantes": f"{T[c].isna().mean() * 100:.1f}", "p": p, "prueba": "Chi-cuadrado"})
tabla1 = pd.DataFrame(rows)
tabla1.to_csv(config.TABLES / "tabla1_descriptiva.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Valores faltantes
# ---------------------------------------------------------------------------
miss = T[G["numeric"] + G["categorical"]].isna().mean().sort_values(ascending=False)
miss = miss[miss > 0]
plots.plot_barh([f"{m}" for m in miss.index], miss.values * 100,
                "Porcentaje de valores faltantes por variable", "% de pacientes", "eda_faltantes",
                color=config.PALETTE[6])

# ---------------------------------------------------------------------------
# 3. Asociación univariante: alteraciones binarias (Fisher + OR con IC 95 %)
# ---------------------------------------------------------------------------
res = []
for c in G["binary"]:
    x = T[c].fillna(0).astype(int)
    a = int(((x == 1) & (y == 1)).sum()); b = int(((x == 1) & (y == 0)).sum())
    cc = int(((x == 0) & (y == 1)).sum()); d = int(((x == 0) & (y == 0)).sum())
    if a + b < 10:
        continue
    _, p = stats.fisher_exact([[a, b], [cc, d]])
    aa, bb, c2, dd = a + .5, b + .5, cc + .5, d + .5  # corrección de Haldane-Anscombe
    or_ = (aa * dd) / (bb * c2)
    se = np.sqrt(1 / aa + 1 / bb + 1 / c2 + 1 / dd)
    res.append({"variable": c, "n_alterados": a + b, "frecuencia_%": (a + b) / len(T) * 100,
                "LMA_en_alterados_%": a / (a + b) * 100, "LMA_en_no_alterados_%": cc / (cc + d) * 100,
                "OR": or_, "OR_IC2.5": np.exp(np.log(or_) - 1.96 * se),
                "OR_IC97.5": np.exp(np.log(or_) + 1.96 * se), "p": p})
assoc = pd.DataFrame(res)
assoc["q_FDR"] = multipletests(assoc["p"], method="fdr_bh")[1]
assoc = assoc.sort_values("p")
assoc.to_csv(config.TABLES / "asociacion_univariante_alteraciones.csv", index=False)

sig = assoc[assoc["q_FDR"] < 0.05].sort_values("OR", ascending=False)
fig, ax = plt.subplots(figsize=(6.4, 0.3 * len(sig) + 1.3))
yy = np.arange(len(sig))
cols = [config.PALETTE[7] if o > 1 else config.PALETTE[0] for o in sig["OR"]]
ax.errorbar(sig["OR"], yy, xerr=[sig["OR"] - sig["OR_IC2.5"], sig["OR_IC97.5"] - sig["OR"]],
            fmt="none", ecolor=plots.INK2, lw=1, capsize=2)
ax.scatter(sig["OR"], yy, color=cols, s=36, zorder=3, edgecolor="white", linewidth=1)
ax.axvline(1, color=plots.INK2, ls="--", lw=1)
ax.set_xscale("log")
ax.set_yticks(yy, [s.replace("MUT_", "").replace("CYTO_", "cito: ") for s in sig["variable"]], fontsize=8)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set(xlabel="Odds ratio de transformación a LMA (escala log, IC 95 %)",
       title="Alteraciones asociadas a progresión a LMA (FDR < 0,05)")
plots.save(fig, "forest_univariante")

# Frecuencia de mutaciones conductoras por grupo
genes = [c for c in G["binary"] if c.startswith("MUT_")]
freq = T[genes].mean().sort_values(ascending=False).head(25).index
f0 = T.loc[y == 0, freq].mean() * 100
f1 = T.loc[y == 1, freq].mean() * 100
fig, ax = plt.subplots(figsize=(9, 3.8))
xx = np.arange(len(freq))
ax.bar(xx - 0.2, f0, 0.38, color=config.PALETTE[0], label="Sin LMA")
ax.bar(xx + 0.2, f1, 0.38, color=config.PALETTE[1], label="Transformación a LMA")
ax.set_xticks(xx, [g.replace("MUT_", "") for g in freq], rotation=60, fontsize=8, style="italic")
ax.set(ylabel="% de pacientes mutados", title="25 genes conductores más frecuentes según desenlace")
ax.legend()
ax.grid(axis="x", visible=False)
plots.save(fig, "frecuencia_mutaciones")

# Asociación univariante de variables continuas (estandarizada)
cont = []
for c in [c for c in G["numeric"] if c in T]:
    a, b = T.loc[y == 0, c].dropna(), T.loc[y == 1, c].dropna()
    _, p = stats.mannwhitneyu(a, b)
    cont.append({"variable": c, "mediana_sin_LMA": a.median(), "mediana_LMA": b.median(), "p": p})
cont = pd.DataFrame(cont)
cont["q_FDR"] = multipletests(cont["p"], method="fdr_bh")[1]
cont.sort_values("p").to_csv(config.TABLES / "asociacion_univariante_continuas.csv", index=False)

# Distribución de blastos, hemoglobina y plaquetas por desenlace
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
for ax, (c, lab, logy) in zip(axes, [("BM_BLAST", "Blastos en MO (%)", False),
                                     ("HB", "Hemoglobina (g/dL)", False),
                                     ("PLT", "Plaquetas (×10⁹/L)", True)]):
    data = [T.loc[y == k, c].dropna() for k in (0, 1)]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={"color": plots.INK, "lw": 1.5})
    for patch, col in zip(bp["boxes"], [config.PALETTE[0], config.PALETTE[1]]):
        patch.set_facecolor(col); patch.set_alpha(0.85); patch.set_edgecolor("white")
    ax.set_xticks([1, 2], ["Sin LMA", "LMA"])
    ax.set(title=lab)
    if logy:
        ax.set_yscale("log")
plots.save(fig, "eda_clinicas_por_desenlace")

# ---------------------------------------------------------------------------
# 4. Incidencia acumulada de LMA y LFS por IPSS-M
# ---------------------------------------------------------------------------
S = T.dropna(subset=["IPSSM"]).copy()
S["evento"] = np.select([S["AML_T"] == 1, S["DEATH_WITHOUT_AML"] == 1], [1, 2], 0)
S["t_anos"] = S["AML_T_MONTHS"] / 12
seq = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]  # secuencial (riesgo)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
cif_rows = []
for col, cat in zip(seq, config.IPSSM_ORDER):
    d = S[S["IPSSM"] == cat]
    ajf = AalenJohansenFitter(calculate_variance=False, seed=config.SEED)
    ajf.fit(d["t_anos"], d["evento"], event_of_interest=1)
    ci = ajf.cumulative_density_
    axes[0].step(ci.index, ci.iloc[:, 0], where="post", color=col, label=f"{cat} (n={len(d)})")
    for yr in (1, 3, 5):
        cif_rows.append({"IPSS-M": cat, "años": yr,
                         "incidencia_acumulada_LMA_%": float(ci[ci.index <= yr].iloc[-1, 0]) * 100})
    kmf = KaplanMeierFitter().fit(d["t_anos"], (d["evento"] > 0).astype(int))
    axes[1].step(kmf.survival_function_.index, kmf.survival_function_.iloc[:, 0], where="post", color=col)
lr = multivariate_logrank_test(S["t_anos"], S["IPSSM"], (S["evento"] > 0).astype(int))
axes[0].set(xlim=(0, 10), ylim=(0, 0.6), xlabel="Años desde el diagnóstico",
            ylabel="Incidencia acumulada de LMA", title="Transformación a LMA por IPSS-M\n(muerte sin LMA = riesgo competitivo)")
axes[1].set(xlim=(0, 10), ylim=(0, 1), xlabel="Años desde el diagnóstico",
            ylabel="Supervivencia libre de leucemia", title=f"LFS por IPSS-M (log-rank p = {lr.p_value:.1e})")
axes[0].legend(fontsize=8, title="IPSS-M", title_fontsize=8)
plots.save(fig, "incidencia_LMA_por_IPSSM")
pd.DataFrame(cif_rows).pivot(index="IPSS-M", columns="años").reindex(config.IPSSM_ORDER).round(1).to_csv(
    config.TABLES / "incidencia_acumulada_LMA_IPSSM.csv")

print(tabla1.head(12).to_string(index=False))
print("\nAlteraciones con FDR < 0,05:", len(sig))
print(assoc.head(15)[["variable", "frecuencia_%", "OR", "q_FDR"]].to_string(index=False))
print(f"\nLog-rank LFS por IPSS-M: p = {lr.p_value:.2e}")
