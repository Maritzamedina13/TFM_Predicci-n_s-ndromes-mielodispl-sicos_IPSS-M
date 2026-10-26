"""Paso 4 — Interpretabilidad con SHAP (Objetivo específico 5).

* Red neuronal: valores SHAP mediante el explicador de permutación (agnóstico al modelo)
  con un fondo de 100 pacientes de entrenamiento.
* Random forest: TreeExplainer (SHAP exacto para árboles).
* Regresión logística: odds ratio por desviación estándar de cada variable.
Se comparan los rankings de importancia y se contrastan con los factores pronósticos
descritos en la literatura (blastos, citogenética de alto riesgo, TP53 multihit, etc.).

Uso:  python scripts/04_interpretabilidad.py   (requiere haber ejecutado el paso 3)
"""
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from tfm_mds import config, plots  # noqa: E402
from tfm_mds.features import predictor_columns  # noqa: E402
from tfm_mds.models import load_dnn_pipeline  # noqa: E402

plots.set_style()
N_EXPLAIN = int(os.environ.get("TFM_SHAP_N", "0")) or None  # None = todo el conjunto de prueba

PRETTY = {
    "AGE": "Edad", "BM_BLAST": "Blastos MO (%)", "PB_BLAST": "Blastos SP (%)", "HB": "Hemoglobina",
    "PLT": "Plaquetas", "WBC": "Leucocitos", "ANC": "Neutrófilos", "MONOCYTES": "Monocitos",
    "TMB_NONSYNONYMOUS": "TMB", "CYTO_IPSSR_ORD": "Riesgo citogenético IPSS-R",
    "N_DRIVER_MUTATIONS": "Nº mutaciones conductoras", "N_DRIVER_GENES": "Nº genes conductores",
    "TP53_N_MUT": "Nº mutaciones TP53", "TP53_MAX_VAF": "VAF máx. TP53", "TP53_MULTIHIT": "TP53 multihit",
    "IPSSM_NRES": "Nº genes residuales IPSS-M", "MAX_DRIVER_VAF": "VAF máx. conductora",
    "COMPLEX_KARYOTYPE": "Cariotipo complejo", "TP53_LOCUS_LOH": "LOH/deleción locus TP53",
    "FLT3_ITD": "FLT3-ITD", "MLL_PTD": "KMT2A-PTD", "FLT3_ANY": "FLT3 (ITD/TKD)",
    "CYTO_DEL5Q": "del(5q)", "CYTO_DEL7_7Q": "−7/del(7q)", "CYTO_DEL17_17P": "−17/del(17p)",
    "CYTO_TRIS8": "+8", "CYTO_DEL20Q": "del(20q)", "CYTO_MINUS_Y": "−Y", "CYTO_DEL11Q": "del(11q)",
    "CYTO_DEL12P": "del(12p)", "CYTO_3Q26": "inv(3)/t(3;3)", "CYTO_NORMAL": "Cariotipo normal",
    "CYTO_MISSING": "Cariotipo no disponible",
}


def pretty(name: str) -> str:
    if name.startswith("MUT_"):
        return f"{name[4:]} mutado"
    if name.startswith("missingindicator_"):
        return f"{PRETTY.get(name[17:], name[17:])} (faltante)"
    for pre in ("SEX_", "MDS_TYPE_", "WHO_2016_"):
        if name.startswith(pre):
            return f"{pre[:-1]}={name[len(pre):]}"
    return PRETTY.get(name, name)


T = pd.read_csv(config.ANALYTIC_DATASET)
G = predictor_columns(T)
FEATS = G["numeric"] + G["categorical"] + G["binary"]
part = pd.read_csv(config.TABLES / "particion_pacientes.csv")
tr = part["particion"].eq("entrenamiento").to_numpy()
Xtr, Xte = T.loc[tr, FEATS], T.loc[~tr, FEATS]

dnn = load_dnn_pipeline(str(config.MODELS / "dnn"))
rf = joblib.load(config.MODELS / "random_forest.joblib")
lr = joblib.load(config.MODELS / "regresion_logistica.joblib")

pre = dnn.named_steps["pre"]
names = list(pre.get_feature_names_out())
nice = [pretty(n) for n in names]
Ztr, Zte = pre.transform(Xtr), pre.transform(Xte)
if N_EXPLAIN:
    Zte, Xte = Zte[:N_EXPLAIN], Xte.iloc[:N_EXPLAIN]

# ---------------------------------------------------------------------------
# SHAP — red neuronal
# ---------------------------------------------------------------------------
clf = dnn.named_steps["clf"]
background = shap.sample(Ztr, 100, random_state=config.SEED)
explainer = shap.Explainer(lambda z: clf.predict_proba(z)[:, 1],
                           shap.maskers.Independent(background, max_samples=100),
                           algorithm="permutation", feature_names=nice, seed=config.SEED)
sv_dnn = explainer(Zte, max_evals=2 * Zte.shape[1] + 1, silent=True)
np.save(config.MODELS / "shap_dnn_test.npy", sv_dnn.values)

shap.plots.beeswarm(sv_dnn, max_display=20, show=False, plot_size=(8, 7))
plt.title("SHAP — red neuronal (conjunto de prueba)", fontweight="bold")
plots.save(plt.gcf(), "shap_dnn_beeswarm")
shap.plots.bar(sv_dnn, max_display=20, show=False)
plt.title("Importancia media |SHAP| — red neuronal", fontweight="bold")
plots.save(plt.gcf(), "shap_dnn_barras")

# Paciente de prueba con mayor riesgo predicho (explicación individual)
i = int(np.argmax(clf.predict_proba(Zte)[:, 1]))
shap.plots.waterfall(sv_dnn[i], max_display=12, show=False)
plt.title("Explicación individual — paciente de mayor riesgo predicho", fontweight="bold")
plots.save(plt.gcf(), "shap_dnn_waterfall_paciente")

# Dependencia en la escala clínica original
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
for ax, col in zip(axes.ravel(), ["BM_BLAST", "PLT", "TP53_MAX_VAF", "AGE"]):
    j = names.index(col)
    xv = Xte[col].to_numpy()
    sc = ax.scatter(xv, sv_dnn.values[:, j], s=12, c=config.PALETTE[0], alpha=0.6, edgecolor="none")
    ax.axhline(0, color=plots.INK2, lw=1, ls="--")
    ax.set(xlabel=pretty(col), ylabel="Valor SHAP (prob. de LMA)")
    if col == "PLT":
        ax.set_xscale("log")
fig.suptitle("Efecto de variables clínicas en la predicción de la red neuronal", fontweight="bold")
plots.save(fig, "shap_dnn_dependencia")

# ---------------------------------------------------------------------------
# SHAP — random forest y coeficientes de la regresión logística
# ---------------------------------------------------------------------------
Zte_rf = rf.named_steps["pre"].transform(Xte)
sv_rf = shap.TreeExplainer(rf.named_steps["clf"]).shap_values(Zte_rf)
sv_rf = sv_rf[:, :, 1] if np.ndim(sv_rf) == 3 else (sv_rf[1] if isinstance(sv_rf, list) else sv_rf)
shap.summary_plot(sv_rf, Zte_rf, feature_names=nice, max_display=20, show=False, plot_size=(8, 7))
plt.title("SHAP — random forest (conjunto de prueba)", fontweight="bold")
plots.save(plt.gcf(), "shap_rf_beeswarm")

coef = lr.named_steps["clf"].coef_.ravel()
lr_names = [pretty(n) for n in lr.named_steps["pre"].get_feature_names_out()]
or_tab = pd.DataFrame({"variable": lr_names, "coeficiente": coef, "OR": np.exp(coef)})
or_tab = or_tab.reindex(or_tab["coeficiente"].abs().sort_values(ascending=False).index)
or_tab.to_csv(config.TABLES / "regresion_logistica_coeficientes.csv", index=False)
top = or_tab.head(20)
plots.plot_barh(top["variable"].tolist(), top["OR"].to_numpy(),
                "Regresión logística — odds ratio (variables escaladas)", "OR (escala log)",
                "lr_odds_ratios", color=config.PALETTE[2], ref=1)

# ---------------------------------------------------------------------------
# Comparación de rankings de importancia
# ---------------------------------------------------------------------------
imp = pd.DataFrame({
    "variable": nice,
    "SHAP_DNN": np.abs(sv_dnn.values).mean(0),
    "SHAP_RF": np.abs(sv_rf).mean(0),
})
imp = imp.merge(or_tab[["variable", "coeficiente"]].assign(LR_abs_coef=lambda d: d["coeficiente"].abs()),
                on="variable", how="left").drop(columns="coeficiente")
for c in ["SHAP_DNN", "SHAP_RF", "LR_abs_coef"]:
    imp[f"rank_{c}"] = imp[c].rank(ascending=False).astype(int)
imp = imp.sort_values("SHAP_DNN", ascending=False)
# Dirección del efecto en la red: correlación entre valor de la variable y su SHAP
direction = []
for j in range(Zte.shape[1]):
    xv = Zte[:, j]
    direction.append(np.corrcoef(xv, sv_dnn.values[:, j])[0, 1] if np.std(xv) > 0 else np.nan)
imp["direccion_DNN"] = pd.Series(np.where(np.array(direction) > 0, "↑ riesgo", "↓ riesgo"),
                                 index=range(len(nice))).reindex(imp.index)
imp.round(5).to_csv(config.TABLES / "importancia_variables_comparada.csv", index=False)
from scipy.stats import spearmanr  # noqa: E402

rho = spearmanr(imp["SHAP_DNN"], imp["SHAP_RF"]).statistic
print(imp.head(20).to_string(index=False))
print(f"\nCorrelación de Spearman entre importancias DNN y RF: {rho:.3f}")
