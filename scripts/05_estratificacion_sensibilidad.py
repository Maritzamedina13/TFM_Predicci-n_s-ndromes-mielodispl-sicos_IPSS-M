"""Paso 5 — Estratificación pronóstica y análisis de sensibilidad (Objetivos 4 y 5).

a) Estratificación del riesgo: se definen tres grupos de riesgo de la red neuronal con
   puntos de corte fijados en entrenamiento (percentiles 50 y 85 de las predicciones OOF
   reconstruidas) y se compara su incidencia acumulada de LMA en prueba con la del IPSS-M
   agrupado (Bajo = Muy bajo/Bajo, Intermedio = Moderado bajo/alto, Alto = Alto/Muy alto).
   Se calcula también el índice C de Harrell para el tiempo hasta LMA.
b) Sensibilidad del desenlace: etiqueta a horizonte fijo (36 meses), que excluye a los
   pacientes censurados vivos antes del horizonte, evaluada por validación cruzada.

Uso:  python scripts/05_estratificacion_sensibilidad.py   (requiere el paso 3)
"""
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import AalenJohansenFitter  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from tfm_mds import config, plots  # noqa: E402
from tfm_mds.features import predictor_columns  # noqa: E402
from tfm_mds.models import make_dnn, make_logistic, make_random_forest  # noqa: E402
from tfm_mds.preprocessing import build_preprocessor  # noqa: E402
from tfm_mds.target import fixed_horizon_label  # noqa: E402

plots.set_style()
DNN, RF, LR = "Red neuronal (DNN)", "Random forest", "Regresión logística"
T = pd.read_csv(config.ANALYTIC_DATASET)
G = predictor_columns(T)
FEATS = G["numeric"] + G["categorical"] + G["binary"]
P = pd.read_csv(config.TABLES / "predicciones_test.csv")
part = pd.read_csv(config.TABLES / "particion_pacientes.csv")
tr = part["particion"].eq("entrenamiento").to_numpy()
best = json.load(open(config.MODELS / "mejores_hiperparametros.json"))
imb = json.load(open(config.MODELS / "umbrales.json"))["estrategia_desbalance_dnn"]
dnn_params = {k.replace("clf__", ""): (tuple(v) if isinstance(v, list) else v)
              for k, v in best[DNN].items()}


def models():
    pre = build_preprocessor(G)
    return {
        LR: make_logistic(pre).set_params(**best[LR]),
        RF: make_random_forest(build_preprocessor(G)).set_params(**best[RF]),
        DNN: make_dnn(build_preprocessor(G), imbalance=imb, **dnn_params),
    }


# ---------------------------------------------------------------------------
# a) Grupos de riesgo de la red neuronal frente a IPSS-M
# ---------------------------------------------------------------------------
cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)
Xtr, ytr = T.loc[tr, FEATS], T.loc[tr, "AML_T"].to_numpy()
oof = cross_val_predict(models()[DNN], Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
cuts = np.quantile(oof, [0.50, 0.85])
grp_lab = ["Bajo", "Intermedio", "Alto"]
P["grupo_DNN"] = pd.cut(P[f"p_{DNN}"], [-np.inf, cuts[0], cuts[1], np.inf], labels=grp_lab)
ipssm3 = {"Very-Low": "Bajo", "Low": "Bajo", "Moderate-Low": "Intermedio",
          "Moderate-High": "Intermedio", "High": "Alto", "Very-High": "Alto"}
P["grupo_IPSSM"] = pd.Categorical(P["IPSSM"].map(ipssm3), categories=grp_lab)
P["evento"] = np.select([P["AML_T"] == 1, P["DEATH_WITHOUT_AML"] == 1], [1, 2], 0)
P["t_anos"] = P["AML_T_MONTHS"] / 12
S = P.dropna(subset=["grupo_IPSSM"])

seq = {"Bajo": "#9ecae1", "Intermedio": "#3182bd", "Alto": "#08306b"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
rows = []
for ax, col, title in [(axes[0], "grupo_IPSSM", "IPSS-M (3 grupos)"),
                       (axes[1], "grupo_DNN", "Red neuronal (3 grupos)")]:
    for g in grp_lab:
        d = S[S[col] == g]
        if len(d) < 5:
            continue
        ajf = AalenJohansenFitter(calculate_variance=False, seed=config.SEED)
        ajf.fit(d["t_anos"], d["evento"], event_of_interest=1)
        ci = ajf.cumulative_density_
        ax.step(ci.index, ci.iloc[:, 0], where="post", color=seq[g], label=f"{g} (n={len(d)})")
        at3 = float(ci[ci.index <= 3].iloc[-1, 0]) * 100 if (ci.index <= 3).any() else np.nan
        rows.append({"clasificación": title, "grupo": g, "n": len(d), "% del total": len(d) / len(S) * 100,
                     "LMA observadas": int(d["AML_T"].sum()), "% LMA": d["AML_T"].mean() * 100,
                     "incidencia acumulada LMA 3 años %": at3})
    ax.set(title=title, xlabel="Años desde el diagnóstico", xlim=(0, 8), ylim=(0, 0.6))
    ax.legend(fontsize=8)
axes[0].set_ylabel("Incidencia acumulada de LMA")
fig.suptitle("Estratificación del riesgo de transformación a LMA — conjunto de prueba", fontweight="bold")
plots.save(fig, "estratificacion_dnn_vs_ipssm")
strat = pd.DataFrame(rows).round(1)
strat.to_csv(config.TABLES / "estratificacion_grupos_riesgo.csv", index=False)

recl = pd.crosstab(S["grupo_IPSSM"], S["grupo_DNN"], margins=True)
rate = S.pivot_table(index="grupo_IPSSM", columns="grupo_DNN", values="AML_T", aggfunc="mean",
                     observed=False) * 100
recl.to_csv(config.TABLES / "reclasificacion_ipssm_vs_dnn_n.csv")
rate.round(1).to_csv(config.TABLES / "reclasificacion_ipssm_vs_dnn_tasa_LMA.csv")

# Índice C de Harrell (tiempo hasta LMA; muerte sin LMA tratada como censura)
cidx = {
    DNN: concordance_index(S["AML_T_MONTHS"], -S[f"p_{DNN}"], S["AML_T"]),
    RF: concordance_index(S["AML_T_MONTHS"], -S[f"p_{RF}"], S["AML_T"]),
    LR: concordance_index(S["AML_T_MONTHS"], -S[f"p_{LR}"], S["AML_T"]),
    "IPSS-M (puntuación)": concordance_index(S["AML_T_MONTHS"], -S["IPSSM_SCORE"], S["AML_T"]),
}
pd.Series(cidx, name="indice_C").round(4).to_csv(config.TABLES / "indice_C_tiempo_LMA.csv")

# ---------------------------------------------------------------------------
# b) Sensibilidad: desenlace a horizonte fijo
# ---------------------------------------------------------------------------
lab = fixed_horizon_label(T, config.HORIZON_MONTHS)
H = T[lab.notna()].copy()
yh = lab[lab.notna()].astype(int).to_numpy()
rows = []
for m, est in models().items():
    p = cross_val_predict(est, H[FEATS], yh, cv=cv, method="predict_proba")[:, 1]
    rows.append({"modelo": m, "AUC-ROC": roc_auc_score(yh, p), "AUC-PR": average_precision_score(yh, p)})
ok = H["IPSSM_SCORE"].notna().to_numpy()
rows.append({"modelo": "IPSS-M (puntuación)*", "AUC-ROC": roc_auc_score(yh[ok], H.loc[ok, "IPSSM_SCORE"]),
             "AUC-PR": average_precision_score(yh[ok], H.loc[ok, "IPSSM_SCORE"])})
sens = pd.DataFrame(rows).round(4)
sens.insert(1, "n", [len(H)] * 3 + [int(ok.sum())])
sens.insert(2, "eventos", [int(yh.sum())] * 3 + [int(yh[ok].sum())])
sens.to_csv(config.TABLES / f"sensibilidad_horizonte_{config.HORIZON_MONTHS}m.csv", index=False)

print(strat.to_string(index=False))
print("\nReclasificación (n):\n", recl)
print("\nTasa de LMA (%) por celda:\n", rate.round(1))
print("\nÍndice C:", {k: round(float(v), 3) for k, v in cidx.items()})
print(f"\nSensibilidad (LMA a {config.HORIZON_MONTHS} meses; validación cruzada):\n", sens.to_string(index=False))
