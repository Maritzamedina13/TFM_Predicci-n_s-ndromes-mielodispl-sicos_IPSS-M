"""Paso 3 — Entrenamiento, ajuste de hiperparámetros y evaluación (Objetivos 2–4).

Diseño:
1. Partición estratificada 80/20: entrenamiento / prueba independiente (no se usa
   durante el desarrollo).
2. En entrenamiento, validación cruzada estratificada de 5 pliegues para:
   a) ajustar hiperparámetros (regresión logística, random forest y red neuronal);
   b) comparar estrategias de desbalance en la red (ninguna, pesos de clase, SMOTE);
   c) obtener predicciones fuera de pliegue (OOF) para fijar el umbral de decisión
      (índice de Youden) y recalibrar probabilidades (Platt).
3. Ajuste final en todo el entrenamiento y evaluación en prueba: AUC-ROC, AUC-PR,
   sensibilidad, especificidad, F1, VPP, VPN, Brier y calibración, con IC 95 % por
   bootstrap y comparación frente al IPSS-M (DeLong y bootstrap pareado).

Uso:  python scripts/03_entrenamiento_evaluacion.py   (≈ 20–30 min en CPU)
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score  # noqa: E402
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_val_predict, cross_validate, train_test_split)

from tfm_mds import config, plots  # noqa: E402
from tfm_mds.evaluation import (PlattRecalibrator, bootstrap_ci, delong_test,  # noqa: E402
                                full_metrics, paired_bootstrap_diff, youden_threshold)
from tfm_mds.features import predictor_columns  # noqa: E402
from tfm_mds.models import (PARAM_DIST_DNN, PARAM_GRID_LR, PARAM_GRID_RF, make_dnn,  # noqa: E402
                            make_logistic, make_random_forest, save_dnn_pipeline)
from tfm_mds.preprocessing import build_preprocessor  # noqa: E402

plots.set_style()
t0 = time.time()
DNN, RF, LR = "Red neuronal (DNN)", "Random forest", "Regresión logística"
IPSSM, IPSSR = "IPSS-M (puntuación)", "IPSS-R (puntuación)"


def log(msg):
    print(f"[{(time.time() - t0) / 60:5.1f} min] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Datos y partición
# ---------------------------------------------------------------------------
T = pd.read_csv(config.ANALYTIC_DATASET)
G = predictor_columns(T)
FEATS = G["numeric"] + G["categorical"] + G["binary"]
X, y = T[FEATS], T["AML_T"].to_numpy()
idx_tr, idx_te = train_test_split(np.arange(len(T)), test_size=config.TEST_SIZE,
                                  stratify=y, random_state=config.SEED)
Xtr, Xte, ytr, yte = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]
pd.DataFrame({"PATIENT_ID": T["PATIENT_ID"], "particion": "entrenamiento"}).assign(
    particion=lambda d: np.where(np.isin(np.arange(len(d)), idx_te), "prueba", "entrenamiento")
).to_csv(config.TABLES / "particion_pacientes.csv", index=False)
log(f"Entrenamiento: {len(idx_tr)} (LMA={ytr.sum()}) | Prueba: {len(idx_te)} (LMA={yte.sum()}) "
    f"| {len(FEATS)} variables de entrada")

cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
pre = build_preprocessor(G)

# ---------------------------------------------------------------------------
# 1. Ajuste de hiperparámetros (CV en entrenamiento, métrica AUC-PR)
# ---------------------------------------------------------------------------
searches = {}
log("Búsqueda en rejilla — regresión logística")
searches[LR] = GridSearchCV(make_logistic(pre), PARAM_GRID_LR, scoring="average_precision",
                            cv=cv, n_jobs=-1).fit(Xtr, ytr)
log("Búsqueda en rejilla — random forest")
searches[RF] = GridSearchCV(make_random_forest(pre), PARAM_GRID_RF, scoring="average_precision",
                            cv=cv, n_jobs=1).fit(Xtr, ytr)
log("Búsqueda aleatoria — red neuronal (TensorFlow/Keras)")
searches[DNN] = RandomizedSearchCV(make_dnn(pre), PARAM_DIST_DNN, n_iter=config.N_ITER_SEARCH_DNN,
                                   scoring="average_precision", cv=cv, n_jobs=1,
                                   random_state=config.SEED).fit(Xtr, ytr)
best_params = {}
for m, s in searches.items():
    best_params[m] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.best_params_.items()}
    r = pd.DataFrame(s.cv_results_)
    keep = [c for c in r.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score", "rank_test_score"]
    r[keep].sort_values("rank_test_score").to_csv(
        config.TABLES / f"busqueda_hiperparametros_{m.split()[0].lower()}.csv", index=False)
    log(f"  {m}: AUC-PR CV = {s.best_score_:.3f} | {best_params[m]}")
json.dump(best_params, open(config.MODELS / "mejores_hiperparametros.json", "w"), indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# 2. Estrategias de desbalance en la red neuronal
# ---------------------------------------------------------------------------
dnn_params = {k.replace("clf__", ""): v for k, v in searches[DNN].best_params_.items()}
imb_rows = []
for strat, lab in [("none", "Sin corrección"), ("class_weight", "Pesos de clase"), ("smote", "SMOTE")]:
    log(f"Desbalance — {lab}")
    r = cross_validate(make_dnn(pre, imbalance=strat, **dnn_params), Xtr, ytr, cv=cv,
                       scoring={"AUC-ROC": "roc_auc", "AUC-PR": "average_precision",
                                "F1": "f1", "Sensibilidad": "recall"})
    imb_rows.append({"estrategia": lab, **{k: f"{np.mean(r['test_' + k]):.3f} ± {np.std(r['test_' + k]):.3f}"
                                           for k in ["AUC-ROC", "AUC-PR", "F1", "Sensibilidad"]},
                     "_ap": np.mean(r["test_AUC-PR"]), "_strat": strat})
imb = pd.DataFrame(imb_rows)
imb.drop(columns=["_ap", "_strat"]).to_csv(config.TABLES / "dnn_estrategias_desbalance.csv", index=False)
best_imb = imb.loc[imb["_ap"].idxmax(), "_strat"]
log(f"Estrategia de desbalance elegida: {best_imb}")

final_models = {
    LR: searches[LR].best_estimator_,
    RF: searches[RF].best_estimator_,
    DNN: make_dnn(pre, imbalance=best_imb, **dnn_params),
}

# ---------------------------------------------------------------------------
# 3. Predicciones fuera de pliegue (OOF) en entrenamiento
# ---------------------------------------------------------------------------
oof, fold_rows = {}, []
folds = list(cv.split(Xtr, ytr))
for m, est in final_models.items():
    log(f"Predicciones OOF — {m}")
    oof[m] = cross_val_predict(clone(est), Xtr, ytr, cv=folds, method="predict_proba")[:, 1]
score_tr = {IPSSM: T["IPSSM_SCORE"].to_numpy()[idx_tr], IPSSR: T["IPSSR_SCORE"].to_numpy()[idx_tr]}
for k, (tr_i, va_i) in enumerate(folds):
    for m in final_models:
        fold_rows.append({"modelo": m, "pliegue": k + 1, "AUC-ROC": roc_auc_score(ytr[va_i], oof[m][va_i]),
                          "AUC-PR": average_precision_score(ytr[va_i], oof[m][va_i])})
    for m, s in score_tr.items():
        ok = va_i[~np.isnan(s[va_i])]
        fold_rows.append({"modelo": m, "pliegue": k + 1, "AUC-ROC": roc_auc_score(ytr[ok], s[ok]),
                          "AUC-PR": average_precision_score(ytr[ok], s[ok])})
cvdf = pd.DataFrame(fold_rows)
cvdf.to_csv(config.TABLES / "cv_metricas_por_pliegue.csv", index=False)
cv_summary = cvdf.groupby("modelo")[["AUC-ROC", "AUC-PR"]].agg(["mean", "std"]).round(3)
cv_summary.to_csv(config.TABLES / "cv_resumen.csv")
plots.plot_cv_scores(cvdf, "AUC-ROC", "cv_auc_roc")
plots.plot_cv_scores(cvdf, "AUC-PR", "cv_auc_pr")

thresholds = {m: youden_threshold(ytr, p) for m, p in oof.items()}
recal = {m: PlattRecalibrator().fit(p, ytr) for m, p in oof.items()}
# Referencias: umbral de Youden para las puntuaciones y modelo logístico puntuación→probabilidad
ref_lr = {}
for m, s in score_tr.items():
    ok = ~np.isnan(s)
    thresholds[m] = youden_threshold(ytr[ok], s[ok])
    ref_lr[m] = LogisticRegression().fit(s[ok].reshape(-1, 1), ytr[ok])

# ---------------------------------------------------------------------------
# 4. Ajuste final y evaluación en el conjunto de prueba
# ---------------------------------------------------------------------------
pred_te, pred_te_cal = {}, {}
for m, est in final_models.items():
    log(f"Ajuste final — {m}")
    est.fit(Xtr, ytr)
    pred_te[m] = est.predict_proba(Xte)[:, 1]
    pred_te_cal[m] = recal[m].transform(pred_te[m])

joblib.dump(final_models[LR], config.MODELS / "regresion_logistica.joblib")
joblib.dump(final_models[RF], config.MODELS / "random_forest.joblib")
save_dnn_pipeline(final_models[DNN], str(config.MODELS / "dnn"))
plots.plot_training_history(final_models[DNN].named_steps["clf"].history_)
json.dump({"umbrales_youden": thresholds, "estrategia_desbalance_dnn": best_imb},
          open(config.MODELS / "umbrales.json", "w"), indent=2, ensure_ascii=False)

score_te = {IPSSM: T["IPSSM_SCORE"].to_numpy()[idx_te], IPSSR: T["IPSSR_SCORE"].to_numpy()[idx_te]}

# 4a. Métricas en toda la prueba (modelos de aprendizaje automático)
rows = []
for m in final_models:
    met = full_metrics(yte, pred_te[m], thresholds[m])
    met["Brier_recalibrado"] = full_metrics(yte, pred_te_cal[m], 0.5)["Brier"]
    rows.append({"modelo": m, "n": len(yte), **met})
pd.DataFrame(rows).round(4).to_csv(config.TABLES / "metricas_test_completo.csv", index=False)

# 4b. Comparación pareada con IPSS-M (pacientes de prueba con IPSS-M disponible)
ok = ~np.isnan(score_te[IPSSM]) & ~np.isnan(score_te[IPSSR])
yc = yte[ok]
comp_scores = {m: pred_te[m][ok] for m in final_models}
comp_scores[IPSSM] = score_te[IPSSM][ok]
comp_scores[IPSSR] = score_te[IPSSR][ok]
comp_probs = {m: pred_te_cal[m][ok] for m in final_models}
for m in (IPSSM, IPSSR):
    comp_probs[m] = ref_lr[m].predict_proba(comp_scores[m].reshape(-1, 1))[:, 1]

rows = []
for m, s in comp_scores.items():
    met = full_metrics(yc, s, thresholds[m])
    cal = full_metrics(yc, comp_probs[m], 0.5)
    met.update({"Brier": cal["Brier"], "calib_pendiente": cal["calib_pendiente"],
                "calib_intercepto": cal["calib_intercepto"]})
    rows.append({"modelo": m, "n": int(ok.sum()), **met})
# IPSS-M categórico con punto de corte clínico (Alto / Muy alto)
cat = T["IPSSM"].to_numpy()[idx_te][ok]
hi = np.isin(cat, ["High", "Very-High"]).astype(float)
rows.append({"modelo": "IPSS-M (Alto/Muy alto)", "n": int(ok.sum()),
             **{k: v for k, v in full_metrics(yc, hi, 0.5).items() if k not in ("AUC-ROC", "AUC-PR", "Brier", "calib_pendiente", "calib_intercepto")}})
comp = pd.DataFrame(rows)
ci_df, boot = bootstrap_ci(yc, comp_scores)
comp = comp.merge(ci_df, on="modelo", how="left")
comp.round(4).to_csv(config.TABLES / "metricas_test_comparacion_IPSSM.csv", index=False)

tests = []
for a, b in [(DNN, IPSSM), (DNN, RF), (DNN, LR), (RF, IPSSM), (LR, IPSSM), (IPSSM, IPSSR)]:
    auc_a, auc_b, p = delong_test(yc, comp_scores[a], comp_scores[b])
    tests.append({"comparación": f"{a} vs {b}", "AUC_a": auc_a, "AUC_b": auc_b,
                  "ΔAUC": auc_a - auc_b, "p_DeLong": p})
pd.DataFrame(tests).round(4).to_csv(config.TABLES / "comparacion_delong.csv", index=False)
pd.DataFrame([paired_bootstrap_diff(boot, a, b, "AUC-PR") for a, b in
              [(DNN, IPSSM), (DNN, RF), (DNN, LR)]]).round(4).to_csv(
    config.TABLES / "comparacion_bootstrap_aucpr.csv", index=False)

# Figuras de prueba
aucs = {m: roc_auc_score(yc, s) for m, s in comp_scores.items()}
aps = {m: average_precision_score(yc, s) for m, s in comp_scores.items()}
plots.plot_roc(yc, comp_scores, aucs=aucs)
plots.plot_pr(yc, comp_scores, aps=aps)
plots.plot_calibration(yc, comp_probs, title="Calibración (probabilidades recalibradas) — prueba")
plots.plot_calibration(yte, pred_te, name="calibracion_test_sin_recalibrar",
                       title="Calibración sin recalibrar — prueba")
plots.plot_confusion(confusion_matrix(yte, (pred_te[DNN] >= thresholds[DNN]).astype(int)),
                     f"{DNN}\numbral de Youden = {thresholds[DNN]:.2f}", "matriz_confusion_dnn")
plots.plot_confusion(confusion_matrix(yc, hi.astype(int)), "IPSS-M Alto/Muy alto",
                     "matriz_confusion_ipssm")

pd.DataFrame({"PATIENT_ID": T["PATIENT_ID"].to_numpy()[idx_te], "AML_T": yte,
              **{f"p_{m}": pred_te[m] for m in final_models},
              **{f"p_recal_{m}": pred_te_cal[m] for m in final_models},
              "IPSSM": T["IPSSM"].to_numpy()[idx_te], "IPSSM_SCORE": score_te[IPSSM],
              "AML_T_MONTHS": T["AML_T_MONTHS"].to_numpy()[idx_te],
              "DEATH_WITHOUT_AML": T["DEATH_WITHOUT_AML"].to_numpy()[idx_te]}).to_csv(
    config.TABLES / "predicciones_test.csv", index=False)

log("Resumen en prueba (pacientes con IPSS-M):")
print(comp[["modelo", "n", "AUC-ROC", "AUC-PR", "sensibilidad", "especificidad", "F1", "Brier"]]
      .round(3).to_string(index=False))
print(pd.DataFrame(tests).round(4).to_string(index=False))
log("Fin.")
