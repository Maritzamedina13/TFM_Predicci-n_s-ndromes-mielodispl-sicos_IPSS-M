"""Métricas de evaluación, intervalos de confianza y comparación de modelos."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)

from . import config


# ----------------------------------------------------------------------
# Umbral y métricas
# ----------------------------------------------------------------------
def youden_threshold(y, p) -> float:
    """Umbral que maximiza el índice de Youden (sensibilidad + especificidad − 1)."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def threshold_metrics(y, p, thr) -> dict:
    pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "umbral": thr,
        "sensibilidad": tp / (tp + fn) if tp + fn else np.nan,
        "especificidad": tn / (tn + fp) if tn + fp else np.nan,
        "VPP": tp / (tp + fp) if tp + fp else np.nan,
        "VPN": tn / (tn + fn) if tn + fn else np.nan,
        "F1": f1_score(y, pred, zero_division=0),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Pendiente e intercepto de calibración (regresión logística sobre logit(p))."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def full_metrics(y, p, thr) -> dict:
    out = {"AUC-ROC": roc_auc_score(y, p), "AUC-PR": average_precision_score(y, p)}
    if np.all((p >= 0) & (p <= 1)):
        out["Brier"] = brier_score_loss(y, p)
        slope, intercept = calibration_slope_intercept(y, p)
        out["calib_pendiente"], out["calib_intercepto"] = slope, intercept
    out.update(threshold_metrics(y, p, thr))
    return out


# ----------------------------------------------------------------------
# Recalibración (Platt) a partir de predicciones fuera de pliegue
# ----------------------------------------------------------------------
class PlattRecalibrator:
    def fit(self, p, y):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
        return self

    def transform(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return self.lr_.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


# ----------------------------------------------------------------------
# IC 95 % por bootstrap y prueba de DeLong
# ----------------------------------------------------------------------
def bootstrap_ci(y, preds: dict[str, np.ndarray], n_boot=config.N_BOOTSTRAP, seed=config.SEED):
    """IC 95 % percentil del AUC-ROC y AUC-PR para cada modelo (remuestreo estratificado)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    res = {m: {"AUC-ROC": [], "AUC-PR": []} for m in preds}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for m, p in preds.items():
            res[m]["AUC-ROC"].append(roc_auc_score(y[idx], p[idx]))
            res[m]["AUC-PR"].append(average_precision_score(y[idx], p[idx]))
    rows = []
    for m, d in res.items():
        row = {"modelo": m}
        for k, v in d.items():
            row[f"{k} IC2.5"], row[f"{k} IC97.5"] = np.percentile(v, [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows), res


def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong_test(y, p1, p2) -> tuple[float, float, float]:
    """Prueba de DeLong para comparar dos AUC correlacionadas (mismos pacientes).

    Devuelve (AUC1, AUC2, p-valor bilateral). Implementación de Sun & Xu (2014).
    """
    y = np.asarray(y)
    order = np.argsort(-y, kind="mergesort")
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    m = int(y.sum())
    n = len(y) - m
    tx = np.array([_midrank(r[:m]) for r in preds])
    ty = np.array([_midrank(r[m:]) for r in preds])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))


def paired_bootstrap_diff(boot: dict, a: str, b: str, metric: str) -> dict:
    d = np.asarray(boot[a][metric]) - np.asarray(boot[b][metric])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"comparación": f"{a} − {b}", "métrica": metric, "diferencia_media": d.mean(),
            "IC2.5": np.percentile(d, 2.5), "IC97.5": np.percentile(d, 97.5),
            "p_bootstrap": min(p, 1.0)}
