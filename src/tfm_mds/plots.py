"""Figuras del TFM (estilo homogéneo, paleta categórica validada para daltonismo)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from . import config  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
        "lines.linewidth": 2,
    })


def save(fig, name):
    fig.savefig(config.FIGURES / f"{name}.png")
    plt.close(fig)


def _color(name, i):
    return config.MODEL_COLORS.get(name, config.PALETTE[i % len(config.PALETTE)])


def plot_roc(y, preds: dict, name="roc_test", title="Curvas ROC — conjunto de prueba", aucs=None):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2)
    for i, (m, p) in enumerate(preds.items()):
        fpr, tpr, _ = roc_curve(y[m] if isinstance(y, dict) else y, p)
        lab = f"{m} (AUC={aucs[m]:.3f})" if aucs else m
        ax.plot(fpr, tpr, color=_color(m, i), label=lab)
    ax.set(xlabel="1 − Especificidad", ylabel="Sensibilidad", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right", fontsize=8)
    save(fig, name)


def plot_pr(y, preds: dict, name="pr_test", title="Curvas precisión-sensibilidad — prueba", aps=None):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    base = None
    for i, (m, p) in enumerate(preds.items()):
        yy = y[m] if isinstance(y, dict) else y
        prec, rec, _ = precision_recall_curve(yy, p)
        lab = f"{m} (AP={aps[m]:.3f})" if aps else m
        ax.plot(rec, prec, color=_color(m, i), label=lab)
        base = np.mean(yy)
    ax.axhline(base, ls="--", lw=1, color=INK2)
    ax.text(0.99, base + 0.01, f"prevalencia = {base:.2f}", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set(xlabel="Sensibilidad (recall)", ylabel="Precisión (VPP)", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper right", fontsize=8)
    save(fig, name)


def plot_calibration(y, preds: dict, name="calibracion_test", title="Calibración — conjunto de prueba"):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2, label="Calibración perfecta")
    for i, (m, p) in enumerate(preds.items()):
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", ms=5, color=_color(m, i), label=m)
    ax.set(xlabel="Probabilidad predicha", ylabel="Proporción observada de LMA", title=title,
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=8)
    save(fig, name)


def plot_confusion(cm: np.ndarray, model: str, name: str):
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    labels = ["Sin LMA", "LMA"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else INK, fontsize=12, fontweight="bold")
    ax.set(xlabel="Predicción", ylabel="Real", title=model)
    save(fig, name)


def plot_cv_scores(df, metric, name):
    """Distribución de la métrica en los pliegues de validación cruzada por modelo."""
    models = list(df["modelo"].unique())
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, m in enumerate(models):
        vals = df.loc[df["modelo"] == m, metric].to_numpy()
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals)), vals,
                   s=40, color=_color(m, i), zorder=3, edgecolor="white", linewidth=1)
        ax.hlines(vals.mean(), i - 0.25, i + 0.25, color=INK, lw=2)
        ax.text(i + 0.28, vals.mean(), f"{vals.mean():.3f}", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(models)), models, rotation=12)
    ax.set(ylabel=metric, title=f"{metric} en validación cruzada ({config.CV_FOLDS} pliegues)")
    ax.set_xlim(-0.5, len(models) - 0.3)
    save(fig, name)


def plot_training_history(hist: dict, name="dnn_curvas_entrenamiento"):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ep = np.arange(1, len(hist["loss"]) + 1)
    axes[0].plot(ep, hist["loss"], color=config.PALETTE[0], label="Entrenamiento")
    axes[0].plot(ep, hist["val_loss"], color=config.PALETTE[1], label="Validación interna")
    axes[0].set(xlabel="Época", ylabel="Entropía cruzada binaria", title="Pérdida")
    axes[1].plot(ep, hist["auc_pr"], color=config.PALETTE[0], label="Entrenamiento")
    axes[1].plot(ep, hist["val_auc_pr"], color=config.PALETTE[1], label="Validación interna")
    best = int(np.argmax(hist["val_auc_pr"])) + 1
    for a in axes:
        a.axvline(best, ls=":", color=INK2, lw=1)
    axes[1].set(xlabel="Época", ylabel="AUC-PR", title=f"AUC-PR (mejor época = {best})")
    axes[0].legend(fontsize=8)
    save(fig, name)


def plot_barh(labels, values, title, xlabel, name, color=None, ci=None, ref=None):
    fig, ax = plt.subplots(figsize=(6.4, 0.28 * len(labels) + 1.2))
    y = np.arange(len(labels))
    ax.barh(y, values, color=color or config.PALETTE[0], height=0.7,
            xerr=ci, error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1, ls="--")
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(title=title, xlabel=xlabel)
    save(fig, name)
