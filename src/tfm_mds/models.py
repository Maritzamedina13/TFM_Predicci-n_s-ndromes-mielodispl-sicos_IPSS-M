"""Modelos: red neuronal profunda (TensorFlow/Keras) y modelos de referencia.

``KerasMLPClassifier`` envuelve una red neuronal densa (perceptrón multicapa) con la API
de scikit-learn (``fit`` / ``predict_proba``) para poder usarla dentro de ``Pipeline``,
``RandomizedSearchCV`` y ``cross_val_predict`` igual que los modelos clásicos.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

from . import config


class KerasMLPClassifier(ClassifierMixin, BaseEstimator):
    """Red neuronal profunda para clasificación binaria (progresión a LMA).

    Arquitectura: [Dense → BatchNorm → ReLU → Dropout] × n capas ocultas → Dense(1, sigmoide).
    Regularización: dropout, penalización L2 de los pesos y parada temprana (early stopping)
    sobre el AUC-PR de una partición interna de validación estratificada.
    Desbalance de clases: pesos de clase inversamente proporcionales a su frecuencia
    (``class_weight='balanced'``) u omisión (``None``) cuando se usa SMOTE.
    """

    def __init__(self, hidden_units=(64, 32), dropout=0.3, l2=1e-4, learning_rate=1e-3,
                 batch_size=64, epochs=300, patience=30, class_weight="balanced",
                 batch_norm=True, validation_fraction=0.15, random_state=config.SEED,
                 verbose=0):
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.l2 = l2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.class_weight = class_weight
        self.batch_norm = batch_norm
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    def _build(self, n_features: int):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers

        tf.get_logger().setLevel("ERROR")
        tf.keras.utils.set_random_seed(self.random_state)
        inputs = keras.Input(shape=(n_features,), name="entrada")
        x = inputs
        for i, units in enumerate(self.hidden_units):
            x = layers.Dense(units, kernel_regularizer=regularizers.l2(self.l2),
                             use_bias=not self.batch_norm, name=f"densa_{i + 1}")(x)
            if self.batch_norm:
                x = layers.BatchNormalization(name=f"batchnorm_{i + 1}")(x)
            x = layers.Activation("relu", name=f"relu_{i + 1}")(x)
            x = layers.Dropout(self.dropout, name=f"dropout_{i + 1}")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="probabilidad_LMA")(x)
        model = keras.Model(inputs, outputs, name="DNN_progresion_LMA")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(curve="PR", name="auc_pr"),
                     keras.metrics.AUC(curve="ROC", name="auc_roc")],
        )
        return model

    def fit(self, X, y):
        from tensorflow import keras

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y).astype("float32").ravel()
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.validation_fraction, stratify=y, random_state=self.random_state)
        cw = None
        if self.class_weight == "balanced":
            w = compute_class_weight("balanced", classes=np.array([0.0, 1.0]), y=y_tr)
            cw = {0: float(w[0]), 1: float(w[1])}

        self.model_ = self._build(X.shape[1])
        stop = keras.callbacks.EarlyStopping(monitor="val_auc_pr", mode="max",
                                             patience=self.patience, restore_best_weights=True)
        reduce = keras.callbacks.ReduceLROnPlateau(monitor="val_auc_pr", mode="max", factor=0.5,
                                                   patience=max(5, self.patience // 3),
                                                   min_lr=1e-5)
        self.history_ = self.model_.fit(
            X_tr, y_tr, validation_data=(X_val, y_val), epochs=self.epochs,
            batch_size=self.batch_size, class_weight=cw, callbacks=[stop, reduce],
            verbose=self.verbose,
        ).history
        self.best_epoch_ = int(np.argmax(self.history_["val_auc_pr"]) + 1)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype="float32")
        p = self.model_.predict(X, verbose=0, batch_size=4096).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags


# ----------------------------------------------------------------------
# Fábricas de pipelines (preprocesado + modelo)
# ----------------------------------------------------------------------
def make_logistic(pre) -> Pipeline:
    clf = LogisticRegression(class_weight="balanced", max_iter=5000, solver="lbfgs")
    return Pipeline([("pre", pre), ("clf", clf)])


def make_random_forest(pre) -> Pipeline:
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                 random_state=config.SEED, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


def make_dnn(pre, imbalance: str = "class_weight", **params):
    """``imbalance``: 'class_weight' (por defecto), 'smote' o 'none'."""
    if imbalance == "smote":
        clf = KerasMLPClassifier(class_weight=None, **params)
        return ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=config.SEED)), ("clf", clf)])
    cw = "balanced" if imbalance == "class_weight" else None
    return Pipeline([("pre", pre), ("clf", KerasMLPClassifier(class_weight=cw, **params))])


# Espacios de búsqueda de hiperparámetros
PARAM_GRID_LR = {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}
PARAM_GRID_RF = {
    "clf__max_depth": [None, 6, 10],
    "clf__min_samples_leaf": [1, 5, 10, 20],
    "clf__max_features": ["sqrt", 0.3],
}
PARAM_DIST_DNN = {
    "clf__hidden_units": [(32,), (64, 32), (128, 64), (128, 64, 32), (256, 128)],
    "clf__dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "clf__l2": [1e-5, 1e-4, 1e-3],
    "clf__learning_rate": [3e-4, 1e-3, 3e-3],
    "clf__batch_size": [32, 64, 128],
}


# ----------------------------------------------------------------------
# Persistencia del pipeline con red neuronal
# ----------------------------------------------------------------------
def save_dnn_pipeline(pipe, path_prefix) -> None:
    """Guarda el pipeline: preprocesador (joblib) + red (formato nativo .keras)."""
    import copy

    import joblib

    clf = pipe.named_steps["clf"]
    clf.model_.save(f"{path_prefix}.keras")
    light = copy.copy(pipe)
    light.steps = [(n, (copy.copy(s) if n == "clf" else s)) for n, s in pipe.steps]
    light.named_steps["clf"].model_ = None
    joblib.dump(light, f"{path_prefix}_pipeline.joblib")


def load_dnn_pipeline(path_prefix):
    import joblib
    from tensorflow import keras

    pipe = joblib.load(f"{path_prefix}_pipeline.joblib")
    pipe.named_steps["clf"].model_ = keras.models.load_model(f"{path_prefix}.keras")
    return pipe

if os.environ.get("TFM_QUICK") == "1":
    PARAM_GRID_RF = {"clf__min_samples_leaf": [5, 10]}
    PARAM_DIST_DNN = {**PARAM_DIST_DNN, "clf__hidden_units": [(32,)]}
