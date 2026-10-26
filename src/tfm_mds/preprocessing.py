"""Preprocesamiento: imputación, transformación, normalización y codificación.

El preprocesador es un ``ColumnTransformer`` de scikit-learn que se integra en un
``Pipeline`` junto al modelo. Así, en cada partición de la validación cruzada y en el
ajuste final, la imputación y la normalización se aprenden **solo** con los datos de
entrenamiento y después se aplican a validación/prueba (evita la fuga de información).
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import LOG_TRANSFORM


def _log1p(x):
    return np.log1p(np.clip(x, 0, None))


def build_preprocessor(groups: dict[str, list[str]]) -> ColumnTransformer:
    numeric = groups["numeric"]
    log_cols = [c for c in numeric if c in LOG_TRANSFORM]
    lin_cols = [c for c in numeric if c not in LOG_TRANSFORM]

    num_log = Pipeline([
        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
        ("log", FunctionTransformer(_log1p, feature_names_out="one-to-one")),
        ("escalar", StandardScaler()),
    ])
    num_lin = Pipeline([
        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
        ("escalar", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    binary = SimpleImputer(strategy="constant", fill_value=0)

    pre = ColumnTransformer(
        [
            ("num_log", num_log, log_cols),
            ("num", num_lin, lin_cols),
            ("cat", cat, groups["categorical"]),
            ("bin", binary, groups["binary"]),
        ],
        verbose_feature_names_out=False,
    )
    return pre
