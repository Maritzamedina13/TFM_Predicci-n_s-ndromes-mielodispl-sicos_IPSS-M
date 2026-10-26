"""Ingeniería de variables clínicas, citogenéticas y mutacionales.

Todas las variables corresponden a información disponible en el momento de la toma de
muestra (diagnóstico). Las variables de seguimiento y las puntuaciones IPSS-R/IPSS-M se
conservan solo como columnas de referencia, nunca como predictores.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# Citogenética
# ---------------------------------------------------------------------------
_MONO = r"(?:^|,)-{chrom}(?=$|,|\[)"
_TRI = r"(?:^|,)\+{chrom}(?=$|,|\[)"

KARYOTYPE_PATTERNS = {
    "CYTO_DEL5Q": [r"del\(5\)\(q", _MONO.format(chrom="5")],
    "CYTO_DEL7_7Q": [r"del\(7\)\(q", _MONO.format(chrom="7")],
    "CYTO_DEL17_17P": [r"del\(17\)\(p", r"i\(17\)\(q", _MONO.format(chrom="17")],
    "CYTO_TRIS8": [_TRI.format(chrom="8")],
    "CYTO_DEL20Q": [r"del\(20\)\(q"],
    "CYTO_MINUS_Y": [_MONO.format(chrom="y")],
    "CYTO_DEL11Q": [r"del\(11\)\(q"],
    "CYTO_DEL12P": [r"del\(12\)\(p", r"t\(12;"],
    "CYTO_3Q26": [r"inv\(3\)", r"t\(3;3\)", r"del\(3\)\(q2[16]"],
}
CNACS_EQUIVALENT = {
    "CYTO_DEL5Q": ["5q", "5"],
    "CYTO_DEL7_7Q": ["7q", "7"],
    "CYTO_DEL17_17P": ["17p", "17"],
    "CYTO_DEL20Q": ["20q", "20"],
    "CYTO_DEL11Q": ["11q"],
    "CYTO_DEL12P": ["12p"],
}


def _is_normal_karyotype(k: str) -> bool:
    clones = [c.strip() for c in k.split("/") if c.strip()]
    return all(re.fullmatch(r"46,x[xy](?:\[\d+\])?", c) for c in clones)


def cytogenetic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Alteraciones citogenéticas binarias a partir del cariotipo y de CNACS (secuenciación)."""
    kary = df["CYTOGENETICS"].fillna("").str.lower().str.replace(" ", "", regex=False)
    arm_loss = df["CNACS_CHRARM_LOSS"].fillna("").str.split(",")
    feats = pd.DataFrame(index=df.index)
    for name, pats in KARYOTYPE_PATTERNS.items():
        hit = kary.apply(lambda k: any(re.search(p, k) for p in pats))
        if name in CNACS_EQUIVALENT:
            arms = set(CNACS_EQUIVALENT[name])
            hit = hit | arm_loss.apply(lambda lst: bool(arms.intersection(lst)))
        feats[name] = hit.astype(int)
    feats["CYTO_NORMAL"] = kary.apply(lambda k: int(bool(k) and _is_normal_karyotype(k)))
    feats["CYTO_MISSING"] = (kary == "").astype(int)
    feats["COMPLEX_KARYOTYPE"] = (df["COMPLEX_KARYOTYPE"] == "complex").astype(int)
    feats["TP53_LOCUS_LOH"] = (df["CHR_TP53_LOCUS"].fillna("normal") != "normal").astype(int)
    feats["FLT3_ITD"] = df["FLT3_ITD"].astype(str).str.upper().eq("TRUE").astype(int)
    feats["MLL_PTD"] = df["MLL_PTD"].astype(str).str.upper().eq("TRUE").astype(int)
    return feats


# ---------------------------------------------------------------------------
# Mutaciones
# ---------------------------------------------------------------------------
def mutation_features(df: pd.DataFrame, maf: pd.DataFrame, min_freq: int) -> pd.DataFrame:
    """Matriz binaria gen-paciente (solo mutaciones conductoras/oncogénicas) y resúmenes.

    Se emplean las mutaciones anotadas por cBioPortal como ``Putative_Driver``
    (patogénicas o probablemente patogénicas), igual que el IPSS-M considera solo
    mutaciones oncogénicas.
    """
    samples = df["SAMPLE_ID"]
    drv = maf[(maf["cbp_driver"] == "Putative_Driver") & maf["Tumor_Sample_Barcode"].isin(samples)]

    gene_mat = (
        drv.assign(v=1)
        .pivot_table(index="Tumor_Sample_Barcode", columns="Hugo_Symbol", values="v", aggfunc="max")
        .reindex(samples)
        .fillna(0)
        .astype(int)
    )
    freq = gene_mat.sum()
    keep = set(freq[freq >= min_freq].index)
    keep |= {g for g in config.IPSSM_MAIN_GENES + config.IPSSM_RESIDUAL_GENES if g in gene_mat.columns}
    genes = sorted(keep)
    out = gene_mat[genes].add_prefix("MUT_")
    out.index = df.index

    # Resúmenes de carga mutacional
    n_mut = drv.groupby("Tumor_Sample_Barcode").size().reindex(samples).fillna(0).to_numpy()
    out["N_DRIVER_MUTATIONS"] = n_mut
    out["N_DRIVER_GENES"] = gene_mat.sum(axis=1).to_numpy()

    # TP53: número de mutaciones, VAF máxima y estado multihit (definición IPSS-M)
    tp = drv[drv["Hugo_Symbol"] == "TP53"]
    tp_n = tp.groupby("Tumor_Sample_Barcode").size().reindex(samples).fillna(0).to_numpy()
    tp_vaf = tp.groupby("Tumor_Sample_Barcode")["VAF"].max().reindex(samples).fillna(0).to_numpy()
    out["TP53_N_MUT"] = tp_n
    out["TP53_MAX_VAF"] = tp_vaf
    loh = (df["CHR_TP53_LOCUS"].fillna("normal") != "normal").to_numpy()
    del17p = df["CNACS_CHRARM_LOSS"].fillna("").str.contains("17p").to_numpy()
    out["TP53_MULTIHIT"] = ((tp_n >= 2) | ((tp_n >= 1) & (loh | del17p | (tp_vaf > 0.55)))).astype(int)

    # FLT3 (ITD o mutación puntual) como en el IPSS-M
    flt3_mut = gene_mat["FLT3"].to_numpy() if "FLT3" in gene_mat.columns else 0
    out["FLT3_ANY"] = ((df["FLT3_ITD"].astype(str).str.upper() == "TRUE").to_numpy() | (flt3_mut == 1)).astype(int)

    # Nº de genes residuales del IPSS-M mutados (truncado en 2)
    res = [g for g in config.IPSSM_RESIDUAL_GENES if g in gene_mat.columns]
    out["IPSSM_NRES"] = np.minimum(gene_mat[res].sum(axis=1).to_numpy(), 2)

    # Máxima VAF de cualquier mutación conductora (tamaño clonal)
    vmax = drv.groupby("Tumor_Sample_Barcode")["VAF"].max().reindex(samples).fillna(0).to_numpy()
    out["MAX_DRIVER_VAF"] = vmax
    return out


# ---------------------------------------------------------------------------
# Conjunto analítico
# ---------------------------------------------------------------------------
NUMERIC_CLINICAL = [
    "AGE", "BM_BLAST", "PB_BLAST", "HB", "PLT", "WBC", "ANC", "MONOCYTES", "TMB_NONSYNONYMOUS",
]
LOG_TRANSFORM = ["PLT", "WBC", "ANC", "MONOCYTES", "TMB_NONSYNONYMOUS"]
CATEGORICAL = ["SEX", "MDS_TYPE", "WHO_2016"]
ORDINAL = ["CYTO_IPSSR_ORD"]


def build_feature_table(df: pd.DataFrame, maf: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla paciente × variables (predictores + desenlace + referencias)."""
    base = df[["PATIENT_ID", "SAMPLE_ID"] + NUMERIC_CLINICAL + CATEGORICAL].copy()
    base["MDS_TYPE"] = base["MDS_TYPE"].replace({"s_t_MDS": "secondary_therapy"})
    # WHO 2016 con categorías poco frecuentes agrupadas
    who_counts = base["WHO_2016"].value_counts()
    rare = who_counts[who_counts < 30].index
    base.loc[base["WHO_2016"].isin(rare), "WHO_2016"] = "Other"
    cyto_map = {c: i for i, c in enumerate(config.CYTO_IPSSR_ORDER)}
    base["CYTO_IPSSR_ORD"] = df["CYTO_IPSSR"].map(cyto_map)

    cyto = cytogenetic_features(df)
    muts = mutation_features(df, maf, config.MIN_GENE_FREQ)
    refs = df[["AML_T", "AML_T_MONTHS", "DEATH_WITHOUT_AML", "IPSSM", "IPSSM_SCORE", "IPSSR",
               "IPSSR_SCORE", "OS_MONTHS", "OS_STATUS", "LFS_MONTHS", "LFS_STATUS"]]
    return pd.concat([base, cyto, muts, refs], axis=1)


def predictor_columns(table: pd.DataFrame) -> dict[str, list[str]]:
    """Devuelve los grupos de predictores (excluye identificadores, desenlace y fuga)."""
    excluded = set(config.LEAKAGE_COLUMNS) | {
        "PATIENT_ID", "SAMPLE_ID", "AML_T", "AML_T_MONTHS", "DEATH_WITHOUT_AML",
    }
    numeric = [c for c in NUMERIC_CLINICAL + ORDINAL if c in table.columns]
    numeric += ["N_DRIVER_MUTATIONS", "N_DRIVER_GENES", "TP53_N_MUT", "TP53_MAX_VAF",
                "IPSSM_NRES", "MAX_DRIVER_VAF"]
    categorical = [c for c in CATEGORICAL if c in table.columns]
    binary = [c for c in table.columns
              if c not in excluded and c not in numeric and c not in categorical
              and (c.startswith(("MUT_", "CYTO_")) or c in
                   {"COMPLEX_KARYOTYPE", "TP53_LOCUS_LOH", "FLT3_ITD", "MLL_PTD",
                    "TP53_MULTIHIT", "FLT3_ANY"})]
    assert not (set(numeric + categorical + binary) & excluded), "Fuga de información detectada"
    return {"numeric": numeric, "categorical": categorical, "binary": binary}
