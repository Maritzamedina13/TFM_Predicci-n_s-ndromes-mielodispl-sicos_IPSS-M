"""Carga de los ficheros del estudio cBioPortal ``mds_iwg_2022``."""
from __future__ import annotations

import pandas as pd

from . import config


def read_cbioportal_table(path) -> pd.DataFrame:
    """Lee una tabla clínica de cBioPortal (las 4 primeras líneas empiezan por '#')."""
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False)


def read_case_list(path) -> list[str]:
    """Devuelve los identificadores de muestra de un fichero ``case_lists``."""
    text = open(path, encoding="utf-8").read()
    ids = text.split("case_list_ids:")[1].split()
    return ids


def load_clinical() -> pd.DataFrame:
    """Une la información clínica a nivel de paciente y de muestra (1 muestra/paciente)."""
    patient = read_cbioportal_table(config.FILE_PATIENT)
    sample = read_cbioportal_table(config.FILE_SAMPLE)
    df = patient.merge(sample, on="PATIENT_ID", how="inner", validate="one_to_one")
    if config.USE_IPSSM_COHORT:
        ids = set(read_case_list(config.FILE_CASES_IPSSM))
        df = df[df["SAMPLE_ID"].isin(ids)].copy()
    return df.reset_index(drop=True)


def load_mutations() -> pd.DataFrame:
    """Carga el fichero MAF con las mutaciones somáticas (panel de 152 genes mieloides)."""
    cols = [
        "Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification",
        "HGVSp_Short", "t_ref_count", "t_alt_count", "cbp_driver",
    ]
    maf = pd.read_csv(config.FILE_MUTATIONS, sep="\t", usecols=cols, low_memory=False)
    maf["VAF"] = maf["t_alt_count"] / (maf["t_alt_count"] + maf["t_ref_count"])
    return maf
