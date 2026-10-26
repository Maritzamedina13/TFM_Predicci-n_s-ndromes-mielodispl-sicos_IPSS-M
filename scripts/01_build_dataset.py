"""Paso 1 — Construcción del conjunto de datos analítico (Objetivo específico 1).

Une datos clínicos, citogenéticos y mutacionales de la cohorte IWG-PM (cBioPortal
``mds_iwg_2022``), define la variable objetivo (transformación a LMA) y excluye las
variables con fuga de información.

Uso:  python scripts/01_build_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from tfm_mds import config  # noqa: E402
from tfm_mds.data_loading import load_clinical, load_mutations  # noqa: E402
from tfm_mds.features import build_feature_table, predictor_columns  # noqa: E402
from tfm_mds.target import derive_aml_transformation  # noqa: E402


def main() -> None:
    clinical = load_clinical()
    maf = load_mutations()
    print(f"Pacientes en la cohorte IPSS-M: {len(clinical)}")

    clinical = derive_aml_transformation(clinical)
    flow = {
        "Cohorte IPSS-M (cBioPortal)": len(clinical),
        "Sin dato de LFS o evento ambiguo (excluidos)": int((~clinical["TARGET_VALID"]).sum()),
    }
    clinical = clinical[clinical["TARGET_VALID"]].reset_index(drop=True)
    flow["Cohorte analítica"] = len(clinical)
    flow["Transformación a LMA (AML_T = 1)"] = int(clinical["AML_T"].sum())
    flow["Sin transformación (AML_T = 0)"] = int((clinical["AML_T"] == 0).sum())
    flow["  · de ellos, muerte sin LMA"] = int(clinical["DEATH_WITHOUT_AML"].sum())

    table = build_feature_table(clinical, maf)
    groups = predictor_columns(table)
    table.to_csv(config.ANALYTIC_DATASET, index=False)

    pd.Series(flow, name="n").to_csv(config.TABLES / "flujo_pacientes.csv")
    rows = [(g, c) for g, cols in groups.items() for c in cols]
    pd.DataFrame(rows, columns=["tipo", "variable"]).to_csv(
        config.TABLES / "diccionario_predictores.csv", index=False)

    for k, v in flow.items():
        print(f"{k:<48}{v:>6}")
    print(f"Predictores: {sum(len(v) for v in groups.values())} "
          f"(numéricos={len(groups['numeric'])}, categóricos={len(groups['categorical'])}, "
          f"binarios={len(groups['binary'])})")
    print(f"Guardado: {config.ANALYTIC_DATASET}")


if __name__ == "__main__":
    main()
