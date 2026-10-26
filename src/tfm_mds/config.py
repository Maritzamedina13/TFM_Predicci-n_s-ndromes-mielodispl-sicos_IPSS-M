"""Configuración global del proyecto TFM.

Predicción pronóstica en síndromes mielodisplásicos mediante Deep Learning e IPSS-M.
Autora: Maritza Medina Giraldo — Máster Universitario en Bioinformática (UNIR).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
MODELS = RESULTS / "models"

for _p in (DATA_PROCESSED, FIGURES, TABLES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# Ficheros del estudio cBioPortal mds_iwg_2022
# https://www.cbioportal.org/study/clinicalData?id=mds_iwg_2022
FILE_PATIENT = DATA_RAW / "data_clinical_patient.txt"
FILE_SAMPLE = DATA_RAW / "data_clinical_sample.txt"
FILE_MUTATIONS = DATA_RAW / "data_mutations.txt"
FILE_CASES_IPSSM = DATA_RAW / "cases_ipssm.txt"

ANALYTIC_DATASET = DATA_PROCESSED / "dataset_analitico.csv"

# ---------------------------------------------------------------------------
# Reproducibilidad y diseño experimental
# ---------------------------------------------------------------------------
SEED = 2026
TEST_SIZE = 0.20          # conjunto de prueba independiente (estratificado)
CV_FOLDS = 5              # validación cruzada estratificada en entrenamiento
N_ITER_SEARCH_DNN = 12    # configuraciones evaluadas en la búsqueda de hiperparámetros
N_BOOTSTRAP = 2000        # remuestreos para IC 95 % en el conjunto de prueba

# Cohorte principal: muestras usadas para desarrollar el IPSS-M (n = 2.957)
USE_IPSSM_COHORT = True

# Horizonte (meses) del análisis de sensibilidad con desenlace a tiempo fijo
HORIZON_MONTHS = 36

# Frecuencia mínima (nº de pacientes) para incluir un gen como variable binaria
MIN_GENE_FREQ = 20

# Variables que NO se usan como predictores por fuga de información
# (contienen el desenlace o se calculan a partir de las mismas variables).
LEAKAGE_COLUMNS = [
    "OS_MONTHS", "OS_STATUS", "LFS_MONTHS", "LFS_STATUS",
    "IPSSR", "IPSSR_SCORE", "IPSSM", "IPSSM_SCORE",
]

# Orden de categorías IPSS-M / IPSS-R (de menor a mayor riesgo)
IPSSM_ORDER = ["Very-Low", "Low", "Moderate-Low", "Moderate-High", "High", "Very-High"]
IPSSR_ORDER = ["Very-Low", "Low", "Int", "High", "Very-High"]
CYTO_IPSSR_ORDER = ["Very-Good", "Good", "Int", "Poor", "Very-Poor"]

# Genes del IPSS-M (Bernard et al., NEJM Evidence 2022)
IPSSM_MAIN_GENES = [
    "TP53", "MLL_PTD", "FLT3", "ASXL1", "CBL", "DNMT3A", "ETV6", "EZH2", "IDH2",
    "KRAS", "NPM1", "NRAS", "RUNX1", "SF3B1", "SRSF2", "U2AF1",
]
IPSSM_RESIDUAL_GENES = [
    "BCOR", "BCORL1", "CEBPA", "ETNK1", "GATA2", "GNB1", "IDH1", "NF1", "PHF6",
    "PPM1D", "PRPF8", "PTPN11", "SETBP1", "STAG2", "WT1",
]

# Paleta (colores categóricos validados para daltonismo, orden fijo)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLORS = {
    "Red neuronal (DNN)": PALETTE[0],
    "Random forest": PALETTE[1],
    "Regresión logística": PALETTE[2],
    "IPSS-M (puntuación)": PALETTE[6],
    "IPSS-R (puntuación)": "#7a7a75",
}

# Modo rápido para pruebas del código (export TFM_QUICK=1): no usar para resultados.
import os as _os  # noqa: E402

if _os.environ.get("TFM_QUICK") == "1":
    CV_FOLDS, N_ITER_SEARCH_DNN, N_BOOTSTRAP = 3, 2, 50
