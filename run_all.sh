#!/usr/bin/env bash
# Ejecuta el pipeline completo del TFM de principio a fin.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/01_build_dataset.py
python scripts/02_eda_estadistica.py
python scripts/03_entrenamiento_evaluacion.py
python scripts/04_interpretabilidad.py
python scripts/05_estratificacion_sensibilidad.py
