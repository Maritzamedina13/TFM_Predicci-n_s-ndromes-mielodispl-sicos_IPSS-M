# Predicción pronóstica en síndromes mielodisplásicos mediante Deep Learning e IPSS-M

**Trabajo Fin de Máster — Máster Universitario en Bioinformática (UNIR)**
Autora: **Maritza Medina Giraldo** · Tipo 3: trabajo original relacionado con bases de datos · Diciembre 2026

Modelo de aprendizaje profundo (TensorFlow/Keras) que estima el riesgo de **progresión a leucemia mieloide aguda (LMA)** en pacientes con síndromes mielodisplásicos (SMD) a partir de variables clínicas, citogenéticas y mutacionales de la cohorte pública **IWG-PM** de cBioPortal, comparado con regresión logística, random forest y el sistema pronóstico **IPSS-M**.

> 📄 La propuesta metodológica completa, la correspondencia con los objetivos y la estructura propuesta de la memoria están en [`docs/PROPUESTA_DESARROLLO.md`](docs/PROPUESTA_DESARROLLO.md).

---

## 1. Datos

| | |
|---|---|
| Fuente | cBioPortal — *Myelodysplastic Syndromes (MDS IWG, IPSSM, NEJM Evidence 2022)* — [`mds_iwg_2022`](https://www.cbioportal.org/study/clinicalData?id=mds_iwg_2022) |
| Cohorte | 2.957 pacientes de desarrollo del IPSS-M (de 3.323), panel de 152 genes mieloides |
| Licencia de los datos | [ODbL 1.0](http://opendatacommons.org/licenses/odbl/1.0/) (`data/raw/LICENSE`) |
| Cita | Bernard E, *et al.* Molecular International Prognostic Scoring System for Myelodysplastic Syndromes. *NEJM Evid.* 2022;1(7):EVIDoa2200008 |

Los ficheros originales de cBioPortal se incluyen en `data/raw/` para que el análisis sea reproducible.

### Variable objetivo
cBioPortal agrupa transformación y muerte en el evento de supervivencia libre de leucemia (LFS). La transformación a LMA se reconstruye combinando LFS y supervivencia global (`src/tfm_mds/target.py`): evento LFS con el paciente vivo o anterior a la última fecha de OS = **LMA**; evento LFS coincidente con la muerte = **muerte sin LMA**.

| Flujo de pacientes | n |
|---|---:|
| Cohorte IPSS-M | 2.957 |
| Excluidos (sin LFS o evento no clasificable) | 347 |
| **Cohorte analítica** | **2.610** |
| Transformación a LMA | 472 (18,1 %) |
| Sin transformación (874 muertes sin LMA) | 2.138 |

### Predictores (89 variables → 114 tras codificación)
Edad, sexo, tipo de SMD, OMS 2016, hemograma y blastos; citogenética (categoría IPSS-R, cariotipo complejo, del(5q), −7/del(7q), −17/del(17p), +8, del(20q), −Y, etc.); estado mutacional de 53 genes conductores (incluidos los 31 del IPSS-M), TP53 *multihit*, VAF de TP53, FLT3-ITD/TKD, KMT2A-PTD y carga mutacional.
**Excluidas por fuga de información:** tiempos/estados de OS y LFS y las puntuaciones/categorías IPSS-R e IPSS-M (usadas solo como comparadores).

## 2. Metodología

```
01_build_dataset ─► 02_eda_estadistica ─► 03_entrenamiento_evaluacion ─► 04_interpretabilidad
                                                                      └─► 05_estratificacion_sensibilidad
```

- **Preprocesamiento** dentro de un `Pipeline` (sin fuga): imputación por mediana con indicadores de faltante, log-transformación de recuentos, estandarización y *one-hot*.
- **Red neuronal**: perceptrón multicapa `Dense → BatchNorm → ReLU → Dropout`, salida sigmoide, Adam, entropía cruzada binaria, L2, *early stopping* sobre AUC-PR y reducción de la tasa de aprendizaje. Búsqueda aleatoria de 12 configuraciones con CV-5.
- **Desbalance**: comparación sin corrección / pesos de clase / SMOTE.
- **Referencias**: regresión logística (L2) y random forest (rejilla de hiperparámetros), IPSS-M e IPSS-R.
- **Validación**: partición estratificada 80/20 + CV estratificada de 5 pliegues en entrenamiento; umbral de Youden y recalibración de Platt aprendidos con predicciones fuera de pliegue.
- **Métricas**: AUC-ROC, AUC-PR, sensibilidad, especificidad, VPP, VPN, F1, Brier y calibración; IC 95 % por *bootstrap* (2.000), prueba de DeLong, índice C para el tiempo hasta LMA.
- **Interpretabilidad**: SHAP (permutación para la red, TreeExplainer para random forest) y odds ratios.

## 3. Resultados principales (conjunto de prueba, n = 470 con IPSS-M disponible, 85 LMA)

| Modelo | AUC-ROC (IC 95 %) | AUC-PR (IC 95 %) | Sensib. | Especif. | F1 |
|---|---|---|---:|---:|---:|
| Random forest | **0,776** (0,726–0,822) | 0,445 (0,363–0,538) | 0,765 | 0,652 | 0,458 |
| Regresión logística | 0,765 (0,709–0,814) | **0,452** (0,371–0,564) | 0,718 | 0,634 | 0,425 |
| **Red neuronal (DNN)** | 0,754 (0,699–0,805) | 0,442 (0,355–0,538) | 0,682 | 0,683 | 0,438 |
| IPSS-M (puntuación) | 0,725 (0,664–0,784) | 0,370 (0,304–0,462) | 0,765 | 0,584 | 0,419 |
| IPSS-R (puntuación) | 0,655 (0,594–0,714) | 0,272 (0,228–0,343) | 0,600 | 0,655 | 0,379 |

- **Red neuronal vs IPSS-M:** ΔAUC-ROC = +0,028 (DeLong p = 0,16); ΔAUC-PR = +0,071 (IC 95 % −0,001 a 0,144; p = 0,055). Mejora numérica, no significativa.
- **Red neuronal vs modelos sencillos:** sin diferencias significativas (vs RF p = 0,16; vs RL p = 0,50). El random forest supera al IPSS-M (p = 0,018). Coherente con la literatura sobre datos tabulares (Shwartz-Ziv y Armon, 2022).
- **Validación cruzada (entrenamiento):** AUC-ROC DNN 0,750 ± 0,016; RF 0,771 ± 0,030; RL 0,765 ± 0,022; IPSS-M 0,751 ± 0,023.
- **Desbalance:** los pesos de clase no mejoran la discriminación de la red (AUC-PR 0,369 vs 0,373 sin corrección) y SMOTE la empeora (0,345); el desbalance se gestiona con el umbral de Youden.
- **Estratificación:** el grupo de *alto riesgo* de la red (13 % de los pacientes) tiene una incidencia acumulada de LMA a 3 años del **51,5 %**, frente al 39,8 % del grupo Alto/Muy alto del IPSS-M (33 % de los pacientes); grupo bajo: 9,7 % vs 7,7 %.
- **Interpretabilidad:** blastos en médula ósea, número de mutaciones conductoras, genes residuales del IPSS-M, mutaciones de TP53, riesgo citogenético y SF3B1/TET2 (protectores) dominan la predicción, en concordancia con la literatura. Los rankings de la red y del random forest se correlacionan (ρ de Spearman = 0,84).
- **Sensibilidad (LMA a 36 meses):** mismas conclusiones (AUC-ROC DNN 0,745; RF 0,774; RL 0,760; IPSS-M 0,745).

> ⚠️ El IPSS-M se derivó con esta misma cohorte, por lo que su rendimiento aquí es optimista; cualquier ventaja de los modelos debe confirmarse en una cohorte externa.

Todas las tablas están en `results/tables/` y las figuras en `results/figures/`. El cuaderno [`notebooks/TFM_resultados.ipynb`](notebooks/TFM_resultados.ipynb) las reúne en orden de lectura.

## 4. Estructura del repositorio

```
├── data/
│   ├── raw/                 # ficheros originales de cBioPortal (mds_iwg_2022, ODbL)
│   └── processed/           # dataset_analitico.csv (generado)
├── docs/PROPUESTA_DESARROLLO.md
├── notebooks/TFM_resultados.ipynb
├── results/
│   ├── figures/             # figuras (PNG, 300 ppp)
│   ├── tables/              # tablas de resultados (CSV)
│   └── models/              # modelos entrenados, hiperparámetros, umbrales, SHAP
├── scripts/                 # pasos 01–05 del pipeline
├── src/tfm_mds/             # paquete: carga, desenlace, variables, preprocesado,
│                            # modelos, evaluación y figuras
├── requirements.txt
└── run_all.sh
```

## 5. Reproducir el análisis

Requiere Python 3.10–3.13.

```bash
git clone https://github.com/Maritzamedina13/TFM_Predicci-n_s-ndromes-mielodispl-sicos_IPSS-M.git
cd TFM_Predicci-n_s-ndromes-mielodispl-sicos_IPSS-M
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash run_all.sh                                         # ≈ 35 min en CPU
```

O paso a paso: `python scripts/01_build_dataset.py`, `02_eda_estadistica.py`, `03_entrenamiento_evaluacion.py`, `04_interpretabilidad.py`, `05_estratificacion_sensibilidad.py`. Los parámetros (semilla 2026, pliegues, horizonte, etc.) están en `src/tfm_mds/config.py`. Con `TFM_QUICK=1` se ejecuta una versión reducida para comprobar el código (no válida para resultados).

## 6. Limitaciones

- Desenlace de LMA reconstruido a partir de LFS/OS (una LMA diagnosticada el día del fallecimiento se clasifica como muerte sin LMA).
- Desenlace binario con seguimiento heterogéneo (mediana ≈ 20 meses); se complementa con índice C y análisis a horizonte fijo.
- Validación interna en una sola cohorte, la misma con la que se construyó el IPSS-M; sin información de tratamientos posteriores.
- Los indicadores de valor faltante (p. ej., monocitos) aparecen entre las variables influyentes y pueden reflejar diferencias entre centros más que biología.

## 7. Referencias clave

1. Bernard E, *et al.* Molecular International Prognostic Scoring System for myelodysplastic syndromes. *NEJM Evid.* 2022;1(7).
2. Cerami E, *et al.* The cBio cancer genomics portal. *Cancer Discov.* 2012;2(5):401-4.
3. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *NeurIPS* 2017.
4. Shwartz-Ziv R, Armon A. Tabular data: deep learning is not all you need. *Inf Fusion.* 2022;81:84-90.
5. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot on imbalanced datasets. *PLoS One.* 2015;10(3):e0118432.
