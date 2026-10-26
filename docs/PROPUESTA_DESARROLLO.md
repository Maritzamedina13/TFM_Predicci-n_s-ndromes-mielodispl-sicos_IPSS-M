# Propuesta de desarrollo del TFM

**Predicción pronóstica en síndromes mielodisplásicos mediante Deep Learning e IPSS-M**
Maritza Medina Giraldo · Máster Universitario en Bioinformática (UNIR) · TFM Tipo 3 (trabajo original con bases de datos)

---

## 1. Pregunta de investigación e hipótesis

> ¿Puede una red neuronal profunda entrenada con los datos clínicos, citogenéticos y mutacionales de la cohorte IWG-PM predecir la progresión a leucemia mieloide aguda (LMA) igual o mejor que modelos más sencillos (regresión logística, random forest) y que el IPSS-M?

**Hipótesis:** la red neuronal puede aprovechar relaciones no lineales e interacciones entre variables (p. ej., blastos × citogenética, co-mutaciones) y mejorar la discriminación del riesgo frente al IPSS-M, que es un modelo de efectos fijos.

## 2. Fuente de datos

| Elemento | Detalle |
|---|---|
| Repositorio | cBioPortal — *Myelodysplastic Syndromes (MDS IWG, IPSSM, NEJM Evidence 2022)* |
| Identificador | `mds_iwg_2022` — <https://www.cbioportal.org/study/clinicalData?id=mds_iwg_2022> |
| Pacientes | 3.323 muestras sin tratamiento previo; 2.957 forman la cohorte de desarrollo del IPSS-M (`cases_ipssm.txt`) |
| Genómica | Panel dirigido de 152 genes mieloides (MAF, hg19), alteraciones de número de copia (CNACS), FLT3-ITD, KMT2A-PTD |
| Seguimiento | Supervivencia global (OS) y supervivencia libre de leucemia (LFS) |
| Licencia | ODC Open Database License (ODbL) — se cita la fuente y se mantiene abierta |
| Cita | Bernard E, et al. *NEJM Evidence* 2022;1(7):EVIDoa2200008 |

Ficheros usados: `data_clinical_patient.txt`, `data_clinical_sample.txt`, `data_mutations.txt`, `cases_ipssm.txt` (carpeta `data/raw/`).

## 3. Variable objetivo (desenlace)

cBioPortal no publica una columna exclusiva de transformación a LMA: el evento de LFS combina **transformación o muerte** (`1:Transformed/Deceased`). La transformación se reconstruye combinando LFS y OS (`src/tfm_mds/target.py`):

| Situación | Clasificación |
|---|---|
| Evento LFS y paciente vivo (`OS_STATUS = 0`) | **LMA = 1** |
| Evento LFS antes del fin de OS (`LFS_MONTHS < OS_MONTHS`) | **LMA = 1** |
| Libre de leucemia al final del seguimiento | LMA = 0 |
| Evento LFS en la misma fecha que la muerte | LMA = 0 (muerte sin LMA, evento competitivo) |
| Sin LFS, o evento LFS sin información de OS | Excluido (no clasificable) |

Resultado: **2.610 pacientes, 472 transformaciones a LMA (18,1 %)** — conjunto desbalanceado. Se añade un análisis de sensibilidad con desenlace a horizonte fijo (LMA en ≤ 36 meses, excluyendo censurados antes del horizonte).

## 4. Variables predictoras (información disponible al diagnóstico)

| Grupo | Variables |
|---|---|
| Demográficas | Edad, sexo |
| Clínicas / hematológicas | Blastos en médula ósea y sangre periférica, hemoglobina, plaquetas, leucocitos, neutrófilos, monocitos |
| Diagnósticas | Tipo de SMD (primario / secundario-relacionado con tratamiento), clasificación OMS 2016 |
| Citogenéticas | Categoría citogenética IPSS-R (ordinal), cariotipo complejo, del(5q), −7/del(7q), −17/del(17p), +8, del(20q), −Y, del(11q), del(12p), inv(3)/t(3;3), cariotipo normal, cariotipo no disponible (cariotipo convencional + pérdidas detectadas por CNACS) |
| Moleculares | Estado mutacional (solo mutaciones conductoras/oncogénicas) de 53 genes: los 31 del IPSS-M + genes mutados en ≥ 20 pacientes; FLT3-ITD/TKD, KMT2A-PTD; TP53: nº de mutaciones, VAF máxima, LOH del locus y estado *multihit*; nº de genes residuales IPSS-M; carga mutacional (TMB, nº de mutaciones y genes conductores, VAF máxima) |

**Excluidas por fuga de información:** tiempos y estados de OS y LFS, categorías y puntuaciones IPSS-R e IPSS-M (se reservan como comparadores). `predictor_columns()` comprueba automáticamente que ninguna de ellas entra al modelo. En total, 89 variables originales → 114 tras codificación.

## 5. Metodología y correspondencia con los objetivos específicos

| Objetivo específico | Método | Script |
|---|---|---|
| **OE1.** Delimitar el conjunto de variables y la variable objetivo sin fuga de información | Unión clínica + MAF + CNACS, ingeniería de variables, reconstrucción del desenlace, exclusión de variables con fuga, diagrama de flujo | `01_build_dataset.py` |
| **OE1–2.** Exploración y relación de mutaciones con la progresión | Descriptiva por desenlace (tabla 1), faltantes, Mann-Whitney U, Fisher exacto con OR (IC 95 %), corrección FDR de Benjamini-Hochberg; incidencia acumulada de LMA por IPSS-M (Aalen-Johansen con riesgo competitivo) y LFS (Kaplan-Meier, log-rank) | `02_eda_estadistica.py` |
| **OE2.** Construir la red neuronal en TensorFlow/Keras con preprocesamiento, desbalance y ajuste de hiperparámetros | Perceptrón multicapa (Dense → BatchNorm → ReLU → Dropout), L2, Adam, early stopping sobre AUC-PR; imputación por mediana + indicadores de faltante, log-transformación, estandarización y one-hot ajustados **solo** en entrenamiento (Pipeline); comparación sin corrección / pesos de clase / SMOTE; búsqueda aleatoria de 12 configuraciones con CV-5 | `03_entrenamiento_evaluacion.py` |
| **OE3.** Evaluar el desempeño y compararlo con modelos de referencia e IPSS-M | Partición 80/20 estratificada; CV estratificada de 5 pliegues; AUC-ROC, AUC-PR, sensibilidad, especificidad, VPP, VPN, F1, Brier, pendiente/intercepto de calibración; umbral de Youden fijado en entrenamiento; IC 95 % bootstrap (2.000 remuestreos); DeLong y bootstrap pareado frente a IPSS-M; índice C para el tiempo hasta LMA; grupos de riesgo y tabla de reclasificación | `03_…`, `05_estratificacion_sensibilidad.py` |
| **OE4.** Interpretar la contribución de las variables | SHAP (permutación) para la red, TreeExplainer para random forest, odds ratios de la regresión logística; comparación de rankings y contraste con la literatura (blastos, TP53 multihit, citogenética de alto riesgo, RUNX1, STAG2, SF3B1…) | `04_interpretabilidad.py` |

### Control de los tres retos metodológicos del planteamiento

1. **Fuga de información:** exclusión explícita de OS/LFS e IPSS-R/IPSS-M; preprocesamiento dentro del Pipeline; conjunto de prueba aislado; umbral y recalibración aprendidos con predicciones fuera de pliegue del entrenamiento.
2. **Desbalance de clases (18 % de eventos):** AUC-PR como métrica principal de ajuste, pesos de clase frente a SMOTE, partición y CV estratificadas.
3. **Sobreajuste:** dropout, penalización L2, batch normalization, parada temprana, reducción de la tasa de aprendizaje, búsqueda de hiperparámetros con CV y comparación con modelos más sencillos.

## 6. Estructura propuesta de la memoria (capítulos pendientes)

Los capítulos 1–5 (introducción, justificación, planteamiento, objetivos y marco teórico) ya están en el borrador. Se proponen los siguientes, alimentados directamente por las salidas de `results/`:

**6. Metodología**
- 6.1 Diseño del estudio y fuente de datos (tabla del apartado 2)
- 6.2 Población y definición del desenlace (`flujo_pacientes.csv`; diagrama de flujo)
- 6.3 Variables y preprocesamiento (`diccionario_predictores.csv`)
- 6.4 Análisis estadístico exploratorio
- 6.5 Modelos: arquitectura de la red, modelos de referencia, hiperparámetros (`busqueda_hiperparametros_*.csv`, `mejores_hiperparametros.json`)
- 6.6 Estrategia de validación y métricas
- 6.7 Interpretabilidad (SHAP)
- 6.8 Herramientas y reproducibilidad (`requirements.txt`, repositorio GitHub, semilla 2026)

**7. Resultados**
- 7.1 Descripción de la cohorte (`tabla1_descriptiva.csv`, `eda_*.png`)
- 7.2 Factores asociados a la progresión (`forest_univariante.png`, `frecuencia_mutaciones.png`, `incidencia_LMA_por_IPSSM.png`)
- 7.3 Ajuste de la red y estrategia de desbalance (`dnn_estrategias_desbalance.csv`, `dnn_curvas_entrenamiento.png`)
- 7.4 Validación cruzada (`cv_resumen.csv`, `cv_auc_*.png`)
- 7.5 Desempeño en prueba y comparación con IPSS-M (`metricas_test_*.csv`, `comparacion_delong.csv`, `roc_test.png`, `pr_test.png`, `calibracion_*.png`, matrices de confusión)
- 7.6 Estratificación y reclasificación (`estratificacion_*`, `reclasificacion_*`, `indice_C_tiempo_LMA.csv`)
- 7.7 Interpretabilidad (`shap_*.png`, `importancia_variables_comparada.csv`)
- 7.8 Análisis de sensibilidad (`sensibilidad_horizonte_36m.csv`)

**8. Discusión** — comparación con IPSS-M, AIPSS-MDS (Mosquera Orgueira 2023), Nazha 2021 y Bersanelli 2021; coherencia biológica de las variables (TP53 multihit, RUNX1, STAG2, ASXL1, SF3B1 protector, blastos); valor añadido real de la red frente a modelos sencillos en datos tabulares (Shwartz-Ziv 2022); limitaciones.

**9. Conclusiones y líneas futuras** — validación externa (p. ej., cohortes de vida real, Sauta 2023), modelos de supervivencia con riesgos competitivos (DeepHit, Fine-Gray), modelos tabulares específicos (TabNet, FT-Transformer).

## 7. Limitaciones que deben declararse

- La transformación a LMA se **reconstruye** a partir de LFS y OS; una LMA diagnosticada el mismo día del fallecimiento se clasifica como muerte sin LMA.
- El desenlace binario ignora el tiempo de seguimiento: pacientes censurados con seguimiento corto pueden progresar más tarde (por eso se incluye el análisis a horizonte fijo y el índice C).
- El IPSS-M se desarrolló **con esta misma cohorte**, por lo que su rendimiento aquí es optimista; la comparación favorece al IPSS-M y cualquier ventaja de los modelos debe confirmarse en validación externa.
- Validación interna en una única cohorte; sin datos de tratamiento posterior (agentes hipometilantes, trasplante), que modifican el riesgo de progresión.
- SMOTE se aplica sobre variables ya codificadas (binarias interpoladas); se compara con pesos de clase.

## 8. Cronograma propuesto (hasta la entrega de diciembre de 2026)

| Semana | Tarea |
|---|---|
| 26 oct – 1 nov | Revisión del código y de la definición del desenlace con el director; ejecutar `run_all.sh` |
| 2 – 8 nov | Redactar Metodología (cap. 6) |
| 9 – 22 nov | Redactar Resultados (cap. 7) con tablas y figuras de `results/` |
| 23 – 29 nov | Discusión, limitaciones y conclusiones (caps. 8–9) |
| 30 nov – 6 dic | Revisión del director, resumen/abstract, formato UNIR y referencias |
| 7 dic → | Correcciones finales y depósito |
