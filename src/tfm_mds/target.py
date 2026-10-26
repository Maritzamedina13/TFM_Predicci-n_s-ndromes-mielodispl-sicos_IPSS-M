"""Definición de la variable objetivo: transformación a leucemia mieloide aguda (LMA).

cBioPortal publica para esta cohorte la *supervivencia libre de leucemia* (LFS), cuyo
evento combina transformación a LMA **o** muerte (``1:Transformed/Deceased``), junto con
la supervivencia global (OS). La transformación a LMA se reconstruye así:

* **LMA = 1** si hay evento LFS y además (a) el paciente sigue vivo
  (``OS_STATUS = 0``) o (b) el evento LFS ocurre antes que el fin del seguimiento de OS
  (``LFS_MONTHS < OS_MONTHS``): el evento LFS no puede ser la muerte, luego es la LMA.
* **LMA = 0** si el paciente está libre de leucemia al final del seguimiento
  (``LFS_STATUS = 0``) o si el evento LFS coincide con la muerte
  (``LFS_MONTHS = OS_MONTHS`` y ``OS_STATUS = 1``): muerte sin transformación previa.
* Se **excluyen** los pacientes sin datos de LFS y los que tienen evento LFS sin
  información de OS (no es posible distinguir muerte de transformación).

Limitación: una transformación diagnosticada el mismo día de la muerte se clasificaría
como muerte sin LMA (posible infraestimación leve de eventos).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_TOL = 1e-6


def derive_aml_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """Añade ``AML_T`` (0/1), ``AML_T_MONTHS`` y ``TARGET_VALID`` al DataFrame."""
    out = df.copy()
    lfs_event = out["LFS_STATUS"].str.startswith("1")
    lfs_free = out["LFS_STATUS"].str.startswith("0")
    os_dead = out["OS_STATUS"].str.startswith("1")
    os_alive = out["OS_STATUS"].str.startswith("0")

    aml = lfs_event & (os_alive | (out["LFS_MONTHS"] < out["OS_MONTHS"] - _TOL))
    death_no_aml = lfs_event & os_dead & ((out["LFS_MONTHS"] - out["OS_MONTHS"]).abs() <= _TOL)

    valid = (aml | death_no_aml | lfs_free).fillna(False)
    out["AML_T"] = np.where(aml.fillna(False), 1, 0)
    out["DEATH_WITHOUT_AML"] = np.where(death_no_aml.fillna(False), 1, 0)
    out["AML_T_MONTHS"] = out["LFS_MONTHS"]
    out["TARGET_VALID"] = valid
    return out


def fixed_horizon_label(df: pd.DataFrame, horizon: float) -> pd.Series:
    """Etiqueta a horizonte fijo (análisis de sensibilidad).

    1 = LMA en los primeros ``horizon`` meses; 0 = sin LMA con seguimiento >= horizonte
    o muerte sin LMA antes del horizonte (evento competitivo); NaN = censurado vivo sin
    LMA antes del horizonte (desenlace desconocido, se excluye).
    """
    t = df["AML_T_MONTHS"]
    lab = pd.Series(np.nan, index=df.index)
    lab[(df["AML_T"] == 1) & (t <= horizon)] = 1
    lab[(df["AML_T"] == 1) & (t > horizon)] = 0
    lab[(df["AML_T"] == 0) & (t >= horizon)] = 0
    lab[df["DEATH_WITHOUT_AML"] == 1] = 0
    return lab
