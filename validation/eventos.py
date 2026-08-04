"""Etiquetado de eventos: default = filing de Chapter 11 en los 12 meses siguientes.

Cada empresa-mes se etiqueta 1 si la empresa se acoge al Chapter 11 dentro
del horizonte (12 meses por defecto) contado desde el fin de mes, 0 si no.

Dos reglas anti-leakage, salidas de la auditoría de Fase 2:
- Las filas post-filing (meses OTC posteriores al Chapter 11) se excluyen:
  o predicen un evento que ya ocurrió o entran como no-default con insumos
  basura, y en ambos casos contaminan el AUC.
- El mes del filing mismo ya viene marcado como post_filing en el panel
  (su fin de mes es posterior al filing), así que el último mes etiquetable
  es el anterior al del evento: solo se usa información previa al default.
"""

import pandas as pd

HORIZONTE_MESES = 12


def etiquetar(panel: pd.DataFrame, horizonte_meses: int = HORIZONTE_MESES) -> pd.DataFrame:
    """Panel de validación: sin filas post-filing y con la columna default_12m."""
    base = panel[~panel["post_filing"]].copy()
    limite = base["mes"] + pd.DateOffset(months=horizonte_meses)
    base["default_12m"] = (
        base["fecha_filing"].notna()
        & (base["fecha_filing"] > base["mes"])
        & (base["fecha_filing"] <= limite)
    ).astype(int)
    return base
