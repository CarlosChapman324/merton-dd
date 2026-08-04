"""Nota de calibración: N(-DD) NO es una probabilidad de default calibrada.

Se compara el pi promedio de cada modelo contra la tasa realizada de
default a 12 meses, global y por decil de pi. En la práctica el DD se usa
como ranking; tomar N(-DD) como PD literal sobreestima o subestima según
el régimen, y esta tabla lo muestra con los datos del panel.
"""

import pandas as pd

from validation.metricas import deciles_por_mes


def calibracion_global(df: pd.DataFrame) -> pd.DataFrame:
    """pi promedio vs tasa realizada, para ambos modelos."""
    tasa = float(df["default_12m"].mean())
    filas = []
    for nombre, col in [("naive", "pi_naive"), ("merton", "pi_merton")]:
        filas.append(
            {
                "modelo": nombre,
                "pi_promedio": float(df[col].mean()),
                "tasa_realizada": tasa,
                "ratio": float(df[col].mean() / tasa) if tasa > 0 else float("nan"),
            }
        )
    return pd.DataFrame(filas)


def calibracion_por_decil(df: pd.DataFrame, col_pi: str) -> pd.DataFrame:
    """pi promedio vs tasa realizada dentro de cada decil de pi."""
    base = df.dropna(subset=[col_pi]).copy()
    base["decil"] = deciles_por_mes(base, col_pi)
    tabla = base.groupby("decil").agg(
        filas=("default_12m", "size"),
        pi_promedio=(col_pi, "mean"),
        tasa_realizada=("default_12m", "mean"),
    )
    return tabla.sort_index(ascending=False)
