"""Tasa libre de riesgo: T-bill de 3 meses (TB3MS) desde FRED, CSV sin key.

TB3MS es mensual y viene en porcentaje anualizado; se convierte a decimal.
"""

from io import StringIO

import pandas as pd
import requests

from data import cache

URL_TB3MS = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS"


def tbill_3m() -> pd.DataFrame:
    """DataFrame mensual con columnas mes (fin de mes) y r (decimal anual)."""
    if cache.tiene("fred_tb3ms"):
        return cache.cargar_df("fred_tb3ms")
    resp = requests.get(URL_TB3MS, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = ["fecha", "tasa"]
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["tasa"] = pd.to_numeric(df["tasa"], errors="coerce")
    df = df.dropna()
    # FRED reporta el primer día del mes; el panel indexa por fin de mes
    df["mes"] = df["fecha"] + pd.offsets.MonthEnd(0)
    df["r"] = df["tasa"] / 100.0
    salida = df[["mes", "r"]].reset_index(drop=True)
    cache.guardar_df(salida, "fred_tb3ms")
    return salida
