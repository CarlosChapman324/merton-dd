"""Precios diarios: yfinance como fuente principal, stockanalysis.com como fallback.

Por qué dos fuentes: yfinance suele borrar el histórico de tickers
deslistados, y las quebradas son justamente la parte valiosa del panel.
El PLAN preveía Stooq como fallback, pero Stooq puso un desafío JavaScript
anti-bot delante de sus CSV (verificado en Fase 1), así que se reemplazó por
la API pública de stockanalysis.com, que conserva series de varios tickers
OTC deslistados e incluye cierre ajustado.

Cada ticker se cachea en data/cache/prices/{ticker}.parquet con columnas:
    date, close (sin ajustar, para market cap), adjclose (ajustado, para
    retornos y volatilidad) y source (yfinance o stockanalysis).
"""

import time

import pandas as pd
import requests
import yfinance as yf

from data import cache

PAUSA = 0.4  # pausa entre tickers para no golpear rate limits
INICIO = "2015-01-01"  # un año antes de la ventana 2016-2025, para volatilidad


def _desde_yfinance(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker,
            start=INICIO,
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )
    except Exception:
        return None
    time.sleep(PAUSA)
    if df is None or df.empty or "Close" not in df.columns:
        return None
    salida = pd.DataFrame(
        {
            "date": df.index,
            "close": df["Close"].to_numpy(),
            "adjclose": df.get("Adj Close", df["Close"]).to_numpy(),
        }
    ).dropna()
    salida["source"] = "yfinance"
    return salida if len(salida) > 0 else None


def _desde_stockanalysis(ticker: str) -> pd.DataFrame | None:
    url = (
        "https://stockanalysis.com/api/symbol/s/"
        f"{ticker.lower()}/history?range=10Y&period=Daily"
    )
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        datos = resp.json().get("data")
    except Exception:
        return None
    time.sleep(PAUSA)
    if not isinstance(datos, list) or not datos:
        return None
    df = pd.DataFrame(datos)
    salida = pd.DataFrame(
        {
            "date": pd.to_datetime(df["t"]),
            "close": pd.to_numeric(df["c"], errors="coerce"),
            # 'a' es el cierre ajustado; si falta, se usa el cierre simple
            "adjclose": pd.to_numeric(df.get("a", df["c"]), errors="coerce"),
        }
    ).dropna()
    salida = salida[salida["date"] >= INICIO]
    salida["source"] = "stockanalysis"
    return salida if len(salida) > 0 else None


def splits(ticker: str) -> pd.DataFrame:
    """Eventos de split del ticker: fecha y ratio (acciones nuevas / viejas,
    4.0 para un split 4:1, 0.05 para un reverse 1:20). Vienen de yfinance y
    se cachean; para tickers deslistados yfinance no los trae y esos casos
    se cubren con SPLITS_MANUALES en data/universe.py."""
    nombre = f"splits/{ticker}"
    if cache.tiene(nombre):
        return cache.cargar_df(nombre)
    try:
        serie = yf.Ticker(ticker).splits
    except Exception:
        serie = None
    time.sleep(PAUSA)
    if serie is None or len(serie) == 0:
        df = pd.DataFrame(
            {"fecha": pd.Series(dtype="datetime64[ns]"), "ratio": pd.Series(dtype="float64")}
        )
    else:
        df = pd.DataFrame(
            {
                "fecha": pd.to_datetime(serie.index).tz_localize(None),
                "ratio": serie.to_numpy(dtype=float),
            }
        )
    cache.guardar_df(df, nombre)
    return df


def precios(ticker: str) -> pd.DataFrame | None:
    """Serie diaria de un ticker, de cache o descargando con fallback."""
    nombre = f"prices/{ticker}"
    if cache.tiene(nombre):
        return cache.cargar_df(nombre)
    df = _desde_yfinance(ticker)
    if df is None:
        df = _desde_stockanalysis(ticker)
    if df is None:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    cache.guardar_df(df, nombre)
    return df
