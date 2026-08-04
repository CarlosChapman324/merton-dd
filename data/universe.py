"""Universo de empresas: vivas (S&P 500 no financiero) y quebradas (Chapter 11).

Vivas: se toma la lista actual del S&P 500 (tabla de Wikipedia, que incluye
sector GICS y CIK), se excluyen Financials y Real Estate (el modelo de Merton
no aplica bien a su estructura de deuda; el paper también las excluye) y se
hace un muestreo estratificado por sector hasta ~130 nombres. El muestreo es
determinista (alfabético dentro de cada sector) para que sea reproducible.

Quebradas: lista curada de Chapter 11 en EE.UU. 2019-2024, verificada en la
Fase 1 contra la disponibilidad real de precios (24+ meses previos al filing).
La verificación descartó a buena parte de la cohorte COVID 2020 (J.C. Penney,
Hertz, Chesapeake, Whiting, Frontier y otras): las fuentes gratuitas purgan
el histórico de tickers deslistados al cabo de unos años. Ese sesgo de
disponibilidad se documenta en el README como limitación del panel. BBBY se
excluyó porque el ticker fue reutilizado por otra empresa (Overstock/Beyond)
y la serie descargable no corresponde a la Bed Bath & Beyond original.

El ticker listado es el que hoy da acceso a la serie de precios (para varias
quebradas es el ticker OTC con sufijo Q, que conserva el histórico previo).
"""

from dataclasses import dataclass
from io import StringIO

import pandas as pd
import requests

from data import cache

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SECTORES_EXCLUIDOS = {"Financials", "Real Estate"}
N_VIVAS = 130


@dataclass(frozen=True)
class Default:
    ticker: str          # ticker que hoy da acceso a la serie de precios
    nombre: str
    fecha_filing: str    # fecha del filing de Chapter 11 (YYYY-MM-DD)
    cik: str | None = None  # override manual si el mapeo de la SEC no lo trae


# Verificadas en Fase 1: precios descargables y 24+ meses previos al filing.
# Los CIK de los tickers OTC se buscaron a mano en EDGAR porque el archivo
# company_tickers.json solo trae registrantes con ticker activo.
DEFAULTS: list[Default] = [
    Default("PCG", "PG&E", "2019-01-29"),
    Default("GTX", "Garrett Motion", "2020-09-20"),
    Default("REVRQ", "Revlon", "2022-06-15", cik="0000887921"),
    Default("PRTYQ", "Party City", "2023-01-17", cik="0001592058"),
    Default("NRDE", "Lordstown Motors (hoy Nu Ride)", "2023-06-27"),
    Default("YELLQ", "Yellow Corp", "2023-08-06", cik="0000716006"),
    Default("PTRAQ", "Proterra", "2023-08-07", cik="0001820630"),
    Default("RADCQ", "Rite Aid", "2023-10-15", cik="0000084129"),
    Default("WEWKQ", "WeWork", "2023-11-06", cik="0001813756"),
    Default("JOANQ", "Joann", "2024-03-18", cik="0001834585"),
    Default("BIGGQ", "Big Lots", "2024-09-09", cik="0000768835"),
]

# Splits de las quebradas cuyos precios vienen de stockanalysis: yfinance no
# tiene eventos de tickers deslistados, y sin esto las acciones as-reported
# de EDGAR quedan en términos pre-split mientras el precio viene restatado
# (E se distorsiona por el factor del split). Ratio = acciones nuevas/viejas.
# Fechas de los registros públicos de cada reverse split.
SPLITS_MANUALES: dict[str, list[tuple[str, float]]] = {
    "WEWKQ": [("2023-04-21", 1 / 40)],  # WeWork, reverse 1:40
    "RADCQ": [("2019-04-22", 1 / 20)],  # Rite Aid, reverse 1:20
}

# Buscadas y descartadas por falta de datos (se reporta como sesgo del panel):
# J.C. Penney, Hertz (solo existe la serie de la nueva Hertz de 2021), Chesapeake,
# Whiting, Frontier, Sears, iHeartMedia, Windstream, Dean Foods, Intelsat, GNC,
# Ascena, Tailored Brands, Valaris, Mallinckrodt, Endo, Stage Stores, Pier 1,
# McDermott, Tupperware, Spirit Airlines, Express, SunPower, Fisker, Nikola,
# Canoo, Bird Global, Cyxtera, Virgin Orbit, Core Scientific, Diebold (solo
# existe la serie posterior a la emergencia) y Bed Bath & Beyond (ticker
# reutilizado por otra empresa).


def _descargar_sp500() -> pd.DataFrame:
    """Baja la tabla de constituyentes del S&P 500 desde Wikipedia."""
    resp = requests.get(WIKI_SP500, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    tabla = pd.read_html(StringIO(resp.text))[0]
    df = tabla.rename(
        columns={
            "Symbol": "ticker",
            "Security": "nombre",
            "GICS Sector": "sector",
            "CIK": "cik",
        }
    )[["ticker", "nombre", "sector", "cik"]]
    # Yahoo usa guion donde la SEC usa punto (BRK.B -> BRK-B)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df["cik"] = df["cik"].astype(int).astype(str).str.zfill(10)
    return df


def vivas() -> pd.DataFrame:
    """Universo de vivas: no financieras del S&P 500, muestreo estratificado.

    Asignación proporcional por sector y orden alfabético dentro de cada
    sector: reproducible y sin discrecionalidad en la selección.
    """
    if cache.tiene("universo_vivas"):
        return cache.cargar_df("universo_vivas")

    sp500 = _descargar_sp500()
    sp500 = sp500[~sp500["sector"].isin(SECTORES_EXCLUIDOS)].copy()
    # PCG (PG&E) es un caso de default del panel; no puede estar en ambos grupos
    sp500 = sp500[~sp500["ticker"].isin({d.ticker for d in DEFAULTS})]

    cuotas = (sp500["sector"].value_counts(normalize=True) * N_VIVAS).round().astype(int)
    partes = []
    for sector, n in cuotas.items():
        bloque = sp500[sp500["sector"] == sector].sort_values("ticker").head(n)
        partes.append(bloque)
    universo = pd.concat(partes).sort_values("ticker").reset_index(drop=True)

    cache.guardar_df(universo, "universo_vivas")
    return universo


def quebradas() -> pd.DataFrame:
    """Universo de quebradas como DataFrame (ticker, nombre, fecha_filing, cik)."""
    df = pd.DataFrame([d.__dict__ for d in DEFAULTS])
    df["fecha_filing"] = pd.to_datetime(df["fecha_filing"])
    return df
