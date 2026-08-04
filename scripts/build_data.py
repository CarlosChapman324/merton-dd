"""Fase 1: descarga todos los datos y construye el panel empresa-mes.

Orden deliberado: primero se verifican las quebradas (disponibilidad de
precios y 24+ meses de historia previa al filing), porque son el riesgo
número uno del proyecto. Solo después se descarga el resto.

Uso: uv run python scripts/build_data.py
Todo queda cacheado; correrlo dos veces no repite ninguna descarga.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import cache, fred, panel, prices, sec, universe


def verificar_quebradas() -> pd.DataFrame:
    """Paso 1: qué quebradas tienen precios y cuánta historia previa al filing."""
    filas = []
    for d in universe.quebradas().itertuples():
        serie = prices.precios(d.ticker)
        if serie is None:
            filas.append(
                {
                    "ticker": d.ticker,
                    "nombre": d.nombre,
                    "fecha_filing": d.fecha_filing,
                    "fuente": None,
                    "meses_previos": 0,
                    "usable": False,
                }
            )
            continue
        previos = serie[serie["date"] < d.fecha_filing]
        meses = previos["date"].dt.to_period("M").nunique()
        filas.append(
            {
                "ticker": d.ticker,
                "nombre": d.nombre,
                "fecha_filing": d.fecha_filing,
                "fuente": serie["source"].iloc[0],
                "meses_previos": meses,
                "usable": meses >= 24,
            }
        )
    reporte = pd.DataFrame(filas)
    cache.guardar_df(reporte, "reporte_quebradas")
    return reporte


def descargar_precios_vivas(vivas: pd.DataFrame) -> list[str]:
    """Paso 2: precios de las vivas. Devuelve los tickers sin datos."""
    sin_datos = []
    for i, ticker in enumerate(vivas["ticker"], start=1):
        if prices.precios(ticker) is None:
            sin_datos.append(ticker)
        if i % 25 == 0:
            print(f"  precios: {i}/{len(vivas)} tickers")
    return sin_datos


def resolver_ciks(vivas: pd.DataFrame, quebradas: pd.DataFrame) -> pd.DataFrame:
    """Paso 3: CIK por empresa. Wikipedia trae el de las vivas; para las
    quebradas se usa el archivo de la SEC más los overrides manuales."""
    mapa = sec.mapa_ticker_cik()
    filas = [
        {"ticker": t, "cik": c, "es_default": False}
        for t, c in zip(vivas["ticker"], vivas["cik"])
    ]
    for d in quebradas.itertuples():
        # pd.isna: con pandas 3 los None del override llegan como NaN (truthy)
        tiene_override = isinstance(d.cik, str) and len(d.cik) > 0
        cik = d.cik if tiene_override else mapa.get(d.ticker.upper())
        filas.append({"ticker": d.ticker, "cik": cik, "es_default": True})
    return pd.DataFrame(filas)


def descargar_fundamentales(ciks: pd.DataFrame) -> tuple[dict, dict, dict, pd.DataFrame]:
    """Paso 4: deuda, acciones y ROA desde companyfacts, con registro de tags."""
    deudas, acciones, roas, registro = {}, {}, {}, []
    total = len(ciks)
    for i, fila in enumerate(ciks.itertuples(), start=1):
        sin_tags = {"tag_corriente": None, "tag_largo": None, "tag_acciones": None}
        if fila.cik is None or (isinstance(fila.cik, float) and pd.isna(fila.cik)):
            registro.append({"ticker": fila.ticker, "estado": "sin CIK", **sin_tags})
            continue
        facts = sec.companyfacts(fila.cik)
        if facts is None:
            registro.append({"ticker": fila.ticker, "estado": "sin companyfacts", **sin_tags})
            continue
        df_deuda, tags = sec.deuda(facts)
        df_acciones, tag_acciones = sec.acciones_en_circulacion(facts)
        df_roa = sec.rentabilidad(facts)
        if not df_roa.empty:
            roas[fila.ticker] = df_roa
        if df_deuda.empty:
            estado = "sin deuda utilizable"
        elif df_acciones.empty:
            estado = "sin acciones utilizables"
        else:
            estado = "ok"
        if not df_deuda.empty:
            deudas[fila.ticker] = df_deuda
        if not df_acciones.empty:
            acciones[fila.ticker] = df_acciones
        registro.append(
            {
                "ticker": fila.ticker,
                "estado": estado,
                "tag_corriente": tags["corriente"],
                "tag_largo": tags["largo"],
                "tag_acciones": tag_acciones,
            }
        )
        if i % 25 == 0:
            print(f"  EDGAR: {i}/{total} empresas")
    reporte = pd.DataFrame(registro)
    cache.guardar_df(reporte, "reporte_edgar")
    return deudas, acciones, roas, reporte


def _splits_de(ticker: str) -> pd.DataFrame:
    """Splits de yfinance más los overrides manuales de las deslistadas."""
    eventos = prices.splits(ticker)
    manuales = universe.SPLITS_MANUALES.get(ticker)
    if manuales:
        extra = pd.DataFrame(
            {"fecha": pd.to_datetime([f for f, _ in manuales]), "ratio": [r for _, r in manuales]}
        )
        eventos = pd.concat([eventos, extra], ignore_index=True).drop_duplicates("fecha")
    return eventos


def construir_panel(ciks: pd.DataFrame, quebradas: pd.DataFrame, deudas: dict, acciones: dict, roas: dict) -> pd.DataFrame:
    """Paso 5: panel empresa-mes con etiquetas de default.

    Las acciones se convierten a términos de hoy con los splits del ticker
    (los precios vienen restatados por splits; ver panel.acciones_en_terminos_actuales).
    """
    tbill = fred.tbill_3m()
    filing = dict(zip(quebradas["ticker"], quebradas["fecha_filing"]))
    paneles = []
    for fila in ciks.itertuples():
        t = fila.ticker
        serie = prices.precios(t)
        if serie is None or t not in deudas or t not in acciones:
            continue
        acc = panel.acciones_en_terminos_actuales(acciones[t], _splits_de(t))
        p = panel.panel_de_empresa(t, serie, deudas[t], acc, tbill, roas.get(t))
        p["es_default"] = fila.es_default
        p["fecha_filing"] = filing.get(t, pd.NaT)
        p["post_filing"] = p["fecha_filing"].notna() & (p["mes"] > p["fecha_filing"])
        paneles.append(p)
    bruto = pd.concat(paneles, ignore_index=True)
    cache.guardar_df(bruto, "panel_bruto")
    limpio = panel.filtrar_completo(bruto)
    cache.guardar_df(limpio, "panel")
    return limpio


def main() -> None:
    print("Paso 1: verificación temprana de las quebradas")
    reporte_q = verificar_quebradas()
    print(reporte_q.to_string(index=False))
    usables = int(reporte_q["usable"].sum())
    print(f"  defaults usables (24+ meses de precios): {usables}/{len(reporte_q)}")

    print("Paso 2: universo de vivas y sus precios")
    vivas = universe.vivas()
    print(f"  universo de vivas: {len(vivas)} empresas no financieras del S&P 500")
    sin_datos = descargar_precios_vivas(vivas)
    if sin_datos:
        print(f"  sin precios (se excluyen): {sin_datos}")

    print("Paso 3: mapeo ticker a CIK")
    quebradas = universe.quebradas()
    ciks = resolver_ciks(vivas, quebradas)
    sin_cik = ciks[ciks["cik"].isna()]["ticker"].tolist()
    if sin_cik:
        print(f"  sin CIK (revisar overrides en universe.py): {sin_cik}")

    print("Paso 4: deuda y acciones desde EDGAR")
    deudas, acciones, roas, reporte_e = descargar_fundamentales(ciks)
    problemas = reporte_e[reporte_e["estado"] != "ok"]
    if not problemas.empty:
        print("  empresas con problemas en EDGAR:")
        print(problemas.to_string(index=False))

    print("Paso 5: construcción del panel empresa-mes")
    limpio = construir_panel(ciks, quebradas, deudas, acciones, roas)
    print(f"  filas utilizables: {len(limpio)}")
    print(f"  empresas: {limpio['ticker'].nunique()}")
    print(f"  rango: {limpio['mes'].min().date()} a {limpio['mes'].max().date()}")
    defaults_en_panel = limpio[limpio["es_default"]]["ticker"].nunique()
    print(f"  quebradas presentes en el panel: {defaults_en_panel}")
    print("Listo. Panel en data/cache/panel.parquet")


if __name__ == "__main__":
    main()
