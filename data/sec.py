"""SEC EDGAR: mapeo ticker a CIK, deuda de balance y acciones en circulación.

Todo viene de la API companyfacts (gratis, sin key, requiere User-Agent con
email) y del archivo público company_tickers.json. Límite de EDGAR: 10
requests por segundo; aquí se deja una pausa fija entre llamadas.

Convención de deuda del paper (vía Vassalou-Xing):
    F = deuda corriente + 0.5 * deuda de largo plazo
Los tags XBRL no se eligen uno por empresa sino por coalescencia fecha a
fecha (ver deuda() y las listas TAGS_*): las empresas cambian de tag con
los años y reportan algunos componentes solo en el 10-K anual, así que
cada fecha de balance toma el tag de mayor prioridad con dato, arrastrando
el último valor conocido hasta 400 días. Los tags usados quedan
registrados por empresa para poder documentarlo.

Punto clave anti lookahead: cada observación conserva su fecha de
publicación (filed). El panel mensual solo usa valores con filed <= fin de
mes, es decir, lo que un analista podía conocer en ese momento.
"""

import os
import time

import numpy as np
import pandas as pd
import requests

from data import cache

# EDGAR exige un User-Agent con nombre y correo de contacto. Se lee del
# entorno para no dejar datos personales en un repo público: antes de
# reconstruir la ingesta, exporta SEC_USER_AGENT="Nombre correo@dominio".
# Con el cache ya poblado no hace falta (nada vuelve a la red).
USER_AGENT = os.environ.get("SEC_USER_AGENT", "Merton DD replication (contacto sin definir)")
HEADERS = {"User-Agent": USER_AGENT}
PAUSA_SEGUNDOS = 0.15  # bien por debajo del límite de 10 req/s

URL_TICKERS = "https://www.sec.gov/files/company_tickers.json"
URL_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Deuda corriente: el tag directo del paper, y como fallback la suma de la
# porción corriente de la deuda larga (dos variantes que las empresas usan
# como sinónimos: la segunda incluye arriendos financieros) más los
# préstamos de corto plazo.
TAG_CORRIENTE_PRINCIPAL = "DebtCurrent"
TAGS_CORRIENTE_PORCION_LT = [
    "LongTermDebtCurrent",
    "LongTermDebtAndCapitalLeaseObligationsCurrent",
]
TAG_CORRIENTE_CORTO = "ShortTermBorrowings"

# Deuda de largo plazo, en orden de prioridad. Las empresas cambian de tag
# con los años (Eaton dejó LongTermDebtNoncurrent muerto en 2014 y siguió
# con LongTermDebt; Air Products migró a las variantes con arriendos en
# 2023), así que la elección es fecha a fecha por coalescencia, no un solo
# tag por empresa. Los dos últimos rescatan casos particulares: AKAM
# reporta su deuda solo como notas convertibles y DHI solo como
# NotesPayable; ambos son totales instantáneos de balance.
TAGS_LARGO = [
    "LongTermDebtNoncurrent",
    "LongTermDebt",
    "LongTermDebtAndCapitalLeaseObligations",
    "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    "ConvertibleLongTermNotesPayable",
    "NotesPayable",
]
# Cadena de fallbacks para acciones en circulación. El tag dei es el
# preferido (portada de cada 10-Q/10-K), pero companyfacts omite los hechos
# dimensionales, así que empresas con varias clases de acciones (Airbnb,
# Alphabet) pueden no traerlo; el promedio ponderado básico del cálculo de
# EPS sí suele venir agregado y sirve de aproximación.
TAGS_ACCIONES = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesIssued"),
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
]
MIN_OBS_ACCIONES = 20


def mapa_ticker_cik() -> dict[str, str]:
    """Diccionario ticker -> CIK (10 dígitos) desde el archivo de la SEC."""
    if not cache.tiene("sec_company_tickers", "json"):
        resp = requests.get(URL_TICKERS, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        cache.guardar_json(resp.json(), "sec_company_tickers")
    crudo = cache.cargar_json("sec_company_tickers")
    return {fila["ticker"].upper(): str(fila["cik_str"]).zfill(10) for fila in crudo.values()}


def companyfacts(cik: str) -> dict | None:
    """Baja (o lee de cache) el companyfacts completo de un CIK."""
    nombre = f"companyfacts/CIK{cik}"
    if cache.tiene(nombre, "json"):
        return cache.cargar_json(nombre)
    resp = requests.get(URL_FACTS.format(cik=cik), headers=HEADERS, timeout=60)
    time.sleep(PAUSA_SEGUNDOS)
    if resp.status_code != 200:
        return None
    cache.guardar_json(resp.json(), nombre)
    return cache.cargar_json(nombre)


def _vacio() -> pd.DataFrame:
    """DataFrame vacío con los dtypes correctos (evita errores de merge en pandas 3)."""
    return pd.DataFrame(
        {
            "end": pd.Series(dtype="datetime64[ns]"),
            "filed": pd.Series(dtype="datetime64[ns]"),
            "val": pd.Series(dtype="float64"),
        }
    )


def _serie_tag(facts: dict, tag: str, unidad: str, taxonomia: str) -> pd.DataFrame:
    """Extrae las observaciones (end, filed, val) de un tag, sin duplicados.

    Los balances aparecen repetidos entre 10-Q, 10-K y enmiendas. Para cada
    fecha de balance (end) se conserva la primera publicación (filed mínimo):
    es el dato point-in-time, tal como se conoció originalmente.
    """
    entradas = facts.get("facts", {}).get(taxonomia, {}).get(tag, {}).get("units", {}).get(unidad)
    if not entradas:
        return _vacio()
    df = pd.DataFrame(entradas)[["end", "filed", "val"]]
    # unidad ns explicita: pandas 3 conserva la unidad de origen y los
    # merge_asof posteriores exigen que todas las fechas coincidan
    df["end"] = pd.to_datetime(df["end"]).dt.as_unit("ns")
    df["filed"] = pd.to_datetime(df["filed"]).dt.as_unit("ns")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna(subset=["val"])
    df = df.sort_values(["end", "filed"]).groupby("end", as_index=False).first()
    return df


# Cuanto tiempo se arrastra el ultimo valor conocido de un componente de
# deuda. Hay empresas que reportan un componente solo en el 10-K anual
# (AbbVie solo trae LongTermDebt en los anuales): sin arrastre, los
# trimestres intermedios quedaban con ese componente en cero y F caia tres
# ordenes de magnitud (hallazgo confirmado de la revision de Fase 2:
# AbbVie con F = 16 millones y deuda real de ~70,000 millones). Los 400
# dias cubren el ciclo anual con margen, pero evitan arrastrar para
# siempre un tag que la empresa dejo de reportar.
ARRASTRE_MAXIMO = pd.Timedelta(days=400)


def _alinear(serie: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Alinea una serie (end, filed, val) a la malla comun de fechas de
    balance, arrastrando el ultimo valor conocido hasta ARRASTRE_MAXIMO.

    El filed viaja con el valor arrastrado: la fila resultante queda
    fechada point-in-time con la publicacion mas reciente involucrada.
    """
    if serie.empty:
        return pd.DataFrame({"end": grid["end"], "filed": pd.NaT, "val": float("nan")})
    return pd.merge_asof(grid, serie.sort_values("end"), on="end", tolerance=ARRASTRE_MAXIMO)


def _coalescer(alineadas: list[pd.DataFrame]) -> pd.DataFrame:
    """Fila a fila, el valor de la primera serie (en orden de prioridad)
    que tenga dato. Devuelve un DataFrame con val y filed."""
    vals = np.column_stack([a["val"].to_numpy(dtype=float) for a in alineadas])
    fileds = np.column_stack([a["filed"].to_numpy() for a in alineadas])
    tiene_dato = ~np.isnan(vals)
    eleccion = tiene_dato.argmax(axis=1)  # primera columna con dato
    fila = np.arange(len(vals))
    val = vals[fila, eleccion]
    filed = fileds[fila, eleccion]
    # argmax devuelve 0 cuando ninguna columna tiene dato: invalidar
    sin_dato = ~tiene_dato.any(axis=1)
    val[sin_dato] = float("nan")
    filed[sin_dato] = np.datetime64("NaT")
    return pd.DataFrame({"val": val, "filed": filed})


def _sumar(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Suma dos series alineadas tratando el NaN como cero si la otra
    existe (min_count=1); el filed es el mas reciente de los presentes."""
    val = pd.concat([a["val"], b["val"]], axis=1).sum(axis=1, min_count=1)
    filed = pd.concat([a["filed"], b["filed"]], axis=1).max(axis=1)
    return pd.DataFrame({"val": val, "filed": filed})


def deuda(facts: dict) -> tuple[pd.DataFrame, dict]:
    """Serie trimestral de F = corriente + 0.5 * largo plazo, con tags usados.

    Todos los tags candidatos se alinean a una malla comun de fechas de
    balance (_alinear, con arrastre limitado) y en cada fecha gana el tag
    de mayor prioridad que tenga dato (_coalescer). Asi una empresa que
    cambio de tag con los anos queda cubierta en todo el periodo, y un
    componente que un trimestre no se reporto pero si hace menos de 400
    dias conserva su ultimo valor conocido.
    """
    principal = _serie_tag(facts, TAG_CORRIENTE_PRINCIPAL, "USD", "us-gaap")
    porciones_lt = [_serie_tag(facts, t, "USD", "us-gaap") for t in TAGS_CORRIENTE_PORCION_LT]
    corto = _serie_tag(facts, TAG_CORRIENTE_CORTO, "USD", "us-gaap")
    largos = [_serie_tag(facts, t, "USD", "us-gaap") for t in TAGS_LARGO]

    hay_corriente = not principal.empty or any(not s.empty for s in porciones_lt) or not corto.empty
    hay_largo = any(not s.empty for s in largos)
    if not hay_corriente and not hay_largo:
        return pd.DataFrame(), {"corriente": None, "largo": None}

    # Malla comun: todas las fechas de balance donde algo se reporto
    todas = [s for s in [principal, corto, *porciones_lt, *largos] if not s.empty]
    grid = pd.DataFrame(
        {"end": pd.concat([s["end"] for s in todas]).drop_duplicates().sort_values().to_numpy()}
    )

    # corriente: DebtCurrent manda; si no hay, porcion corriente de la
    # deuda larga (coalescencia entre sus dos variantes) + corto plazo
    fallback_corriente = _sumar(
        _coalescer([_alinear(s, grid) for s in porciones_lt]),
        _alinear(corto, grid),
    )
    corriente = _coalescer([_alinear(principal, grid), fallback_corriente])
    largo = _coalescer([_alinear(s, grid) for s in largos])

    df = grid.copy()
    df["deuda_corriente"] = corriente["val"]
    df["filed_c"] = corriente["filed"]
    df["deuda_largo"] = largo["val"]
    df["filed_l"] = largo["filed"]

    # fila utilizable: al menos un componente conocido tras el arrastre
    df = df[df["deuda_corriente"].notna() | df["deuda_largo"].notna()].copy()
    df["filed"] = df[["filed_c", "filed_l"]].max(axis=1)
    df["deuda_corriente"] = df["deuda_corriente"].fillna(0.0)
    df["deuda_largo"] = df["deuda_largo"].fillna(0.0)
    df["F"] = df["deuda_corriente"] + 0.5 * df["deuda_largo"]
    df = df[["end", "filed", "deuda_corriente", "deuda_largo", "F"]].sort_values("end")

    tags = {
        "corriente": TAG_CORRIENTE_PRINCIPAL if not principal.empty
        else ("+".join([TAGS_CORRIENTE_PORCION_LT[0], TAG_CORRIENTE_CORTO]) if hay_corriente else None),
        "largo": "|".join(t for t, s in zip(TAGS_LARGO, largos) if not s.empty) or None,
    }
    return df.reset_index(drop=True), tags


TAG_ACTIVOS = "Assets"
TAGS_RESULTADO = ["NetIncomeLoss", "ProfitLoss"]
DIAS_ANUAL = (330, 400)  # duración de un ejercicio anual, con margen


def _serie_tag_duracion(facts: dict, tag: str, taxonomia: str = "us-gaap") -> pd.DataFrame:
    """Como _serie_tag pero para hechos de duración (flujos), quedándose
    solo con los ejercicios anuales (330-400 días): el ROA del baseline
    usa el resultado del año completo, no trimestres que habría que
    reconstruir (el Q4 de XBRL suele venir solo como acumulado)."""
    entradas = facts.get("facts", {}).get(taxonomia, {}).get(tag, {}).get("units", {}).get("USD")
    if not entradas:
        return _vacio()
    df = pd.DataFrame(entradas)
    if "start" not in df.columns:
        return _vacio()
    df = df[["start", "end", "filed", "val"]].dropna(subset=["start"])
    df["start"] = pd.to_datetime(df["start"]).dt.as_unit("ns")
    df["end"] = pd.to_datetime(df["end"]).dt.as_unit("ns")
    df["filed"] = pd.to_datetime(df["filed"]).dt.as_unit("ns")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    duracion = (df["end"] - df["start"]).dt.days
    df = df[(duracion >= DIAS_ANUAL[0]) & (duracion <= DIAS_ANUAL[1])].dropna(subset=["val"])
    if df.empty:
        return _vacio()
    df = df.sort_values(["end", "filed"]).groupby("end", as_index=False).first()
    return df[["end", "filed", "val"]]


def rentabilidad(facts: dict) -> pd.DataFrame:
    """ROA anual: resultado neto del ejercicio / activos totales al cierre.

    Es el ratio de rentabilidad del baseline contable (mini Shumway 2001).
    Se une el NI anual con los activos del mismo cierre de ejercicio; la
    granularidad es anual (asof por filed en el panel), suficiente para un
    ratio que cambia lento y honesto con lo que XBRL da sin reconstruir
    trimestres.
    """
    ni = _vacio()
    for tag in TAGS_RESULTADO:
        ni = _serie_tag_duracion(facts, tag)
        if not ni.empty:
            break
    activos = _serie_tag(facts, TAG_ACTIVOS, "USD", "us-gaap")
    if ni.empty or activos.empty:
        return pd.DataFrame(columns=["end", "filed", "roa"])
    df = pd.merge(
        ni.rename(columns={"val": "ni"}),
        activos.rename(columns={"val": "activos"}),
        on="end",
        suffixes=("_ni", "_a"),
    )
    df = df[df["activos"] > 0].copy()
    df["roa"] = df["ni"] / df["activos"]
    df["filed"] = df[["filed_ni", "filed_a"]].max(axis=1)
    return df[["end", "filed", "roa"]].sort_values("end").reset_index(drop=True)


def acciones_en_circulacion(facts: dict) -> tuple[pd.DataFrame, str | None]:
    """Serie de acciones en circulación y el tag usado.

    Se usa EDGAR y no yfinance porque cubre también a las empresas
    deslistadas, con la misma definición para todo el panel. Se recorre la
    cadena de fallbacks y se toma el primer tag con cobertura razonable
    (MIN_OBS_ACCIONES); si ninguno llega, el mejor poblado.
    """
    candidatas = []
    for taxonomia, tag in TAGS_ACCIONES:
        serie = _serie_tag(facts, tag, "shares", taxonomia)
        if not serie.empty:
            candidatas.append((tag, serie))
            if len(serie) >= MIN_OBS_ACCIONES:
                break
    if not candidatas:
        return _vacio().rename(columns={"val": "acciones"}), None
    tag, serie = max(candidatas, key=lambda par: len(par[1]))
    # Guardia de escala: ninguna cotizada tiene menos de 10,000 acciones.
    # Joann reportó 40.7 (millones sin reescalar) en dos portadas de 2023 y
    # eso desplomaba E a 70 dólares (hallazgo de la auditoría de Fase 2);
    # se descarta la observación y el as-of arrastra el reporte anterior.
    serie = serie[serie["val"] >= 1e4]
    return serie.rename(columns={"val": "acciones"}), tag
