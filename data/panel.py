"""Construcción del panel empresa-mes.

Cada fila es una empresa en un fin de mes, con los insumos del modelo:
    E        market cap = precio de cierre sin ajustar x acciones (EDGAR)
    F        deuda corriente + 0.5 x largo plazo (EDGAR, ultimo balance publicado)
    sigma_E  volatilidad anualizada de retornos diarios del último año
    ret_12m  retorno simple de la acción en los 12 meses previos (para el naive)
    r        T-bill 3m anualizada (FRED)

Regla anti lookahead: en cada mes solo se usa el balance más reciente cuya
fecha de publicación (filed) sea anterior o igual al fin de mes. La deuda
trimestral queda así interpolada a mensual como escalón, sin mirar al futuro.
"""

import numpy as np
import pandas as pd

DIAS_HABILES_ANO = 252
MIN_DIAS_VOLATILIDAD = 126


def _ns(fechas) -> pd.Series | pd.DatetimeIndex:
    """Normaliza fechas a nanosegundos. Pandas 3 conserva la unidad de origen
    (Parquet entrega ms, EDGAR us) y los merges exigen unidades iguales."""
    if isinstance(fechas, pd.DatetimeIndex):
        return fechas.as_unit("ns")
    return pd.to_datetime(fechas).dt.as_unit("ns")


def asof_sin_lookahead(trimestral: pd.DataFrame, meses: pd.DatetimeIndex, columnas: list[str]) -> pd.DataFrame:
    """Para cada fin de mes, el último valor trimestral ya publicado.

    trimestral debe tener columnas end, filed y las de `columnas`.
    Devuelve un DataFrame indexado por mes con esas columnas.
    """
    if trimestral.empty:
        return pd.DataFrame(index=meses, columns=columnas, dtype=float)
    base = trimestral.sort_values("filed").reset_index(drop=True)
    base["filed"] = _ns(base["filed"])
    meses_df = pd.DataFrame({"mes": _ns(meses)})
    unido = pd.merge_asof(meses_df, base, left_on="mes", right_on="filed")
    return unido.set_index("mes")[columnas]


def metricas_de_precios(precios: pd.DataFrame) -> pd.DataFrame:
    """Métricas mensuales a partir de la serie diaria de un ticker.

    Devuelve, por fin de mes: close (sin ajustar), sigma_E y ret_12m.
    sigma_E usa retornos logarítmicos diarios del último año (ventana de 252
    días hábiles, mínimo 126 para no reportar volatilidades de casi nada).
    """
    df = precios.assign(date=_ns(precios["date"])).set_index("date").sort_index()
    logret = np.log(df["adjclose"]).diff()
    sigma = logret.rolling(DIAS_HABILES_ANO, min_periods=MIN_DIAS_VOLATILIDAD).std() * np.sqrt(
        DIAS_HABILES_ANO
    )

    mensual = pd.DataFrame(
        {
            "close": df["close"].resample("ME").last(),
            "adjclose": df["adjclose"].resample("ME").last(),
            "sigma_E": sigma.resample("ME").last(),
            "dias_con_precio": df["close"].resample("ME").count(),
        }
    )
    mensual["ret_12m"] = mensual["adjclose"] / mensual["adjclose"].shift(12) - 1.0
    # Un mes sin cotizaciones (ticker ya deslistado) no debe generar fila
    mensual = mensual[mensual["dias_con_precio"] > 0]
    return mensual.drop(columns=["adjclose", "dias_con_precio"])


def panel_de_empresa(
    ticker: str,
    precios: pd.DataFrame,
    deuda_trimestral: pd.DataFrame,
    acciones_trimestrales: pd.DataFrame,
    tbill: pd.DataFrame,
    roa_anual: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Panel mensual de una empresa, todavía sin filtrar filas incompletas."""
    mensual = metricas_de_precios(precios)
    meses = mensual.index

    deuda_asof = asof_sin_lookahead(deuda_trimestral, meses, ["F", "deuda_corriente", "deuda_largo"])
    acciones_asof = asof_sin_lookahead(acciones_trimestrales, meses, ["acciones"])

    panel = mensual.join(deuda_asof).join(acciones_asof)
    if roa_anual is not None and not roa_anual.empty:
        panel = panel.join(asof_sin_lookahead(roa_anual, meses, ["roa"]))
    else:
        panel["roa"] = float("nan")
    panel["E"] = panel["close"] * panel["acciones"]
    panel = panel.reset_index().rename(columns={"index": "mes", "date": "mes"})
    panel = panel.merge(tbill.assign(mes=_ns(tbill["mes"])), on="mes", how="left")
    panel.insert(0, "ticker", ticker)
    return panel


def acciones_en_terminos_actuales(acciones: pd.DataFrame, splits: pd.DataFrame | None) -> pd.DataFrame:
    """Convierte las acciones as-reported de EDGAR a términos de hoy.

    Los precios de Yahoo y stockanalysis vienen restatados retroactivamente
    por splits (verificado en la auditoría de Fase 2: AAPL de 2019 aparece a
    49 dólares cuando cotizaba a ~197), pero las acciones de EDGAR son las
    reportadas en su momento. Sin este ajuste, E = precio x acciones queda
    mal por el factor acumulado de los splits posteriores: NVDA 40x abajo
    antes de 2021, Rite Aid 20x arriba antes de su reverse split de 2019.

    Multiplicar cada reporte por los ratios de los splits posteriores lo
    corrige de forma exacta: el producto telescopa, incluso si el split cae
    entre el reporte y el mes valuado.
    """
    if splits is None or splits.empty or acciones.empty:
        return acciones
    resultado = acciones.copy()
    factores = np.ones(len(resultado))
    fechas_reporte = _ns(resultado["end"]).to_numpy()
    for evento in splits.itertuples():
        anteriores = fechas_reporte < np.datetime64(pd.Timestamp(evento.fecha).as_unit("ns"))
        factores[anteriores] *= float(evento.ratio)
    resultado["acciones"] = resultado["acciones"] * factores
    return resultado


def trayectoria_equity(
    fechas: np.ndarray,
    adjclose: np.ndarray,
    mes,
    E_mes: float,
    ventana: int = 253,
    minimo: int = 127,
) -> np.ndarray | None:
    """Trayectoria diaria del equity del último año, insumo del solver de Merton.

    Toma los últimos `ventana` precios ajustados con fecha <= mes (253
    precios = 252 retornos) y los escala para que el último punto sea el
    market cap del mes: E_d = E_mes x adj_d / adj_t. Así los cambios en el
    número de acciones no meten saltos espurios en la volatilidad de V
    (misma lógica con la que el panel calcula sigma_E). Sin lookahead:
    estrictamente precios hasta el fin de mes. Devuelve None si hay menos
    de `minimo` observaciones.
    """
    pos = int(np.searchsorted(fechas, np.datetime64(mes), side="right"))
    valores = adjclose[max(0, pos - ventana) : pos]
    if len(valores) < minimo:
        return None
    # el parentesis importa: v/v[-1] hace exacto el 1.0 del ultimo punto
    return E_mes * (valores / valores[-1])


def filtrar_completo(panel: pd.DataFrame, inicio: str = "2016-01-01") -> pd.DataFrame:
    """Filas utilizables por el modelo: E > 0, F > 0, sigma_E y r presentes.

    F == 0 (empresa sin deuda reportada) se excluye: el DD diverge sin deuda
    y el paper también descarta esas observaciones.
    """
    completo = panel[
        (panel["mes"] >= pd.Timestamp(inicio))
        & (panel["E"] > 0)
        & (panel["F"] > 0)
        & panel["sigma_E"].notna()
        & (panel["sigma_E"] > 0)
        & panel["r"].notna()
    ].copy()
    return completo.reset_index(drop=True)
