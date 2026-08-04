"""Fase 1: lógica del panel con datos sintéticos, sin tocar la red.

Lo crítico es el anti lookahead: un balance publicado después del fin de mes
no puede aparecer en ese mes, aunque su fecha de balance sea anterior.
"""

import numpy as np
import pandas as pd
import pytest

from data.panel import (
    acciones_en_terminos_actuales,
    asof_sin_lookahead,
    filtrar_completo,
    metricas_de_precios,
    trayectoria_equity,
)


def _trimestral():
    return pd.DataFrame(
        {
            "end": pd.to_datetime(["2020-03-31", "2020-06-30"]),
            "filed": pd.to_datetime(["2020-05-10", "2020-08-08"]),
            "F": [100.0, 200.0],
        }
    )


def test_asof_no_mira_al_futuro():
    meses = pd.DatetimeIndex(["2020-04-30", "2020-05-31", "2020-07-31", "2020-08-31"])
    resultado = asof_sin_lookahead(_trimestral(), meses, ["F"])
    # Abril: el balance de marzo aún no se ha publicado (filed 10 de mayo)
    assert np.isnan(resultado.loc["2020-04-30", "F"])
    # Mayo: ya se publicó el balance de marzo
    assert resultado.loc["2020-05-31", "F"] == 100.0
    # Julio: el balance de junio aún no se publica (filed 8 de agosto)
    assert resultado.loc["2020-07-31", "F"] == 100.0
    # Agosto: ya está el de junio
    assert resultado.loc["2020-08-31", "F"] == 200.0


def test_asof_vacio_devuelve_nan():
    meses = pd.DatetimeIndex(["2020-04-30"])
    vacio = pd.DataFrame(columns=["end", "filed", "F"])
    resultado = asof_sin_lookahead(vacio, meses, ["F"])
    assert resultado["F"].isna().all()


def _precios_sinteticos(n_dias=600, sigma_diaria=0.02, semilla=7):
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range("2019-01-01", periods=n_dias)
    logret = rng.normal(0.0, sigma_diaria, n_dias)
    precios = 50.0 * np.exp(np.cumsum(logret))
    return pd.DataFrame(
        {"date": fechas, "close": precios, "adjclose": precios, "source": "test"}
    )


def test_sigma_anualizada_recupera_la_verdadera():
    sigma_diaria = 0.02
    df = metricas_de_precios(_precios_sinteticos(sigma_diaria=sigma_diaria))
    esperada = sigma_diaria * np.sqrt(252)
    ultima = df["sigma_E"].dropna().iloc[-1]
    assert abs(ultima - esperada) / esperada < 0.15


def test_ret_12m_necesita_12_meses():
    df = metricas_de_precios(_precios_sinteticos())
    assert df["ret_12m"].iloc[:12].isna().all()
    assert df["ret_12m"].iloc[12:].notna().any()


def test_acciones_en_terminos_actuales():
    # reportes trimestrales de 100 acciones; split 4:1 en 2020-08-01 y
    # reverse 1:20 en 2021-02-01. Los reportes anteriores a ambos eventos
    # se multiplican por 4 * 1/20; el intermedio solo por 1/20; el
    # posterior queda igual.
    acciones = pd.DataFrame(
        {
            "end": pd.to_datetime(["2020-06-30", "2020-09-30", "2021-03-31"]),
            "filed": pd.to_datetime(["2020-08-05", "2020-11-05", "2021-05-05"]),
            "acciones": [100.0, 400.0, 20.0],
        }
    )
    splits = pd.DataFrame(
        {
            "fecha": pd.to_datetime(["2020-08-01", "2021-02-01"]),
            "ratio": [4.0, 1 / 20],
        }
    )
    resultado = acciones_en_terminos_actuales(acciones, splits)
    assert resultado["acciones"].tolist() == [100.0 * 4 / 20, 400.0 / 20, 20.0]


def test_acciones_sin_splits_quedan_iguales():
    acciones = pd.DataFrame(
        {
            "end": pd.to_datetime(["2020-06-30"]),
            "filed": pd.to_datetime(["2020-08-05"]),
            "acciones": [100.0],
        }
    )
    vacio = pd.DataFrame({"fecha": pd.Series(dtype="datetime64[ns]"), "ratio": pd.Series(dtype="float64")})
    assert acciones_en_terminos_actuales(acciones, vacio)["acciones"].tolist() == [100.0]
    assert acciones_en_terminos_actuales(acciones, None)["acciones"].tolist() == [100.0]


def test_trayectoria_equity_sin_lookahead_y_escalada():
    fechas = pd.bdate_range("2020-01-01", periods=300).to_numpy()
    adj = np.linspace(10.0, 40.0, 300)
    mes = fechas[259]  # un dia intermedio: lo posterior no puede entrar
    tray = trayectoria_equity(fechas, adj, mes, E_mes=1000.0)
    assert tray is not None
    assert len(tray) == 253
    # el ultimo punto es exactamente el market cap del mes
    assert tray[-1] == 1000.0
    # y la trayectoria replica los retornos ajustados de la ventana
    assert tray[0] == pytest.approx(1000.0 * adj[259 - 252] / adj[259], rel=1e-12)
    # dias posteriores al mes no participan: recortar la serie despues del
    # mes produce exactamente la misma trayectoria
    tray_recortada = trayectoria_equity(fechas[:260], adj[:260], mes, E_mes=1000.0)
    assert np.array_equal(tray, tray_recortada)


def test_trayectoria_equity_exige_minimo_de_dias():
    fechas = pd.bdate_range("2020-01-01", periods=100).to_numpy()
    adj = np.full(100, 5.0)
    assert trayectoria_equity(fechas, adj, fechas[-1], E_mes=100.0) is None


def test_filtrar_completo_exige_insumos_positivos():
    base = pd.DataFrame(
        {
            "mes": pd.to_datetime(["2020-01-31"] * 4),
            "E": [10.0, 0.0, 10.0, 10.0],
            "F": [5.0, 5.0, 0.0, 5.0],
            "sigma_E": [0.3, 0.3, 0.3, np.nan],
            "r": [0.02, 0.02, 0.02, 0.02],
        }
    )
    limpio = filtrar_completo(base)
    assert len(limpio) == 1
