"""Fase 4: baseline, K-fold agrupado, curvas alineadas y calibración."""

import numpy as np
import pandas as pd
import pytest

from validation.baseline import auc_cv_por_grupos
from validation.calibracion import calibracion_global, calibracion_por_decil
from validation.casos import curvas_alineadas, resumen_anticipacion


def test_cv_agrupado_sin_leakage_ni_sesgo_de_tasa_base():
    # Senal verdadera -> AUC alto out-of-fold. Ruido puro -> AUC ~0.5:
    # esta segunda mitad es la que cazo el sesgo del leave-one-out (daba
    # 0.08 con ruido por el desajuste de tasas base entre folds).
    rng = np.random.default_rng(4)
    n_firmas, meses = 30, 12
    filas = []
    for i in range(n_firmas):
        en_default = i < 5
        for m in range(meses):
            filas.append(
                {
                    "ticker": f"T{i}",
                    "default_12m": int(en_default),
                    "senal": (1.0 if en_default else 0.0) + rng.normal(0, 0.3),
                    "ruido": rng.normal(),
                }
            )
    df = pd.DataFrame(filas)
    auc_senal, pred = auc_cv_por_grupos(df, ["senal"])
    auc_ruido, _ = auc_cv_por_grupos(df, ["ruido"])
    assert auc_senal > 0.9
    assert abs(auc_ruido - 0.5) < 0.12
    assert pred.notna().all()


def _panel_casos():
    meses = pd.date_range("2021-01-31", "2022-12-31", freq="ME")
    filing = pd.Timestamp("2022-06-15")
    filas = []
    for m in meses:
        filas.append(
            {
                "ticker": "QUEB",
                "mes": m,
                "fecha_filing": filing,
                "post_filing": m > filing,
                "es_default": True,
                "dd_merton": 5.0 - 0.2 * len(filas),
                "dd_naive": 5.0 - 0.2 * len(filas),
                "pi_merton": 0.1,
                "pi_naive": 0.1,
            }
        )
    return pd.DataFrame(filas)


def test_curvas_alineadas_cuenta_meses_al_evento():
    curvas = curvas_alineadas(_panel_casos())
    # el ultimo mes previo (mayo 2022) es meses_al_filing = 1
    assert curvas["meses_al_filing"].min() == 1
    ultimo = curvas[curvas["meses_al_filing"] == 1]["mes"].iloc[0]
    assert ultimo == pd.Timestamp("2022-05-31")
    # nada posterior al filing entra en las curvas
    assert (curvas["mes"] < pd.Timestamp("2022-06-15")).all()


def test_resumen_anticipacion_encuentra_el_cruce():
    curvas = pd.DataFrame(
        {
            "ticker": ["Q"] * 6,
            "meses_al_filing": [1, 2, 3, 4, 5, 6],
            # DD >= 2 a los 5-6 meses, < 2 del mes 4 en adelante
            "dd_merton": [-1.0, 0.0, 1.0, 1.9, 2.5, 3.0],
            "dd_naive": [-1.0, 0.0, 1.0, 1.9, 2.5, 3.0],
        }
    )
    resumen = resumen_anticipacion(curvas, umbral_dd=2.0)
    assert resumen["meses_alerta_merton"].iloc[0] == 4
    assert not resumen["meses_alerta_merton_censurado"].iloc[0]


def test_resumen_anticipacion_marca_censura():
    # nunca estuvo sana en la ventana: el valor es un minimo (censurado)
    curvas = pd.DataFrame(
        {
            "ticker": ["Q"] * 3,
            "meses_al_filing": [1, 2, 3],
            "dd_merton": [-1.0, 0.0, 1.0],
            "dd_naive": [-1.0, 0.0, 1.0],
        }
    )
    resumen = resumen_anticipacion(curvas, umbral_dd=2.0)
    assert resumen["meses_alerta_merton"].iloc[0] == 3
    assert resumen["meses_alerta_merton_censurado"].iloc[0]


def test_calibracion_recupera_tasas():
    rng = np.random.default_rng(8)
    n = 4000
    meses = np.tile(pd.date_range("2020-01-31", periods=40, freq="ME"), n // 40)
    pi = rng.uniform(0, 0.2, n)
    df = pd.DataFrame(
        {
            "mes": meses,
            "pi_naive": pi,
            "pi_merton": pi,
            "default_12m": rng.binomial(1, pi),  # perfectamente calibrado
        }
    )
    global_ = calibracion_global(df)
    assert global_["ratio"].iloc[0] == pytest.approx(1.0, abs=0.15)
    por_decil = calibracion_por_decil(df, "pi_naive")
    top = por_decil.loc[10]
    assert top["pi_promedio"] == pytest.approx(top["tasa_realizada"], abs=0.05)