"""Fase 3: etiquetado, AUC, deciles, bootstrap y logit con datos sintéticos."""

import numpy as np
import pandas as pd
import pytest

from validation.eventos import etiquetar
from validation.metricas import (
    accuracy_ratio,
    auc,
    bootstrap_auc_pareado,
    resumen_bootstrap,
    tabla_deciles,
)
from validation.modelos import logit_cluster


def _panel_sintetico():
    """Una quebrada (filing 2020-06-15) y una viva, meses de 2019 a 2021."""
    meses = pd.date_range("2019-01-31", "2021-12-31", freq="ME")
    filas = []
    for m in meses:
        filas.append(
            {
                "ticker": "QUEB",
                "mes": m,
                "fecha_filing": pd.Timestamp("2020-06-15"),
                "post_filing": m > pd.Timestamp("2020-06-15"),
            }
        )
        filas.append(
            {"ticker": "VIVA", "mes": m, "fecha_filing": pd.NaT, "post_filing": False}
        )
    return pd.DataFrame(filas)


def test_etiquetado_ventana_de_12_meses():
    etiquetado = etiquetar(_panel_sintetico())
    queb = etiquetado[etiquetado.ticker == "QUEB"].set_index("mes")
    # exactamente 12 meses antes del filing: junio 2019 a mayo 2020 -> 1
    assert queb.loc["2020-05-31", "default_12m"] == 1
    assert queb.loc["2019-06-30", "default_12m"] == 1
    # 13 meses antes: 0
    assert queb.loc["2019-05-31", "default_12m"] == 0
    # el mes del filing y los posteriores no existen (post_filing)
    assert pd.Timestamp("2020-06-30") not in queb.index
    # la viva nunca esta etiquetada y conserva todos sus meses
    viva = etiquetado[etiquetado.ticker == "VIVA"]
    assert viva.default_12m.sum() == 0 and len(viva) == 36


def test_auc_casos_conocidos():
    # a mano: pares (pos, neg) donde pos > neg: 3 de 4 -> 0.75
    assert auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert auc([0, 1], [0.1, 0.9]) == 1.0
    assert auc([0, 1], [0.9, 0.1]) == 0.0
    assert auc([0, 1], [0.5, 0.5]) == 0.5  # empate: medio punto
    assert np.isnan(auc([0, 0], [0.1, 0.2]))
    assert accuracy_ratio(0.75) == pytest.approx(0.5)


def test_tabla_deciles_separacion_perfecta():
    # 3 meses x 20 empresas; las 2 con mas score de cada mes en default
    filas = []
    for m in pd.date_range("2020-01-31", periods=3, freq="ME"):
        for i in range(20):
            filas.append(
                {"mes": m, "ticker": f"T{i}", "score": i, "default_12m": int(i >= 18)}
            )
    df = pd.DataFrame(filas)
    tabla = tabla_deciles(df, "score")
    assert tabla.loc[10, "pct_defaults"] == pytest.approx(1.0)
    assert tabla.loc[10, "defaults"] == 6


def test_bootstrap_pareado_scores_identicos():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "ticker": np.repeat([f"T{i}" for i in range(20)], 10),
            "default_12m": np.repeat([1, 1, 0, 0] + [0] * 16, 10),
            "s1": rng.normal(size=200),
        }
    )
    df["s2"] = df["s1"]
    replicas = bootstrap_auc_pareado(df, "s1", "s2", n_replicas=50, semilla=7)
    assert (replicas["diff"] == 0).all()
    resumen = resumen_bootstrap(replicas)
    assert resumen["prob_a_mayor_igual_b"] == 1.0
    # reproducible con la misma semilla
    replicas2 = bootstrap_auc_pareado(df, "s1", "s2", n_replicas=50, semilla=7)
    assert replicas["auc_a"].equals(replicas2["auc_a"])


def test_logit_recupera_el_signo_y_es_significativo():
    rng = np.random.default_rng(11)
    n_empresas, meses = 50, 24
    ticker = np.repeat([f"T{i}" for i in range(n_empresas)], meses)
    x = rng.normal(size=n_empresas * meses)
    prob = 1 / (1 + np.exp(-(-2.0 - 1.5 * x)))  # beta negativo: mas DD, menos default
    y = rng.binomial(1, prob)
    df = pd.DataFrame({"ticker": ticker, "dd": x, "default_12m": y})
    modelo = logit_cluster(df, ["dd"])
    beta = modelo.params[1]
    assert beta == pytest.approx(-1.5, abs=0.3)
    assert modelo.pvalues[1] < 0.01