"""Fase 2: Black-Scholes-Merton contra valores conocidos y propiedades básicas."""

import numpy as np
import pytest

from model.black_scholes import call_price, d1, d2, implied_asset_value


def test_precio_contra_valores_de_libro():
    # Caso clásico: S=100, K=100, r=5%, sigma=20%, T=1 -> C = 10.4506
    assert call_price(100, 100, 0.05, 0.2, 1.0) == pytest.approx(10.4506, abs=1e-4)
    # Hull, ejemplo 15.6: S=42, K=40, r=10%, sigma=20%, T=0.5 -> C = 4.76
    assert call_price(42, 40, 0.10, 0.2, 0.5) == pytest.approx(4.759, abs=1e-3)


def test_d2_es_d1_menos_sigma_raiz_t():
    esperado = d1(100, 80, 0.03, 0.3, 1.0) - 0.3
    assert d2(100, 80, 0.03, 0.3, 1.0) == pytest.approx(esperado)


def test_cotas_de_la_call():
    # intrinseco descontado <= call < V, para una malla amplia de parametros.
    # La cota inferior no puede ser estricta: en deep ITM con sigma pequeña
    # el valor temporal cae por debajo de la resolucion de float64.
    for V in [10.0, 100.0, 1e12]:
        for razon_deuda in [0.1, 0.5, 0.9, 2.0]:
            for sigma in [0.05, 0.3, 1.5]:
                F = V * razon_deuda
                c = call_price(V, F, 0.03, sigma, 1.0)
                intrinseco = max(V - F * np.exp(-0.03), 0.0)
                assert intrinseco - 1e-9 * V <= c < V


def test_creciente_en_v_y_en_sigma():
    c_base = call_price(100, 80, 0.03, 0.3, 1.0)
    assert call_price(110, 80, 0.03, 0.3, 1.0) > c_base
    assert call_price(100, 80, 0.03, 0.5, 1.0) > c_base


def test_vectorizado_coincide_con_escalar():
    V = np.array([50.0, 100.0, 200.0])
    vec = call_price(V, 80.0, 0.03, 0.3, 1.0)
    esc = [call_price(v, 80.0, 0.03, 0.3, 1.0) for v in V]
    assert np.allclose(vec, esc)


def test_inversion_recupera_v():
    # round trip: V -> E = call(V) -> implied_asset_value(E) == V
    for V in [15.0, 100.0, 5e11]:
        for razon_deuda in [0.05, 0.6, 1.5]:
            for sigma in [0.05, 0.4, 1.2]:
                F = V * razon_deuda
                E = call_price(V, F, 0.02, sigma, 1.0)
                V_hat = implied_asset_value(E, F, 0.02, sigma, 1.0)
                assert V_hat == pytest.approx(V, rel=1e-8)


def test_inversion_valida_entradas():
    with pytest.raises(ValueError):
        implied_asset_value(-5.0, 80.0, 0.03, 0.3, 1.0)
    with pytest.raises(ValueError):
        implied_asset_value(100.0, 0.0, 0.03, 0.3, 1.0)
    with pytest.raises(ValueError):
        implied_asset_value(100.0, 80.0, 0.03, -0.1, 1.0)
