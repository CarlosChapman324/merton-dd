"""Fase 2: el naive contra un cálculo a mano y sus propiedades."""

import numpy as np
import pytest

from model.naive import naive_dd


def test_contra_calculo_a_mano():
    # E=80, F=20, sigma_E=0.4, ret=0.1:
    #   peso equity = 0.8
    #   sigma_V = 0.8*0.4 + 0.2*(0.05 + 0.25*0.4) = 0.32 + 0.03 = 0.35
    #   dd = [ln(100/20) + (0.1 - 0.5*0.35^2)] / 0.35
    dd, pi = naive_dd(80.0, 20.0, 0.4, 0.1)
    esperado = (np.log(5.0) + (0.1 - 0.5 * 0.35**2)) / 0.35
    assert dd == pytest.approx(esperado, rel=1e-12)
    assert pi == pytest.approx(0.0, abs=1e-5)  # dd ~ 4.7, pi minuscula


def test_horizonte_distinto_de_uno():
    # con T=4, sqrt(T)=2: si el codigo omitiera el sqrt del denominador o
    # el T del numerador, el valor se aleja mucho del calculado a mano
    dd, _ = naive_dd(80.0, 20.0, 0.4, 0.1, T=4.0)
    sigma_v = 0.35
    esperado = (np.log(5.0) + (0.1 - 0.5 * sigma_v**2) * 4.0) / (sigma_v * 2.0)
    assert dd == pytest.approx(esperado, rel=1e-12)


def test_vectorizado_coincide_con_escalar():
    E = np.array([80.0, 50.0, 120.0])
    F = np.array([20.0, 60.0, 5.0])
    sigma = np.array([0.4, 0.9, 0.2])
    ret = np.array([0.1, -0.3, 0.25])
    dd_vec, pi_vec = naive_dd(E, F, sigma, ret)
    for i in range(3):
        dd_i, pi_i = naive_dd(E[i], F[i], sigma[i], ret[i])
        assert dd_vec[i] == pytest.approx(dd_i)
        assert pi_vec[i] == pytest.approx(pi_i)


def test_dd_baja_con_apalancamiento_y_volatilidad():
    dd_base, _ = naive_dd(80.0, 20.0, 0.4, 0.1)
    dd_mas_deuda, _ = naive_dd(80.0, 60.0, 0.4, 0.1)
    dd_mas_vol, _ = naive_dd(80.0, 20.0, 0.8, 0.1)
    assert dd_mas_deuda < dd_base
    assert dd_mas_vol < dd_base


def test_pi_siempre_en_cero_uno():
    rng = np.random.default_rng(9)
    E = rng.uniform(1.0, 1e12, 500)
    F = rng.uniform(0.5, 1e12, 500)
    sigma = rng.uniform(0.01, 3.0, 500)
    ret = rng.uniform(-0.95, 5.0, 500)
    _, pi = naive_dd(E, F, sigma, ret)
    assert np.all((pi >= 0.0) & (pi <= 1.0))


def test_nan_se_propaga_sin_romper():
    dd, pi = naive_dd(np.array([80.0, 80.0]), 20.0, 0.4, np.array([0.1, np.nan]))
    assert np.isfinite(dd[0])
    assert np.isnan(dd[1]) and np.isnan(pi[1])
