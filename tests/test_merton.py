"""Fase 2: el solver iterativo recupera la verdad en escenarios sintéticos.

La prueba fuerte: se simula un browniano geométrico para V con sigma
conocido, se construye el equity E = call(V) día a día, y el solver debe
recuperar la serie de V casi exacta y un sigma_V igual a la volatilidad
realizada del camino simulado (contra la realizada, no contra el sigma
poblacional: con 252 días el error muestral del estimador es real).
"""

import numpy as np
import pytest
from scipy.special import ndtr

from model.black_scholes import call_price, implied_asset_value
from model.merton_iterativo import _invertir_serie, solve_merton

R = 0.03
T = 1.0


def _dd_analitico(V, F, r, T_horizonte):
    """DD calculado por fuera del solver, con los momentos realizados de la
    serie V verdadera: la referencia contra la que se compara solve_merton.
    mu_v es el retorno realizado anualizado, la convencion del solver."""
    sigma = np.diff(np.log(V)).std(ddof=1) * np.sqrt(252)
    mu_v = (V[-1] / V[0]) ** (252 / (len(V) - 1)) - 1.0
    mu = max(mu_v, r)
    dd = (np.log(V[-1] / F) + (mu - 0.5 * sigma**2) * T_horizonte) / (
        sigma * np.sqrt(T_horizonte)
    )
    return dd, mu_v


def _camino_gbm(sigma=0.30, mu=0.10, v0=100.0, n_dias=252, semilla=11):
    rng = np.random.default_rng(semilla)
    dt = 1.0 / 252
    logret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.normal(size=n_dias)
    return v0 * np.exp(np.cumsum(logret))


def test_newton_vectorizado_equivale_a_brentq():
    V_verdad = _camino_gbm()
    F = 60.0
    E = call_price(V_verdad, F, R, 0.30, T)
    V_newton = _invertir_serie(E, F, R, 0.30, T)
    V_brentq = np.array([implied_asset_value(e, F, R, 0.30, T) for e in E])
    assert np.allclose(V_newton, V_brentq, rtol=1e-8)


def test_recupera_v_y_sigma_en_sintetico():
    sigma_verdad = 0.30
    V_verdad = _camino_gbm(sigma=sigma_verdad)
    realizada = np.diff(np.log(V_verdad)).std(ddof=1) * np.sqrt(252)
    F = 60.0
    E = call_price(V_verdad, F, R, sigma_verdad, T)

    res = solve_merton(E, F, R, T)

    assert res.convergio
    # la tolerancia del iterativo es 1e-3 sobre sigma (la del paper); eso
    # deja al punto fijo a unos pocos milesimos del sigma realizado y a V
    # con un error relativo del mismo orden via la vega. Se pide lo que el
    # procedimiento garantiza, no mas.
    assert res.V == pytest.approx(V_verdad[-1], rel=5e-3)
    assert res.sigma_V == pytest.approx(realizada, abs=1e-2)
    assert 0.0 <= res.pi <= 1.0


@pytest.mark.parametrize("apalancamiento_alto,apalancamiento_bajo", [(80.0, 20.0)])
def test_dd_baja_cuando_sube_el_apalancamiento(apalancamiento_alto, apalancamiento_bajo):
    V_verdad = _camino_gbm()
    E_bajo = call_price(V_verdad, apalancamiento_bajo, R, 0.30, T)
    E_alto = call_price(V_verdad, apalancamiento_alto, R, 0.30, T)
    dd_bajo = solve_merton(E_bajo, apalancamiento_bajo, R, T).dd
    dd_alto = solve_merton(E_alto, apalancamiento_alto, R, T).dd
    assert dd_alto < dd_bajo


def test_dd_baja_cuando_sube_la_volatilidad():
    F = 60.0
    dd_por_sigma = []
    for sigma in [0.2, 0.6]:
        V = _camino_gbm(sigma=sigma, semilla=3)
        E = call_price(V, F, R, sigma, T)
        dd_por_sigma.append(solve_merton(E, F, R, T).dd)
    assert dd_por_sigma[1] < dd_por_sigma[0]


def test_naive_y_completo_correlacionan_en_panel_sintetico():
    # Correlacion de rangos (Spearman), no de Pearson: el naive produce
    # colas extremas cuando F es minuscula (ln((E+F)/F) explota) y ambos
    # modelos se usan como ranking en la validacion (deciles, AUC).
    from scipy.stats import spearmanr

    from model.naive import naive_dd

    rng = np.random.default_rng(5)
    dd_completo, dd_naive = [], []
    for i in range(60):
        sigma = rng.uniform(0.15, 0.9)
        F = rng.uniform(10.0, 150.0)
        V = _camino_gbm(sigma=sigma, mu=rng.uniform(-0.1, 0.3), semilla=100 + i)
        E = call_price(V, F, R, sigma, T)
        res = solve_merton(E, F, R, T)
        sigma_E = np.diff(np.log(E)).std(ddof=1) * np.sqrt(252)
        ret_12m = E[-1] / E[0] - 1.0
        dd_n, _ = naive_dd(E[-1], F, sigma_E, ret_12m, T)
        dd_completo.append(res.dd)
        dd_naive.append(dd_n)
    correlacion = spearmanr(dd_completo, dd_naive).statistic
    assert correlacion > 0.8


def test_dd_y_pi_contra_formula_analitica():
    # El valor exacto del DD: se calcula por fuera con la serie V verdadera
    # y el solver debe reproducirlo. Holgura 0.1: la tolerancia 1e-3 del
    # iterativo deja el punto fijo a unos milesimos del sigma realizado y
    # la sensibilidad d(DD)/d(sigma) es ~5 en este caso; un signo volteado
    # o un termino omitido mueve el DD en mas de 0.5, asi que 0.1 sigue
    # discriminando cualquier mutacion real.
    F = 60.0
    V = _camino_gbm(sigma=0.30, mu=0.15, semilla=21)
    E = call_price(V, F, R, 0.30, T)
    res = solve_merton(E, F, R, T)
    dd_esperado, _ = _dd_analitico(V, F, R, T)
    assert res.dd == pytest.approx(dd_esperado, abs=0.1)
    # pi debe ser N(-DD), no N(+DD): el signo volteado daria ~0.65 en vez
    # de ~0.35 (holgura 0.04 = la del dd propagada por la densidad normal)
    assert res.pi == pytest.approx(float(ndtr(-dd_esperado)), abs=0.04)
    assert res.pi < 0.5


def test_pi_mayor_a_medio_para_empresa_en_distress():
    # apalancamiento brutal y drift negativo: DD < 0 y pi > 0.5
    V = _camino_gbm(sigma=0.6, mu=-0.3, v0=100.0, semilla=8)
    F = 400.0
    E = call_price(V, F, R, 0.6, T)
    res = solve_merton(E, F, R, T)
    assert res.dd < 0
    assert res.pi > 0.5


def test_mu_usa_el_piso_de_la_tasa_libre_de_riesgo():
    # Regla documentada del PLAN: mu = max(mu_V, r). Con drift muy negativo
    # el DD debe calcularse con r; sin el piso, el DD seria claramente menor.
    F = 60.0
    V = _camino_gbm(sigma=0.30, mu=-0.60, semilla=33)
    E = call_price(V, F, R, 0.30, T)
    res = solve_merton(E, F, R, T)
    dd_esperado, mu_v = _dd_analitico(V, F, R, T)
    assert mu_v < R, "premisa del test: el drift estimado debe quedar bajo r"
    assert res.mu_V == pytest.approx(mu_v, rel=0.05)
    assert res.dd == pytest.approx(dd_esperado, abs=0.05)
    sigma_real = np.diff(np.log(V)).std(ddof=1) * np.sqrt(252)
    dd_sin_piso = (np.log(V[-1] / F) + (mu_v - 0.5 * sigma_real**2)) / sigma_real
    assert res.dd > dd_sin_piso + 0.1


def test_horizonte_distinto_de_un_ano():
    # T entra en el descuento, en el termino de drift y en el sqrt(T) del
    # denominador; con T=0.5 una omision de sqrt(T) mueve el DD ~40%.
    T_medio = 0.5
    F = 60.0
    V = _camino_gbm(sigma=0.30, mu=0.10, semilla=13)
    E = call_price(V, F, R, 0.30, T_medio)
    res = solve_merton(E, F, R, T=T_medio)
    dd_esperado, _ = _dd_analitico(V, F, R, T_medio)
    assert res.convergio
    assert res.dd == pytest.approx(dd_esperado, abs=0.07)


def test_sin_convergencia_devuelve_flag_false():
    V = _camino_gbm()
    E = call_price(V, 60.0, R, 0.30, T)
    res = solve_merton(E, 60.0, R, T, max_iteraciones=0)
    assert not res.convergio
    assert np.isfinite(res.dd)


def test_valida_entradas():
    with pytest.raises(ValueError):
        solve_merton([100.0], 60.0, R)
    with pytest.raises(ValueError):
        solve_merton([100.0, -5.0, 100.0], 60.0, R)
    with pytest.raises(ValueError):
        solve_merton(_camino_gbm(), 0.0, R)


def test_valida_nan_explicitamente():
    # NaN <= 0 es False: sin el chequeo explicito de isfinite, F o r en NaN
    # producirian resultados NaN silenciosos en vez de una excepcion
    V = _camino_gbm()
    E = call_price(V, 60.0, R, 0.30, T)
    with pytest.raises(ValueError):
        solve_merton(E, float("nan"), R)
    with pytest.raises(ValueError):
        solve_merton(E, 60.0, float("nan"))
    with pytest.raises(ValueError):
        solve_merton(np.array([100.0, float("nan"), 101.0]), 60.0, R)


def test_sigma_en_el_piso_se_reporta_como_no_convergido():
    # precios casi constantes (OTC iliquido con cierres repetidos): sigma
    # colapsa al piso y el DD seria arbitrario; el flag debe avisar
    E = np.full(252, 50.0)
    E[::50] += 1e-9  # variacion infima para pasar la validacion de entrada
    res = solve_merton(E, 30.0, R, T)
    assert not res.convergio
