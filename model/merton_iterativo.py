"""Merton DD completo: el procedimiento iterativo de Bharath-Shumway / Crosbie-Bohn.

Algoritmo, tal como lo describe el paper:
1. Arrancar con sigma_V = sigma_E * E / (E + F).
2. Con ese sigma_V, invertir Black-Scholes-Merton día a día para obtener
   la serie diaria de V implícito del último año (F y r fijos en sus
   valores del mes, como hace el paper).
3. Recalcular sigma_V como la volatilidad anualizada de los retornos
   diarios logarítmicos de ese V implícito.
4. Repetir hasta que el cambio en sigma_V sea menor que la tolerancia
   (1e-3), con tope de iteraciones.

Con el sigma_V convergido se estima mu_V como el retorno realizado de V en
la ventana, anualizado: (V_final / V_inicial)^(252/n) - 1. Es lo que hace
el paper (el retorno del activo del año previo) y es robusto: telescopa,
así que solo dependen el primer y el último V. La alternativa (media
aritmética de retornos diarios anualizada) explota con volatilidad alta:
una acción que cae 70% con rebotes de +/-20% diarios puede dar una media
aritmética de +80% anual e inflar el DD justo en las empresas quebradas
(hallazgo de la revisión de Fase 2). Después:

    DD = [ln(V/F) + (mu - 0.5 sigma_V^2) T] / (sigma_V sqrt(T))
    pi_merton = N(-DD)

con mu = max(mu_V, r): un drift estimado por debajo de la tasa libre de
riesgo suele ser ruido de la ventana de un año y produce DD absurdos
(decisión documentada en el PLAN, igual que hacen varias réplicas).

Matemática pura: recibe arrays de numpy, no toca la red.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from model import black_scholes as bs

DIAS_HABILES_ANO = 252
TOLERANCIA_SIGMA = 1e-3
MAX_ITERACIONES = 50
SIGMA_MINIMO = 1e-4


@dataclass(frozen=True)
class ResultadoMerton:
    """Salida del solver para un empresa-mes."""

    V: float           # valor de los activos al cierre del mes
    sigma_V: float     # volatilidad anualizada de los activos
    mu_V: float        # retorno simple medio anualizado de los activos
    dd: float          # distancia al default
    pi: float          # N(-DD)
    convergio: bool
    iteraciones: int


def _invertir_serie(E: np.ndarray, F: float, r: float, sigma_V: float, T: float) -> np.ndarray:
    """Para cada equity diario, el V que resuelve call_price(V) = E.

    Newton-Raphson vectorizado. La derivada del precio de la call respecto
    de V es N(d1), así que cada paso es V <- V - (call(V) - E) / N(d1).
    Se arranca en la cota superior V0 = E + F e^(-rT); como la call es
    convexa y creciente en V, los iterados bajan monótonamente hacia la
    raíz sin salirse del bracket. Equivale a brentq sobre cada día (hay
    test de eso) pero resuelve los 252 días de una pasada.
    """
    V = E + F * np.exp(-r * T)
    tolerancia = 1e-10 * np.maximum(E, 1.0)
    for _ in range(100):
        error = bs.call_price(V, F, r, sigma_V, T) - E
        if np.all(np.abs(error) <= tolerancia):
            break
        derivada = ndtr(bs.d1(V, F, r, sigma_V, T))
        V = V - error / np.maximum(derivada, 1e-12)
        # protección numérica: la raíz siempre está por encima de E
        V = np.maximum(V, E)
    return V


def solve_merton(
    equity_series,
    F: float,
    r: float,
    T: float = 1.0,
    tolerancia: float = TOLERANCIA_SIGMA,
    max_iteraciones: int = MAX_ITERACIONES,
) -> ResultadoMerton:
    """Corre el iterativo sobre un año de valores diarios de equity.

    equity_series: valores diarios de mercado del equity (el último es el
    del cierre del mes para el que se calcula el DD). F y r son la deuda y
    la tasa de ese mes.
    """
    E = np.asarray(equity_series, dtype=float)
    if E.ndim != 1 or len(E) < 2:
        raise ValueError("equity_series debe ser una serie con al menos 2 valores")
    if np.any(~np.isfinite(E)) or np.any(E <= 0):
        raise ValueError("equity_series debe ser positiva y finita")
    # los NaN pasan silenciosos por las comparaciones (NaN <= 0 es False),
    # asi que la condicion se escribe al reves: exigir finito y positivo
    if not (np.isfinite(F) and F > 0 and np.isfinite(r) and np.isfinite(T) and T > 0):
        raise ValueError("F, r y T deben ser finitos, con F y T positivos")

    E_t = E[-1]

    # Paso 1 del paper: semilla sigma_V = sigma_E * E / (E + F)
    sigma_E = np.diff(np.log(E)).std(ddof=1) * np.sqrt(DIAS_HABILES_ANO)
    sigma_V = max(sigma_E * E_t / (E_t + F), SIGMA_MINIMO)

    convergio = False
    iteraciones = 0
    for iteraciones in range(1, max_iteraciones + 1):
        V = _invertir_serie(E, F, r, sigma_V, T)
        sigma_nuevo = np.diff(np.log(V)).std(ddof=1) * np.sqrt(DIAS_HABILES_ANO)
        if not np.isfinite(sigma_nuevo) or sigma_nuevo <= 0:
            break
        if abs(sigma_nuevo - sigma_V) < tolerancia:
            sigma_V = max(sigma_nuevo, SIGMA_MINIMO)
            convergio = True
            break
        sigma_V = max(sigma_nuevo, SIGMA_MINIMO)

    # sigma en el piso: los precios de la ventana estan casi constantes
    # (tipico de OTC ilíquido con cierres repetidos) y el DD resultante
    # seria numericamente arbitrario; se reporta como no convergido
    if sigma_V <= SIGMA_MINIMO * 1.001:
        convergio = False

    # Serie final de V con el sigma definitivo, y momentos sobre ella.
    # mu_V: retorno realizado anualizado (telescopa: robusto a la
    # volatilidad de la ventana, ver docstring del modulo)
    V = _invertir_serie(E, F, r, sigma_V, T)
    n_retornos = len(V) - 1
    mu_V = float((V[-1] / V[0]) ** (DIAS_HABILES_ANO / n_retornos) - 1.0)
    V_t = float(V[-1])

    mu = max(mu_V, r)
    dd = (np.log(V_t / F) + (mu - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    return ResultadoMerton(
        V=V_t,
        sigma_V=float(sigma_V),
        mu_V=mu_V,
        dd=float(dd),
        pi=float(ndtr(-dd)),
        convergio=convergio,
        iteraciones=iteraciones,
    )
