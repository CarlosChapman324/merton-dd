"""Black-Scholes-Merton: precio de una call europea y sus piezas.

En el modelo de Merton (1974) el patrimonio E de una empresa es una call
sobre el valor de sus activos V, con strike igual al valor nominal de la
deuda F y vencimiento T:

    E = V N(d1) - e^(-rT) F N(d2)
    d1 = [ln(V/F) + (r + 0.5 sigma_V^2) T] / (sigma_V sqrt(T))
    d2 = d1 - sigma_V sqrt(T)

Matemática pura: sin red, sin pandas, testeable offline. Las funciones
aceptan escalares o arrays de numpy (se vectorizan solas). Como normal
acumulada se usa scipy.special.ndtr, que es la misma N de norm.cdf pero
llamada directa en C: el solver iterativo la evalúa millones de veces.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr


def d1(V, F, r, sigma_V, T):
    """d1 de Black-Scholes-Merton. Acepta escalares o arrays."""
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=float)
    return (np.log(V / F) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))


def d2(V, F, r, sigma_V, T):
    """d2 = d1 - sigma_V sqrt(T)."""
    return d1(V, F, r, sigma_V, T) - sigma_V * np.sqrt(T)


def call_price(V, F, r, sigma_V, T):
    """Precio de la call europea: el valor del equity según Merton."""
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=float)
    precio = V * ndtr(d1(V, F, r, sigma_V, T)) - F * np.exp(-r * T) * ndtr(
        d2(V, F, r, sigma_V, T)
    )
    # Si la entrada era escalar, devolver escalar
    return float(precio) if np.ndim(precio) == 0 else precio


def implied_asset_value(E: float, F: float, r: float, sigma_V: float, T: float) -> float:
    """Invierte la ecuación de la call: dado E observado, encuentra V.

    La call es estrictamente creciente en V y la raíz está encerrada:
        E < V* <= E + F e^(-rT)
    porque la call vale menos que V (el strike es positivo) y al menos su
    valor intrínseco descontado V - F e^(-rT). Con ese bracket, brentq
    converge garantizado. Esta versión escalar es la referencia legible;
    el solver iterativo usa una inversión vectorizada equivalente (ver
    merton_iterativo) validada contra esta en los tests.
    """
    if E <= 0 or F <= 0 or sigma_V <= 0 or T <= 0:
        raise ValueError("E, F, sigma_V y T deben ser positivos")
    lo = E
    # margen relativo minúsculo: con sigma casi cero la raíz cae justo en
    # la cota superior y brentq necesita que el bracket la contenga
    hi = (E + F * np.exp(-r * T)) * (1.0 + 1e-12)
    return float(
        brentq(
            lambda V: call_price(V, F, r, sigma_V, T) - E,
            lo,
            hi,
            xtol=1e-12 * (E + F),
            maxiter=200,
        )
    )
