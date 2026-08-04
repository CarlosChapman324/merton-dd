"""Merton DD naive: la alternativa de Bharath & Shumway (2008) sin resolver nada.

Definiciones exactas del paper:
    sigma_V_naive = (E/(E+F)) * sigma_E + (F/(E+F)) * (0.05 + 0.25 * sigma_E)
    mu_naive      = retorno simple de la acción en los 12 meses previos
    DD_naive      = [ln((E+F)/F) + (mu_naive - 0.5 sigma_V_naive^2) T]
                    / (sigma_V_naive sqrt(T))
    pi_naive      = N(-DD_naive)

La gracia del naive: aproxima el valor de los activos con E + F y la
volatilidad de la deuda con 0.05 + 0.25 sigma_E (5 puntos de vol de piso
más un cuarto de la del equity), sin resolver ningún sistema. La forma
funcional es la misma que la del Merton completo; si predice igual de
bien, lo que importa es la forma, no la solución (esa es la tesis del
paper que esta réplica pone a prueba).

Matemática pura, vectorizada con numpy: acepta escalares o arrays.
"""

import numpy as np
from scipy.special import ndtr


def naive_dd(E, F, sigma_E, ret_12m, T: float = 1.0):
    """DD y pi naive. Devuelve (dd, pi); escalares si la entrada es escalar.

    Donde algún insumo sea NaN (por ejemplo ret_12m en los primeros doce
    meses de cotización) el resultado es NaN, sin romper el resto del array.
    """
    E = np.asarray(E, dtype=float)
    F = np.asarray(F, dtype=float)
    sigma_E = np.asarray(sigma_E, dtype=float)
    ret_12m = np.asarray(ret_12m, dtype=float)

    peso_equity = E / (E + F)
    sigma_V = peso_equity * sigma_E + (1.0 - peso_equity) * (0.05 + 0.25 * sigma_E)
    dd = (np.log((E + F) / F) + (ret_12m - 0.5 * sigma_V**2) * T) / (
        sigma_V * np.sqrt(T)
    )
    pi = ndtr(-dd)
    if np.ndim(dd) == 0:
        return float(dd), float(pi)
    return dd, pi
