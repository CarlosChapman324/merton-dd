"""Logit pooled con errores agrupados por empresa.

Réplica en espíritu de los hazard models del paper. Con 11 eventos el panel
no da para un Cox fino, así que se usa logit pooled sobre empresa-mes con
errores estándar cluster por empresa (los meses de una firma no son
independientes) y se dice explícitamente, como pide el PLAN.

El DD se winsoriza al 1%/99% solo para el logit: las colas de DD > 40 de
empresas casi sin deuda (Accenture) distorsionarían la pendiente sin
aportar información de ranking. Deciles y AUC son rank-based y no se
winsorizan.
"""

import pandas as pd
import statsmodels.api as sm

WINSOR = (0.01, 0.99)


def logit_cluster(
    df: pd.DataFrame,
    columnas_score: list[str],
    col_label: str = "default_12m",
    winsor: tuple[float, float] = WINSOR,
):
    """Logit de default_12m sobre uno o más scores, SE cluster por empresa.

    Devuelve el resultado de statsmodels (params, bse, pvalues, prsquared).
    """
    base = df.dropna(subset=columnas_score).copy()
    for col in columnas_score:
        lo, hi = base[col].quantile(list(winsor))
        base[col] = base[col].clip(lo, hi)
    X = sm.add_constant(base[columnas_score].to_numpy(dtype=float))
    y = base[col_label].to_numpy(dtype=float)
    grupos = base["ticker"].astype("category").cat.codes.to_numpy()
    modelo = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": grupos})
    return modelo


def resumen_logit(modelo, nombres: list[str]) -> pd.DataFrame:
    """Tabla legible: coeficiente, SE cluster, z y p por regresor."""
    etiquetas = ["const"] + nombres
    return pd.DataFrame(
        {
            "coef": modelo.params,
            "se_cluster": modelo.bse,
            "z": modelo.tvalues,
            "p": modelo.pvalues,
        },
        index=etiquetas,
    )
