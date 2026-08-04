"""Baseline contable (mini Shumway 2001) y la pregunta de señal incremental.

El baseline es un logit con ratios simples, todos ya en el panel:
    apalancamiento = F / (E + F)
    tamano         = ln(E)
    ret_12m        = retorno de la acción 12 meses
    sigma_E        = volatilidad del equity
    roa            = resultado neto anual / activos (EDGAR)

La pregunta de la mejora, la que haría un equipo de riesgo: en un logit
conjunto (ratios + DD), ¿el DD sigue siendo significativo? ¿Mejora el AUC
fuera de muestra?

Fuera de muestra = K-fold agrupado por empresa y estratificado: K = número
de quebradas; cada fold contiene UNA quebrada y una cuota determinista de
vivas (round-robin alfabético). Se ajusta sin el fold, se predice el fold,
y el AUC se calcula sobre el pool de predicciones out-of-fold. Ninguna
firma se predice con información de sí misma, y con 11 eventos un split
temporal dejaría casi todos los defaults a un solo lado.

Por qué NO leave-one-firm-out: con folds de una sola firma, las quebradas
se predicen con modelos entrenados con menos positivos (tasa base menor,
intercepto más bajo) y las sanas con modelos que tienen todos los
positivos; ese desajuste sistemático de interceptos entre folds hunde el
AUC de un predictor NULO muy por debajo de 0.5 (~0.08 en el test que cazó
este sesgo). Con folds quebrada + cuota de vivas, las tasas base de
entrenamiento son casi idénticas entre folds y el pooling es válido.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

RATIOS = ["apalancamiento", "tamano", "ret_12m", "sigma_E", "roa"]
WINSOR = (0.01, 0.99)


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Añade los ratios derivados y filtra a filas con todo el baseline.

    Winsoriza 1/99 todos los continuos no acotados, incluida sigma_E
    (hallazgo de la revisión adversarial: dejarla sin winsorizar era
    inconsistente con la política y sensibilizaba la pendiente a las colas
    post-COVID). El apalancamiento no se toca: está acotado en [0, 1].
    Nota honesta: los percentiles se calculan sobre la muestra completa
    antes del split de folds; es un leakage técnico cuantificado como
    inmaterial en la revisión (recorta 1% de colas, no ordena).
    """
    base = df.copy()
    base["apalancamiento"] = base["F"] / (base["E"] + base["F"])
    base["tamano"] = np.log(base["E"])
    base = base.dropna(subset=RATIOS + ["dd_naive", "dd_merton"]).reset_index(drop=True)
    for col in ["ret_12m", "roa", "tamano", "sigma_E", "dd_naive", "dd_merton"]:
        lo, hi = base[col].quantile(list(WINSOR))
        base[col] = base[col].clip(lo, hi)
    return base


def _ajustar(y: np.ndarray, X: np.ndarray):
    """Logit MLE simple; regularización mínima si la separación es perfecta."""
    try:
        return sm.Logit(y, X).fit(disp=0, maxiter=200)
    except Exception:
        # separación casi perfecta en algún fold: ridge suave como respaldo
        return sm.Logit(y, X).fit_regularized(alpha=1e-4, disp=0)


def auc_cv_por_grupos(df: pd.DataFrame, columnas: list[str], col_label: str = "default_12m") -> tuple[float, pd.Series]:
    """AUC sobre predicciones out-of-fold con K-fold agrupado y estratificado.

    K = número de empresas con eventos; el fold k contiene la quebrada k
    (todos sus meses) y las vivas asignadas por round-robin alfabético.
    Ver el docstring del módulo para el porqué frente al leave-one-out.
    Devuelve (auc, serie de predicciones alineada con df).
    """
    from validation.metricas import auc

    tiene_evento = df.groupby("ticker")[col_label].transform("max") == 1
    quebradas = sorted(df.loc[tiene_evento, "ticker"].unique())
    vivas = sorted(set(df["ticker"].unique()) - set(quebradas))
    k_folds = max(len(quebradas), 2)
    fold_de = {t: i for i, t in enumerate(quebradas)}
    fold_de.update({t: j % k_folds for j, t in enumerate(vivas)})
    asignacion = df["ticker"].map(fold_de).to_numpy()

    X_all = sm.add_constant(df[columnas].to_numpy(dtype=float))
    y_all = df[col_label].to_numpy(dtype=float)
    pred = np.full(len(df), np.nan)
    for k in range(k_folds):
        prueba = asignacion == k
        modelo = _ajustar(y_all[~prueba], X_all[~prueba])
        pred[prueba] = modelo.predict(X_all[prueba])
    return auc(y_all.astype(int), pred), pd.Series(pred, index=df.index)
