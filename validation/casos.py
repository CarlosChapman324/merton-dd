"""Casos de estudio: las curvas de DD alineadas al evento (el gráfico estrella).

Para cada quebrada, la trayectoria de DD (completo y naive) en los meses
previos al filing, con el eje en tiempo-al-evento (t = -24 ... -1). Como
contraste se calcula la banda de las vivas (mediana y cuartiles del DD del
panel sano): si el DD "vio venir" los defaults, las curvas deben descender
y separarse de la banda al acercarse a t = 0.
"""

import pandas as pd

MESES_MAX = 24


def curvas_alineadas(panel_dd: pd.DataFrame, meses_max: int = MESES_MAX) -> pd.DataFrame:
    """DD de cada quebrada por meses al filing (1 = último mes previo)."""
    q = panel_dd[panel_dd["fecha_filing"].notna() & ~panel_dd["post_filing"]].copy()
    q["meses_al_filing"] = (
        q["fecha_filing"].dt.to_period("M").astype(int)
        - q["mes"].dt.to_period("M").astype(int)
    )
    q = q[(q["meses_al_filing"] >= 1) & (q["meses_al_filing"] <= meses_max)]
    return q[["ticker", "mes", "meses_al_filing", "dd_merton", "dd_naive", "pi_merton", "pi_naive"]].sort_values(
        ["ticker", "meses_al_filing"]
    )


def banda_vivas(panel_dd: pd.DataFrame) -> pd.DataFrame:
    """Mediana y cuartiles del DD de las empresas vivas (referencia de sano)."""
    vivas = panel_dd[~panel_dd["es_default"]]
    filas = []
    for col in ["dd_merton", "dd_naive"]:
        filas.append(
            {
                "modelo": col,
                "p25": float(vivas[col].quantile(0.25)),
                "mediana": float(vivas[col].median()),
                "p75": float(vivas[col].quantile(0.75)),
            }
        )
    return pd.DataFrame(filas)


def resumen_anticipacion(curvas: pd.DataFrame, umbral_dd: float = 2.0) -> pd.DataFrame:
    """Con cuánta antelación cruzó cada quebrada el umbral de peligro.

    Devuelve, por empresa y modelo, el primer mes (contando hacia atrás
    desde el filing) en que el DD quedó por debajo del umbral y ya no lo
    volvió a superar. DD < 2 es una zona convencional de alerta (pi > 2%).

    Censura (hallazgo de la revisión adversarial): si la empresa nunca
    estuvo sana dentro de la ventana observada, el valor es un mínimo, no
    un cruce real; la columna *_censurado lo marca y el reporte debe
    mostrarlo como "N+". La regla es de alerta SOSTENIDA: un solo mes de
    rebote por encima del umbral reinicia el conteo (Revlon con el naive
    es el ejemplo: un squeeze de 2021 rompe la racha y el valor cae de
    24+ a 7 sin que los modelos discrepen de fondo).
    """
    filas = []
    for t, g in curvas.groupby("ticker"):
        g = g.sort_values("meses_al_filing")  # 1, 2, ..., 24
        fila = {"ticker": t}
        for col in ["dd_merton", "dd_naive"]:
            nombre = "meses_alerta_" + col.split("_")[1]
            debajo = g[g[col] >= umbral_dd]["meses_al_filing"]
            censurado = len(debajo) == 0
            fila[nombre] = int(debajo.min() - 1) if not censurado else int(g["meses_al_filing"].max())
            fila[nombre + "_censurado"] = censurado
        filas.append(fila)
    return pd.DataFrame(filas)
