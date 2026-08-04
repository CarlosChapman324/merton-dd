"""AUC, accuracy ratio, deciles y bootstrap por empresa.

Todo rank-based y sin dependencias más allá de numpy/pandas: el paper
compara los modelos como rankings de riesgo, no por el nivel de pi.
"""

import numpy as np
import pandas as pd


def auc(etiquetas, score) -> float:
    """AUC por Mann-Whitney U con empates a rango medio.

    Equivale a la probabilidad de que una observación en default reciba un
    score mayor que una sana (con medio punto por los empates).
    """
    y = np.asarray(etiquetas)
    s = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rangos = pd.Series(s).rank(method="average").to_numpy()
    u = rangos[y == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def accuracy_ratio(valor_auc: float) -> float:
    """CAP / accuracy ratio (Gini): 2 AUC - 1."""
    return 2.0 * valor_auc - 1.0


def deciles_por_mes(df: pd.DataFrame, col_score: str) -> pd.Series:
    """Decil de riesgo 1-10 dentro de cada mes (10 = mayor score = más riesgo).

    Cross-seccional como en el paper: cada empresa se compara contra las
    demás del mismo mes, no contra otros años con regímenes distintos.
    """
    pct = df.groupby("mes")[col_score].rank(pct=True)
    return np.ceil(pct * 10).clip(1, 10).astype(int)


def tabla_deciles(df: pd.DataFrame, col_score: str, col_label: str = "default_12m") -> pd.DataFrame:
    """Cuántos empresa-mes en default caen en cada decil del score."""
    base = df.dropna(subset=[col_score]).copy()
    base["decil"] = deciles_por_mes(base, col_score)
    tabla = base.groupby("decil").agg(
        filas=(col_label, "size"), defaults=(col_label, "sum")
    )
    tabla["pct_defaults"] = tabla["defaults"] / tabla["defaults"].sum()
    return tabla.sort_index(ascending=False)


def bootstrap_auc_pareado(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    col_label: str = "default_12m",
    n_replicas: int = 2000,
    semilla: int = 42,
) -> pd.DataFrame:
    """Bootstrap estratificado por empresa para comparar dos AUC pareados.

    Por qué por empresa (cluster bootstrap): los meses de una misma firma
    están fuertemente autocorrelacionados; remuestrear filas fingiría
    ~14,000 observaciones independientes y daría intervalos irrealmente
    angostos. Se remuestrean empresas enteras, con reemplazo.

    Por qué estratificado: con ~11 quebradas, un remuestreo simple dejaría
    réplicas sin ningún default. Se remuestrean por separado las empresas
    con algún mes en default y las demás, preservando ambas clases.

    Devuelve un DataFrame con auc_a, auc_b y diff (a - b) por réplica.
    """
    rng = np.random.default_rng(semilla)
    y = df[col_label].to_numpy()
    a = df[col_a].to_numpy(dtype=float)
    b = df[col_b].to_numpy(dtype=float)

    # groupby.indices da posiciones enteras dentro de df, listas para indexar y/a/b
    indices_por_empresa = {t: np.asarray(idx) for t, idx in df.groupby("ticker").indices.items()}
    con_default = [t for t, idx in indices_por_empresa.items() if y[idx].sum() > 0]
    sin_default = [t for t in indices_por_empresa if t not in set(con_default)]

    filas = []
    for _ in range(n_replicas):
        eleccion = list(rng.choice(con_default, size=len(con_default), replace=True)) + list(
            rng.choice(sin_default, size=len(sin_default), replace=True)
        )
        idx = np.concatenate([indices_por_empresa[t] for t in eleccion])
        auc_a = auc(y[idx], a[idx])
        auc_b = auc(y[idx], b[idx])
        filas.append({"auc_a": auc_a, "auc_b": auc_b, "diff": auc_a - auc_b})
    return pd.DataFrame(filas)


def resumen_bootstrap(replicas: pd.DataFrame) -> dict:
    """Intervalos percentiles al 95% y probabilidad bootstrap de a >= b."""
    q = replicas.quantile([0.025, 0.975])
    return {
        "auc_a_ic95": (float(q.loc[0.025, "auc_a"]), float(q.loc[0.975, "auc_a"])),
        "auc_b_ic95": (float(q.loc[0.025, "auc_b"]), float(q.loc[0.975, "auc_b"])),
        "diff_ic95": (float(q.loc[0.025, "diff"]), float(q.loc[0.975, "diff"])),
        "prob_a_mayor_igual_b": float((replicas["diff"] >= 0).mean()),
    }
