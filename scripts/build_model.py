"""Fase 2: corre el Merton completo y el naive sobre el panel empresa-mes.

Para cada fila del panel limpio:
- Naive: directo de las columnas E, F, sigma_E y ret_12m (vectorizado).
- Completo: reconstruye la trayectoria diaria del equity del último año
  y corre el solver iterativo con la F y la r de ese mes.

La trayectoria diaria se reconstruye escalando el market cap del mes por
los retornos ajustados: E_d = E_t x (adjclose_d / adjclose_t). Así los
cambios en el número de acciones (buybacks, splits) no meten saltos
espurios en la volatilidad de V; equivale a asumir acciones constantes
dentro de la ventana, y es la misma lógica con la que el panel calcula
sigma_E. La ventana usa solo precios hasta el fin de mes (sin lookahead).

Los meses donde el solver no converge quedan registrados con su flag;
la validación decidirá qué hacer con ellos (el PLAN sugiere usar el naive).

Uso: uv run python scripts/build_model.py
Lee todo de data/cache, no toca la red. Resultado: data/cache/panel_dd.parquet
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import cache
from data.panel import trayectoria_equity
from model.merton_iterativo import solve_merton
from model.naive import naive_dd


def merton_por_fila(panel: pd.DataFrame) -> pd.DataFrame:
    """Corre el solver iterativo fila por fila, agrupando por ticker."""
    filas = []
    tickers = panel["ticker"].unique()
    t0 = time.time()
    for n, ticker in enumerate(tickers, start=1):
        grupo = panel[panel["ticker"] == ticker]
        serie = cache.cargar_df(f"prices/{ticker}")
        fechas = pd.to_datetime(serie["date"]).dt.as_unit("ns").to_numpy()
        adj = serie["adjclose"].to_numpy(dtype=float)

        for fila in grupo.itertuples():
            base = {"ticker": ticker, "mes": fila.mes}
            E_path = trayectoria_equity(fechas, adj, fila.mes, fila.E)
            if E_path is None:
                filas.append({**base, "estado": "ventana corta"})
                continue
            try:
                res = solve_merton(E_path, fila.F, fila.r)
            except ValueError:
                filas.append({**base, "estado": "entrada invalida"})
                continue
            filas.append(
                {
                    **base,
                    "estado": "ok" if res.convergio else "sin convergencia",
                    "V": res.V,
                    "sigma_V": res.sigma_V,
                    "mu_V": res.mu_V,
                    "dd_merton": res.dd,
                    "pi_merton": res.pi,
                    "iteraciones": res.iteraciones,
                }
            )
        if n % 20 == 0:
            print(f"  merton: {n}/{len(tickers)} tickers ({time.time()-t0:.0f}s)")
    return pd.DataFrame(filas)


def main() -> None:
    panel = cache.cargar_df("panel")
    print(f"Panel: {len(panel)} filas, {panel['ticker'].nunique()} empresas")

    print("Naive (vectorizado)")
    dd_naive, pi_naive = naive_dd(
        panel["E"].to_numpy(),
        panel["F"].to_numpy(),
        panel["sigma_E"].to_numpy(),
        panel["ret_12m"].to_numpy(),
    )
    panel = panel.assign(dd_naive=dd_naive, pi_naive=pi_naive)

    print("Merton completo (iterativo)")
    merton = merton_por_fila(panel)
    salida = panel.merge(merton, on=["ticker", "mes"], how="left")
    cache.guardar_df(salida, "panel_dd")

    print()
    print("=== Resumen ===")
    print(salida["estado"].value_counts().to_string())
    ok = salida[salida["estado"] == "ok"]
    tasa = len(ok) / salida["estado"].notna().sum()
    print(f"convergencia: {tasa:.1%}")
    ambos = salida.dropna(subset=["dd_merton", "dd_naive"])
    rho = ambos["dd_merton"].corr(ambos["dd_naive"], method="spearman")
    print(f"spearman dd_merton vs dd_naive: {rho:.3f}")
    print(f"dd_merton: mediana {ambos['dd_merton'].median():.2f}, "
          f"p5 {ambos['dd_merton'].quantile(0.05):.2f}, "
          f"p95 {ambos['dd_merton'].quantile(0.95):.2f}")
    print("Listo. Resultados en data/cache/panel_dd.parquet")


if __name__ == "__main__":
    main()
