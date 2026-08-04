"""Fase 5: exporta las figuras del README a docs/img (PNG vía kaleido).

Son las mismas figuras del dashboard (app/figuras.py): una sola fuente de
verdad visual. Uso: uv run python scripts/build_figuras.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import figuras
from data import cache

DESTINO = Path(__file__).resolve().parents[1] / "docs" / "img"
FONDO = "#0b0e14"


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    curvas = cache.cargar_df("mejora_curvas_alineadas")
    banda = cache.cargar_df("mejora_banda_vivas")
    deciles_n = cache.cargar_df("replica_deciles_naive")
    deciles_m = cache.cargar_df("replica_deciles_merton")
    calibracion = cache.cargar_df("mejora_calibracion_deciles")
    mejora = json.loads(Path("data/cache/mejora_conclusion.json").read_text())

    salidas = {
        "curvas_alineadas": figuras.fig_curvas_alineadas(curvas, banda, "dd_merton"),
        "deciles": figuras.fig_deciles(deciles_n, deciles_m),
        "auc_cv": figuras.fig_auc_cv(mejora["aucs_cv"]),
        "calibracion": figuras.fig_calibracion(calibracion),
        "payoff": figuras.fig_payoff(),
    }
    for nombre, fig in salidas.items():
        # el PNG necesita fondo solido (el dashboard usa transparente)
        fig.update_layout(paper_bgcolor=FONDO, plot_bgcolor=FONDO)
        ruta = DESTINO / f"{nombre}.png"
        fig.write_image(str(ruta), width=1100, height=fig.layout.height or 450, scale=2)
        print(f"  {ruta.name}")
    print(f"Listo: {len(salidas)} figuras en docs/img/")


if __name__ == "__main__":
    main()
