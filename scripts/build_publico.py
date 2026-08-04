"""Fase 6: copia a data/publico/ lo que el dashboard necesita para el deploy.

El cache completo pesa ~515 MB (companyfacts crudos de EDGAR) y no va a
git. El dashboard, en cambio, solo lee resultados derivados: el panel con
los DD y las tablas de la réplica y la mejora, unos 2 MB en total. Eso sí
se versiona, y es lo que permite que Streamlit Community Cloud levante la
app sin descargar nada.

Uso: uv run python scripts/build_publico.py (correr tras build_mejora.py)
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.cache import CACHE_DIR, PUBLICO_DIR

# Exactamente lo que carga app/streamlit_app.py, ni un archivo más
ARTEFACTOS = [
    ("panel_dd", "parquet"),
    ("replica_deciles_naive", "parquet"),
    ("replica_deciles_merton", "parquet"),
    ("replica_deciles_eventos", "parquet"),
    ("replica_leave_one_out", "parquet"),
    ("mejora_curvas_alineadas", "parquet"),
    ("mejora_banda_vivas", "parquet"),
    ("mejora_anticipacion", "parquet"),
    ("mejora_calibracion_deciles", "parquet"),
    ("mejora_calibracion_global", "parquet"),
    ("replica_conclusion", "json"),
    ("mejora_conclusion", "json"),
    # de apoyo para documentar el panel en el repo
    ("reporte_quebradas", "parquet"),
    ("reporte_edgar", "parquet"),
]


def main() -> None:
    PUBLICO_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    faltantes = []
    for nombre, ext in ARTEFACTOS:
        origen = CACHE_DIR / f"{nombre}.{ext}"
        if not origen.exists():
            faltantes.append(f"{nombre}.{ext}")
            continue
        destino = PUBLICO_DIR / f"{nombre}.{ext}"
        shutil.copy2(origen, destino)
        total += destino.stat().st_size
    if faltantes:
        print(f"  FALTAN (corre antes los build_*.py): {faltantes}")
    print(f"{len(ARTEFACTOS) - len(faltantes)} artefactos en data/publico/ "
          f"({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
