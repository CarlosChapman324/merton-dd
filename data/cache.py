"""Cache local en disco para todas las descargas.

Regla del proyecto: ninguna función de análisis vuelve a llamar a la red
si el dato ya existe en data/cache/. Tablas en Parquet, respuestas crudas
de APIs en JSON.

Dos ubicaciones, con precedencia deliberada:
- data/cache/ es la fuente: todo lo descargado y todo lo derivado. Pesa
  cientos de MB (companyfacts crudos) y no va a git.
- data/publico/ es el subconjunto derivado que el dashboard necesita para
  correr sin el cache completo (unos 2 MB), y sí va a git: es lo que hace
  posible el deploy a Streamlit Community Cloud. Lo genera
  scripts/build_publico.py desde el cache.

La lectura prueba el cache primero y cae al público: en local mandan los
datos frescos, y en el deploy (donde no hay cache) manda lo publicado.
"""

import json
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "cache"
PUBLICO_DIR = Path(__file__).resolve().parent / "publico"


def _ruta(nombre: str, ext: str) -> Path:
    ruta = CACHE_DIR / f"{nombre}.{ext}"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    return ruta


def _ruta_lectura(nombre: str, ext: str) -> Path:
    """Cache si existe; si no, el respaldo publicado."""
    en_cache = CACHE_DIR / f"{nombre}.{ext}"
    if en_cache.exists():
        return en_cache
    return PUBLICO_DIR / f"{nombre}.{ext}"


def tiene(nombre: str, ext: str = "parquet") -> bool:
    return _ruta_lectura(nombre, ext).exists()


def guardar_df(df: pd.DataFrame, nombre: str) -> None:
    df.to_parquet(_ruta(nombre, "parquet"))


def cargar_df(nombre: str) -> pd.DataFrame:
    return pd.read_parquet(_ruta_lectura(nombre, "parquet"))


def guardar_json(obj, nombre: str) -> None:
    _ruta(nombre, "json").write_text(json.dumps(obj))


def cargar_json(nombre: str):
    return json.loads(_ruta_lectura(nombre, "json").read_text())
