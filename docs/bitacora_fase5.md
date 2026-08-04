# Bitácora Fase 5: dashboard y README

## Dashboard (app/)

- `app/streamlit_app.py`: 6 pestañas (Metodología, Panel, Réplica, Defaults,
  Mejora, Limitaciones). Identidad de terminal de datos oscura con acento
  azul: CSS inyectado + `.streamlit/config.toml` + template Plotly propio.
- `app/figuras.py`: TODAS las figuras viven aquí (template "terminal",
  paleta, payoff de Merton, curvas alineadas con banda de vivas, deciles,
  AUC CV, calibración, serie por empresa). El README usa exactamente las
  mismas figuras vía `scripts/build_figuras.py` (export PNG con kaleido):
  una sola fuente de verdad visual.
- Los números del dashboard se leen de `replica_conclusion.json` y
  `mejora_conclusion.json`: si se reconstruyen los datos, el dashboard se
  actualiza solo (los textos con cifras usan f-strings sobre los JSON).
- La tabla de anticipación muestra la censura como "N+" (hallazgo de la
  revisión de Fase 4).
- Verificado en navegador: renderizado de la home, métricas (el cuarto
  metric truncaba y se bajó la fuente a 1.6rem), pestaña Defaults con el
  gráfico estrella y la banda de vivas.
- Arranque local: `uv run streamlit run app/streamlit_app.py` (hay
  `.claude/launch.json` para el preview).

## README

Estructura pedida: original -> réplica -> mejora -> resultados, con las 4
figuras clave embebidas (docs/img/), las dos tablas centrales, la sección
de honestidad estadística y el bloque de reproducción (uv sync + scripts en
orden + 46 tests). Cifras verificadas contra los JSON de conclusión tras
los ajustes de la revisión de Fase 4 (ganancia del DD +0.018, z -0.83,
calibración sobre la muestra de la réplica).

## Pendiente (Fase 6)

Deploy a Streamlit Community Cloud, git init + commits limpios, revisión
final de textos.
