# Bitácora Fase 6: pulido y deploy

## El bloqueador que había que resolver

El dashboard leía de `data/cache/`, que pesa ~515 MB (companyfacts crudos
de EDGAR) y está en `.gitignore`. Tal cual, un deploy habría levantado la
app sin un solo dato.

Solución: `data/publico/`, un subconjunto versionado de 1.8 MB con los 14
artefactos derivados que el dashboard realmente lee (panel con los DD,
tablas de réplica y mejora, los dos JSON de conclusión). Lo genera
`scripts/build_publico.py` desde el cache, y `data/cache.py` lee el cache
primero y cae al público: en local mandan los datos frescos, en el
servidor manda lo publicado. Verificado apuntando `CACHE_DIR` a un
directorio inexistente (lo que ve Streamlit Cloud): el panel, las
conclusiones y las seis figuras cargan.

`requirements.txt` trae solo lo que la app necesita (streamlit, plotly,
pandas, pyarrow, numpy): Streamlit Cloud no usa uv, y ni la ingesta ni el
modelado corren allí. El pipeline completo sigue en `pyproject.toml`.

## Privacidad antes de publicar

El User-Agent de EDGAR tenía el correo personal hardcodeado. Pasa a
leerse de `SEC_USER_AGENT` (documentado en el README); con el cache
poblado nada vuelve a la red, así que no rompe la reproducción. Los
commits usan el correo `noreply` de GitHub.

## Consistencia final

Se re-corrió la cadena completa (`build_validation` -> `build_mejora` ->
`build_figuras` -> `build_publico`) porque los artefactos de la réplica
eran anteriores al panel con ROA. Todos los números reprodujeron
idénticos (AUC 0.9834 / 0.9804, ganancia del DD +0.0177), lo que confirma
que añadir ROA no tocó el modelo. 46 tests en verde.

Revisión de textos: sin TODOs, sin `NotImplementedError`, sin dobles
guiones en prosa, y el docstring de `data/sec.py` actualizado (describía
la cadena de fallbacks vieja, no la coalescencia por fecha que se
implementó en la Fase 2).

## Historia de git

Siete commits temáticos que cuentan el proyecto en orden (setup, datos,
modelo, réplica, mejora, dashboard, deploy) más uno de pulido del README.
Cada mensaje explica el porqué de las decisiones no obvias, no solo el
qué.

Publicado en https://github.com/CarlosChapman324/merton-dd

## Pendiente (solo lo puede hacer el dev)

Conectar el repo en https://share.streamlit.io: entrar con la cuenta de
GitHub, "New app", elegir `CarlosChapman324/merton-dd`, rama `main`,
archivo `app/streamlit_app.py`, Deploy. Requiere autenticación con la
cuenta personal.
