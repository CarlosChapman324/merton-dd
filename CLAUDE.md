# CLAUDE.md — Merton Distance to Default: réplica y extensión de Bharath & Shumway (2008)

## Qué es
Réplica académica reproducible del paper Bharath & Shumway (2008), "Forecasting Default with the Merton Distance to Default Model" (Review of Financial Studies), aplicada a empresas cotizadas de Estados Unidos con datos 100% públicos y gratuitos.

El paper implementa el modelo estructural de Merton (1974): el patrimonio de una empresa se trata como una opción call sobre sus activos, con strike igual al valor de su deuda. De ahí se deriva la distancia al default (DD) y una probabilidad de default. El hallazgo central del paper, y lo que esta réplica pone a prueba, es contraintuitivo: una versión "naive" del modelo (misma forma funcional, sin resolver el sistema iterativo) predice default igual o ligeramente mejor que el modelo completo. La conclusión es que lo valioso del Merton es su forma funcional, no la solución precisa.

Estructura del entregable: original -> réplica -> mejora -> resultados. Es una pieza de investigación de portafolio, no una herramienta de inversión.

## Diferenciadores (no descuidar)
1. **Réplica fiel y verificable.** Implementar el modelo completo (solución iterativa) Y el naive exactamente como los define el paper, y comprobar si el resultado naive >= completo se sostiene en datos recientes de EE.UU.
2. **Validación contra defaults reales.** Panel que incluye empresas que efectivamente quebraron (Chapter 11) en 2018-2025, con las curvas de DD en los meses previos al default como evidencia visual central.
3. **Mejora propia.** Comparar ambos DD contra un baseline contable (logit con ratios financieros) y medir si el DD aporta señal incremental. Es la pregunta que un equipo de riesgo haría.
4. **Honestidad estadística.** Los defaults son eventos raros: reportar la incertidumbre, el sesgo de supervivencia de los datos y el hecho de que N(-DD) no es una probabilidad calibrada. Decir lo que el modelo NO puede hacer es parte del valor.

## Stack
- Python 3.12 con uv (mismo setup que el proyecto anterior del dev)
- pandas, numpy para datos; scipy para el solver iterativo (Black-Scholes) y estadística
- statsmodels para los modelos logit / hazard de la validación
- yfinance para precios de mercado; SEC EDGAR (data.sec.gov, API companyfacts, gratis, solo requiere User-Agent) para deuda de los balances; FRED (CSV público) para la tasa libre de riesgo
- Streamlit + Plotly para el dashboard
- pytest para tests
- Almacenamiento local en Parquet (cache de todas las descargas)

## Convenciones
- El modelo vive aislado en `model/`: matemática pura, sin llamadas de red, testeable sin internet. La ingesta vive en `data/`, la validación en `validation/`, la app en `app/`.
- Toda descarga se cachea en disco; ninguna función de análisis vuelve a llamar a la red si el dato ya existe.
- Respetar los límites de EDGAR (máximo 10 requests/segundo, User-Agent identificado) y los rate limits de yfinance (pausas entre tickers).
- Código legible y comentado: el dev tiene que poder explicar cada ecuación y cada decisión en una entrevista. Claridad sobre astucia.
- Nada de "--" (doble guion) en código ni en texto.
- README con la estructura: paper original (qué dice) -> réplica (qué hice y qué encontré) -> mejora (qué añadí) -> resultados (tablas y gráficos clave).

## Diseño del dashboard
Misma identidad visual que el otro proyecto del dev (predictor Mundial): terminal de datos oscura, acento azul, números grandes, visualizaciones custom de Plotly. Los dos proyectos deben verse como de la misma persona.

## Contexto del dev
Carlos Chapman, economista (Barranquilla, Colombia), trabaja en cartera/riesgo de crédito y arma portafolio para roles de finanzas cuantitativas y másters en Europa. Fuerte en R, consolidando Python (este es su segundo proyecto Python). Quiere ENTENDER lo que se construye: explica el porqué de cada decisión técnica a medida que avanzas y construye por fases. El detalle de implementación vive en PLAN.md.
