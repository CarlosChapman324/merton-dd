# PLAN.md — Merton DD: réplica y extensión de Bharath & Shumway (2008)

Plan de implementación por fases. Objetivo: replicar el modelo completo y el naive del paper sobre un panel de empresas de EE.UU. con datos públicos, validar contra quiebras reales, añadir una mejora propia (baseline contable y análisis de señal incremental) y presentarlo en un dashboard + README con estructura original -> réplica -> mejora -> resultados.

---

## Arquitectura general

```
Ingesta (yfinance, EDGAR, FRED)  ->  cache en Parquet
        |
   Modelo (model/, matemática pura, sin red)
        |  DD completo (iterativo) y DD naive por empresa-mes
        |
   Validación (validation/): réplica del test naive vs completo + mejora
        |
   Dashboard (Streamlit) + README de investigación
```

Estructura de carpetas:
```
data/          ingesta y normalización (con cache)
model/         black_scholes.py, merton_iterativo.py, naive.py
validation/    panel, logit/hazard, métricas, casos de default
app/           streamlit_app.py
scripts/       build_data.py, build_model.py, build_validation.py
tests/         pytest
```

---

## Universo y datos (todo gratis)

### Universo de empresas
- **Vivas:** las empresas no financieras del S&P 500 actual (excluir bancos, aseguradoras y financieras: el modelo de Merton no aplica bien a su estructura de deuda; el propio paper las excluye).
- **Quebradas (la parte valiosa):** una lista curada de empresas de EE.UU. que se acogieron a Chapter 11 entre 2018 y 2025 y cuyos datos de precios aún son descargables. Candidatas a verificar (confirmar ticker, fecha de filing y disponibilidad de datos antes de incluir): Hertz (2020), J.C. Penney (2020), Chesapeake Energy (2020), Whiting Petroleum (2020), Frontier Communications (2020), Revlon (2022), Bed Bath & Beyond (2023), Party City (2023), Yellow Corp (2023), Rite Aid (2023), WeWork (2023), Lordstown Motors (2023), Spirit Airlines (2024), Big Lots (2024), Tupperware (2024), Express (2024). Objetivo: al menos 12-15 defaults verificados con datos suficientes (24+ meses de precios previos al filing).
- Panel objetivo: 100-150 empresas vivas + los defaults, frecuencia mensual, ventana 2016-2025.

### Fuentes
1. **Precios y market cap:** yfinance. Precio ajustado diario y acciones en circulación. Para tickers deslistados, intentar igualmente con yfinance; si no hay datos, probar Stooq como fallback y documentar los que se pierdan (sesgo de supervivencia: reportarlo).
2. **Deuda:** SEC EDGAR, API companyfacts (https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json). Gratis, sin key, requiere header User-Agent con email. Convención del paper (vía Vassalou-Xing): F = deuda corriente + 0.5 x deuda de largo plazo. Tags XBRL con fallbacks en este orden: para corriente `DebtCurrent`, si no `LongTermDebtCurrent` + `ShortTermBorrowings`; para largo plazo `LongTermDebtNoncurrent`, si no `LongTermDebt`. Documentar la elección por empresa. Interpolar la deuda trimestral a mensual (valor del último trimestre reportado disponible en cada mes, sin mirar al futuro).
3. **Tasa libre de riesgo:** T-bill de 3 meses desde FRED en CSV público sin key: https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS
4. **Mapeo ticker -> CIK:** archivo público de la SEC https://www.sec.gov/files/company_tickers.json

---

## El modelo (model/)

### Insumos por empresa-mes
E = market cap (precio x acciones), F = deuda según convención, r = T-bill 3m anualizada, sigma_E = volatilidad anualizada de retornos diarios del último año, T = 1 año.

### Modelo completo (iterativo, el "Merton DD" del paper)
1. Ecuaciones de Black-Scholes-Merton que ligan equity y activos:
   E = V N(d1) - e^(-rT) F N(d2), con d1 = [ln(V/F) + (r + 0.5 sigma_V^2) T] / (sigma_V sqrt(T)) y d2 = d1 - sigma_V sqrt(T)
2. Procedimiento iterativo (el estándar de Bharath-Shumway / Crosbie-Bohn):
   - Arrancar con sigma_V = sigma_E x E/(E+F).
   - Con ese sigma_V, invertir la ecuación de B-S-M día a día (solver de scipy) para obtener la serie diaria de V del último año.
   - Recalcular sigma_V como la volatilidad anualizada de los retornos diarios de ese V implícito, y mu_V como su retorno medio anualizado.
   - Repetir hasta convergencia (cambio en sigma_V < 1e-3, con tope de iteraciones).
3. Distancia al default y probabilidad:
   DD = [ln(V/F) + (mu - 0.5 sigma_V^2) T] / (sigma_V sqrt(T)), con mu = max(mu_V estimado, r) para evitar drifts negativos absurdos (documentarlo).
   pi_Merton = N(-DD)

### Modelo naive (la alternativa del paper, sin resolver nada)
- sigma_V_naive = (E/(E+F)) x sigma_E + (F/(E+F)) x (0.05 + 0.25 x sigma_E)
- mu_naive = retorno de la acción en los 12 meses previos
- DD_naive = [ln((E+F)/F) + (mu_naive - 0.5 sigma_V_naive^2) T] / (sigma_V_naive sqrt(T))
- pi_naive = N(-DD_naive)

Tests con pytest: precio B-S-M contra valores conocidos; el solver recupera V y sigma_V en casos sintéticos donde se conoce la verdad; DD baja cuando sube el apalancamiento o la volatilidad; naive y completo correlacionan alto en el panel sintético; pi en [0,1] siempre.

---

## Réplica (validation/) — el corazón del proyecto

Replicar el test central del paper en este panel:
1. **Definición de evento:** default = filing de Chapter 11. Cada empresa-mes se etiqueta 1 si la empresa quiebra en los 12 meses siguientes, 0 si no.
2. **Poder predictivo por deciles:** ordenar empresa-mes por pi y verificar en qué decil caen los defaults (el paper concentra la gran mayoría en el decil superior).
3. **Comparación naive vs completo:** AUC (y CAP/accuracy ratio) de pi_naive vs pi_Merton. La hipótesis del paper: naive >= completo. Reportar si se replica o no, con intervalo (bootstrap).
4. **Modelo de riesgo:** logit de default a 12 meses sobre pi (o sobre DD), replicando en espíritu los hazard models del paper. Si el panel lo permite, Cox con statsmodels/lifelines; si no, logit pooled con errores agrupados por empresa y decirlo.

---

## Mejora propia (el sello del dev)

1. **Baseline contable:** logit con ratios simples de los balances de EDGAR (apalancamiento = F/(E+F), rentabilidad, tamaño = ln(E), retorno de la acción 12m, volatilidad sigma_E). Es un mini Shumway (2001).
2. **Pregunta de señal incremental:** en un logit conjunto (ratios + DD), ¿el DD sigue siendo significativo? ¿Mejora el AUC fuera de muestra? Esa es la pregunta que un equipo de riesgo real le haría al modelo, y la respuesta honesta (sea cual sea) es el resultado principal de la mejora.
3. **Casos de estudio (el gráfico estrella):** curvas de DD (completo y naive) de las empresas quebradas en los 24 meses previos al filing, alineadas en el tiempo del evento. Mostrar si el DD "vio venir" el default y con cuánta antelación. Incluir 2-3 controles que no quebraron para contraste.
4. **Nota de calibración:** mostrar que N(-DD) NO es una probabilidad calibrada (comparar pi promedio contra la tasa real de default del panel) y explicar por qué en la práctica se usa el DD como ranking, no como PD literal.

---

## Dashboard (app/)

Pestañas: (1) Metodología, explicando Merton en lenguaje claro con un diagrama; (2) Panel, DD por empresa con serie temporal y ranking actual; (3) Réplica, naive vs completo con AUC y deciles; (4) Defaults, las curvas de los casos reales alineadas al evento; (5) Mejora, señal incremental y calibración; (6) Limitaciones. Estética idéntica al proyecto del Mundial (terminal oscura, acento azul).

---

## Fases

### Fase 0 — Setup (medio día)
Entorno con uv, estructura de carpetas, dependencias, pytest configurado, esqueleto de módulos.

### Fase 1 — Datos (1-2 días, la fase pesada)
Mapeo ticker-CIK, descarga y cache de precios (vivas + quebradas), deuda desde EDGAR con los fallbacks de tags, T-bill de FRED. VERIFICAR TEMPRANO la lista de quebradas: confirmar cuáles tienen precios descargables y 24+ meses de historia previa al filing; reportar cuáles se pierden. Construir el panel empresa-mes limpio en Parquet.

### Fase 2 — Modelo (1-2 días)
black_scholes.py, el solver iterativo del Merton completo, el naive. Tests exhaustivos con casos sintéticos antes de tocar datos reales. Correr el modelo sobre el panel y guardar DD y pi de ambos modelos por empresa-mes.

### Fase 3 — Réplica (1 día)
Etiquetado de eventos, deciles, AUC naive vs completo con bootstrap, logit/hazard. Conclusión explícita: ¿se replica el hallazgo del paper?

### Fase 4 — Mejora (1 día)
Baseline contable, señal incremental, casos de estudio con las curvas alineadas al evento, análisis de calibración.

### Fase 5 — Dashboard y README (1-2 días)
Las 6 pestañas, y el README de investigación con estructura original -> réplica -> mejora -> resultados, con las tablas y gráficos clave embebidos.

### Fase 6 — Pulido y deploy (medio día)
Deploy a Streamlit Community Cloud, revisión final de textos, commits limpios.

---

## Riesgos

- **Tickers deslistados sin datos:** el riesgo número uno. Mitigación: verificarlos en la Fase 1 antes de construir nada encima; fallback a Stooq; si la muestra de defaults queda por debajo de ~10, ampliar la ventana temporal hacia atrás y documentar el sesgo de supervivencia.
- **Tags XBRL inconsistentes entre empresas:** usar la cadena de fallbacks, registrar qué tag se usó por empresa, y excluir (documentando) las que no reporten deuda utilizable.
- **Pocos defaults para inferencia fina:** la validación se apoya en deciles, AUC con bootstrap y los casos de estudio; no prometer significancia que la muestra no da.
- **Convergencia del solver:** en empresas con apalancamiento extremo el iterativo puede fallar; usar bounds y tope de iteraciones, registrar los fallos y usar el naive como valor de esos meses (documentado).
- **Rate limits:** EDGAR max 10 req/s con User-Agent; yfinance con pausas y cache. Todo se descarga una vez y se cachea.
- **Framing:** es investigación reproducible de portafolio; no es asesoría de inversión y el README lo dice.
