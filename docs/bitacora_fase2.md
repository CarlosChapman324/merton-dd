# Bitácora Fase 2: el modelo

## Decisiones de implementación

- **Inversión día a día con Newton vectorizado.** La versión de referencia
  (escalar, legible) usa brentq con la raíz encerrada en
  E < V <= E + F e^(-rT). El solver iterativo usa Newton-Raphson
  vectorizado sobre los 252 días a la vez: la derivada de la call respecto
  de V es N(d1), la call es convexa y creciente en V, y arrancando desde la
  cota superior el descenso es monótono. Hay un test que verifica que ambas
  inversiones coinciden a 1e-8. Resultado: el panel completo corre en
  segundos, contra decenas de minutos de la versión escalar.
- **mu_V = retorno realizado anualizado de V** ((V_fin/V_ini)^(252/n) - 1),
  no la media aritmética de retornos diarios. La media aritmética explota
  con volatilidad alta (una acción que cae 70% con rebotes diarios de
  +/-20% puede dar media anualizada de +80%) e inflaría el DD justo en las
  quebradas. El retorno realizado telescopa (solo dependen el primer y el
  último V) y es lo que usa el paper: el retorno del activo del año
  previo. Hallazgo de la revisión adversarial.
- **mu = max(mu_V, r)** como documenta el PLAN: un drift estimado por
  debajo de la tasa libre de riesgo en una ventana de un año suele ser
  ruido y produce DD sin sentido económico. Testeado.
- **sigma en el piso (1e-4) se reporta como no convergido**: una ventana de
  precios casi constantes (OTC ilíquido con cierres repetidos) daría un DD
  numéricamente arbitrario; el flag avisa en vez de fingir convergencia.
- **Trayectoria diaria del equity reconstruida con retornos ajustados**
  (E_d = E_t x adjclose_d / adjclose_t): los cambios de acciones en
  circulación no meten saltos espurios en sigma_V. Función pura
  (data/panel.py, trayectoria_equity) con test de no-lookahead.
- **Tolerancia del iterativo 1e-3** (la del paper). Los tests piden la
  precisión que el procedimiento garantiza (DD +/- 0.1), no más.

## Revisión adversarial (workflows multi-agente)

Cuatro dimensiones de revisión (fidelidad al paper, numérica, suficiencia
de tests, integración con el panel) más paneles de verificación por
lentes, y una auditoría final de resultados. Lo que dejó:

- **Fidelidad al paper: cero hallazgos.**
- **Tests: 7 huecos señalados, 5 aceptados y cerrados** (valor exacto del
  DD contra fórmula analítica externa, signo de pi, regla del piso de mu,
  horizonte T distinto de 1, y extracción de trayectoria_equity para
  testearla). Se descartaron 2 con razón documentada: ddof y 250 vs 252
  días están dentro de la ambigüedad legítima del paper y mueven el DD
  menos de 0.02; la semilla del iterativo no afecta el punto fijo.
- **Numérica: el hallazgo más valioso de la fase, que resultó ser de
  DATOS** (confirmado 3/3): mega-caps con F desplomada tres órdenes de
  magnitud (AbbVie F = 16 millones vs deuda real ~70,000 millones), porque
  hay empresas que reportan componentes solo en el 10-K anual y otras que
  cambian de tag XBRL con los años (Eaton dejó morir LongTermDebtNoncurrent
  en 2014; Air Products migró a las variantes con arriendos en 2023).

## El arreglo de la deuda (data/sec.py)

Coalescencia por prioridad fecha a fecha: todos los tags candidatos se
alinean a una malla común de fechas de balance con arrastre máximo de 400
días (cubre reportes anuales sin arrastrar para siempre tags muertos) y en
cada fecha gana el tag de mayor prioridad con dato. Cadena de largo plazo
ampliada con las variantes con arriendos financieros. Verificación contra
deuda real conocida: AbbVie 40.6B, Eaton 6.2B, Air Products 8.3B, Apple
52.8B. EA volvió al panel (134 empresas). Las filas con dd_merton > 40
cayeron de 187 a 67, y las que quedan son empresas genuinamente casi sin
deuda (tipo Accenture, deuda ~10M sobre market cap de cientos de miles de
millones): colas reales, no errores. El paper trabaja con rankings, así
que no afectan el orden; si se winsoriza para gráficos se decidirá en
Fase 3.

## Auditoría de resultados (workflow final) y arreglo de splits

La auditoría sobre panel_dd.parquet confirmó 4 hallazgos y destapó el más
serio de la fase al perseguir uno no verificado:

1. **E distorsionada por splits (arreglado).** Los precios de Yahoo y de
   stockanalysis vienen restatados retroactivamente por splits (AAPL de
   2019 aparece a 49 cuando cotizaba a ~197; WEWKQ es continua a través de
   su reverse 1:40), pero las acciones de EDGAR son as-reported. E quedaba
   mal por el factor acumulado de splits posteriores: AAPL 4x abajo antes
   de 2020, Rite Aid 20x arriba antes de 2019, Lordstown 15x. Arreglo:
   acciones convertidas a términos de hoy multiplicando por los splits
   posteriores a cada balance (panel.acciones_en_terminos_actuales, con
   test); splits desde yfinance cacheados más overrides manuales para las
   deslistadas (WEWKQ, RADCQ en universe.py). Verificado contra market
   caps conocidos: AAPL 2019-06 911B (real ~910B), Rite Aid 2018-06 1.85B
   (real ~1.7B).
2. **275 filas post-filing en el panel (flag añadido).** Los meses OTC
   posteriores al Chapter 11 (centavos, sigma_E de hasta 2400%) generaban
   toda la cola extrema, y PG&E/Garrett post-emergencia son empresas sanas
   con es_default=True. El panel ahora trae la columna post_filing; la
   Fase 3 y la Fase 4 deben excluirlas del etiquetado y de las curvas.
3. **Joann con acciones en millones sin reescalar (arreglado).** Dos
   portadas de 2023 reportaron 40.7 en vez de 40,700,000; E caía a 70
   dólares y eran las únicas 2 filas sin converger. Guardia de escala en
   sec.acciones_en_circulacion (ninguna cotizada tiene < 10,000 acciones);
   el as-of arrastra el reporte anterior. Convergencia ahora 100.0%.
4. **mu_V sin techo (documentado, sin acción).** 54 filas con drift > 200%
   (AppLovin hasta 679%) inflan el DD de empresas con momentum extremo. Es
   fiel al paper (B&S no winsorizan el drift); si los deciles superiores
   de la Fase 3 se ven raros, winsorizar ahí.
5. **Muestras distintas para el AUC (nota para Fase 3).** 64 filas tienen
   dd_merton válido pero dd_naive NaN (ret_12m requiere 12 meses de
   historia). La comparación naive vs completo debe restringirse a la
   intersección de ambos DD no-NaN.

## Resultado de la corrida sobre el panel (post-arreglos)

- 14,821 filas, 134 empresas, convergencia 100.0% (cero filas sin
  converger tras el arreglo de Joann).
- Spearman entre DD naive y DD completo: 0.992 en el panel real.
- Vivas: dd_merton mediano 10.3, p5 2.9 (el S&P 500 sano está lejos del
  default). 71 filas con dd > 40: empresas genuinamente casi sin deuda
  (Accenture, Copart), colas reales.
- Quebradas, último mes antes del filing (excluyendo post_filing): 8 de
  11 con pi_naive > 0.69 (Rite Aid 0.995, Joann 0.994, Big Lots 0.993,
  Revlon 0.986, Party City 0.980, Yellow 0.943, Garrett 0.879, WeWork
  0.698), con apalancamiento 0.77-1.00 en esas ocho.
- El naive es sistemáticamente más pesimista cerca del default que el
  completo (pi_merton 0.19-0.85 en esos mismos meses): primer indicio del
  hallazgo central del paper.
- Señal débil con explicación económica real, no error de datos: PG&E
  (pi 0.32: su equity sobrevivió al Chapter 11), Lordstown (pi 0.31
  naive: quebró con leverage 0.20 por el pleito con Foxconn; un modelo
  estructural no puede ver defaults estratégicos), Proterra (0.47, siete
  meses de caída entre su último dato y el filing) y WeWork (0.70: su
  pasivo real eran arriendos operativos invisibles para los tags de
  deuda). Casos de estudio para la Fase 4.

Salida: data/cache/panel_dd.parquet (panel + dd_merton, pi_merton,
dd_naive, pi_naive, sigma_V, mu_V, V, estado, iteraciones).
