# Bitácora Fase 4: la mejora propia

## Diseño

- **Baseline contable (mini Shumway 2001):** logit con apalancamiento
  F/(E+F), tamaño ln(E), ret_12m, sigma_E y ROA. El ROA es el único insumo
  nuevo: resultado neto del ejercicio anual (tags NetIncomeLoss/ProfitLoss,
  duración 330-400 días) sobre activos totales, del mismo cierre, asof por
  fecha de publicación. Cobertura 97.6% del panel; anual porque el Q4 de
  XBRL suele venir solo acumulado y reconstruir trimestres no paga.
- **Fuera de muestra = K-fold agrupado y estratificado por empresa**
  (K = 11; cada fold contiene una quebrada y una cuota determinista de
  vivas). La primera versión usaba leave-one-firm-out y el test sintético
  de ruido puro la cazó: daba AUC 0.08 en vez de ~0.5, porque las
  quebradas se predicen con modelos entrenados con menos positivos
  (intercepto más bajo) que las sanas, y ese desajuste de tasas base entre
  folds hunde el AUC del pool. Es una patología conocida del LOO-CV para
  AUC con clases desbalanceadas; el K-fold estratificado la elimina
  (tasas base de entrenamiento casi idénticas entre folds). Queda el test
  que lo detectó, como regresión.
- **Winsorización 1/99** de ret_12m, roa, tamaño, sigma_E y ambos DD, una
  sola vez en la preparación (misma política que Fase 3). La revisión
  adversarial detectó que la primera versión dejaba sigma_E sin winsorizar
  y la ganancia del DD era sensible a ello (+0.029 con la cola de sigma
  libre, +0.018 winsorizada): se reporta la versión consistente, que es la
  menor. Los percentiles se calculan antes del split de folds (leakage
  técnico inmaterial: recorta colas, no ordena).

## Resultados

**Logit in-sample (SE cluster por empresa):**

| especificación | pseudo-R2 | z del DD |
|---|---|---|
| baseline contable | 0.604 | |
| solo dd_naive | 0.522 | -6.23 |
| baseline + dd_naive | 0.611 | -1.45 (p 0.15) |
| baseline + dd_merton | 0.612 | -1.44 (p 0.15) |

En el conjunto, ni el DD ni el apalancamiento son individualmente
significativos: el DD ES una función de apalancamiento y volatilidad, así
que compite con sus propios ingredientes (colinealidad estructural). El
único ratio robusto es el tamaño.

**AUC fuera de muestra (K-fold agrupado):**

| modelo | AUC CV |
|---|---|
| baseline contable | 0.9146 |
| baseline + dd_naive | 0.9323 |
| baseline + dd_merton | 0.9281 |
| solo dd_naive | 0.9797 |

La respuesta a la pregunta de la mejora, en dos partes:
1. **Sí, el DD aporta señal incremental:** añadirlo al baseline sube el
   AUC out-of-sample de 0.915 a 0.932 (+0.018), aunque su z in-sample
   (-0.83) no supere el umbral con 11 clusters (anticonservador y
   colineal: el DD es función de apalancamiento y volatilidad).
2. **El hallazgo inesperado: el DD solo generaliza mejor que el conjunto**
   (0.980 vs 0.932). Con 11 eventos, cada parámetro extra se paga fuera
   de muestra; la revisión matizó el mecanismo: parte es sobreajuste y
   parte descalibración entre folds al agrupar predicciones de modelos
   con escalas distintas (concentrada en los folds de GTX y PCG). En
   ambos casos la lectura práctica es la misma: la forma funcional del
   Merton ya condensa apalancamiento y volatilidad mejor que un logit
   que intenta recombinarlos con 11 eventos.

**Anticipación de la alerta (DD < 2 sostenido hasta el filing):** Rite Aid
y Revlon 24+ meses (censurado: nunca estuvieron sanas en la ventana),
Joann 23, Big Lots 21, Yellow 18, Party City y WeWork 14, Garrett 7-9,
Proterra 5, PG&E 2, Lordstown 1-4. La aparente divergencia de Revlon
(merton 24+ vs naive 7) es un artefacto de la regla de alerta SOSTENIDA:
el squeeze de 2021 hizo rebotar el DD naive por encima de 2 un momento y
reinició su conteo; los modelos no discrepan de fondo (hallazgo de la
revisión). Las curvas quedaron en mejora_curvas_alineadas.parquet con la
banda de vivas de referencia (dd mediano de sanas: 10.5) y la censura
marcada por columna.

**Calibración (la nota honesta):** calculada sobre la misma muestra de la
réplica (la calibración no necesita ratios contables; usar la muestra
filtrada de la mejora cambiaba la tasa base reportada, hallazgo de la
revisión). Tasa realizada a 12 meses: 0.90%; pi_naive promedio 1.69%
(sobreestima 1.9x), pi_merton 0.66% (subestima 0.73x). En el decil 10,
pi_naive promete 16.2% y ocurre 8.1%. Fuera de los deciles 9-10 la tasa
realizada es exactamente 0. Conclusión: N(-DD) ordena bien pero NO es una
probabilidad calibrada; en la práctica se usa el ranking.

Salidas: mejora_curvas_alineadas / mejora_banda_vivas / mejora_anticipacion /
mejora_calibracion_global / mejora_calibracion_deciles (parquet) y
mejora_conclusion.json, todo en data/cache.
