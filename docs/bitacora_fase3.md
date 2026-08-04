# Bitácora Fase 3: la réplica

## Diseño del test

- **Etiquetado:** cada empresa-mes vale 1 si la empresa se acoge al
  Chapter 11 en los 12 meses siguientes al fin de mes. Las 275 filas
  post-filing se excluyen (leakage); el mes del filing ya cae en esa
  exclusión, así que solo entra información previa al evento. Test
  unitario de la ventana.
- **Muestra:** 14,482 filas con AMBOS DD no-NaN (misma muestra para los
  dos modelos; las 64 filas excluidas por falta de ret_12m contienen cero
  positivos, así que no hay sesgo de exclusión: verificado e impreso).
  130 empresa-mes positivos que son ventanas solapadas de 11 eventos: el
  N efectivo es 11 quiebras y todo el diseño inferencial lo asume.
- **Deciles cross-seccionales por mes** (cada empresa contra las demás
  del mismo mes, como el paper): rank-based, sin winsorizar.
- **AUC por Mann-Whitney** con bootstrap estratificado por empresa
  (cluster bootstrap: los meses de una firma están autocorrelacionados;
  estratificado porque con 11 quebradas un remuestreo simple dejaría
  réplicas sin defaults). Diferencia pareada en la misma réplica.
- **Logit pooled** con SE cluster por empresa, DD winsorizado 1/99 solo
  aquí (las colas de DD > 40 de empresas casi sin deuda distorsionarían
  la pendiente; deciles y AUC son rank-based y no lo necesitan).

## Resultados

| métrica | naive | Merton completo |
|---|---|---|
| AUC (IC95 bootstrap) | 0.9834 [0.973, 0.993] | 0.9804 [0.961, 0.994] |
| Accuracy ratio (CAP) | 0.967 | 0.961 |
| decil 10 captura | 93.8% | 96.2% |
| deciles 9-10 capturan | 100% | 100% |
| logit solo: z del DD | -6.23 | -4.36 |
| pseudo-R2 | 0.522 | 0.517 |

- Las 11 quebradas estaban en el decil 10 un mes antes del filing según
  ambos modelos (a 3, 6 y 12 meses casi todas siguen en el 10; PCG cae al
  9 en merton a 3m, PTRAQ al 9 a 6-12m, NRDE no tiene observación a 12m).
- Diferencia de AUC (naive - completo): +0.0030, IC95 [-0.0042, +0.0139],
  probabilidad bootstrap de naive >= completo: 64.9%.
- Leave-one-out por quebrada: el signo de la diferencia se invierte solo
  al quitar PCG (+0.0030 pasa a -0.0018). Guardado en
  replica_leave_one_out.parquet.
- Logit conjunto: ninguno significativo (z -1.0 y -0.6) y añadir el
  completo al naive mueve el pseudo-R2 de 0.522 a 0.526. Con Spearman
  0.99 entre ambos DD y 11 clusters tratados, esto NO reproduce el
  horse-race del hazard del paper: se reporta como evidencia descriptiva
  de redundancia, nada más (hallazgo de la revisión adversarial).

## Conclusión (formulada sobre el bootstrap, no sobre el punto)

**Se replica la sustancia del hallazgo de Bharath & Shumway: la solución
iterativa no añade poder predictivo sobre la forma funcional naive.** La
diferencia de AUC es un empate estadístico cuyo signo depende de una sola
firma (PG&E). No se puede afirmar que el naive sea superior, y tampoco
hace falta: la tesis del paper es que lo valioso del Merton es su forma
funcional, no el solver. La regla que decide el booleano
replica_hallazgo_paper del JSON es "el completo NO supera al naive con el
IC95 entero por debajo de cero", no la comparación puntual (hallazgo de
la revisión adversarial: con otra semilla u otra cohorte el punto podía
voltearse).

## Honestidad estadística (lo que estos números NO dicen)

1. **Los niveles no son comparables con el paper.** AUC ~0.98 y pseudo-R2
   ~0.52 contra accuracy ratios ~0.65-0.87 y pseudo-R2 ~0.09 del paper:
   nuestro panel (123 mega-caps sobrevivientes del S&P 500 actual contra
   11 quebradas OTC en colapso) hace la separación casi trivial. Separar
   Apple de Rite Aid seis meses antes del Chapter 11 no tiene mérito.
2. **Efecto techo:** con ambos AUC ~0.98, las diferencias entre modelos se
   comprimen mecánicamente hacia cero; el empate es en parte artefacto
   del diseño muestral. La comparación interna sigue siendo válida
   (mismas 14,482 filas para ambos), pero no es una réplica plena sobre
   un universo como el de CRSP.
3. **11 eventos son 11 eventos:** el percentile bootstrap con 11 clusters
   es poco fiable (la dependencia de PCG es el síntoma) y los z del logit
   con 11 clusters tratados son anticonservadores. Por eso el peso de la
   evidencia está en los deciles y en los casos de estudio, no en la
   significancia fina.

Salidas: replica_deciles_naive/merton.parquet, replica_deciles_eventos.parquet,
replica_leave_one_out.parquet, replica_conclusion.json (todo en data/cache).
