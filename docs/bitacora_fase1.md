# Bitácora Fase 1: datos y decisiones

Registro de lo encontrado al construir el panel, para el README final.

## Verificación de las quebradas (el riesgo número uno del plan)

La lista candidata original (16 Chapter 11 de 2018-2025) se verificó ticker
por ticker contra las fuentes gratuitas. Resultado central: **las fuentes
gratuitas purgan el histórico de precios de los tickers deslistados al cabo
de unos años**, y eso eliminó a casi toda la cohorte COVID de 2020.

Detalle de la verificación:

- yfinance devuelve vacío para J.C. Penney, Chesapeake, Whiting, Frontier,
  Revlon, Party City, Rite Aid, WeWork, Express, Tupperware, Spirit, etc.,
  tanto con el ticker original como con el ticker OTC con sufijo Q.
- Hertz existe en yfinance pero la serie arranca en julio de 2021: es la
  Hertz re-listada tras la quiebra, inservible para el filing de mayo 2020.
- BBBY existe con historia completa, pero cotizaba a 17.86 USD el 21 de
  abril de 2023, cuando la Bed Bath & Beyond original valía centavos: el
  ticker fue reutilizado por otra empresa (Overstock/Beyond). Se excluye.
- Stooq (el fallback previsto en el PLAN) puso un desafío JavaScript
  anti-bot delante de sus CSV; ya no es utilizable programáticamente.
  Se reemplazó por la API pública de stockanalysis.com, que sí conserva
  varios tickers OTC deslistados con cierre ajustado.
- La descarga masiva de Stooq (static.stooq.com/db) pide autenticación y
  el CSV histórico de WSJ está detrás de un captcha. Nasdaq y Macrotrends
  no tienen los deslistados.

## Lista final de defaults (11, verificados con 24+ meses previos al filing)

| Empresa | Ticker de datos | Filing Ch11 | Fuente de precios |
|---|---|---|---|
| PG&E | PCG | 2019-01-29 | yfinance (siguió listada durante el proceso) |
| Garrett Motion | GTX | 2020-09-20 | yfinance (al límite: cotiza desde 2018-09) |
| Revlon | REVRQ | 2022-06-15 | stockanalysis |
| Party City | PRTYQ | 2023-01-17 | stockanalysis |
| Lordstown Motors | NRDE | 2023-06-27 | yfinance (renombrada Nu Ride, conserva historial) |
| Yellow Corp | YELLQ | 2023-08-06 | yfinance |
| Proterra | PTRAQ | 2023-08-07 | stockanalysis |
| Rite Aid | RADCQ | 2023-10-15 | stockanalysis |
| WeWork | WEWKQ | 2023-11-06 | stockanalysis |
| Joann | JOANQ | 2024-03-18 | stockanalysis |
| Big Lots | BIGGQ | 2024-09-09 | yfinance |

El objetivo del PLAN era 12-15; quedan 11 y se documenta el porqué. Sesgos
que esto introduce y que el README debe decir con claridad:

1. **Sesgo de disponibilidad:** los defaults que sobreviven en las fuentes
   gratuitas son los recientes (2022-2024) o los que siguieron cotizando
   (PG&E). La cohorte 2020 casi desaparece. El panel no es una muestra
   aleatoria de quiebras.
2. **Empresas jóvenes vía SPAC:** Lordstown, Proterra y WeWork cotizan
   pocos años antes de quebrar (la ventana previa incluye su etapa SPAC).
3. **Sesgo de supervivencia en las vivas:** el S&P 500 actual son las
   ganadoras. Igual que en el punto 1, se reporta, no se corrige.

## Otras decisiones de datos

- **Acciones en circulación desde EDGAR** (tag dei
  EntityCommonStockSharesOutstanding) y no desde yfinance: es la única
  fuente consistente que cubre también a las deslistadas. Market cap =
  cierre sin ajustar x acciones del último informe publicado.
- **Anti lookahead:** cada dato de balance conserva su fecha de publicación
  (filed) y el panel mensual solo usa lo publicado hasta ese fin de mes.
  Hay test unitario de esa regla.
- **Precios ajustados solo para retornos y volatilidad;** cierre sin
  ajustar para market cap. En stockanalysis el campo a es el ajustado.
- **Limitación conocida:** si hay un split entre dos informes trimestrales,
  el market cap de esas semanas queda distorsionado (caso WeWork, reverse
  split 1:40 en abril 2023). Se revisará en los casos de estudio.
- **Tags de deuda ampliados:** a la cadena del paper se añadieron
  ConvertibleLongTermNotesPayable (rescata a AKAM) y NotesPayable (rescata a
  DHI); ambos son totales instantáneos de balance. Con eso solo quedan fuera
  AES y EA (deuda solo en hechos dimensionales que companyfacts omite),
  ALGN y EXPD (sin deuda material) y FDXF (registrante nuevo sin historia).
- **CIK de los tickers OTC:** company_tickers.json solo trae registrantes
  activos; los CIK de las 8 quebradas con ticker Q se buscaron a mano en
  EDGAR y quedaron como overrides en universe.py.

## Resultado final de la fase

Panel limpio en data/cache/panel.parquet: 14,197 filas empresa-mes, 133
empresas (122 vivas + 11 quebradas), enero 2016 a junio 2026. Chequeos de
sanidad: market cap de AAPL 4.25 billones (trillions) USD en 2026, mediana
de sigma_E 0.285, mediana de apalancamiento 0.068 en las vivas. Las
quebradas llegan al mes previo al filing con apalancamiento y volatilidad
extremos (Yellow 1.00 y 1.68; Rite Aid 0.98 y 1.26), como predice el modelo.

Casos con señal débil a vigilar en la Fase 4:
- Lordstown (NRDE): apalancamiento 0.02 antes del filing; quebró casi sin
  deuda por el pleito con Foxconn. El DD estructural no puede ver eso, y es
  un buen ejemplo de los límites del modelo.
- WeWork (WEWKQ): apalancamiento 0.35 pre-filing, sospechosamente bajo. Su
  pasivo real eran sobre todo arriendos operativos, que no entran en los
  tags de deuda del paper, y el reverse split 1:40 de abril 2023 puede
  distorsionar E entre informes trimestrales.
- GTX y NRDE tienen ventanas pre-filing cortas en el panel (18 y 10 meses):
  cotizaban desde hacía poco (spinoff 2018 y SPAC 2020).
