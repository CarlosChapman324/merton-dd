"""Dashboard del proyecto: réplica y extensión de Bharath & Shumway (2008).

Uso: uv run streamlit run app/streamlit_app.py
Lee resultados ya calculados (data/cache en local, data/publico en el
deploy; ver data/cache.py); no toca la red ni recalcula nada.
"""

import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from app import figuras  # registra el template plotly "terminal"
from data import cache

st.set_page_config(page_title="Merton DD", page_icon="📉", layout="wide")

CSS = """
<style>
.stApp { background-color: #0b0e14; }
html, body, [class*="css"] { font-family: "IBM Plex Mono", "JetBrains Mono", monospace; }
h1, h2, h3 { color: #e6edf3 !important; letter-spacing: -0.5px; }
[data-testid="stMetric"] {
  background: #11151d; border: 1px solid #1f2937; border-radius: 8px; padding: 14px 18px;
}
[data-testid="stMetricValue"] { color: #60a5fa; font-size: 1.6rem; }
[data-testid="stMetricLabel"] { color: #8b949e; }
.stTabs [data-baseweb="tab"] { color: #8b949e; }
.stTabs [aria-selected="true"] { color: #60a5fa !important; }
div[data-testid="stMarkdownContainer"] { color: #c9d1d9; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def cargar():
    return {
        "panel": cache.cargar_df("panel_dd"),
        "deciles_naive": cache.cargar_df("replica_deciles_naive"),
        "deciles_merton": cache.cargar_df("replica_deciles_merton"),
        "deciles_eventos": cache.cargar_df("replica_deciles_eventos"),
        "loo": cache.cargar_df("replica_leave_one_out"),
        "curvas": cache.cargar_df("mejora_curvas_alineadas"),
        "banda": cache.cargar_df("mejora_banda_vivas"),
        "anticipacion": cache.cargar_df("mejora_anticipacion"),
        "calibracion_deciles": cache.cargar_df("mejora_calibracion_deciles"),
        "calibracion_global": cache.cargar_df("mejora_calibracion_global"),
        "replica": cache.cargar_json("replica_conclusion"),
        "mejora": cache.cargar_json("mejora_conclusion"),
    }


datos = cargar()
panel = datos["panel"]
replica = datos["replica"]
mejora = datos["mejora"]

st.title("Merton Distance to Default")
st.caption(
    "Réplica y extensión de Bharath & Shumway (2008) con datos públicos: "
    "S&P 500 no financiero + 11 Chapter 11 verificados, 2016-2026. "
    "Investigación reproducible de portafolio; no es asesoría de inversión."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("empresa-mes", f"{len(panel):,}")
c2.metric("empresas", panel["ticker"].nunique())
c3.metric("quiebras verificadas", int(panel[panel["es_default"]]["ticker"].nunique()))
c4.metric("AUC naive / completo", f"{replica['auc_naive']:.3f} / {replica['auc_merton']:.3f}")

tabs = st.tabs(["Metodología", "Panel", "Réplica", "Defaults", "Mejora", "Limitaciones"])

with tabs[0]:
    st.subheader("El modelo de Merton en una frase")
    st.markdown(
        """
El patrimonio de una empresa es **una opción call sobre sus activos** con strike igual a su
deuda: si en el horizonte los activos valen más que la deuda, los accionistas se quedan el
excedente; si no, entregan la empresa a los acreedores. De ahí salen dos cantidades:

- **DD (distancia al default):** cuántas desviaciones estándar separan el valor de los
  activos del nivel de la deuda. `DD = [ln(V/F) + (mu - sigma_V²/2)T] / (sigma_V √T)`
- **pi = N(-DD):** la probabilidad de que los activos terminen por debajo de la deuda.

El **modelo completo** resuelve V y sigma_V (no observables) con el sistema iterativo de
Black-Scholes-Merton. El **naive** de Bharath & Shumway los aproxima sin resolver nada:
`V = E + F` y `sigma_V = (E/(E+F))·sigma_E + (F/(E+F))·(0.05 + 0.25·sigma_E)`.
La pregunta del paper: ¿pierde algo el que no resuelve el sistema?
        """
    )
    st.plotly_chart(figuras.fig_payoff(), use_container_width=True)
    st.markdown(
        """
**Datos:** precios de yfinance (fallback stockanalysis.com para deslistadas), deuda y
acciones de SEC EDGAR (companyfacts, point-in-time por fecha de publicación), T-bill de
FRED. Convención de deuda del paper: `F = deuda corriente + 0.5 × deuda de largo plazo`.
        """
    )

with tabs[1]:
    st.subheader("DD por empresa")
    ultimo_mes = panel["mes"].max()
    vigente = panel[panel["mes"] == ultimo_mes].dropna(subset=["dd_merton"])
    col_izq, col_der = st.columns([2, 1])
    with col_izq:
        ticker = st.selectbox("Empresa", sorted(panel["ticker"].unique()), index=0)
        st.plotly_chart(figuras.fig_serie_empresa(panel, ticker), use_container_width=True)
    with col_der:
        st.markdown(f"**Ranking de riesgo, {ultimo_mes.date()}** (menor DD = más riesgo)")
        ranking = vigente.nsmallest(15, "dd_merton")[["ticker", "dd_merton", "dd_naive", "pi_naive"]]
        st.dataframe(ranking.round(3), hide_index=True, height=430)

with tabs[2]:
    st.subheader("La réplica: ¿el naive predice igual que el completo?")
    r1, r2, r3 = st.columns(3)
    r1.metric("AUC naive", f"{replica['auc_naive']:.4f}")
    r2.metric("AUC completo", f"{replica['auc_merton']:.4f}")
    r3.metric("diferencia (IC95)", f"{replica['diff']:+.4f}",
              f"[{replica['diff_ic95'][0]:+.3f}, {replica['diff_ic95'][1]:+.3f}]", delta_color="off")
    st.plotly_chart(
        figuras.fig_deciles(datos["deciles_naive"], datos["deciles_merton"]),
        use_container_width=True,
    )
    st.markdown(
        f"""
**Conclusión (formulada sobre el bootstrap por empresa, no sobre el punto):** se replica la
sustancia del hallazgo del paper. La diferencia de AUC es un **empate estadístico**
(probabilidad bootstrap de naive ≥ completo: {replica['prob_naive_mayor_igual']:.0%}; el
signo del punto depende solo de {', '.join(replica['signo_depende_de'])}, ver leave-one-out
abajo). La solución iterativa no añade poder predictivo sobre la forma funcional: **lo
valioso del Merton es la forma, no el solver.** Las 11 quiebras estaban en el decil 10 de
riesgo un mes antes del filing según ambos modelos.
        """
    )
    with st.expander("Leave-one-out por quebrada (sensibilidad del signo)"):
        st.dataframe(datos["loo"].round(4), hide_index=True)
    with st.expander("Decil de cada quebrada a 1, 3, 6 y 12 meses del filing"):
        st.dataframe(datos["deciles_eventos"], hide_index=True)

with tabs[3]:
    st.subheader("Las quiebras reales: ¿el DD las vio venir?")
    modelo_sel = st.radio("Modelo", ["dd_merton", "dd_naive"], horizontal=True,
                          format_func=lambda c: "completo" if c == "dd_merton" else "naive")
    st.plotly_chart(
        figuras.fig_curvas_alineadas(datos["curvas"], datos["banda"], modelo_sel),
        use_container_width=True,
    )
    st.markdown(
        """
Lectura: las minoristas sobre-endeudadas (Rite Aid, Joann, Big Lots, Party City) pasan
**más de un año** por debajo de DD = 2 antes del filing. Los dos casos donde el modelo
llega tarde son informativos por qué fallan: **PG&E** (el equity conservó valor durante el
proceso y de hecho sobrevivió al Chapter 11) y **Lordstown** (quebró con caja y sin deuda
por el pleito con Foxconn: un modelo estructural no puede ver defaults estratégicos).
        """
    )
    st.markdown("**Antelación de la alerta** (meses con DD < 2 sostenido hasta el filing; "
                "N+ = nunca estuvo sana en la ventana observada):")
    ant = datos["anticipacion"].copy()
    for m in ["merton", "naive"]:
        ant[f"alerta_{m}"] = ant.apply(
            lambda f: f"{f[f'meses_alerta_{m}']}+" if f[f"meses_alerta_{m}_censurado"]
            else str(f[f"meses_alerta_{m}"]), axis=1)
    st.dataframe(ant[["ticker", "alerta_merton", "alerta_naive"]], hide_index=True)

with tabs[4]:
    st.subheader("La mejora: ¿el DD aporta algo sobre un baseline contable?")
    m1, m2, m3 = st.columns(3)
    m1.metric("AUC CV baseline contable", f"{mejora['aucs_cv']['baseline contable']:.4f}")
    m2.metric("AUC CV baseline + DD", f"{mejora['aucs_cv']['baseline + dd_naive']:.4f}",
              f"{mejora['ganancia_dd_sobre_baseline']:+.4f}")
    m3.metric("AUC CV solo DD naive", f"{mejora['aucs_cv']['solo dd_naive']:.4f}")
    st.plotly_chart(figuras.fig_auc_cv(mejora["aucs_cv"]), use_container_width=True)
    st.markdown(
        f"""
Dos respuestas a la pregunta que haría un equipo de riesgo:

1. **Sí, el DD aporta señal incremental:** añadirlo al logit de ratios contables sube el
   AUC fuera de muestra en {mejora['ganancia_dd_sobre_baseline']:+.3f}. Su z in-sample
   ({mejora['z_dd_en_conjunto']:.2f}) no cruza el umbral, pero con 11 clusters de eventos y
   colinealidad estructural (el DD *es* una función de apalancamiento y volatilidad) eso
   era esperable.
2. **El hallazgo incómodo: el DD solo generaliza mejor que el modelo conjunto**
   ({mejora['aucs_cv']['solo dd_naive']:.3f} vs {mejora['aucs_cv']['baseline + dd_naive']:.3f}).
   Con 11 eventos, cada parámetro extra se paga fuera de muestra (parte sobreajuste, parte
   descalibración entre folds del pooling). La forma funcional del Merton ya condensa
   apalancamiento y volatilidad mejor que un logit que intenta recombinarlos.
        """
    )
    st.plotly_chart(figuras.fig_calibracion(datos["calibracion_deciles"]), use_container_width=True)
    cal = datos["calibracion_global"]
    st.markdown(
        f"""
**Nota de calibración:** la tasa realizada de default a 12 meses del panel es
{cal['tasa_realizada'].iloc[0]:.2%}. El pi naive promedio ({cal['pi_promedio'].iloc[0]:.2%})
la sobreestima ~{cal['ratio'].iloc[0]:.1f}x y el pi del completo
({cal['pi_promedio'].iloc[1]:.2%}) la subestima. **N(-DD) ordena bien, pero no es una
probabilidad calibrada:** en la práctica se usa como ranking, no como PD literal.
        """
    )

with tabs[5]:
    st.subheader("Lo que estos resultados NO dicen")
    st.markdown(
        """
1. **Los niveles no son comparables con el paper.** Nuestro AUC ~0.98 contra accuracy
   ratios de ~0.65-0.87 en B&S: el panel (mega-caps sobrevivientes del S&P 500 actual
   contra 11 quebradas OTC en colapso) hace la separación casi trivial. Separar a Apple de
   Rite Aid seis meses antes del Chapter 11 no tiene mérito. La comparación **interna**
   naive vs completo sí es válida: mismas filas para ambos modelos.
2. **Efecto techo.** Con ambos AUC ~0.98, las diferencias entre modelos se comprimen
   mecánicamente hacia cero: el empate es en parte artefacto del diseño muestral.
3. **11 eventos son 11 eventos.** Los 130 empresa-mes positivos son ventanas solapadas de
   11 quiebras. El bootstrap es por empresa y aun así el percentil con 11 clusters es
   frágil (el signo de la diferencia de AUC depende de PG&E). Los z del logit con 11
   clusters tratados son anticonservadores.
4. **Sesgo de disponibilidad en las quebradas.** Las fuentes gratuitas purgan los tickers
   deslistados: la cohorte COVID 2020 (J.C. Penney, Hertz, Chesapeake...) se perdió casi
   entera y BBBY resultó ser otra empresa con el ticker reciclado. Los defaults que quedan
   son los recientes o los que siguieron cotizando.
5. **La deuda de XBRL no captura todo.** WeWork quebró por arriendos operativos que no
   entran en los tags de deuda del paper: su apalancamiento pre-filing (0.35) está
   subestimado y su señal llega tarde por eso.
6. **N(-DD) no es una PD calibrada** (ver pestaña Mejora) y mu con piso en r es una
   convención documentada, no una estimación del retorno esperado real.
7. **No es asesoría de inversión.** Es una réplica académica reproducible con datos
   públicos, construida como pieza de portafolio.
        """
    )

st.divider()
st.caption(
    "Código y bitácoras: scripts/build_*.py reconstruyen todo desde cero con cache local. "
    "Carlos Chapman, 2026."
)
