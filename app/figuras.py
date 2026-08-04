"""Figuras Plotly del proyecto: una sola fuente para el dashboard y el README.

Identidad visual: terminal de datos oscura con acento azul (la misma del
predictor del Mundial). Todo pasa por el template "terminal" que se
registra al importar este módulo.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

FONDO = "#0b0e14"
PANEL = "#11151d"
REJILLA = "#1f2937"
TEXTO = "#e6edf3"
GRIS = "#8b949e"
AZUL = "#3b82f6"
AZUL_CLARO = "#60a5fa"
ROJO = "#f87171"
VERDE = "#34d399"
AMBAR = "#fbbf24"

pio.templates["terminal"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, JetBrains Mono, monospace", color=TEXTO, size=13),
        xaxis=dict(gridcolor=REJILLA, zerolinecolor=REJILLA, linecolor=REJILLA),
        yaxis=dict(gridcolor=REJILLA, zerolinecolor=REJILLA, linecolor=REJILLA),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        margin=dict(l=50, r=20, t=50, b=40),
        colorway=[AZUL, ROJO, VERDE, AMBAR, AZUL_CLARO, GRIS],
        hoverlabel=dict(bgcolor=PANEL, font=dict(family="IBM Plex Mono, monospace")),
    )
)
pio.templates.default = "terminal"


def fig_payoff() -> go.Figure:
    """El diagrama de la metodología: el equity es una call sobre los activos."""
    V = np.linspace(0, 200, 300)
    F = 100.0
    equity = np.maximum(V - F, 0.0)
    deuda = np.minimum(V, F)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=V, y=equity, name="Equity = max(V - F, 0)",
                             line=dict(color=AZUL, width=3)))
    fig.add_trace(go.Scatter(x=V, y=deuda, name="Deuda = min(V, F)",
                             line=dict(color=GRIS, width=2, dash="dash")))
    fig.add_vline(x=F, line=dict(color=ROJO, width=1, dash="dot"))
    fig.add_annotation(x=F, y=95, text="F (deuda)", showarrow=False,
                       font=dict(color=ROJO), xshift=42)
    fig.update_layout(
        title="En el vencimiento: el patrimonio es una opción call sobre los activos",
        xaxis_title="Valor de los activos V en T",
        yaxis_title="Pago a cada parte",
        height=420,
    )
    return fig


def fig_curvas_alineadas(curvas: pd.DataFrame, banda: pd.DataFrame, col: str = "dd_merton") -> go.Figure:
    """El gráfico estrella: DD de cada quebrada, alineado al filing (t=0)."""
    fila_banda = banda[banda["modelo"] == col].iloc[0]
    fig = go.Figure()
    # banda de referencia de las vivas
    fig.add_hrect(y0=fila_banda["p25"], y1=fila_banda["p75"],
                  fillcolor=AZUL, opacity=0.10, line_width=0)
    fig.add_hline(y=fila_banda["mediana"], line=dict(color=AZUL, width=1, dash="dot"))
    fig.add_annotation(x=-23.5, y=fila_banda["mediana"], text="mediana de las vivas",
                       showarrow=False, font=dict(color=AZUL_CLARO, size=11), yshift=10)
    fig.add_hline(y=0, line=dict(color=ROJO, width=1, dash="dash"))
    for ticker, g in curvas.groupby("ticker"):
        g = g.sort_values("meses_al_filing")
        fig.add_trace(
            go.Scatter(
                x=-g["meses_al_filing"],
                y=g[col],
                name=ticker,
                mode="lines",
                line=dict(width=2),
                hovertemplate=f"{ticker}: DD %{{y:.2f}} a %{{x}} meses<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Distancia al default ({'completo' if col == 'dd_merton' else 'naive'}) en los 24 meses previos al Chapter 11",
        xaxis_title="Meses hasta el filing (0 = evento)",
        yaxis_title="DD",
        height=520,
        colorway=[ROJO, AMBAR, "#fb923c", "#f472b6", "#e879f9", "#c084fc",
                  "#a78bfa", "#818cf8", AZUL_CLARO, "#22d3ee", VERDE],
    )
    return fig


def fig_deciles(tabla_naive: pd.DataFrame, tabla_merton: pd.DataFrame) -> go.Figure:
    """Porcentaje de empresa-mes en default capturado por cada decil de pi."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tabla_naive["decil"], y=tabla_naive["pct_defaults"],
                         name="pi naive", marker_color=AZUL))
    fig.add_trace(go.Bar(x=tabla_merton["decil"], y=tabla_merton["pct_defaults"],
                         name="pi Merton completo", marker_color=GRIS))
    fig.update_layout(
        title="¿En qué decil de riesgo caen los empresa-mes en default?",
        xaxis_title="Decil de pi (10 = mayor riesgo)",
        yaxis_title="% de los defaults",
        yaxis_tickformat=".0%",
        barmode="group",
        height=420,
        xaxis=dict(dtick=1),
    )
    return fig


def fig_auc(conclusion: dict) -> go.Figure:
    """AUC de ambos modelos con su intervalo bootstrap."""
    nombres = ["naive", "Merton completo"]
    valores = [conclusion["auc_naive"], conclusion["auc_merton"]]
    # el JSON guarda el IC de cada AUC bajo diff/…; los IC individuales
    # se pasan por separado si están; si no, solo barras
    fig = go.Figure(
        go.Bar(x=nombres, y=valores, marker_color=[AZUL, GRIS],
               text=[f"{v:.4f}" for v in valores], textposition="outside")
    )
    fig.update_layout(
        title="AUC a 12 meses: naive vs completo (empate estadístico)",
        yaxis_title="AUC",
        yaxis_range=[0.9, 1.0],
        height=420,
    )
    return fig


def fig_auc_cv(aucs_cv: dict) -> go.Figure:
    """AUC fuera de muestra (K-fold agrupado) de la mejora."""
    orden = ["baseline contable", "baseline + dd_merton", "baseline + dd_naive", "solo dd_naive"]
    valores = [aucs_cv[k] for k in orden]
    colores = [GRIS, AZUL_CLARO, AZUL_CLARO, AZUL]
    fig = go.Figure(
        go.Bar(x=orden, y=valores, marker_color=colores,
               text=[f"{v:.4f}" for v in valores], textposition="outside")
    )
    fig.update_layout(
        title="AUC fuera de muestra (K-fold agrupado por empresa)",
        yaxis_title="AUC out-of-fold",
        yaxis_range=[0.85, 1.0],
        height=420,
    )
    return fig


def fig_calibracion(por_decil: pd.DataFrame) -> go.Figure:
    """pi prometido vs tasa realizada por decil: N(-DD) no es una PD."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=por_decil["decil"], y=por_decil["pi_promedio"],
                         name="pi naive promedio", marker_color=AZUL))
    fig.add_trace(go.Bar(x=por_decil["decil"], y=por_decil["tasa_realizada"],
                         name="tasa realizada 12m", marker_color=ROJO))
    fig.update_layout(
        title="Calibración: lo que pi promete vs lo que ocurre",
        xaxis_title="Decil de pi naive",
        yaxis_title="Probabilidad",
        yaxis_tickformat=".1%",
        barmode="group",
        height=420,
        xaxis=dict(dtick=1),
    )
    return fig


def fig_serie_empresa(panel: pd.DataFrame, ticker: str) -> go.Figure:
    """Serie temporal del DD de una empresa, con su filing si lo hay."""
    g = panel[panel["ticker"] == ticker].sort_values("mes")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["mes"], y=g["dd_merton"], name="DD completo",
                             line=dict(color=AZUL, width=2)))
    fig.add_trace(go.Scatter(x=g["mes"], y=g["dd_naive"], name="DD naive",
                             line=dict(color=AMBAR, width=2, dash="dot")))
    fig.add_hline(y=0, line=dict(color=ROJO, width=1, dash="dash"))
    filing = g["fecha_filing"].dropna()
    if len(filing):
        fig.add_vline(x=filing.iloc[0], line=dict(color=ROJO, width=2))
        fig.add_annotation(x=filing.iloc[0], y=1.02, yref="paper", text="Chapter 11",
                           showarrow=False, font=dict(color=ROJO, size=11))
    fig.update_layout(
        title=f"{ticker}: distancia al default",
        yaxis_title="DD",
        height=440,
    )
    return fig
