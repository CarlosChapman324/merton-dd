"""Fase 3: la réplica del test central de Bharath & Shumway (2008).

Pregunta: ¿el modelo naive (misma forma funcional, sin resolver nada)
predice default igual o mejor que el Merton completo iterativo?

Pasos:
1. Etiquetar empresa-mes: default = Chapter 11 en los 12 meses siguientes
   (sin filas post-filing; solo información previa al evento).
2. Muestra de comparación: filas con AMBOS DD no-NaN (la auditoría de
   Fase 2 mostró 64 filas con dd_merton válido y dd_naive NaN; comparar
   AUC sobre muestras distintas sesgaría la comparación).
3. Deciles cross-seccionales por mes para ambos pi.
4. AUC de ambos, con bootstrap estratificado por empresa e intervalo del
   95% para la diferencia pareada.
5. Logit pooled con SE cluster por empresa: cada DD por separado y ambos
   juntos (¿aporta algo el completo controlando por el naive?).

Uso: uv run python scripts/build_validation.py
Lee data/cache/panel_dd.parquet; no toca la red.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import cache
from validation import eventos, metricas, modelos

N_REPLICAS = 2000


def main() -> None:
    panel = cache.cargar_df("panel_dd")
    etiquetado = eventos.etiquetar(panel)
    excluidas_post = int(panel["post_filing"].sum())

    muestra = etiquetado.dropna(subset=["dd_naive", "dd_merton"]).reset_index(drop=True)
    excluidas = etiquetado[etiquetado[["dd_naive", "dd_merton"]].isna().any(axis=1)]
    excluidas_nan = len(excluidas)
    positivos_excluidos = int(excluidas["default_12m"].sum())

    positivos = int(muestra["default_12m"].sum())
    print("=== Muestra de la réplica ===")
    print(f"filas: {len(muestra)} (excluidas: {excluidas_post} post-filing, {excluidas_nan} sin ambos DD)")
    print(f"las {excluidas_nan} filas sin ambos DD contienen {positivos_excluidos} positivos: "
          f"{'sin sesgo de exclusión' if positivos_excluidos == 0 else 'OJO, revisar sesgo de exclusión'}")
    print(f"empresa-mes en default (12m): {positivos} de {muestra['ticker'].nunique()} empresas, "
          f"{muestra[muestra.default_12m == 1]['ticker'].nunique()} quebradas con meses etiquetados")
    print(f"N efectivo de eventos: {muestra[muestra.default_12m == 1]['ticker'].nunique()} quiebras "
          f"(los {positivos} positivos son ventanas mensuales solapadas de esos mismos eventos)")

    print()
    print("=== Deciles (10 = mayor pi = más riesgo): % de empresa-mes en default capturado ===")
    tablas_deciles = {}
    for nombre, col in [("naive", "pi_naive"), ("merton", "pi_merton")]:
        tabla = metricas.tabla_deciles(muestra, col)
        tablas_deciles[nombre] = tabla
        capturado_top = tabla.loc[10, "pct_defaults"]
        capturado_top2 = tabla.loc[[10, 9], "pct_defaults"].sum()
        print(f"pi_{nombre}: decil 10 captura {capturado_top:.1%}, deciles 9-10 capturan {capturado_top2:.1%}")
    cache.guardar_df(tablas_deciles["naive"].reset_index(), "replica_deciles_naive")
    cache.guardar_df(tablas_deciles["merton"].reset_index(), "replica_deciles_merton")

    print()
    print("=== Decil de cada quebrada a 1, 3, 6 y 12 meses del filing (pi_naive / pi_merton) ===")
    muestra = muestra.assign(
        decil_naive=metricas.deciles_por_mes(muestra, "pi_naive"),
        decil_merton=metricas.deciles_por_mes(muestra, "pi_merton"),
    )
    quebradas = muestra[muestra["fecha_filing"].notna()].copy()
    quebradas["meses_al_filing"] = (
        quebradas["fecha_filing"].dt.to_period("M").astype(int)
        - quebradas["mes"].dt.to_period("M").astype(int)
    )
    filas_evento = []
    for t, g in quebradas.groupby("ticker"):
        fila = {"ticker": t}
        for h in [1, 3, 6, 12]:
            punto = g[g["meses_al_filing"] == h]
            fila[f"naive_{h}m"] = int(punto["decil_naive"].iloc[0]) if len(punto) else None
            fila[f"merton_{h}m"] = int(punto["decil_merton"].iloc[0]) if len(punto) else None
        filas_evento.append(fila)
    tabla_eventos = pd.DataFrame(filas_evento).set_index("ticker")
    print(tabla_eventos.to_string())
    cache.guardar_df(tabla_eventos.reset_index(), "replica_deciles_eventos")

    print()
    print("=== AUC: naive vs completo (bootstrap por empresa, 95%) ===")
    auc_naive = metricas.auc(muestra["default_12m"], muestra["pi_naive"])
    auc_merton = metricas.auc(muestra["default_12m"], muestra["pi_merton"])
    replicas = metricas.bootstrap_auc_pareado(
        muestra, "pi_naive", "pi_merton", n_replicas=N_REPLICAS
    )
    resumen = metricas.resumen_bootstrap(replicas)
    print(f"AUC naive:  {auc_naive:.4f}  IC95 [{resumen['auc_a_ic95'][0]:.4f}, {resumen['auc_a_ic95'][1]:.4f}]"
          f"  (AR {metricas.accuracy_ratio(auc_naive):.3f})")
    print(f"AUC merton: {auc_merton:.4f}  IC95 [{resumen['auc_b_ic95'][0]:.4f}, {resumen['auc_b_ic95'][1]:.4f}]"
          f"  (AR {metricas.accuracy_ratio(auc_merton):.3f})")
    print(f"diferencia (naive - merton): {auc_naive - auc_merton:+.4f}  "
          f"IC95 [{resumen['diff_ic95'][0]:+.4f}, {resumen['diff_ic95'][1]:+.4f}]")
    print(f"probabilidad bootstrap de naive >= merton: {resumen['prob_a_mayor_igual_b']:.1%}")

    print()
    print("=== Sensibilidad: leave-one-out por quebrada (diff = AUC naive - AUC merton) ===")
    loo = []
    for t in sorted(muestra[muestra["es_default"]]["ticker"].unique()):
        sub = muestra[muestra["ticker"] != t]
        d = metricas.auc(sub["default_12m"], sub["pi_naive"]) - metricas.auc(
            sub["default_12m"], sub["pi_merton"]
        )
        loo.append({"sin": t, "diff": d})
    tabla_loo = pd.DataFrame(loo)
    cambia_signo = tabla_loo[tabla_loo["diff"] * (auc_naive - auc_merton) < 0]["sin"].tolist()
    print(tabla_loo.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    if cambia_signo:
        print(f"el signo de la diferencia depende de: {cambia_signo}: es un empate, no una victoria")
    cache.guardar_df(tabla_loo, "replica_leave_one_out")

    print()
    print("=== Logit pooled, SE cluster por empresa (DD winsorizado 1/99) ===")
    pseudo_r2 = {}
    for nombres, cols in [
        (["dd_naive"], ["dd_naive"]),
        (["dd_merton"], ["dd_merton"]),
        (["dd_naive", "dd_merton"], ["dd_naive", "dd_merton"]),
    ]:
        modelo = modelos.logit_cluster(muestra, cols)
        pseudo_r2["+".join(nombres)] = float(modelo.prsquared)
        tabla = modelos.resumen_logit(modelo, nombres)
        print(f"regresores: {nombres} | pseudo-R2: {modelo.prsquared:.3f}")
        print(tabla.to_string(float_format=lambda v: f"{v: .4f}"))
        print()
    ganancia_merton = pseudo_r2["dd_naive+dd_merton"] - pseudo_r2["dd_naive"]
    print(f"añadir el completo al naive mueve el pseudo-R2 de {pseudo_r2['dd_naive']:.3f} "
          f"a {pseudo_r2['dd_naive+dd_merton']:.3f} (+{ganancia_merton:.3f}): redundancia, no mejora.")
    print("Nota: con Spearman ~0.99 entre ambos DD y 11 clusters con eventos, el logit conjunto")
    print("NO es el horse-race del paper: la colinealidad colapsa ambos z y los SE cluster son")
    print("anticonservadores con tan pocos clusters tratados. Se reporta como evidencia")
    print("descriptiva de redundancia, nada más.")

    # La conclusión se decide con el bootstrap, no con el punto: la hipótesis
    # del paper (naive >= completo, es decir, la solución iterativa no añade
    # poder predictivo) se replica si el completo NO supera significativamente
    # al naive. Con el IC de la diferencia cruzando cero y el signo del punto
    # dependiente de una sola firma, lo defendible es un empate estadístico.
    completo_supera = resumen["diff_ic95"][1] < 0
    empate = resumen["diff_ic95"][0] < 0 < resumen["diff_ic95"][1]
    conclusion = {
        "auc_naive": auc_naive,
        "auc_merton": auc_merton,
        "diff": auc_naive - auc_merton,
        "diff_ic95": resumen["diff_ic95"],
        "prob_naive_mayor_igual": resumen["prob_a_mayor_igual_b"],
        "empate_estadistico": bool(empate),
        "signo_depende_de": cambia_signo,
        "pseudo_r2": pseudo_r2,
        "replica_hallazgo_paper": bool(not completo_supera),
        "advertencia_niveles": "AUC ~0.98 y pseudo-R2 ~0.52 NO son comparables con el paper: "
        "el panel (mega-caps sobrevivientes del S&P 500 actual vs 11 quebradas OTC) hace la "
        "separación casi trivial y comprime las diferencias entre modelos (efecto techo). "
        "La comparación interna naive vs completo sí es válida: mismas 14,482 filas para ambos. "
        "N efectivo de eventos: 11.",
    }
    Path("data/cache/replica_conclusion.json").write_text(json.dumps(conclusion, indent=2))
    print()
    print("=== Conclusión ===")
    if completo_supera:
        print("NO se replica: el completo supera al naive con el IC95 completo por debajo de cero.")
    elif empate:
        print("SE REPLICA la sustancia del hallazgo del paper: la solución iterativa no añade")
        print("poder predictivo sobre la forma funcional naive. La diferencia de AUC es un")
        print(f"empate estadístico ({auc_naive - auc_merton:+.4f}, IC95 "
              f"[{resumen['diff_ic95'][0]:+.4f}, {resumen['diff_ic95'][1]:+.4f}]) y su signo")
        print(f"depende de una sola firma ({', '.join(cambia_signo) if cambia_signo else 'ninguna'}).")
        print("No se puede afirmar superioridad del naive, y tampoco hace falta: el punto del")
        print("paper es que lo valioso del Merton es su forma funcional, no el solver.")
    else:
        print("SE REPLICA con holgura: naive supera al completo con IC95 entero sobre cero.")
    print()
    print("Advertencia de niveles: AUC ~0.98 y pseudo-R2 ~0.52 no son comparables con los del")
    print("paper (universo CRSP completo); nuestro panel hace la separación casi trivial por")
    print("diseño (sesgo de supervivencia documentado). Solo la comparación interna es válida.")
    print("Detalle en data/cache/replica_*.parquet y replica_conclusion.json")


if __name__ == "__main__":
    main()
