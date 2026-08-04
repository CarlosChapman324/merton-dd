"""Fase 4: la mejora propia sobre la réplica.

1. Baseline contable (mini Shumway): logit con apalancamiento, tamaño,
   ret_12m, sigma_E y ROA. ¿Cuánto predice sin ningún Merton?
2. Señal incremental: ratios + DD. ¿El DD sigue significativo? ¿Mejora el
   AUC fuera de muestra (K-fold agrupado por empresa)?
3. Casos de estudio: curvas de DD alineadas al evento + banda de vivas +
   antelación de la alerta por empresa.
4. Calibración: pi promedio vs tasa realizada, global y por decil.

Uso: uv run python scripts/build_mejora.py (lee data/cache, sin red)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import cache
from validation import baseline, calibracion, casos, eventos, metricas, modelos

RATIOS = baseline.RATIOS


def main() -> None:
    panel = cache.cargar_df("panel_dd")
    etiquetado = eventos.etiquetar(panel)
    muestra = baseline.preparar(etiquetado)
    positivos = int(muestra["default_12m"].sum())
    print("=== Muestra de la mejora ===")
    print(f"filas con baseline completo y ambos DD: {len(muestra)} "
          f"({positivos} positivos de {muestra[muestra.default_12m == 1]['ticker'].nunique()} quiebras)")

    print()
    print("=== Logit in-sample, SE cluster por empresa ===")
    especificaciones = [
        ("baseline contable", RATIOS),
        ("solo dd_naive", ["dd_naive"]),
        ("baseline + dd_naive", RATIOS + ["dd_naive"]),
        ("baseline + dd_merton", RATIOS + ["dd_merton"]),
    ]
    resumen_logits = {}
    for nombre, cols in especificaciones:
        modelo = modelos.logit_cluster(muestra, cols, winsor=(0.0, 1.0))  # ya winsorizado en preparar
        tabla = modelos.resumen_logit(modelo, cols)
        resumen_logits[nombre] = {"pseudo_r2": float(modelo.prsquared), "tabla": tabla}
        print(f"[{nombre}] pseudo-R2: {modelo.prsquared:.3f}")
        print(tabla.to_string(float_format=lambda v: f"{v: .4f}"))
        print()

    print("=== AUC fuera de muestra (K-fold agrupado por empresa) ===")
    aucs_cv = {}
    for nombre, cols in especificaciones:
        valor, _ = baseline.auc_cv_por_grupos(muestra, cols)
        aucs_cv[nombre] = float(valor)
        print(f"{nombre:22} AUC CV: {valor:.4f}")
    ganancia = aucs_cv["baseline + dd_naive"] - aucs_cv["baseline contable"]
    print(f"ganancia del DD sobre el baseline contable: {ganancia:+.4f}")

    print()
    print("=== Casos de estudio: curvas alineadas al evento ===")
    curvas = casos.curvas_alineadas(panel)
    banda = casos.banda_vivas(panel)
    anticipacion = casos.resumen_anticipacion(curvas)
    cache.guardar_df(curvas, "mejora_curvas_alineadas")
    cache.guardar_df(banda, "mejora_banda_vivas")
    cache.guardar_df(anticipacion, "mejora_anticipacion")
    print(f"curvas: {curvas['ticker'].nunique()} quebradas, banda de vivas "
          f"(dd_merton mediana {banda[banda.modelo == 'dd_merton']['mediana'].iloc[0]:.1f})")
    print("meses de antelación con DD < 2 sostenido hasta el filing:")
    print(anticipacion.to_string(index=False))

    print()
    print("=== Calibración: N(-DD) no es una PD calibrada ===")
    # Sobre la MISMA muestra de la réplica (etiquetado + ambos DD), no la de
    # la mejora: la calibración no necesita ratios contables y filtrar por
    # ellos cambiaría la tasa base reportada (hallazgo de la revisión).
    muestra_replica = etiquetado.dropna(subset=["dd_naive", "dd_merton"]).reset_index(drop=True)
    global_ = calibracion.calibracion_global(muestra_replica)
    print(global_.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    por_decil = calibracion.calibracion_por_decil(muestra_replica, "pi_naive")
    cache.guardar_df(global_, "mejora_calibracion_global")
    cache.guardar_df(por_decil.reset_index(), "mejora_calibracion_deciles")
    print("por decil (pi_naive):")
    print(por_decil.to_string(float_format=lambda v: f"{v:.4f}"))

    # significancia del DD en el conjunto (la pregunta de la mejora)
    tabla_conjunto = resumen_logits["baseline + dd_naive"]["tabla"]
    z_dd = float(tabla_conjunto.loc["dd_naive", "z"])
    p_dd = float(tabla_conjunto.loc["dd_naive", "p"])
    conclusion = {
        "aucs_cv": aucs_cv,
        "ganancia_dd_sobre_baseline": ganancia,
        "z_dd_en_conjunto": z_dd,
        "p_dd_en_conjunto": p_dd,
        "pseudo_r2": {k: v["pseudo_r2"] for k, v in resumen_logits.items()},
        "calibracion_global": global_.to_dict(orient="records"),
        "advertencia": "AUC y pseudo-R2 no comparables con el paper (efecto techo del panel, "
        "N efectivo 11 eventos); z con 11 clusters tratados es anticonservador.",
    }
    Path("data/cache/mejora_conclusion.json").write_text(json.dumps(conclusion, indent=2))
    print()
    print("=== Conclusión de la mejora ===")
    print(f"El DD naive en el logit conjunto: z = {z_dd:.2f} (p = {p_dd:.4f})")
    print(f"AUC CV: baseline {aucs_cv['baseline contable']:.4f} -> "
          f"baseline + DD {aucs_cv['baseline + dd_naive']:.4f} ({ganancia:+.4f})")
    print("Detalle en data/cache/mejora_*.parquet y mejora_conclusion.json")


if __name__ == "__main__":
    main()
