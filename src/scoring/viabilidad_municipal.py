from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PVOUT_PATH = PROJECT_ROOT / "data" / "clean" / "pvout_municipios" / "pvout_puntos_extraidos.csv"
DEFAULT_SLOPE_PATH = (
    PROJECT_ROOT / "data" / "clean" / "pendientes_municipios" / "pendiente_puntos_extraidos.csv"
)
DEFAULT_GRID_DISTANCE_PATH = (
    PROJECT_ROOT / "data" / "clean" / "subestaciones_upme" / "distancia_subestacion_municipios.csv"
)
DEFAULT_RUNAP_PATH = (
    PROJECT_ROOT / "data" / "clean" / "runap_protegidas" / "runap_restricciones_municipios.csv"
)
DEFAULT_DEMAND_PATH = (
    PROJECT_ROOT / "data" / "clean" / "xm_demanda_municipal" / "xm_demanda_municipal_proxy.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "viabilidad_municipal"


OFFICIAL_WEIGHTS = {
    "w_s": 0.30,
    "w_d": 0.15,
    "w_g": 0.25,
    "w_p": 0.20,
    "w_u": 0.10,
}

PRELIMINARY_DISTANCE_LIMIT_KM = 50.0


def minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Normaliza una variable numerica en escala 0-1."""

    values = pd.to_numeric(series, errors="coerce")
    minimum = values.min(skipna=True)
    maximum = values.max(skipna=True)
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series([pd.NA] * len(series), index=series.index, dtype="Float64")

    if higher_is_better:
        score = (values - minimum) / (maximum - minimum)
    else:
        score = (maximum - values) / (maximum - minimum)
    return score.clip(0, 1)


def classify_percentile(score: pd.Series) -> pd.Series:
    """Clasifica score preliminar por percentiles."""

    numeric = pd.to_numeric(score, errors="coerce")
    positive = numeric[numeric > 0]
    if positive.empty:
        return numeric.map(lambda value: "sin_datos" if pd.isna(value) else "excluida_preliminar")

    p50 = positive.quantile(0.50)
    p75 = positive.quantile(0.75)
    p90 = positive.quantile(0.90)

    def classify(value: Any) -> str:
        if pd.isna(value):
            return "sin_datos"
        if value <= 0:
            return "excluida_preliminar"
        if value >= p90:
            return "muy_alta_preliminar"
        if value >= p75:
            return "alta_preliminar"
        if value >= p50:
            return "media_preliminar"
        return "baja_preliminar"

    return numeric.map(classify)


def load_inputs(
    pvout_path: Path,
    slope_path: Path,
    grid_distance_path: Path | None,
    runap_path: Path | None,
    demand_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Carga fuentes municipales ya limpias."""

    if not pvout_path.exists():
        raise FileNotFoundError(f"No existe el archivo PVOUT municipal: {pvout_path}")
    if not slope_path.exists():
        raise FileNotFoundError(f"No existe el archivo de pendiente municipal: {slope_path}")

    pvout = pd.read_csv(pvout_path, dtype={"codigo_dane": "string"})
    slope = pd.read_csv(slope_path, dtype={"codigo_dane": "string"})
    for frame_name, frame in {"PVOUT": pvout, "pendiente": slope}.items():
        if "codigo_dane" not in frame.columns:
            raise ValueError(f"{frame_name} no tiene columna codigo_dane.")
        frame["codigo_dane"] = frame["codigo_dane"].astype("string").str.zfill(5)

    grid_distance = None
    if grid_distance_path is not None and grid_distance_path.exists():
        grid_distance = pd.read_csv(grid_distance_path, dtype={"codigo_dane": "string"})
        if "codigo_dane" not in grid_distance.columns:
            raise ValueError("Distancia a red no tiene columna codigo_dane.")
        grid_distance["codigo_dane"] = grid_distance["codigo_dane"].astype("string").str.zfill(5)

    runap = None
    if runap_path is not None and runap_path.exists():
        runap = pd.read_csv(runap_path, dtype={"codigo_dane": "string"})
        if "codigo_dane" not in runap.columns:
            raise ValueError("Restricciones RUNAP no tiene columna codigo_dane.")
        runap["codigo_dane"] = runap["codigo_dane"].astype("string").str.zfill(5)

    demand = None
    if demand_path is not None and demand_path.exists():
        demand = pd.read_csv(demand_path, dtype={"codigo_dane": "string"})
        if "codigo_dane" not in demand.columns:
            raise ValueError("Demanda XM municipal no tiene columna codigo_dane.")
        demand["codigo_dane"] = demand["codigo_dane"].astype("string").str.zfill(5)
    return pvout, slope, grid_distance, runap, demand


def build_preliminary_score(
    pvout: pd.DataFrame,
    slope: pd.DataFrame,
    grid_distance: pd.DataFrame | None,
    runap: pd.DataFrame | None,
    demand: pd.DataFrame | None,
    distance_limit_km: float = PRELIMINARY_DISTANCE_LIMIT_KM,
) -> pd.DataFrame:
    """Construye tabla municipal preliminar con variables disponibles."""

    base_columns = [
        "codigo_dane",
        "municipio",
        "departamento",
        "categoria_nombre",
        "area_km2_igac",
        "altitud_m",
        "lon",
        "lat",
        "pvout_kwh_kwp_day",
        "annual_yield_kwh_kw_year",
    ]
    available_base = [column for column in base_columns if column in pvout.columns]
    result = pvout[available_base].copy()

    slope_columns = [
        "codigo_dane",
        "pendiente_igac",
        "viabilidad_pendiente",
        "score_pendiente",
        "identify_ok",
        "identify_mensaje",
    ]
    available_slope = [column for column in slope_columns if column in slope.columns]
    result = result.merge(
        slope[available_slope],
        on="codigo_dane",
        how="left",
        validate="one_to_one",
    )

    if grid_distance is not None:
        distance_columns = [
            "codigo_dane",
            "dist_subestacion_km",
            "subestacion_mas_cercana",
            "nivel_tension_mas_cercana",
            "tension_mas_cercana",
            "capacidad_mva_mas_cercana",
            "subestacion_lon",
            "subestacion_lat",
            "criterio_red",
        ]
        available_distance = [column for column in distance_columns if column in grid_distance.columns]
        result = result.merge(
            grid_distance[available_distance],
            on="codigo_dane",
            how="left",
            validate="one_to_one",
        )
    else:
        result["dist_subestacion_km"] = pd.NA

    if runap is not None:
        runap_columns = [
            "codigo_dane",
            "area_municipio_km2_calc",
            "area_protegida_km2_runap",
            "pct_area_protegida_runap",
            "area_no_protegida_km2_runap",
            "u_i_no_protegido_runap",
            "r_i_runap",
            "clasificacion_restriccion_runap",
            "criterio_runap",
        ]
        available_runap = [column for column in runap_columns if column in runap.columns]
        result = result.merge(
            runap[available_runap],
            on="codigo_dane",
            how="left",
            validate="one_to_one",
        )
    else:
        result["u_i_no_protegido_runap"] = pd.NA
        result["r_i_runap"] = pd.NA

    if demand is not None:
        demand_columns = [
            "codigo_dane",
            "zona_xm_demanda",
            "tipo_mapeo_demanda",
            "confianza_mapeo_demanda",
            "flag_revision_demanda",
            "flag_atipico_eda_demanda",
            "nota_mapeo_demanda",
            "demanda_xm_valor_promedio",
            "demanda_xm_valor_p95",
            "demanda_xm_proxy_mwh_o_unidad_fuente",
            "d_i_demanda",
            "demanda_xm_disponible",
            "decision_eda_demanda",
        ]
        available_demand = [column for column in demand_columns if column in demand.columns]
        result = result.merge(
            demand[available_demand],
            on="codigo_dane",
            how="left",
            validate="one_to_one",
        )
    else:
        result["d_i_demanda"] = pd.NA
        result["flag_revision_demanda"] = 1
        result["flag_atipico_eda_demanda"] = 1
        result["tipo_mapeo_demanda"] = "sin_fuente_demanda"

    result["s_i_solar"] = minmax_score(result["pvout_kwh_kwp_day"], higher_is_better=True)
    result["p_i_pendiente_proxy"] = pd.to_numeric(result["score_pendiente"], errors="coerce")
    result["g_i_red"] = minmax_score(result["dist_subestacion_km"], higher_is_better=False)
    result["u_i_uso_suelo"] = pd.to_numeric(result["u_i_no_protegido_runap"], errors="coerce")
    result["d_i_demanda"] = pd.to_numeric(result["d_i_demanda"], errors="coerce")

    # Restriccion preliminar: usa la pendiente puntual disponible y distancia
    # maxima a subestacion de alta tension mas RUNAP. La restriccion oficial
    # debe recalcularse con area apta municipal, lineas, capacidad y permisos.
    slope_restriction = result["viabilidad_pendiente"].map(
        {
            "viable": 1.0,
            "condicional": 1.0,
            "no_viable": 0.0,
            "desconocida": 0.0,
        }
    )
    distance_restriction = pd.Series(1.0, index=result.index)
    if "dist_subestacion_km" in result.columns:
        distance_restriction = (
            pd.to_numeric(result["dist_subestacion_km"], errors="coerce") <= distance_limit_km
        ).astype(float)
        distance_restriction = distance_restriction.where(result["dist_subestacion_km"].notna(), 0.0)
    runap_restriction = pd.Series(1.0, index=result.index)
    if "r_i_runap" in result.columns:
        runap_restriction = pd.to_numeric(result["r_i_runap"], errors="coerce").fillna(1.0)
    result["r_i_preliminar"] = (
        pd.to_numeric(slope_restriction, errors="coerce").fillna(0.0)
        * distance_restriction
        * runap_restriction
    )

    available_terms = [
        ("s_i_solar", OFFICIAL_WEIGHTS["w_s"]),
        ("p_i_pendiente_proxy", OFFICIAL_WEIGHTS["w_p"]),
        ("g_i_red", OFFICIAL_WEIGHTS["w_g"]),
        ("u_i_uso_suelo", OFFICIAL_WEIGHTS["w_u"]),
    ]
    usable_terms = [
        (column, weight)
        for column, weight in available_terms
        if column in result.columns and pd.to_numeric(result[column], errors="coerce").notna().any()
    ]
    available_weight_sum = sum(weight for _, weight in usable_terms)
    weighted_sum = sum(pd.to_numeric(result[column], errors="coerce").fillna(0.0) * weight for column, weight in usable_terms)
    result["score_preliminar_solar_red_pendiente_runap"] = result["r_i_preliminar"] * (
        weighted_sum / available_weight_sum
    )

    full_components = [
        "s_i_solar",
        "d_i_demanda",
        "g_i_red",
        "p_i_pendiente_proxy",
        "u_i_uso_suelo",
    ]
    has_full_components = result[full_components].notna().all(axis=1)
    result["v_i_modelo_proxy_xm"] = pd.NA
    result.loc[has_full_components, "v_i_modelo_proxy_xm"] = result.loc[
        has_full_components, "r_i_preliminar"
    ] * (
        OFFICIAL_WEIGHTS["w_s"] * result.loc[has_full_components, "s_i_solar"]
        + OFFICIAL_WEIGHTS["w_d"] * result.loc[has_full_components, "d_i_demanda"]
        + OFFICIAL_WEIGHTS["w_g"] * result.loc[has_full_components, "g_i_red"]
        + OFFICIAL_WEIGHTS["w_p"] * result.loc[has_full_components, "p_i_pendiente_proxy"]
        + OFFICIAL_WEIGHTS["w_u"] * result.loc[has_full_components, "u_i_uso_suelo"]
    )
    result["clasificacion_preliminar"] = classify_percentile(
        result["v_i_modelo_proxy_xm"]
    )

    result["v_i_modelo_oficial"] = result["v_i_modelo_proxy_xm"]
    result["estado_modelo_oficial"] = (
        "modelo_proxy_xm: demanda asignada por subarea XM; revisar flags de demanda y limitaciones RUNAP/POT"
    )
    result.loc[result.get("flag_revision_demanda", 0).eq(1), "estado_modelo_oficial"] = (
        "modelo_proxy_xm_con_demanda_en_revision"
    )
    result.loc[result["d_i_demanda"].isna(), "estado_modelo_oficial"] = (
        "sin_v_i: no hay demanda XM asignada al municipio"
    )
    result["notas_metodologicas"] = (
        "Score preliminar usa PVOUT puntual municipal, distancia a subestacion UPME "
        "clase de pendiente IGAC en punto interno, proporcion no protegida RUNAP y demanda XM pron_areas "
        "asignada por proxy regional/departamental."
    )

    sort_columns = [
        "v_i_modelo_proxy_xm",
        "score_preliminar_solar_red_pendiente_runap",
        "s_i_solar",
        "d_i_demanda",
        "g_i_red",
        "p_i_pendiente_proxy",
        "u_i_uso_suelo",
    ]
    return result.sort_values(sort_columns, ascending=False, na_position="last")


def write_observations(output_path: Path, scored: pd.DataFrame) -> None:
    """Documenta alcance, pesos y limitaciones del score preliminar."""

    counts = scored["clasificacion_preliminar"].value_counts(dropna=False)
    slope_counts = scored["viabilidad_pendiente"].value_counts(dropna=False)
    lines = [
        "Viabilidad municipal preliminar",
        "================================",
        "",
        "Formula oficial del proyecto:",
        "V_i = R_i(0.30*S_i + 0.15*D_i + 0.25*G_i + 0.20*P_i + 0.10*U_i)",
        "",
        "Estado actual:",
        "- S_i solar: disponible con PVOUT municipal puntual.",
        "- P_i pendiente: disponible como proxy puntual IGAC, no como proporcion de area apta.",
        "- G_i red: disponible como distancia minima a subestacion UPME en servicio de nivel 4/5.",
        "- U_i uso del suelo/restriccion: disponible parcialmente como proporcion no protegida RUNAP.",
        "- D_i demanda: disponible como proxy municipal desde subareas XM pron_areas.",
        "- V_i oficial: se exporta como v_i_modelo_proxy_xm porque D_i no es demanda municipal directa.",
        "",
        "Score principal exportado:",
        "v_i_modelo_proxy_xm = R_preliminar * (0.30*S_i + 0.15*D_i + 0.25*G_i + 0.20*P_i + 0.10*U_i)",
        "",
        "Score auxiliar sin demanda:",
        "score_preliminar_solar_red_pendiente_runap = R_preliminar * ((0.30*S_i + 0.25*G_i + 0.20*P_i + 0.10*U_i) / 0.85)",
        "",
        "Restriccion preliminar:",
        "- R_preliminar = 1 si la clase IGAC es viable o condicional.",
        "- R_preliminar = 0 si la clase IGAC es no_viable o desconocida.",
        f"- R_preliminar = 0 si distancia a subestacion > {PRELIMINARY_DISTANCE_LIMIT_KM:.0f} km.",
        "- R_preliminar = 0 si RUNAP cubre al menos 80% del municipio.",
        "",
        "Conteo por clasificacion preliminar:",
    ]
    for label, count in counts.items():
        lines.append(f"- {label}: {count}")

    lines.append("")
    if "tipo_mapeo_demanda" in scored.columns:
        lines.append("Conteo por tipo de mapeo de demanda XM:")
        for label, count in scored["tipo_mapeo_demanda"].value_counts(dropna=False).items():
            lines.append(f"- {label}: {count}")
        lines.append(
            f"- Municipios con flag_revision_demanda=1: {int(scored['flag_revision_demanda'].fillna(0).eq(1).sum())}"
        )
        lines.append(
            f"- Municipios con flag_atipico_eda_demanda=1: {int(scored['flag_atipico_eda_demanda'].fillna(0).eq(1).sum())}"
        )
        lines.append("")
    lines.append("Conteo por pendiente IGAC:")
    for label, count in slope_counts.items():
        lines.append(f"- {label}: {count}")

    lines.extend(
        [
            "",
            "Limitaciones criticas:",
            "- Esta tabla sirve para avanzar el pipeline, no para afirmar los mejores municipios finales.",
            "- La pendiente se muestrea en un punto interno; el modelo final debe usar proporcion de area apta.",
            "- La distancia a red es euclidiana a subestacion; no incluye lineas, servidumbres ni capacidad disponible real.",
            "- RUNAP cubre areas protegidas registradas; no reemplaza uso/cobertura completa del suelo ni licenciamiento ambiental.",
            "- La demanda XM usada es proxy regional/subarea; los flags permiten excluir casos ambiguos en analisis de sensibilidad.",
            "- El uso del suelo compatible con pastoreo sigue pendiente de una capa de cobertura/uso por area o de POT robusto por poligono.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_scoring(
    pvout_path: Path = DEFAULT_PVOUT_PATH,
    slope_path: Path = DEFAULT_SLOPE_PATH,
    grid_distance_path: Path | None = DEFAULT_GRID_DISTANCE_PATH,
    runap_path: Path | None = DEFAULT_RUNAP_PATH,
    demand_path: Path | None = DEFAULT_DEMAND_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    top_n: int = 10,
    distance_limit_km: float = PRELIMINARY_DISTANCE_LIMIT_KM,
) -> dict[str, Path]:
    """Ejecuta scoring preliminar municipal."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pvout, slope, grid_distance, runap, demand = load_inputs(
        pvout_path,
        slope_path,
        grid_distance_path,
        runap_path,
        demand_path,
    )
    scored = build_preliminary_score(
        pvout,
        slope,
        grid_distance,
        runap,
        demand,
        distance_limit_km=distance_limit_km,
    )

    scored_path = output_dir / "viabilidad_municipal_preliminar.csv"
    top_path = output_dir / f"top{top_n}_modelo_proxy_xm_etiquetado.csv"
    top_clean_path = output_dir / f"top{top_n}_modelo_proxy_xm_sin_atipicos_demanda.csv"
    observations_path = output_dir / "observaciones_viabilidad_municipal.txt"

    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    scored.head(top_n).to_csv(top_path, index=False, encoding="utf-8-sig")
    clean_mask = (
        scored["v_i_modelo_proxy_xm"].notna()
        & scored["flag_atipico_eda_demanda"].fillna(1).eq(0)
    )
    scored[clean_mask].head(top_n).to_csv(top_clean_path, index=False, encoding="utf-8-sig")
    write_observations(observations_path, scored)

    return {
        "scored": scored_path,
        "top": top_path,
        "top_sin_atipicos_demanda": top_clean_path,
        "observations": observations_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye score municipal preliminar con PVOUT y pendiente, sin diagramas."
    )
    parser.add_argument("--pvout-path", type=Path, default=DEFAULT_PVOUT_PATH)
    parser.add_argument("--slope-path", type=Path, default=DEFAULT_SLOPE_PATH)
    parser.add_argument("--grid-distance-path", type=Path, default=DEFAULT_GRID_DISTANCE_PATH)
    parser.add_argument("--runap-path", type=Path, default=DEFAULT_RUNAP_PATH)
    parser.add_argument("--demand-path", type=Path, default=DEFAULT_DEMAND_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--distance-limit-km", type=float, default=PRELIMINARY_DISTANCE_LIMIT_KM)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_scoring(
        pvout_path=args.pvout_path,
        slope_path=args.slope_path,
        grid_distance_path=args.grid_distance_path,
        runap_path=args.runap_path,
        demand_path=args.demand_path,
        output_dir=args.output_dir,
        top_n=args.top_n,
        distance_limit_km=args.distance_limit_km,
    )
    print("Score municipal preliminar generado.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print("Advertencia: V_i se calcula como proxy XM; revise flags de demanda antes de sustentar resultados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
