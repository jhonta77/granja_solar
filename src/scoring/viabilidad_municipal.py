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
DEFAULT_POT_PATH = (
    PROJECT_ROOT / "data" / "clean" / "usos_suelo_pot" / "usos_pot_puntos_extraidos.csv"
)
DEFAULT_DEMAND_PATH = (
    PROJECT_ROOT / "data" / "clean" / "xm_demanda_municipal" / "xm_demanda_municipal_proxy.csv"
)
DEFAULT_COSTS_RISKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "costos_riesgos_municipales"
    / "costos_riesgos_municipales.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "viabilidad_municipal"


RURAL_WEIGHTS = {
    "w_s": 0.35,
    "w_g": 0.30,
    "w_p": 0.25,
    "w_u": 0.10,
}
DEMAND_BONUS_WEIGHT = 0.05
ECONOMIC_CLIMATE_WEIGHTS = {
    "s_i_solar": 0.30,
    "g_i_red": 0.25,
    "p_i_pendiente_proxy": 0.20,
    "u_i_uso_suelo": 0.10,
    "score_tierra": 0.08,
    "score_agua": 0.04,
    "score_riesgo_viento": 0.03,
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
    pot_path: Path | None,
    demand_path: Path | None,
    costs_risks_path: Path | None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
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

    pot = None
    if pot_path is not None and pot_path.exists():
        pot = pd.read_csv(pot_path, dtype={"codigo_dane": "string"})
        if "codigo_dane" not in pot.columns:
            raise ValueError("Usos POT no tiene columna codigo_dane.")
        pot["codigo_dane"] = pot["codigo_dane"].astype("string").str.zfill(5)
        if "tipo_capa_pot" in pot.columns:
            layer_priority = pot["tipo_capa_pot"].astype("string").str.lower().map(
                {"urbana": 0, "rural": 1}
            )
            pot = (
                pot.assign(_prioridad_capa_pot=layer_priority.fillna(2))
                .sort_values(["codigo_dane", "_prioridad_capa_pot"])
                .drop_duplicates(subset=["codigo_dane"], keep="first")
                .drop(columns=["_prioridad_capa_pot"])
            )
        else:
            pot = pot.drop_duplicates(subset=["codigo_dane"], keep="last")

    demand = None
    if demand_path is not None and demand_path.exists():
        demand = pd.read_csv(demand_path, dtype={"codigo_dane": "string"})
        if "codigo_dane" not in demand.columns:
            raise ValueError("Demanda XM municipal no tiene columna codigo_dane.")
        demand["codigo_dane"] = demand["codigo_dane"].astype("string").str.zfill(5)

    costs_risks = None
    if costs_risks_path is not None and costs_risks_path.exists():
        costs_risks = pd.read_csv(costs_risks_path, dtype={"codigo_dane": "string"})
        if "codigo_dane" not in costs_risks.columns:
            raise ValueError("Costos/riesgos municipales no tiene columna codigo_dane.")
        costs_risks["codigo_dane"] = costs_risks["codigo_dane"].astype("string").str.zfill(5)
    return pvout, slope, grid_distance, runap, pot, demand, costs_risks


def impute_score_for_model(
    df: pd.DataFrame,
    score_column: str,
    flag_column: str,
) -> tuple[pd.Series, pd.Series]:
    """Imputa scores faltantes con mediana departamental y luego nacional."""

    values = pd.to_numeric(df[score_column], errors="coerce") if score_column in df.columns else pd.Series(pd.NA, index=df.index)
    imputation = pd.Series("dato_observado", index=df.index, dtype="string")
    missing = values.isna()
    if flag_column in df.columns:
        missing = missing | pd.to_numeric(df[flag_column], errors="coerce").fillna(0).eq(0)

    if "departamento" in df.columns:
        dept_median = values.groupby(df["departamento"]).transform("median")
        dept_fill = missing & dept_median.notna()
        values = values.where(~dept_fill, dept_median)
        imputation.loc[dept_fill] = "mediana_departamental"
        missing = values.isna()

    national_median = values.median(skipna=True)
    if pd.notna(national_median):
        national_fill = missing
        values = values.fillna(national_median)
        imputation.loc[national_fill] = "mediana_nacional"
    else:
        imputation.loc[missing] = "sin_dato_para_imputar"
    return values.clip(0, 1), imputation


def build_preliminary_score(
    pvout: pd.DataFrame,
    slope: pd.DataFrame,
    grid_distance: pd.DataFrame | None,
    runap: pd.DataFrame | None,
    pot: pd.DataFrame | None,
    demand: pd.DataFrame | None,
    costs_risks: pd.DataFrame | None,
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

    if pot is not None:
        pot_columns = [
            "codigo_dane",
            "layer_id_pot",
            "layer_name_pot",
            "tipo_capa_pot",
            "uso_pot",
            "tipo_uso_pot",
            "observacion_pot",
            "categoria_aptitud_pot",
            "u_i_uso_suelo_proxy",
            "apto_doble_uso_pastoreo",
            "restriccion_territorial_proxy",
            "criterio_clasificacion",
            "pot_identify_ok",
            "pot_mensaje",
            "municipio_pot",
        ]
        available_pot = [column for column in pot_columns if column in pot.columns]
        result = result.merge(
            pot[available_pot],
            on="codigo_dane",
            how="left",
            validate="one_to_one",
        )
    else:
        result["tipo_capa_pot"] = pd.NA
        result["categoria_aptitud_pot"] = pd.NA
        result["u_i_uso_suelo_proxy"] = pd.NA
        result["restriccion_territorial_proxy"] = pd.NA
        result["pot_identify_ok"] = pd.NA
        result["pot_mensaje"] = "POT municipal no disponible; no se puede excluir zona urbana por capa."

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

    if costs_risks is not None:
        costs_columns = [
            "codigo_dane",
            "precio_tierra_ha_cop",
            "score_tierra",
            "tarifa_acueducto_m3_cop",
            "score_agua",
            "velocidad_viento_ms",
            "velocidad_viento_max_ms",
            "score_riesgo_viento",
            "flag_dato_tierra",
            "flag_dato_agua",
            "flag_dato_viento",
            "tipo_cruce_tierra",
            "tipo_cruce_agua",
            "tipo_cruce_viento",
        ]
        available_costs = [column for column in costs_columns if column in costs_risks.columns]
        result = result.merge(
            costs_risks[available_costs],
            on="codigo_dane",
            how="left",
            validate="one_to_one",
        )
    else:
        result["precio_tierra_ha_cop"] = pd.NA
        result["score_tierra"] = pd.NA
        result["tarifa_acueducto_m3_cop"] = pd.NA
        result["score_agua"] = pd.NA
        result["velocidad_viento_ms"] = pd.NA
        result["velocidad_viento_max_ms"] = pd.NA
        result["score_riesgo_viento"] = pd.NA
        result["flag_dato_tierra"] = 0
        result["flag_dato_agua"] = 0
        result["flag_dato_viento"] = 0

    result["s_i_solar"] = minmax_score(result["pvout_kwh_kwp_day"], higher_is_better=True)
    result["p_i_pendiente_proxy"] = pd.to_numeric(result["score_pendiente"], errors="coerce")
    result["g_i_red"] = minmax_score(result["dist_subestacion_km"], higher_is_better=False)
    result["u_i_no_protegido_runap"] = pd.to_numeric(result["u_i_no_protegido_runap"], errors="coerce")
    result["u_i_pot_compatible"] = pd.to_numeric(result["u_i_uso_suelo_proxy"], errors="coerce")
    result["u_i_uso_suelo"] = result["u_i_no_protegido_runap"]
    has_pot_score = result["u_i_pot_compatible"].notna()
    result.loc[has_pot_score, "u_i_uso_suelo"] = pd.concat(
        [
            result.loc[has_pot_score, "u_i_no_protegido_runap"],
            result.loc[has_pot_score, "u_i_pot_compatible"],
        ],
        axis=1,
    ).min(axis=1, skipna=True)
    result["d_i_demanda"] = pd.to_numeric(result["d_i_demanda"], errors="coerce")

    # Restriccion preliminar: usa la pendiente puntual disponible y distancia
    # maxima a subestacion de alta tension mas RUNAP/POT. La restriccion oficial
    # debe recalcularse con area rural apta municipal, lineas, capacidad y permisos.
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
    urban_zone = result["tipo_capa_pot"].astype("string").str.lower().eq("urbana")
    result["r_i_zona_urbana_pot"] = pd.Series(1.0, index=result.index)
    result.loc[urban_zone, "r_i_zona_urbana_pot"] = 0.0
    pot_territorial_restriction = pd.Series(1.0, index=result.index)
    if "restriccion_territorial_proxy" in result.columns:
        pot_restricted = pd.to_numeric(
            result["restriccion_territorial_proxy"], errors="coerce"
        ).fillna(0.0).eq(1)
        pot_territorial_restriction = (~pot_restricted).astype(float)
    result["r_i_restriccion_pot"] = pot_territorial_restriction
    result["r_i_preliminar"] = (
        pd.to_numeric(slope_restriction, errors="coerce").fillna(0.0)
        * distance_restriction
        * runap_restriction
        * result["r_i_zona_urbana_pot"]
        * result["r_i_restriccion_pot"]
    )

    available_terms = [
        ("s_i_solar", RURAL_WEIGHTS["w_s"]),
        ("p_i_pendiente_proxy", RURAL_WEIGHTS["w_p"]),
        ("g_i_red", RURAL_WEIGHTS["w_g"]),
        ("u_i_uso_suelo", RURAL_WEIGHTS["w_u"]),
    ]
    usable_terms = [
        (column, weight)
        for column, weight in available_terms
        if column in result.columns and pd.to_numeric(result[column], errors="coerce").notna().any()
    ]
    available_weight_sum = sum(weight for _, weight in usable_terms)
    weighted_sum = sum(pd.to_numeric(result[column], errors="coerce").fillna(0.0) * weight for column, weight in usable_terms)
    result["v_i_modelo_rural"] = result["r_i_preliminar"] * (
        weighted_sum / available_weight_sum
    )
    result["score_preliminar_solar_red_pendiente_runap"] = result["v_i_modelo_rural"]
    result["bono_demanda_favorable"] = (
        result["r_i_preliminar"]
        * pd.to_numeric(result["d_i_demanda"], errors="coerce").fillna(0.0)
        * DEMAND_BONUS_WEIGHT
    )
    result["score_rural_con_bono_demanda"] = (
        result["v_i_modelo_rural"] + result["bono_demanda_favorable"]
    ).clip(upper=1.0)

    result["v_i_modelo_proxy_xm"] = result["v_i_modelo_rural"]
    result["clasificacion_preliminar"] = classify_percentile(
        result["v_i_modelo_rural"]
    )

    for score_column, flag_column in [
        ("score_tierra", "flag_dato_tierra"),
        ("score_agua", "flag_dato_agua"),
        ("score_riesgo_viento", "flag_dato_viento"),
    ]:
        result[score_column] = pd.to_numeric(result[score_column], errors="coerce")
        result[f"{score_column}_modelo"], result[f"imputacion_{score_column}"] = impute_score_for_model(
            result,
            score_column,
            flag_column,
        )

    economic_sum = sum(
        pd.to_numeric(
            result.get(f"{column}_modelo", result[column]),
            errors="coerce",
        ).fillna(0.0)
        * weight
        for column, weight in ECONOMIC_CLIMATE_WEIGHTS.items()
        if column in result.columns
    )
    result["v_i_modelo_rural_economico_climatico"] = result["r_i_preliminar"] * economic_sum

    result["v_i_modelo_oficial"] = result["v_i_modelo_rural"]
    result["estado_modelo_oficial"] = (
        "modelo_rural_sin_demanda: demanda XM solo se reporta como factor favorable no determinante"
    )
    missing_pot = result["tipo_capa_pot"].isna()
    if "pot_mensaje" in result.columns:
        result.loc[missing_pot, "pot_mensaje"] = result.loc[missing_pot, "pot_mensaje"].fillna(
            "Sin muestreo POT para el municipio; no se puede descartar zona urbana."
        )
    result.loc[missing_pot, "estado_modelo_oficial"] = (
        "modelo_rural_sin_demanda_con_pot_pendiente: no hay capa POT municipal para descartar zona urbana"
    )
    result.loc[urban_zone, "estado_modelo_oficial"] = "excluido_zona_urbana_pot"
    result.loc[result["r_i_restriccion_pot"].eq(0), "estado_modelo_oficial"] = (
        "excluido_restriccion_territorial_pot"
    )
    result.loc[result.get("flag_revision_demanda", 0).eq(1), "estado_modelo_oficial"] = (
        result["estado_modelo_oficial"] + "; demanda_en_revision_no_usada_en_v_i"
    )
    result["notas_metodologicas"] = (
        "Score preliminar usa PVOUT puntual municipal, distancia a subestacion UPME "
        "clase de pendiente IGAC en punto interno, proporcion no protegida RUNAP y POT si existe. "
        "La demanda XM pron_areas no entra al V_i rural; se reporta como bono/contexto favorable. "
        "El score economico-climatico agrega tierra, agua y viento con imputacion de medianas cuando faltan datos."
    )

    sort_columns = [
        "v_i_modelo_rural",
        "score_preliminar_solar_red_pendiente_runap",
        "s_i_solar",
        "g_i_red",
        "p_i_pendiente_proxy",
        "u_i_uso_suelo",
        "v_i_modelo_rural_economico_climatico",
        "score_tierra",
        "score_agua",
        "score_riesgo_viento",
        "d_i_demanda",
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
        "Formula rural del proyecto:",
        "V_i = R_i(0.35*S_i + 0.30*G_i + 0.25*P_i + 0.10*U_i)",
        "",
        "Estado actual:",
        "- S_i solar: disponible con PVOUT municipal puntual.",
        "- P_i pendiente: disponible como proxy puntual IGAC, no como proporcion de area apta.",
        "- G_i red: disponible como distancia minima a subestacion UPME en servicio de nivel 4/5.",
        "- U_i uso del suelo/restriccion: combina proporcion no protegida RUNAP y POT cuando existe.",
        "- Zonas urbanas POT: si el punto municipal cae en capa urbana, R_i = 0.",
        "- D_i demanda: disponible como proxy municipal desde subareas XM pron_areas, pero no entra al V_i rural.",
        "- V_i oficial: se exporta como v_i_modelo_rural; v_i_modelo_proxy_xm queda como alias compatible.",
        "- T_i/A_i/W_i: tierra, agua y viento se integran en un score adicional economico-climatico.",
        "",
        "Score principal exportado:",
        "v_i_modelo_rural = R_preliminar * (0.35*S_i + 0.30*G_i + 0.25*P_i + 0.10*U_i)",
        "",
        "Score adicional economico-climatico:",
        "v_i_modelo_rural_economico_climatico = R_i(0.30*S_i + 0.25*G_i + 0.20*P_i + 0.10*U_i + 0.08*T_i + 0.04*A_i + 0.03*W_i)",
        "- T_i = score_tierra, A_i = score_agua, W_i = score_riesgo_viento.",
        "- Si faltan T_i, A_i o W_i, el valor del modelo se imputa con mediana departamental y luego nacional.",
        "",
        "Demanda como factor favorable no determinante:",
        f"score_rural_con_bono_demanda = v_i_modelo_rural + ({DEMAND_BONUS_WEIGHT:.2f} * D_i * R_i)",
        "",
        "Restriccion preliminar:",
        "- R_preliminar = 1 si la clase IGAC es viable o condicional.",
        "- R_preliminar = 0 si la clase IGAC es no_viable o desconocida.",
        f"- R_preliminar = 0 si distancia a subestacion > {PRELIMINARY_DISTANCE_LIMIT_KM:.0f} km.",
        "- R_preliminar = 0 si RUNAP cubre al menos 80% del municipio.",
        "- R_preliminar = 0 si POT indica capa urbana en el punto municipal.",
        "- R_preliminar = 0 si POT indica restriccion territorial estricta.",
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
        lines.append("- Estos flags no excluyen municipios del V_i rural; solo documentan incertidumbre de demanda.")
        lines.append("")
    if "tipo_capa_pot" in scored.columns:
        lines.append("Conteo por capa POT muestreada:")
        pot_counts = scored["tipo_capa_pot"].fillna("sin_pot_muestreado").value_counts(dropna=False)
        for label, count in pot_counts.items():
            lines.append(f"- {label}: {count}")
        lines.append(
            f"- Municipios excluidos por capa urbana POT: {int(scored['r_i_zona_urbana_pot'].eq(0).sum())}"
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
            "- La demanda XM usada es proxy regional/subarea; se conserva como contexto favorable, no como criterio determinante.",
            "- La exclusion urbana POT actual depende de muestreo puntual si no existe interseccion poligonal completa.",
            "- Para quitar zonas urbanas dentro de cada municipio de forma robusta se necesita area rural apta por poligono o buffer a casco urbano.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_scoring(
    pvout_path: Path = DEFAULT_PVOUT_PATH,
    slope_path: Path = DEFAULT_SLOPE_PATH,
    grid_distance_path: Path | None = DEFAULT_GRID_DISTANCE_PATH,
    runap_path: Path | None = DEFAULT_RUNAP_PATH,
    pot_path: Path | None = DEFAULT_POT_PATH,
    demand_path: Path | None = DEFAULT_DEMAND_PATH,
    costs_risks_path: Path | None = DEFAULT_COSTS_RISKS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    top_n: int = 10,
    distance_limit_km: float = PRELIMINARY_DISTANCE_LIMIT_KM,
) -> dict[str, Path]:
    """Ejecuta scoring preliminar municipal."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pvout, slope, grid_distance, runap, pot, demand, costs_risks = load_inputs(
        pvout_path,
        slope_path,
        grid_distance_path,
        runap_path,
        pot_path,
        demand_path,
        costs_risks_path,
    )
    scored = build_preliminary_score(
        pvout,
        slope,
        grid_distance,
        runap,
        pot,
        demand,
        costs_risks,
        distance_limit_km=distance_limit_km,
    )

    scored_path = output_dir / "viabilidad_municipal_preliminar.csv"
    top_path = output_dir / f"top{top_n}_modelo_rural_sin_demanda.csv"
    top_economic_climate_path = output_dir / f"top{top_n}_modelo_rural_economico_climatico.csv"
    top_bonus_path = output_dir / f"top{top_n}_sensibilidad_demanda_favorable.csv"
    observations_path = output_dir / "observaciones_viabilidad_municipal.txt"

    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    scored.head(top_n).to_csv(top_path, index=False, encoding="utf-8-sig")
    (
        scored.dropna(subset=["v_i_modelo_rural_economico_climatico"])
        .sort_values("v_i_modelo_rural_economico_climatico", ascending=False)
        .head(top_n)
        .to_csv(top_economic_climate_path, index=False, encoding="utf-8-sig")
    )
    (
        scored.dropna(subset=["score_rural_con_bono_demanda"])
        .sort_values("score_rural_con_bono_demanda", ascending=False)
        .head(top_n)
        .to_csv(top_bonus_path, index=False, encoding="utf-8-sig")
    )
    write_observations(observations_path, scored)

    return {
        "scored": scored_path,
        "top": top_path,
        "top_economico_climatico": top_economic_climate_path,
        "top_sensibilidad_demanda": top_bonus_path,
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
    parser.add_argument("--pot-path", type=Path, default=DEFAULT_POT_PATH)
    parser.add_argument("--demand-path", type=Path, default=DEFAULT_DEMAND_PATH)
    parser.add_argument("--costs-risks-path", type=Path, default=DEFAULT_COSTS_RISKS_PATH)
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
        pot_path=args.pot_path,
        demand_path=args.demand_path,
        costs_risks_path=args.costs_risks_path,
        output_dir=args.output_dir,
        top_n=args.top_n,
        distance_limit_km=args.distance_limit_km,
    )
    print("Score municipal preliminar generado.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print("Advertencia: V_i rural no usa demanda; POT urbano solo se excluye si existe capa municipal muestreada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
