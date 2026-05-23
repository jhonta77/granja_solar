"""Score de rentabilidad municipal por hectarea para granjas solares.

Este modulo no reemplaza el score territorial/multidimensional. Construye una
lectura economica separada: ingresos, costos anualizados, margen y porcentaje
de aporte de cada componente a la rentabilidad estimada.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIABILITY_PATH = (
    PROJECT_ROOT / "data" / "clean" / "viabilidad_municipal" / "viabilidad_municipal_multidimensional.csv"
)
DEFAULT_SOLAR_COSTS_PATH = (
    PROJECT_ROOT / "data" / "clean" / "solar_costs" / "solar_escenarios_por_hectarea.csv"
)
DEFAULT_INVIAS_PATH = PROJECT_ROOT / "data" / "clean" / "invias_vias" / "distancia_vias_municipios.csv"
DEFAULT_ERA5_PATH = PROJECT_ROOT / "data" / "clean" / "copernicus_era5" / "era5_resumen_municipios.csv"
DEFAULT_SUI_AGUA_PATH = PROJECT_ROOT / "data" / "clean" / "sui_agua" / "sui_costo_agua_municipal.csv"
DEFAULT_SUI_ASEO_PATH = PROJECT_ROOT / "data" / "clean" / "sui_aseo" / "sui_aseo_municipal.csv"
DEFAULT_EVA_BOVINO_PATH = (
    PROJECT_ROOT / "data" / "clean" / "upra_agropecuario" / "eva_inventario_bovino_municipal.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "rentabilidad_municipal"

DEFAULT_EXCHANGE_RATE_COP_USD = 4000.0
DEFAULT_PPA_COP_KWH = 160.0
DEFAULT_WACC = 0.08
DEFAULT_PROJECT_LIFE_YEARS = 25
DEFAULT_PROJECT_AREA_HA = 100.0
DEFAULT_LINE_COST_USD_KM = 120_000.0
DEFAULT_ROAD_LOGISTICS_COP_HA_KM_YEAR = 35_000.0
DEFAULT_OPEX_PCT_CAPEX_YEAR = 0.018
DEFAULT_WATER_M3_HA_YEAR_BASE = 60.0
DEFAULT_WATER_COST_COP_M3 = 3500.0
DEFAULT_CATTLE_OPPORTUNITY_COP_HEAD_YEAR = 180_000.0

COMPONENT_COLUMNS = [
    "ingreso_energia_cop_ha_year",
    "costo_capex_anual_cop_ha_year",
    "costo_interconexion_cop_ha_year",
    "costo_logistica_vias_cop_ha_year",
    "costo_opex_cop_ha_year",
    "costo_agua_limpieza_cop_ha_year",
    "costo_oportunidad_agro_cop_ha_year",
    "costo_riesgo_climatico_cop_ha_year",
]


def _crf(wacc: float, years: int) -> float:
    return wacc / (1 - (1 + wacc) ** -years)


def _minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    vmin = values.min(skipna=True)
    vmax = values.max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series(pd.NA, index=series.index, dtype="Float64")
    if higher_is_better:
        return ((values - vmin) / (vmax - vmin)).clip(0, 1)
    return ((vmax - values) / (vmax - vmin)).clip(0, 1)


def _load_optional(path: Path, key: str = "codigo_dane") -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path, dtype={key: "string"})
    if key in df.columns:
        df[key] = df[key].astype("string").str.zfill(5)
    return df if not df.empty else None


def _load_base_solar_assumptions(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de escenarios solares: {path}")
    df = pd.read_csv(path)
    scenario = df[df["scenario_name"].astype("string").str.lower().eq("base")]
    if scenario.empty:
        scenario = df.head(1)
    row = scenario.iloc[0]
    return {
        "capacidad_kw_por_hectarea": float(row["capacidad_kw_por_hectarea"]),
        "costo_usd_por_hectarea": float(row["costo_usd_por_hectarea"]),
        "land_use_hectares_per_mw": float(row["land_use_hectares_per_mw"]),
    }


def _merge_optional(base: pd.DataFrame, other: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    if other is None or "codigo_dane" not in other.columns:
        return base
    available = ["codigo_dane"] + [c for c in columns if c in other.columns]
    if len(available) == 1:
        return base
    return base.merge(other[available], on="codigo_dane", how="left")


def _water_need_m3_ha_year(prec_mm_year: pd.Series) -> pd.Series:
    """Proxy simple: menos lluvia implica mas lavado de paneles."""
    prec = pd.to_numeric(prec_mm_year, errors="coerce")
    water = pd.Series(DEFAULT_WATER_M3_HA_YEAR_BASE, index=prec.index, dtype="float64")
    water = water.where(~prec.ge(2500), 35.0)
    water = water.where(~prec.between(1200, 2500, inclusive="left"), 60.0)
    water = water.where(~prec.lt(1200), 100.0)
    return water


def build_profitability_score(
    viability: pd.DataFrame,
    solar_assumptions: dict[str, float],
    invias: pd.DataFrame | None = None,
    era5: pd.DataFrame | None = None,
    sui_agua: pd.DataFrame | None = None,
    sui_aseo: pd.DataFrame | None = None,
    eva_bovino: pd.DataFrame | None = None,
    ppa_cop_kwh: float = DEFAULT_PPA_COP_KWH,
    exchange_rate_cop_usd: float = DEFAULT_EXCHANGE_RATE_COP_USD,
    project_area_ha: float = DEFAULT_PROJECT_AREA_HA,
    wacc: float = DEFAULT_WACC,
    project_life_years: int = DEFAULT_PROJECT_LIFE_YEARS,
) -> pd.DataFrame:
    result = viability.copy()
    result["codigo_dane"] = result["codigo_dane"].astype("string").str.zfill(5)

    result = _merge_optional(result, invias, ["dist_via_primaria_km"])
    result = _merge_optional(result, era5, ["prec_suma_mm_year", "t2m_media_c", "ws10m_media_m_s"])
    result = _merge_optional(result, sui_agua, ["costo_agua_cop_m3"])
    result = _merge_optional(result, sui_aseo, ["costo_aseo_cop_ton"])
    result = _merge_optional(result, eva_bovino, ["inventario_bovinos", "fincas_con_bovinos"])

    crf = _crf(wacc, project_life_years)
    capacity_kw_ha = solar_assumptions["capacidad_kw_por_hectarea"]
    capex_cop_ha = solar_assumptions["costo_usd_por_hectarea"] * exchange_rate_cop_usd

    annual_yield = pd.to_numeric(result["annual_yield_kwh_kw_year"], errors="coerce")
    result["generacion_kwh_ha_year"] = annual_yield * capacity_kw_ha
    result["precio_venta_energia_cop_kwh"] = ppa_cop_kwh
    result["ingreso_energia_cop_ha_year"] = result["generacion_kwh_ha_year"] * ppa_cop_kwh

    result["costo_capex_anual_cop_ha_year"] = capex_cop_ha * crf

    dist_grid = pd.to_numeric(result.get("dist_subestacion_km"), errors="coerce")
    result["costo_interconexion_cop_ha_year"] = (
        dist_grid
        * DEFAULT_LINE_COST_USD_KM
        * exchange_rate_cop_usd
        * crf
        / project_area_ha
    )

    dist_road = pd.to_numeric(result.get("dist_via_primaria_km"), errors="coerce")
    result["costo_logistica_vias_cop_ha_year"] = dist_road * DEFAULT_ROAD_LOGISTICS_COP_HA_KM_YEAR

    result["costo_opex_cop_ha_year"] = capex_cop_ha * DEFAULT_OPEX_PCT_CAPEX_YEAR
    if "costo_aseo_cop_ton" in result.columns:
        aseo_norm = _minmax(result["costo_aseo_cop_ton"], higher_is_better=True).fillna(0.0)
        result["costo_opex_cop_ha_year"] = result["costo_opex_cop_ha_year"] * (1 + 0.10 * aseo_norm)

    prec_series = result.get("era5_prec_suma_mm_year", result.get("prec_suma_mm_year"))
    if prec_series is None:
        prec_series = result.get("prectotcorr_suma_mm_year", pd.Series(pd.NA, index=result.index))
    result["agua_limpieza_m3_ha_year"] = _water_need_m3_ha_year(prec_series)
    if "costo_agua_cop_m3" in result.columns:
        water_cost = pd.to_numeric(result["costo_agua_cop_m3"], errors="coerce").fillna(DEFAULT_WATER_COST_COP_M3)
    else:
        water_cost = pd.Series(DEFAULT_WATER_COST_COP_M3, index=result.index, dtype="float64")
    result["costo_agua_limpieza_cop_ha_year"] = result["agua_limpieza_m3_ha_year"] * water_cost

    cattle = pd.to_numeric(result.get("inventario_bovinos"), errors="coerce")
    area_km2 = pd.to_numeric(result.get("area_km2_igac"), errors="coerce")
    area_total_ha = area_km2 * 100.0
    if "area_no_protegida_km2_runap" in result.columns:
        area_base_agro_ha = pd.to_numeric(result["area_no_protegida_km2_runap"], errors="coerce") * 100.0
        area_base_agro_ha = area_base_agro_ha.where(area_base_agro_ha.gt(0), area_total_ha)
    else:
        area_base_agro_ha = area_total_ha

    pot_factor = pd.Series(1.0, index=result.index, dtype="float64")
    for col in ["u_i_pot_compatible", "u_i_uso_suelo_proxy", "u_i_uso_suelo"]:
        if col in result.columns:
            values = pd.to_numeric(result[col], errors="coerce")
            compatible_values = values.where(values.notna() & values.gt(0), 1.0)
            pot_factor = compatible_values.clip(0, 1)
            break
    if "tipo_capa_pot" in result.columns:
        is_urban = result["tipo_capa_pot"].astype("string").str.lower().eq("urbana").fillna(False)
        pot_factor = pot_factor.where(~is_urban, 0.0)

    result["area_agro_rural_proxy_ha"] = (area_base_agro_ha * pot_factor).where(
        (area_base_agro_ha * pot_factor).gt(0)
    )
    cattle_density_head_ha = cattle / result["area_agro_rural_proxy_ha"]
    result["carga_bovina_proxy_cabezas_ha_municipio"] = cattle_density_head_ha
    result["costo_oportunidad_agro_cop_ha_year"] = (
        cattle_density_head_ha.clip(lower=0, upper=5) * DEFAULT_CATTLE_OPPORTUNITY_COP_HEAD_YEAR
    )
    result["calidad_costo_oportunidad_agro"] = "proxy_bovinos_sobre_area_no_runap_por_factor_pot"
    result.loc[
        ~pd.to_numeric(result.get("u_i_pot_compatible"), errors="coerce").notna(),
        "calidad_costo_oportunidad_agro",
    ] = "proxy_bovinos_sobre_area_no_runap_sin_area_pot"
    result.loc[
        cattle.isna() | area_km2.isna() | area_km2.le(0) | result["area_agro_rural_proxy_ha"].isna(),
        "calidad_costo_oportunidad_agro",
    ] = "sin_dato"

    risk_penalty_rate = (1 - pd.to_numeric(result.get("score_riesgo"), errors="coerce")).clip(0, 1).fillna(0.0)
    result["costo_riesgo_climatico_cop_ha_year"] = result["ingreso_energia_cop_ha_year"] * 0.04 * risk_penalty_rate

    cost_cols = [c for c in COMPONENT_COLUMNS if c != "ingreso_energia_cop_ha_year"]
    result["costo_total_estimado_cop_ha_year"] = result[cost_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    result["margen_estimado_cop_ha_year"] = (
        result["ingreso_energia_cop_ha_year"] - result["costo_total_estimado_cop_ha_year"]
    )
    result["relacion_beneficio_costo_rentabilidad"] = (
        result["ingreso_energia_cop_ha_year"] / result["costo_total_estimado_cop_ha_year"].where(
            result["costo_total_estimado_cop_ha_year"] > 0
        )
    )

    result["score_margen"] = _minmax(result["margen_estimado_cop_ha_year"], higher_is_better=True)
    result["score_beneficio_costo"] = _minmax(result["relacion_beneficio_costo_rentabilidad"], higher_is_better=True)
    result["score_rentabilidad"] = (
        0.70 * pd.to_numeric(result["score_margen"], errors="coerce")
        + 0.30 * pd.to_numeric(result["score_beneficio_costo"], errors="coerce")
    ).clip(0, 1)
    result["clasificacion_rentabilidad"] = _classify(result["score_rentabilidad"])
    r_i = pd.to_numeric(result.get("r_i_preliminar", pd.Series(1.0, index=result.index)), errors="coerce").fillna(0.0)
    result["score_rentabilidad_ajustada"] = (result["score_rentabilidad"] * r_i).clip(0, 1)
    result["clasificacion_rentabilidad_ajustada"] = _classify(result["score_rentabilidad_ajustada"])

    abs_total = result[COMPONENT_COLUMNS].abs().sum(axis=1).replace(0, pd.NA)
    pct_names = {
        "ingreso_energia_cop_ha_year": "pct_driver_ingreso_energia",
        "costo_capex_anual_cop_ha_year": "pct_driver_capex",
        "costo_interconexion_cop_ha_year": "pct_driver_interconexion",
        "costo_logistica_vias_cop_ha_year": "pct_driver_logistica_vias",
        "costo_opex_cop_ha_year": "pct_driver_opex",
        "costo_agua_limpieza_cop_ha_year": "pct_driver_agua",
        "costo_oportunidad_agro_cop_ha_year": "pct_driver_oportunidad_agro",
        "costo_riesgo_climatico_cop_ha_year": "pct_driver_riesgo_climatico",
    }
    for source, target in pct_names.items():
        result[target] = (result[source].abs() / abs_total * 100.0).astype("Float64")

    result["supuesto_capacidad_kw_ha"] = capacity_kw_ha
    result["supuesto_capex_cop_ha"] = capex_cop_ha
    result["supuesto_wacc"] = wacc
    result["supuesto_vida_util_anios"] = project_life_years
    result["supuesto_area_proyecto_ha"] = project_area_ha
    result["supuesto_trm_cop_usd"] = exchange_rate_cop_usd

    result["calidad_precio_energia"] = "supuesto_ppa_cop_kwh"
    result["calidad_costo_tierra"] = "capex_escenario_base_no_precio_predial"
    result["calidad_agua"] = "sui_agua" if "costo_agua_cop_m3" in result.columns else "proxy_tarifa_default"
    if "costo_agua_cop_m3" in result.columns:
        result.loc[result["costo_agua_cop_m3"].isna(), "calidad_agua"] = "proxy_tarifa_default"

    return result.sort_values(
        ["score_rentabilidad_ajustada", "score_rentabilidad"],
        ascending=False,
        na_position="last",
    )


def _classify(score: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(score, errors="coerce")
    positive = numeric[numeric > 0].dropna()
    if positive.empty:
        return numeric.map(lambda v: "sin_datos" if pd.isna(v) else "baja")
    p50 = positive.quantile(0.50)
    p75 = positive.quantile(0.75)
    p90 = positive.quantile(0.90)

    def label(value: Any) -> str:
        if pd.isna(value):
            return "sin_datos"
        if value >= p90:
            return "muy_alta"
        if value >= p75:
            return "alta"
        if value >= p50:
            return "media"
        return "baja"

    return numeric.map(label)


def write_observations(path: Path, result: pd.DataFrame) -> None:
    lines = [
        "Score de rentabilidad municipal por hectarea",
        "============================================",
        "",
        "Formula economica base:",
        "  margen = ingreso_energia - capex_anualizado - interconexion - logistica - opex - agua - oportunidad_agro - riesgo",
        "",
        "Score exportado:",
        "  score_rentabilidad = 0.70 * score_margen + 0.30 * score_beneficio_costo",
        "  score_rentabilidad_ajustada = score_rentabilidad * r_i_preliminar",
        "",
        "Porcentajes de drivers:",
        "  pct_driver_* = abs(componente) / sum(abs(componentes)) * 100",
        "",
        f"Municipios procesados: {len(result)}",
        f"Municipios con score_rentabilidad: {result['score_rentabilidad'].notna().sum()}",
        "",
        "Supuestos principales:",
        f"  PPA energia: {DEFAULT_PPA_COP_KWH:,.0f} COP/kWh",
        f"  TRM: {DEFAULT_EXCHANGE_RATE_COP_USD:,.0f} COP/USD",
        f"  WACC: {DEFAULT_WACC:.2%}",
        f"  Vida util: {DEFAULT_PROJECT_LIFE_YEARS} anios",
        f"  Area proyecto para prorratear interconexion: {DEFAULT_PROJECT_AREA_HA:,.0f} ha",
        f"  Linea electrica: {DEFAULT_LINE_COST_USD_KM:,.0f} USD/km",
        "",
        "Limitaciones:",
        "  - Es un modelo preliminar por hectarea, no un VPN/TIR/LCOE bancable.",
        "  - El precio de energia es un supuesto PPA uniforme.",
        "  - La interconexion usa distancia euclidiana a subestacion y se prorratea por area de proyecto.",
        "  - El costo de oportunidad agropecuario usa inventario bovino / area municipal como proxy.",
        "  - Falta integrar precio rural UPRA y tarifa real SUI acueducto cuando esten disponibles.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_profitability(
    viability_path: Path = DEFAULT_VIABILITY_PATH,
    solar_costs_path: Path = DEFAULT_SOLAR_COSTS_PATH,
    invias_path: Path = DEFAULT_INVIAS_PATH,
    era5_path: Path = DEFAULT_ERA5_PATH,
    sui_agua_path: Path = DEFAULT_SUI_AGUA_PATH,
    sui_aseo_path: Path = DEFAULT_SUI_ASEO_PATH,
    eva_bovino_path: Path = DEFAULT_EVA_BOVINO_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    top_n: int = 20,
    ppa_cop_kwh: float = DEFAULT_PPA_COP_KWH,
    exchange_rate_cop_usd: float = DEFAULT_EXCHANGE_RATE_COP_USD,
    project_area_ha: float = DEFAULT_PROJECT_AREA_HA,
) -> dict[str, Path]:
    if not viability_path.exists():
        raise FileNotFoundError(f"No existe la tabla de viabilidad: {viability_path}")

    viability = pd.read_csv(viability_path, dtype={"codigo_dane": "string"})
    assumptions = _load_base_solar_assumptions(solar_costs_path)
    result = build_profitability_score(
        viability=viability,
        solar_assumptions=assumptions,
        invias=_load_optional(invias_path),
        era5=_load_optional(era5_path),
        sui_agua=_load_optional(sui_agua_path),
        sui_aseo=_load_optional(sui_aseo_path),
        eva_bovino=_load_optional(eva_bovino_path),
        ppa_cop_kwh=ppa_cop_kwh,
        exchange_rate_cop_usd=exchange_rate_cop_usd,
        project_area_ha=project_area_ha,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rentabilidad_municipal.csv"
    top_path = output_dir / f"top{top_n}_rentabilidad.csv"
    observations_path = output_dir / "rentabilidad_observaciones.txt"

    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    result.head(top_n).to_csv(top_path, index=False, encoding="utf-8-sig")
    write_observations(observations_path, result)

    return {
        "rentabilidad": output_path,
        "top_rentabilidad": top_path,
        "observaciones": observations_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calcula score de rentabilidad municipal por hectarea.")
    parser.add_argument("--viability-path", type=Path, default=DEFAULT_VIABILITY_PATH)
    parser.add_argument("--solar-costs-path", type=Path, default=DEFAULT_SOLAR_COSTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--ppa-cop-kwh", type=float, default=DEFAULT_PPA_COP_KWH)
    parser.add_argument("--exchange-rate-cop-usd", type=float, default=DEFAULT_EXCHANGE_RATE_COP_USD)
    parser.add_argument("--project-area-ha", type=float, default=DEFAULT_PROJECT_AREA_HA)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_profitability(
        viability_path=args.viability_path,
        solar_costs_path=args.solar_costs_path,
        output_dir=args.output_dir,
        top_n=args.top_n,
        ppa_cop_kwh=args.ppa_cop_kwh,
        exchange_rate_cop_usd=args.exchange_rate_cop_usd,
        project_area_ha=args.project_area_ha,
    )
    print("Score de rentabilidad municipal generado.")
    for label, path in outputs.items():
        print(f"- {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
