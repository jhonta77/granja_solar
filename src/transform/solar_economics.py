from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ACRES_TO_HECTARES = 0.40468564224

IRENA_2024_URL = (
    "https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2025/Jul/"
    "IRENA_TEC_RPGC_in_2024_Summary_2025.pdf"
)
NREL_LAND_USE_URL = "https://docs.nrel.gov/docs/fy13osti/56290.pdf"
WORLD_BANK_PVOUT_URL = (
    "https://datacatalog.worldbank.org/search/dataset/0038379/"
    "global-photovoltaic-power-potential-by-country"
)
WORLD_BANK_PVOUT_XLSX_URL = (
    "https://datacatalogfiles.worldbank.org/ddh-published/0038379/1/DR0046831/"
    "solargis_pvpotential_countryranking_2020_data.xlsx"
)
WORLD_BANK_COLOMBIA_MAP_URL = (
    "https://datacatalog.worldbank.org/search/dataset/0040261/"
    "colombia-solar-irradiation-and-pv-power-potential-map"
)
UPME_ATLAS_URL = (
    "https://www1.upme.gov.co/Hemeroteca/Impresos/Atlas_Radiacion_Solar_2005/"
    "1-Atlas_Radiacion_Solar.pdf"
)


REQUIRED_ASSUMPTION_COLUMNS = {
    "scenario_name",
    "capex_usd_per_kw",
    "land_use_hectares_per_mw",
    "annual_yield_kwh_per_kw_year",
}


def project_root_from_file() -> Path:
    """Resuelve la raiz del proyecto desde este archivo."""

    return Path(__file__).resolve().parents[2]


def hectares_from_acres(acres_per_mw: float) -> float:
    """Convierte acres/MW a hectareas/MW."""

    return acres_per_mw * ACRES_TO_HECTARES


def annual_yield_from_daily_pvout(pvout_kwh_per_kwp_day: float) -> float:
    """Convierte PVOUT diario en kWh/kWp-dia a kWh/kW-anio."""

    return pvout_kwh_per_kwp_day * 365.0


def annual_yield_from_ghi_with_country_ratio(
    regional_ghi_kwh_m2_year: float,
    country_pvout_kwh_kwp_day: float = 4.0492,
    country_ghi_kwh_m2_day: float = 4.867,
) -> float:
    """
    Aproxima PVOUT regional desde GHI regional.

    La razon PVOUT/GHI sale de World Bank/ESMAP para Colombia:
    PVOUT promedio pais = 4.0492 kWh/kWp-dia y GHI promedio pais =
    4.867 kWh/m2-dia. Se usa solo como adaptacion preliminar cuando la
    fuente nacional reporta irradiacion y no energia AC/DC especifica.
    """

    conversion_ratio = country_pvout_kwh_kwp_day / country_ghi_kwh_m2_day
    return regional_ghi_kwh_m2_year * conversion_ratio


def default_assumptions() -> pd.DataFrame:
    """Construye escenarios trazables con fuentes tecnicas reconocidas."""

    base_yield = annual_yield_from_daily_pvout(4.0492)
    conservative_yield = annual_yield_from_ghi_with_country_ratio(1278.0)
    optimistic_yield = annual_yield_from_ghi_with_country_ratio(2190.0)

    rows = [
        {
            "scenario_name": "conservador",
            "capex_usd_per_kw": 691.0,
            "land_use_hectares_per_mw": hectares_from_acres(8.3),
            "annual_yield_kwh_per_kw_year": conservative_yield,
            "exchange_rate_cop_per_usd": pd.NA,
            "source_capex": "IRENA Renewable Power Generation Costs in 2024, TIC solar PV 2024.",
            "source_land_use": "NREL land-use report, large PV 1-axis total area: 8.3 acres/MWac.",
            "source_yield": (
                "UPME/IDEAM Atlas, Costa Pacifica 1,278 kWh/m2-year; "
                "adaptado con razon PVOUT/GHI Colombia de World Bank/ESMAP."
            ),
            "source_capex_url": IRENA_2024_URL,
            "source_land_use_url": NREL_LAND_USE_URL,
            "source_yield_url": UPME_ATLAS_URL,
            "dato_fuente_original": (
                "CAPEX: 691 USD/kW; uso suelo: 8.3 acres/MWac; "
                "GHI regional: 1,278 kWh/m2-year; PVOUT/GHI Colombia: 4.0492/4.867."
            ),
            "supuesto_adoptado": (
                "Escenario de menor productividad: mayor uso de suelo, CAPEX observado "
                "global reciente y rendimiento derivado de una region colombiana de menor radiacion."
            ),
            "formula_yield": (
                "annual_yield = GHI_regional_anual * "
                "(PVOUT_promedio_Colombia_diario / GHI_promedio_Colombia_diario)"
            ),
            "notas_metodologicas": (
                "No representa diseno de ingenieria; usa total area y una adaptacion "
                "GHI->PVOUT para mantener trazabilidad con fuentes publicas."
            ),
        },
        {
            "scenario_name": "base",
            "capex_usd_per_kw": 599.0,
            "land_use_hectares_per_mw": hectares_from_acres(7.9),
            "annual_yield_kwh_per_kw_year": base_yield,
            "exchange_rate_cop_per_usd": pd.NA,
            "source_capex": "IRENA Renewable Power Generation Costs in 2024, proyeccion TIC solar PV cercana a 2025.",
            "source_land_use": "NREL land-use report, large PV total area average: 7.9 acres/MWac.",
            "source_yield": "World Bank/ESMAP Global Solar Atlas, Colombia PVOUT Level 1 promedio: 4.0492 kWh/kWp-dia.",
            "source_capex_url": IRENA_2024_URL,
            "source_land_use_url": NREL_LAND_USE_URL,
            "source_yield_url": WORLD_BANK_PVOUT_URL,
            "dato_fuente_original": (
                "CAPEX: 599 USD/kW en proyeccion IRENA; uso suelo: 7.9 acres/MWac; "
                "PVOUT Colombia: 4.0492 kWh/kWp-day."
            ),
            "supuesto_adoptado": (
                "Escenario medio: costo proyectado de corto plazo, uso de suelo promedio "
                "NREL para grandes PV y PVOUT promedio Colombia."
            ),
            "formula_yield": "annual_yield = PVOUT_diario_Colombia * 365",
            "notas_metodologicas": (
                "PVOUT es promedio pais; para scoring final debe reemplazarse por PVOUT "
                "espacial de cada celda o zona candidata."
            ),
        },
        {
            "scenario_name": "optimista",
            "capex_usd_per_kw": 534.0,
            "land_use_hectares_per_mw": hectares_from_acres(7.5),
            "annual_yield_kwh_per_kw_year": optimistic_yield,
            "exchange_rate_cop_per_usd": pd.NA,
            "source_capex": "IRENA Renewable Power Generation Costs in 2024, proyeccion TIC solar PV cercana a 2026.",
            "source_land_use": "NREL land-use report, large PV fixed total area: 7.5 acres/MWac.",
            "source_yield": (
                "UPME/IDEAM Atlas, Guajira 2,190 kWh/m2-year; "
                "adaptado con razon PVOUT/GHI Colombia de World Bank/ESMAP."
            ),
            "source_capex_url": IRENA_2024_URL,
            "source_land_use_url": NREL_LAND_USE_URL,
            "source_yield_url": UPME_ATLAS_URL,
            "dato_fuente_original": (
                "CAPEX: 534 USD/kW en proyeccion IRENA; uso suelo: 7.5 acres/MWac; "
                "GHI Guajira: 2,190 kWh/m2-year; PVOUT/GHI Colombia: 4.0492/4.867."
            ),
            "supuesto_adoptado": (
                "Escenario de mayor productividad: menor uso de suelo, menor CAPEX "
                "proyectado y rendimiento derivado de region colombiana de alta radiacion."
            ),
            "formula_yield": (
                "annual_yield = GHI_regional_anual * "
                "(PVOUT_promedio_Colombia_diario / GHI_promedio_Colombia_diario)"
            ),
            "notas_metodologicas": (
                "El resultado depende fuertemente de localizar realmente el proyecto en "
                "zonas de alta radiacion y con restricciones de suelo compatibles."
            ),
        },
    ]

    return pd.DataFrame(rows)


def validate_assumptions(df: pd.DataFrame) -> None:
    """Valida columnas y valores minimos para calcular escenarios."""

    missing = REQUIRED_ASSUMPTION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "El archivo de supuestos no contiene las columnas requeridas: "
            + ", ".join(sorted(missing))
        )

    for column in [
        "capex_usd_per_kw",
        "land_use_hectares_per_mw",
        "annual_yield_kwh_per_kw_year",
    ]:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"La columna '{column}' contiene valores no numericos.")
        if (values <= 0).any():
            raise ValueError(f"La columna '{column}' debe tener valores positivos.")


def load_assumptions(assumptions_path: Path | None = None) -> pd.DataFrame:
    """Carga supuestos manuales o usa escenarios por defecto documentados."""

    if assumptions_path is None:
        assumptions = default_assumptions()
    else:
        if not assumptions_path.exists():
            raise FileNotFoundError(f"No existe el archivo de supuestos: {assumptions_path}")
        assumptions = pd.read_csv(assumptions_path)

    validate_assumptions(assumptions)
    return assumptions


def fetch_worldbank_pvout_country(iso3: str = "COL") -> pd.DataFrame:
    """
    Descarga la tabla oficial World Bank/ESMAP/Solargis y extrae un pais.

    No hace scraping HTML: lee el archivo XLSX publicado en el Data Catalog.
    """

    raw = pd.read_excel(WORLD_BANK_PVOUT_XLSX_URL, header=1)
    raw.columns = [str(column).strip() for column in raw.columns]
    country = raw[raw.iloc[:, 0].astype(str).str.upper() == iso3.upper()].copy()
    if country.empty:
        raise ValueError(f"No se encontro el pais ISO_A3={iso3} en la tabla World Bank.")

    selected = pd.DataFrame(
        {
            "iso_a3": country.iloc[:, 0].astype(str).values,
            "country_or_region": country.iloc[:, 1].astype(str).values,
            "ghi_kwh_m2_day": pd.to_numeric(country.iloc[:, 10], errors="coerce").values,
            "pvout_kwh_kwp_day": pd.to_numeric(country.iloc[:, 11], errors="coerce").values,
            "lcoe_usd_kwh_2018": pd.to_numeric(country.iloc[:, 12], errors="coerce").values,
            "pv_seasonality_index": pd.to_numeric(country.iloc[:, 13], errors="coerce").values,
            "pv_equivalent_area_pct_total_area": pd.to_numeric(
                country.iloc[:, 14], errors="coerce"
            ).values,
        }
    )
    selected["annual_yield_kwh_per_kw_year"] = selected["pvout_kwh_kwp_day"] * 365.0
    selected["source"] = (
        "World Bank Group, ESMAP, Solargis (2020). "
        "Global Photovoltaic Power Potential by Country."
    )
    selected["source_url"] = WORLD_BANK_PVOUT_URL
    selected["download_url"] = WORLD_BANK_PVOUT_XLSX_URL
    selected["nota"] = (
        "PVOUT pais promedio; no reemplaza PVOUT geoespacial por zona candidata."
    )
    return selected


def calculate_metrics_per_hectare(
    assumptions: pd.DataFrame,
    exchange_rate_cop_per_usd: float | None = None,
) -> pd.DataFrame:
    """Calcula las metricas economicas y productivas por hectarea."""

    results = assumptions.copy()

    for column in [
        "capex_usd_per_kw",
        "land_use_hectares_per_mw",
        "annual_yield_kwh_per_kw_year",
    ]:
        results[column] = pd.to_numeric(results[column], errors="raise")

    if exchange_rate_cop_per_usd is not None:
        results["exchange_rate_cop_per_usd"] = exchange_rate_cop_per_usd
    elif "exchange_rate_cop_per_usd" in results.columns:
        results["exchange_rate_cop_per_usd"] = pd.to_numeric(
            results["exchange_rate_cop_per_usd"], errors="coerce"
        )
    else:
        results["exchange_rate_cop_per_usd"] = pd.NA

    results["capacidad_kw_por_hectarea"] = (
        1000.0 / results["land_use_hectares_per_mw"]
    )
    results["costo_usd_por_hectarea"] = (
        results["capacidad_kw_por_hectarea"] * results["capex_usd_per_kw"]
    )
    results["generacion_kwh_por_hectarea_anual"] = (
        results["capacidad_kw_por_hectarea"]
        * results["annual_yield_kwh_per_kw_year"]
    )
    results["generacion_mwh_por_hectarea_anual"] = (
        results["generacion_kwh_por_hectarea_anual"] / 1000.0
    )
    results["costo_usd_por_mwh_anual_simple"] = (
        results["costo_usd_por_hectarea"]
        / results["generacion_mwh_por_hectarea_anual"]
    )
    results["costo_cop_por_hectarea"] = (
        results["costo_usd_por_hectarea"] * results["exchange_rate_cop_per_usd"]
    )

    ordered_columns = [
        "scenario_name",
        "source_capex",
        "source_land_use",
        "source_yield",
        "capex_usd_per_kw",
        "land_use_hectares_per_mw",
        "annual_yield_kwh_per_kw_year",
        "capacidad_kw_por_hectarea",
        "costo_usd_por_hectarea",
        "generacion_kwh_por_hectarea_anual",
        "generacion_mwh_por_hectarea_anual",
        "costo_usd_por_mwh_anual_simple",
        "exchange_rate_cop_per_usd",
        "costo_cop_por_hectarea",
        "notas_metodologicas",
    ]
    extra_columns = [column for column in results.columns if column not in ordered_columns]
    return results[ordered_columns + extra_columns]


def build_source_table(assumptions: pd.DataFrame) -> pd.DataFrame:
    """Genera una tabla de trazabilidad fuente-supuesto-formula."""

    source_columns = [
        "scenario_name",
        "capex_usd_per_kw",
        "land_use_hectares_per_mw",
        "annual_yield_kwh_per_kw_year",
        "exchange_rate_cop_per_usd",
        "source_capex",
        "source_land_use",
        "source_yield",
        "source_capex_url",
        "source_land_use_url",
        "source_yield_url",
        "dato_fuente_original",
        "supuesto_adoptado",
        "formula_yield",
        "notas_metodologicas",
    ]
    available_columns = [column for column in source_columns if column in assumptions.columns]
    return assumptions[available_columns].copy()


def build_observation_lines(results: pd.DataFrame) -> list[str]:
    """Redacta observaciones metodologicas para informe academico."""

    lines: list[str] = []
    lines.append("Estimacion economica y productiva solar por hectarea")
    lines.append("")
    lines.append("Fuentes usadas:")
    lines.append(
        "- IRENA: total installed cost de solar PV utility-scale y proyecciones de corto plazo."
    )
    lines.append(
        "- NREL: requerimientos de uso de suelo para plantas solares utility-scale en acres/MWac."
    )
    lines.append(
        "- World Bank/ESMAP/Global Solar Atlas: PVOUT promedio de Colombia y mapa de potencial solar."
    )
    lines.append(
        "- UPME/IDEAM Atlas de Radiacion Solar: irradiacion solar regional de Colombia."
    )
    lines.append("")
    lines.append("Valores adoptados por escenario:")

    for row in results.to_dict("records"):
        lines.append(
            "- {scenario}: CAPEX={capex:.2f} USD/kW; suelo={land:.4f} ha/MW; "
            "yield={yield_value:.2f} kWh/kW-year; capacidad={capacity:.2f} kW/ha; "
            "generacion={generation:.2f} MWh/ha-year; costo simple={cost:.2f} USD/MWh anual.".format(
                scenario=row["scenario_name"],
                capex=row["capex_usd_per_kw"],
                land=row["land_use_hectares_per_mw"],
                yield_value=row["annual_yield_kwh_per_kw_year"],
                capacity=row["capacidad_kw_por_hectarea"],
                generation=row["generacion_mwh_por_hectarea_anual"],
                cost=row["costo_usd_por_mwh_anual_simple"],
            )
        )

    lines.append("")
    lines.append("Formulas implementadas:")
    lines.append("- capacidad_kw_por_hectarea = 1000 / land_use_hectares_per_mw")
    lines.append("- costo_usd_por_hectarea = capacidad_kw_por_hectarea * capex_usd_per_kw")
    lines.append(
        "- generacion_kwh_por_hectarea_anual = capacidad_kw_por_hectarea * annual_yield_kwh_per_kw_year"
    )
    lines.append("- generacion_mwh_por_hectarea_anual = generacion_kwh_por_hectarea_anual / 1000")
    lines.append(
        "- costo_usd_por_mwh_anual_simple = costo_usd_por_hectarea / generacion_mwh_por_hectarea_anual"
    )
    lines.append("- costo_cop_por_hectarea = costo_usd_por_hectarea * exchange_rate_cop_per_usd")
    lines.append("")
    lines.append("Advertencias y limitaciones:")
    lines.append(
        "- El costo_usd_por_mwh_anual_simple no es LCOE: no incluye vida util, O&M, degradacion, impuestos, WACC, conexion ni reposiciones."
    )
    lines.append(
        "- Los valores de NREL son de proyectos de Estados Unidos y se usan como referencia tecnica por ausencia de un valor oficial colombiano por hectarea."
    )
    lines.append(
        "- Los rendimientos regionales derivados desde GHI son aproximaciones; para integracion final se recomienda usar PVOUT geoespacial por zona candidata."
    )
    lines.append(
        "- Esta estimacion preliminar no reemplaza estudios de ingenieria de detalle, interconexion, topografia, suelos, permisos ni diseno financiero."
    )
    return lines


def export_results(
    assumptions: pd.DataFrame,
    results: pd.DataFrame,
    output_dir: Path,
    make_figure: bool = False,
) -> dict[str, Path]:
    """Exporta supuestos, resultados, observaciones y figura opcional."""

    output_dir.mkdir(parents=True, exist_ok=True)

    assumptions_path = output_dir / "solar_supuestos_fuente.csv"
    results_path = output_dir / "solar_escenarios_por_hectarea.csv"
    observations_path = output_dir / "solar_observaciones.txt"

    build_source_table(assumptions).to_csv(
        assumptions_path, index=False, encoding="utf-8-sig"
    )
    results.to_csv(results_path, index=False, encoding="utf-8-sig")

    observation_lines = build_observation_lines(results)
    observations_path.write_text("\n".join(observation_lines), encoding="utf-8")

    exported = {
        "assumptions": assumptions_path,
        "results": results_path,
        "observations": observations_path,
    }

    if make_figure:
        figure_path = output_dir / "solar_escenarios_comparacion.png"
        create_scenario_figure(results, figure_path)
        exported["figure"] = figure_path
    else:
        stale_figure = output_dir / "solar_escenarios_comparacion.png"
        if stale_figure.exists():
            stale_figure.unlink()

    return exported


def export_worldbank_reference(output_dir: Path, iso3: str = "COL") -> Path:
    """Exporta la referencia PVOUT pais desde World Bank/ESMAP/Solargis."""

    output_dir.mkdir(parents=True, exist_ok=True)
    reference = fetch_worldbank_pvout_country(iso3=iso3)
    output_path = output_dir / f"worldbank_esmap_solargis_{iso3.lower()}_pvout.csv"
    reference.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def create_scenario_figure(results: pd.DataFrame, output_path: Path) -> None:
    """Crea una figura comparativa de escenarios."""

    scenario_names = results["scenario_name"].astype(str)
    metrics = [
        ("costo_usd_por_hectarea", "USD/ha"),
        ("generacion_mwh_por_hectarea_anual", "MWh/ha-year"),
        ("costo_usd_por_mwh_anual_simple", "USD/MWh anual simple"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, (column, label) in zip(axes, metrics):
        axis.bar(scenario_names, results[column])
        axis.set_title(label)
        axis.set_xlabel("Escenario")
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_console_summary(results: pd.DataFrame, exported: dict[str, Path]) -> None:
    """Imprime un resumen claro al finalizar."""

    print("Calculo solar por hectarea finalizado.")
    print("")
    print("Fuentes usadas:")
    print("- IRENA: CAPEX utility-scale solar PV.")
    print("- NREL: uso de suelo utility-scale PV.")
    print("- World Bank/ESMAP/Global Solar Atlas: PVOUT Colombia.")
    print("- UPME/IDEAM: irradiacion regional Colombia.")
    print("")
    print("Escenarios calculados:")
    for row in results.to_dict("records"):
        print(
            "- {scenario}: {capacity:.2f} kW/ha, {generation:.2f} MWh/ha-year, "
            "{cost_ha:.2f} USD/ha, {cost_mwh:.2f} USD/MWh anual simple.".format(
                scenario=row["scenario_name"],
                capacity=row["capacidad_kw_por_hectarea"],
                generation=row["generacion_mwh_por_hectarea_anual"],
                cost_ha=row["costo_usd_por_hectarea"],
                cost_mwh=row["costo_usd_por_mwh_anual_simple"],
            )
        )

    print("")
    print("Salidas exportadas:")
    for label, path in exported.items():
        print(f"- {label}: {path}")

    print("")
    print(
        "Advertencia: estimacion preliminar para EDA/scoring. No reemplaza ingenieria "
        "de detalle ni un calculo LCOE financiero completo."
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Configura argumentos de ejecucion."""

    parser = argparse.ArgumentParser(
        description="Calcula metricas economicas y productivas solares por hectarea."
    )
    parser.add_argument(
        "--assumptions",
        type=str,
        default=None,
        help="CSV opcional con supuestos manuales por escenario.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Carpeta de salida. Por defecto: data/clean/solar_costs.",
    )
    parser.add_argument(
        "--exchange-rate",
        type=float,
        default=None,
        help="Tasa COP/USD opcional para calcular costo_cop_por_hectarea.",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Compatibilidad: no genera la figura PNG comparativa.",
    )
    parser.add_argument(
        "--make-figure",
        action="store_true",
        help="Genera PNG comparativo. Por defecto no se crean diagramas.",
    )
    parser.add_argument(
        "--refresh-worldbank-pvout",
        action="store_true",
        help="Descarga y exporta referencia Colombia PVOUT de World Bank/ESMAP/Solargis.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada CLI."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    project_root = project_root_from_file()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else project_root / "data" / "clean" / "solar_costs"
    )
    assumptions_path = Path(args.assumptions).resolve() if args.assumptions else None

    try:
        assumptions = load_assumptions(assumptions_path)
        results = calculate_metrics_per_hectare(
            assumptions, exchange_rate_cop_per_usd=args.exchange_rate
        )
        exported = export_results(
            assumptions,
            results,
            output_dir,
            make_figure=args.make_figure and not args.no_figure,
        )
        if args.refresh_worldbank_pvout:
            exported["worldbank_pvout"] = export_worldbank_reference(output_dir, iso3="COL")
    except Exception as error:
        print(f"ERROR: {error}")
        return 1

    print_console_summary(results, exported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
