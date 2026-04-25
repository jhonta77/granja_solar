from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

import streamlit_app as dashboard


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "visualizaciones_municipios" / "streamlit_dashboard"


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG guardado: {path.relative_to(PROJECT_ROOT)}")


def save_deck(deck, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        deck.to_html(str(path), open_browser=False)
    except TypeError:
        path.write_text(deck.to_html(as_string=True), encoding="utf-8")
    print(f"HTML guardado: {path.relative_to(PROJECT_ROOT)}")


def make_solar_costs_chart(solar_costs: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=dashboard.scaled_figsize(9, 4.5), dpi=dashboard.FIGURE_DPI)
    if solar_costs.empty:
        ax.text(0.5, 0.5, "Sin datos suficientes", ha="center", va="center")
        ax.axis("off")
        return fig

    ax.bar(solar_costs["scenario_name"], solar_costs["generacion_mwh_por_hectarea_anual"])
    ax.set_ylabel("MWh/ha-anio")
    ax.set_title("Generacion anual estimada por hectarea")
    fig.tight_layout(pad=0.7)
    return fig


def export_dashboard_charts() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = dashboard.load_viability()
    filtered = dashboard.filter_data(df, departments=[], hide_demand_outliers=False, only_eligible=True)
    top = dashboard.top_municipalities(filtered, 10)

    cluster_profile = dashboard.load_cluster_profile()
    cluster_evaluation = dashboard.load_cluster_evaluation()
    solar_costs = dashboard.load_solar_costs()
    energy_prices = dashboard.load_energy_prices()

    save_figure(dashboard.make_top_score_chart(top), OUTPUT_DIR / "top_10_municipios_score_rural.png")
    save_figure(dashboard.make_components_chart(top), OUTPUT_DIR / "top_10_aportes_componentes_score.png")
    save_figure(dashboard.make_component_heatmap(top), OUTPUT_DIR / "top_10_heatmap_componentes.png")
    save_figure(dashboard.make_scatter_solar_grid(filtered), OUTPUT_DIR / "scatter_pvout_vs_distancia_red.png")
    save_figure(dashboard.make_score_histogram(filtered), OUTPUT_DIR / "histograma_score_viabilidad_rural.png")
    save_figure(dashboard.make_classification_chart(df), OUTPUT_DIR / "conteo_clasificacion_preliminar.png")
    save_figure(dashboard.make_map_scatter(filtered, top), OUTPUT_DIR / "mapa_estatico_score_top_10.png")
    save_deck(dashboard.make_colombia_heatmap(filtered, top), OUTPUT_DIR / "mapa_interactivo_heatmap_viabilidad.html")

    if not cluster_evaluation.empty:
        save_figure(dashboard.make_k_evaluation_chart(cluster_evaluation), OUTPUT_DIR / "evaluacion_kmeans_codo_silhouette.png")

    if not cluster_profile.empty:
        save_figure(dashboard.make_cluster_profile_chart(cluster_profile), OUTPUT_DIR / "perfil_promedio_clusters.png")

    if not solar_costs.empty:
        save_figure(make_solar_costs_chart(solar_costs), OUTPUT_DIR / "escenarios_generacion_mwh_por_hectarea.png")

    if not solar_costs.empty and not energy_prices.empty:
        scenario_names = solar_costs["scenario_name"].astype(str).tolist()
        default_scenario_name = "base" if "base" in scenario_names else scenario_names[0]
        selected_scenario = solar_costs[
            solar_costs["scenario_name"].astype(str).eq(default_scenario_name)
        ].iloc[0]
        price_base = dashboard.merge_energy_prices(filtered, energy_prices)
        if "precio_compra_cop_kwh" in price_base.columns:
            energy_value = dashboard.build_energy_value_estimate(
                price_base,
                selected_scenario,
                hectares=1.0,
                price_column="precio_compra_cop_kwh",
            )
            if not energy_value.empty:
                save_figure(
                    dashboard.make_energy_value_chart(energy_value, 10),
                    OUTPUT_DIR / "valor_anual_energia_top_10.png",
                )
                save_figure(
                    dashboard.make_energy_value_scatter(energy_value),
                    OUTPUT_DIR / "scatter_viabilidad_vs_valor_anual.png",
                )

    manifest = pd.DataFrame(
        [
            {
                "archivo": str(path.relative_to(PROJECT_ROOT)),
                "tipo": path.suffix.lstrip("."),
                "bytes": path.stat().st_size,
            }
            for path in sorted(OUTPUT_DIR.iterdir())
            if path.is_file()
        ]
    )
    manifest.to_csv(OUTPUT_DIR / "manifest_streamlit_dashboard.csv", index=False, encoding="utf-8-sig")
    print(f"Manifest guardado: {(OUTPUT_DIR / 'manifest_streamlit_dashboard.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    export_dashboard_charts()
