from __future__ import annotations

from pathlib import Path
from textwrap import shorten

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
VIABILITY_PATH = PROJECT_ROOT / "data" / "clean" / "viabilidad_municipal" / "viabilidad_municipal_preliminar.csv"
CLUSTERS_PATH = PROJECT_ROOT / "data" / "clean" / "clusters_municipios" / "municipios_clusters_kmeans.csv"
CLUSTER_PROFILE_PATH = PROJECT_ROOT / "data" / "clean" / "clusters_municipios" / "perfil_clusters_kmeans.csv"
SOLAR_COSTS_PATH = PROJECT_ROOT / "data" / "clean" / "solar_costs" / "solar_escenarios_por_hectarea.csv"

SCORE_COL = "v_i_modelo_proxy_xm"
WEIGHTS = {
    "s_i_solar": 0.30,
    "d_i_demanda": 0.15,
    "g_i_red": 0.25,
    "p_i_pendiente_proxy": 0.20,
    "u_i_uso_suelo": 0.10,
}
COMPONENT_LABELS = {
    "s_i_solar": "Solar",
    "d_i_demanda": "Demanda",
    "g_i_red": "Red",
    "p_i_pendiente_proxy": "Pendiente",
    "u_i_uso_suelo": "Uso / restriccion",
}
DISPLAY_COLUMNS = [
    "ranking",
    "municipio",
    "departamento",
    SCORE_COL,
    "clasificacion_preliminar",
    "pvout_kwh_kwp_day",
    "annual_yield_kwh_kw_year",
    "dist_subestacion_km",
    "pendiente_igac",
    "pct_area_protegida_runap",
    "zona_xm_demanda",
    "tipo_mapeo_demanda",
    "cluster_kmeans_label",
    "por_que_aparece",
]


st.set_page_config(
    page_title="Granja solar Colombia",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_viability() -> pd.DataFrame:
    if not VIABILITY_PATH.exists():
        raise FileNotFoundError(f"No existe {VIABILITY_PATH}")

    df = pd.read_csv(VIABILITY_PATH, dtype={"codigo_dane": "string"})
    df["codigo_dane"] = df["codigo_dane"].astype("string").str.zfill(5)

    if CLUSTERS_PATH.exists():
        clusters = pd.read_csv(CLUSTERS_PATH, dtype={"codigo_dane": "string"})
        clusters["codigo_dane"] = clusters["codigo_dane"].astype("string").str.zfill(5)
        cluster_cols = [
            column
            for column in ["codigo_dane", "cluster_kmeans", "cluster_kmeans_label"]
            if column in clusters.columns
        ]
        if "cluster_kmeans" in cluster_cols and "cluster_kmeans" not in df.columns:
            df = df.merge(clusters[cluster_cols], on="codigo_dane", how="left")

    numeric_cols = [
        SCORE_COL,
        "v_i_modelo_oficial",
        "score_preliminar_solar_red_pendiente_runap",
        "pvout_kwh_kwp_day",
        "annual_yield_kwh_kw_year",
        "dist_subestacion_km",
        "pct_area_protegida_runap",
        "area_km2_igac",
        "r_i_preliminar",
        "flag_atipico_eda_demanda",
        "flag_revision_demanda",
        "demanda_xm_proxy_mwh_o_unidad_fuente",
        *WEIGHTS.keys(),
    ]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column, weight in WEIGHTS.items():
        if column in df.columns:
            df[f"aporte_{column}"] = df[column] * weight

    df["municipio_departamento"] = df["municipio"].astype(str) + " - " + df["departamento"].astype(str)
    df["por_que_aparece"] = df.apply(build_reason, axis=1)
    return df


@st.cache_data(show_spinner=False)
def load_cluster_profile() -> pd.DataFrame:
    if not CLUSTER_PROFILE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CLUSTER_PROFILE_PATH)


@st.cache_data(show_spinner=False)
def load_solar_costs() -> pd.DataFrame:
    if not SOLAR_COSTS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SOLAR_COSTS_PATH)


def build_reason(row: pd.Series) -> str:
    reasons: list[str] = []

    if row.get("s_i_solar", 0) >= 0.75:
        reasons.append("alto recurso solar")
    elif row.get("s_i_solar", 0) >= 0.60:
        reasons.append("recurso solar competitivo")

    if row.get("g_i_red", 0) >= 0.95:
        distance = row.get("dist_subestacion_km")
        if pd.notna(distance):
            reasons.append(f"muy cerca de subestacion ({distance:.1f} km)")
        else:
            reasons.append("muy cerca de red")

    if row.get("p_i_pendiente_proxy", 0) >= 1:
        reasons.append("pendiente viable")
    elif row.get("p_i_pendiente_proxy", 0) >= 0.5:
        reasons.append("pendiente condicional")

    if row.get("u_i_uso_suelo", 0) >= 0.90:
        reasons.append("baja restriccion RUNAP")

    if row.get("d_i_demanda", 0) >= 0.50:
        reasons.append("demanda proxy alta")

    if row.get("flag_atipico_eda_demanda", 0) == 1:
        reasons.append("demanda marcada para revision")

    return "; ".join(reasons) if reasons else "score explicado por combinacion de criterios"


def filter_data(
    df: pd.DataFrame,
    departments: list[str],
    include_demand_outliers: bool,
    only_eligible: bool,
) -> pd.DataFrame:
    filtered = df.copy()
    if departments:
        filtered = filtered[filtered["departamento"].isin(departments)]
    if not include_demand_outliers and "flag_atipico_eda_demanda" in filtered.columns:
        filtered = filtered[filtered["flag_atipico_eda_demanda"].fillna(0).ne(1)]
    if only_eligible and "r_i_preliminar" in filtered.columns:
        filtered = filtered[filtered["r_i_preliminar"].fillna(0).eq(1)]
    return filtered


def top_municipalities(df: pd.DataFrame, n: int) -> pd.DataFrame:
    top = df.dropna(subset=[SCORE_COL]).sort_values(SCORE_COL, ascending=False).head(n).copy()
    top.insert(0, "ranking", range(1, len(top) + 1))
    return top


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in DISPLAY_COLUMNS if column in df.columns]
    table = df[available].copy()
    rename = {
        SCORE_COL: "score_viabilidad",
        "pvout_kwh_kwp_day": "pvout_kwh_kwp_dia",
        "annual_yield_kwh_kw_year": "kwh_kw_anio",
        "dist_subestacion_km": "dist_red_km",
        "pct_area_protegida_runap": "pct_runap",
    }
    table = table.rename(columns=rename)
    return table


def make_top_score_chart(top: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(top))))
    data = top.sort_values(SCORE_COL, ascending=True)
    labels = data["municipio"].astype(str) + " (" + data["departamento"].astype(str) + ")"
    ax.barh(labels, data[SCORE_COL])
    ax.set_xlabel("Score V_i")
    ax.set_title("Top municipios por score de viabilidad")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


def make_components_chart(top: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, max(4, 0.48 * len(top))))
    data = top.sort_values(SCORE_COL, ascending=True).copy()
    y_labels = data["municipio"].astype(str) + " (" + data["departamento"].astype(str) + ")"
    left = pd.Series(0.0, index=data.index)
    for column, label in COMPONENT_LABELS.items():
        contribution_col = f"aporte_{column}"
        if contribution_col not in data.columns:
            continue
        values = data[contribution_col].fillna(0)
        ax.barh(y_labels, values, left=left, label=label)
        left = left + values
    ax.set_xlabel("Aporte ponderado al score")
    ax.set_title("Por que aparecen: aportes ponderados por criterio")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def make_component_heatmap(top: pd.DataFrame) -> plt.Figure:
    component_cols = [column for column in WEIGHTS if column in top.columns]
    matrix = top.set_index("municipio")[component_cols].fillna(0)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(matrix))))
    image = ax.imshow(matrix.values, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(component_cols)))
    ax.set_xticklabels([COMPONENT_LABELS.get(column, column) for column in component_cols], rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Componentes normalizados del top")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04, label="Score 0-1")
    fig.tight_layout()
    return fig


def make_scatter_solar_grid(df: pd.DataFrame) -> plt.Figure:
    plot = df.dropna(subset=["pvout_kwh_kwp_day", "dist_subestacion_km", SCORE_COL]).copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        plot["pvout_kwh_kwp_day"],
        plot["dist_subestacion_km"],
        c=plot[SCORE_COL],
        alpha=0.75,
    )
    ax.set_xlabel("PVOUT kWh/kWp/dia")
    ax.set_ylabel("Distancia a subestacion km")
    ax.set_title("Relacion solar vs cercania a red")
    ax.invert_yaxis()
    fig.colorbar(scatter, ax=ax, label="V_i")
    fig.tight_layout()
    return fig


def make_score_histogram(df: pd.DataFrame) -> plt.Figure:
    scores = df[SCORE_COL].dropna()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(scores, bins=25)
    ax.set_xlabel("Score V_i")
    ax.set_ylabel("Municipios")
    ax.set_title("Distribucion del score de viabilidad")
    fig.tight_layout()
    return fig


def make_classification_chart(df: pd.DataFrame) -> plt.Figure:
    counts = df["clasificacion_preliminar"].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_ylabel("Municipios")
    ax.set_title("Conteo por clasificacion preliminar")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def make_map_scatter(df: pd.DataFrame, top: pd.DataFrame) -> plt.Figure:
    plot = df.dropna(subset=["lon", "lat", SCORE_COL]).copy()
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(plot["lon"], plot["lat"], c=plot[SCORE_COL], s=18, alpha=0.65)
    if not top.empty:
        ax.scatter(top["lon"], top["lat"], s=70, facecolors="none", edgecolors="black", linewidths=1.2)
        for _, row in top.head(10).iterrows():
            ax.text(row["lon"], row["lat"], str(int(row["ranking"])), fontsize=8)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Mapa simple: municipios por score y top resaltado")
    fig.colorbar(scatter, ax=ax, label="V_i")
    fig.tight_layout()
    return fig


def make_cluster_profile_chart(profile: pd.DataFrame) -> plt.Figure:
    metrics = [
        "s_i_solar_promedio",
        "d_i_demanda_promedio",
        "g_i_red_promedio",
        "p_i_pendiente_promedio",
        "u_i_uso_suelo_promedio",
    ]
    existing = [column for column in metrics if column in profile.columns]
    matrix = profile.set_index("cluster_kmeans")[existing].fillna(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(matrix.values, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(existing)))
    ax.set_xticklabels(
        [column.replace("_promedio", "").replace("s_i_", "").replace("d_i_", "").replace("g_i_", "").replace("p_i_", "").replace("u_i_", "") for column in existing],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([f"cluster {value}" for value in matrix.index])
    ax.set_title("Perfil promedio de clusters")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04, label="Score 0-1")
    fig.tight_layout()
    return fig


def metric_value(value: float | int | None, decimals: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/D"
    return f"{value:.{decimals}f}"


def main() -> None:
    st.title("Dashboard de viabilidad para granja solar en Colombia")
    st.caption(
        "Modelo preliminar municipal. El ranking usa V_i y debe leerse con los flags de demanda y restricciones."
    )

    try:
        df = load_viability()
    except Exception as exc:
        st.error(f"No se pudo cargar la tabla de viabilidad: {exc}")
        st.stop()

    cluster_profile = load_cluster_profile()
    solar_costs = load_solar_costs()

    with st.sidebar:
        st.header("Filtros")
        include_outliers = st.checkbox(
            "Incluir atipicos de demanda",
            value=False,
            help="Si se activa, entran municipios con demanda proxy marcada para revision.",
        )
        only_eligible = st.checkbox(
            "Solo municipios no excluidos por R_i",
            value=True,
            help="Filtra R_i_preliminar = 1.",
        )
        top_n = st.slider("Numero de municipios", 5, 30, 10)
        departments_available = sorted(df["departamento"].dropna().astype(str).unique().tolist())
        departments = st.multiselect("Departamentos", departments_available)

    filtered = filter_data(df, departments, include_outliers, only_eligible)
    top = top_municipalities(filtered, top_n)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Municipios base", f"{len(df):,}")
    col2.metric("Municipios filtrados", f"{len(filtered):,}")
    col3.metric("Top mostrado", f"{len(top):,}")
    col4.metric("Max score filtrado", metric_value(filtered[SCORE_COL].max() if not filtered.empty else None))

    if filtered.empty:
        st.warning("No hay municipios despues de aplicar los filtros.")
        st.stop()

    if not include_outliers:
        st.info(
            "Vista limpia: se excluyen municipios con flag_atipico_eda_demanda=1. "
            "Esto evita que el ranking sea dominado por asignaciones ambiguas de demanda."
        )
    else:
        st.warning(
            "Vista de sensibilidad: incluye municipios con demanda marcada para revision. "
            "No debe presentarse como ranking definitivo sin explicar la incertidumbre."
        )

    tab_top, tab_explain, tab_map, tab_clusters, tab_costs, tab_data = st.tabs(
        ["Top municipios", "Por que aparecen", "Mapa", "Clusters", "Economia", "Datos"]
    )

    with tab_top:
        st.subheader(f"Top {len(top)} municipios mas probables")
        st.pyplot(make_top_score_chart(top), clear_figure=True)
        st.dataframe(
            format_table(top),
            use_container_width=True,
            hide_index=True,
        )

    with tab_explain:
        st.subheader("Explicacion del score")
        left, right = st.columns([1.2, 1])
        with left:
            st.pyplot(make_components_chart(top), clear_figure=True)
        with right:
            st.pyplot(make_component_heatmap(top), clear_figure=True)

        st.markdown("**Formula usada**")
        st.code("V_i = R_i(0.30*S_i + 0.15*D_i + 0.25*G_i + 0.20*P_i + 0.10*U_i)")

        st.markdown("**Lectura rapida del top**")
        for _, row in top.head(10).iterrows():
            st.write(
                f"{int(row['ranking'])}. **{row['municipio']} ({row['departamento']})**: "
                f"{shorten(str(row['por_que_aparece']), width=180, placeholder='...')}"
            )

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(make_scatter_solar_grid(filtered), clear_figure=True)
        with c2:
            st.pyplot(make_score_histogram(filtered), clear_figure=True)

    with tab_map:
        st.subheader("Ubicacion de municipios")
        st.pyplot(make_map_scatter(filtered, top), clear_figure=True)
        map_df = top[["lat", "lon", "municipio", "departamento", SCORE_COL]].dropna(subset=["lat", "lon"])
        if not map_df.empty:
            st.map(map_df, latitude="lat", longitude="lon", size=80)

    with tab_clusters:
        st.subheader("Agrupamiento K-Means")
        if cluster_profile.empty:
            st.warning("No existe perfil de clusters. Ejecuta src/scoring/kmeans_municipios.py.")
        else:
            st.pyplot(make_cluster_profile_chart(cluster_profile), clear_figure=True)
            st.dataframe(cluster_profile, use_container_width=True, hide_index=True)
        if "cluster_kmeans_label" in top.columns:
            st.markdown("**Clusters presentes en el top**")
            st.dataframe(
                top[["ranking", "municipio", "departamento", "cluster_kmeans_label", SCORE_COL]],
                use_container_width=True,
                hide_index=True,
            )

    with tab_costs:
        st.subheader("Escenarios economicos por hectarea")
        if solar_costs.empty:
            st.warning("No existe data/clean/solar_costs/solar_escenarios_por_hectarea.csv")
        else:
            show_cols = [
                "scenario_name",
                "capex_usd_per_kw",
                "land_use_hectares_per_mw",
                "annual_yield_kwh_per_kw_year",
                "capacidad_kw_por_hectarea",
                "costo_usd_por_hectarea",
                "generacion_mwh_por_hectarea_anual",
                "costo_usd_por_mwh_anual_simple",
                "notas_metodologicas",
            ]
            st.dataframe(
                solar_costs[[column for column in show_cols if column in solar_costs.columns]],
                use_container_width=True,
                hide_index=True,
            )
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.bar(solar_costs["scenario_name"], solar_costs["generacion_mwh_por_hectarea_anual"])
            ax.set_ylabel("MWh/ha-anio")
            ax.set_title("Generacion anual estimada por hectarea")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

    with tab_data:
        st.subheader("Control de calidad y datos")
        st.pyplot(make_classification_chart(df), clear_figure=True)
        st.markdown("**Tabla filtrada completa**")
        st.dataframe(
            filtered.sort_values(SCORE_COL, ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Descargar ranking filtrado CSV",
            data=filtered.sort_values(SCORE_COL, ascending=False).to_csv(index=False).encode("utf-8-sig"),
            file_name="ranking_viabilidad_filtrado.csv",
            mime="text/csv",
        )

    st.caption(
        "Limitacion: PVOUT y pendiente son proxies puntuales; demanda es proxy regional XM; "
        "el resultado es priorizacion preliminar, no seleccion final de predios."
    )


if __name__ == "__main__":
    main()
