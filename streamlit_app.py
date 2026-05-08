"""Dashboard de viabilidad solar municipal — Modelo agrivoltaico Colombia."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MULTIDIM_PATH = (
    PROJECT_ROOT
    / "data" / "clean" / "viabilidad_municipal"
    / "viabilidad_municipal_multidimensional.csv"
)

# ---------------------------------------------------------------------------
# Constantes de scoring
# ---------------------------------------------------------------------------
DIMS = {
    "score_fisico":       ("Fisico",       0.30, "#2563EB"),
    "score_electrico":    ("Electrico",    0.25, "#16A34A"),
    "score_economico":    ("Economico",    0.20, "#D97706"),
    "score_agropecuario": ("Agropecuario", 0.15, "#DC2626"),
    "score_riesgo":       ("Riesgo",       0.10, "#7C3AED"),
}
TOP5_COLORS = ["#1D4ED8", "#15803D", "#B45309", "#B91C1C", "#6D28D9"]

# Variables que componen cada score
FISICO_VARS = {
    "allsky_sfc_sw_dwn_media_kwh_m2_day": ("Irradiacion solar (kWh/m²/dia)", "NASA Power"),
    "pvout_kwh_kwp_day":                  ("Rendimiento PV (kWh/kWp/dia)", "Solargis PVOUT"),
    "t2m_media":                          ("Temperatura media 2m (°C)", "NASA Power"),
    "ws10m_media":                        ("Viento medio 10m (m/s)", "NASA Power"),
    "prectotcorr_suma_mm_year":           ("Precipitacion anual (mm/año)", "NASA Power"),
    "pendiente_igac":                     ("Clase de pendiente", "IGAC"),
    "score_pendiente":                    ("Score pendiente (0-1)", "IGAC calculado"),
    "score_fisico":                       (">>> SCORE FISICO (peso 30%)", "Calculado"),
}
ELECTRICO_VARS = {
    "dist_subestacion_km":       ("Distancia subestacion (km)", "UPME"),
    "subestacion_mas_cercana":   ("Subestacion mas cercana", "UPME"),
    "nivel_tension_mas_cercana": ("Nivel de tension", "UPME"),
    "tension_mas_cercana":       ("Tension (kV)", "UPME"),
    "capacidad_mva_mas_cercana": ("Capacidad (MVA)", "UPME"),
    "g_i_red":                   ("Score red normalizado (0-1)", "Calculado"),
    "score_electrico":           (">>> SCORE ELECTRICO (peso 25%)", "Calculado"),
}
ECONOMICO_VARS = {
    "score_tierra_modelo":       ("Score precio tierra (0-1)", "UPRA / imputado"),
    "imputacion_score_tierra":   ("Metodo imputacion tierra", "Pipeline"),
    "score_agua_modelo":         ("Score tarifa agua (0-1)", "SUI / imputado"),
    "imputacion_score_agua":     ("Metodo imputacion agua", "Pipeline"),
    "score_economico":           (">>> SCORE ECONOMICO (peso 20%)", "Calculado"),
}
AGRO_VARS = {
    "ugg_ha_proxy":              ("UGG por hectarea proxy", "EVA-ICA"),
    "sistema_productivo":        ("Sistema productivo", "EVA-ICA"),
    "pct_area_protegida_runap":  ("% area protegida RUNAP", "RUNAP"),
    "u_i_no_protegido_runap":    ("Fraccion no protegida (0-1)", "RUNAP"),
    "score_agropecuario":        (">>> SCORE AGROPECUARIO (peso 15%)", "Calculado"),
}
RIESGO_VARS = {
    "riesgo_inundacion_idx":     ("Indice riesgo inundacion (0-1)", "IMRC DNP"),
    "riesgo_sequia_idx":         ("Indice riesgo sequia (0-1)", "IMRC DNP"),
    "score_riesgo_viento_modelo":("Score riesgo viento (0-1)", "NASA / imputado"),
    "imputacion_score_riesgo_viento": ("Metodo imputacion viento", "Pipeline"),
    "score_riesgo":              (">>> SCORE RIESGO (peso 10%)", "Calculado"),
}
TOTAL_VARS = {
    "score_fisico":       ("Score Fisico",       "x 0.30"),
    "score_electrico":    ("Score Electrico",    "x 0.25"),
    "score_economico":    ("Score Economico",    "x 0.20"),
    "score_agropecuario": ("Score Agropecuario", "x 0.15"),
    "score_riesgo":       ("Score Riesgo",       "x 0.10"),
    "v_i_multidimensional": (">>> SCORE TOTAL",  "Suma ponderada"),
}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(MULTIDIM_PATH, dtype={"codigo_dane": "string"})
    return df.sort_values("v_i_multidimensional", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Graficas
# ---------------------------------------------------------------------------
def chart_municipio(row: pd.Series, rank: int, color: str) -> plt.Figure:
    """Barra vertical con el score total de un municipio."""
    score = float(row.get("v_i_multidimensional", 0) or 0)
    municipio = str(row["municipio"])
    depto = str(row["departamento"])

    fig, ax = plt.subplots(figsize=(3.2, 4.5), dpi=110)
    ax.bar(["Score"], [score], color=color, width=0.5, zorder=3)
    ax.bar(["Score"], [1 - score], bottom=[score], color="#E5E7EB", width=0.5, zorder=2)

    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)

    ax.text(
        0, score + 0.02, f"{score:.4f}",
        ha="center", va="bottom", fontsize=13,
        fontweight="bold", color=color,
    )
    ax.set_title(
        f"#{rank}\n{municipio}\n{depto}",
        fontsize=9.5,
        fontweight="bold" if rank == 1 else "normal",
        pad=6,
    )
    ax.set_xlabel("Score multidimensional", fontsize=8)
    fig.tight_layout()
    return fig


def chart_radar(row: pd.Series) -> plt.Figure:
    """Radar (spider) con los 5 scores discriminados del #1."""
    keys = list(DIMS.keys())
    labels = [DIMS[k][0] for k in keys]
    weights = [DIMS[k][1] for k in keys]
    values = [float(row.get(k, 0) or 0) for k in keys]
    colors = [DIMS[k][2] for k in keys]

    N = len(keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals_p = values + [values[0]]
    angs_p = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=110, subplot_kw={"polar": True})

    # Referencia maxima
    ax.plot(angs_p, [1.0] * (N + 1), color="#D1D5DB", linewidth=1, linestyle="--")
    ax.fill(angles, [1.0] * N, color="#F9FAFB", alpha=0.4)

    # Area real
    ax.fill(angles, values, color="#2563EB", alpha=0.18)
    ax.plot(angs_p, vals_p, color="#2563EB", linewidth=2)

    # Puntos por dimension con color propio
    for angle, val, col in zip(angles, values, colors):
        ax.plot(angle, val, "o", color=col, markersize=9, zorder=5)

    ax.set_xticks(angles)
    ax.set_xticklabels(
        [f"{lbl}\n(w={w:.0%})" for lbl, w in zip(labels, weights)],
        fontsize=8.5,
    )
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="#9CA3AF")

    for angle, val, key in zip(angles, values, keys):
        color = DIMS[key][2]
        ax.annotate(
            f"{val:.3f}",
            xy=(angle, val),
            xytext=(angle, min(val + 0.12, 0.98)),
            ha="center", va="center",
            fontsize=9, color=color, fontweight="bold",
        )

    score_total = float(row.get("v_i_multidimensional", 0) or 0)
    ax.set_title(
        f"#{1} {row['municipio']} ({row['departamento']})\nScore total: {score_total:.4f}",
        fontsize=10.5, fontweight="bold", pad=22,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tablas de desglose
# ---------------------------------------------------------------------------
def build_table(row: pd.Series, var_map: dict) -> pd.DataFrame:
    """Construye tabla de variables con su valor y fuente."""
    records = []
    for col, (label, fuente) in var_map.items():
        val = row.get(col, None)
        if pd.isna(val) or val is None:
            val_str = "— (sin dato)"
        elif isinstance(val, float):
            val_str = f"{val:.4f}"
        else:
            val_str = str(val)
        records.append({
            "Variable": label,
            "Valor": val_str,
            "Fuente": fuente,
        })
    return pd.DataFrame(records)


def style_table(df: pd.DataFrame) -> pd.DataFrame.style:
    def highlight_score(row):
        if str(row["Variable"]).startswith(">>>"):
            return ["background-color: #EFF6FF; font-weight: bold; color: #1D4ED8"] * len(row)
        return [""] * len(row)
    return df.style.apply(highlight_score, axis=1)


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Solar Agrivoltaico — Colombia",
        page_icon="☀️",
        layout="wide",
    )

    st.title("☀️ Viabilidad Solar Agrivoltaica — Colombia")
    st.caption("Modelo multidimensional: Fisico (30%) · Electrico (25%) · Economico (20%) · Agropecuario (15%) · Riesgo (10%)")

    if not MULTIDIM_PATH.exists():
        st.error(f"No se encontro el archivo de scores: {MULTIDIM_PATH}")
        st.info("Ejecuta primero: `python -m src.scoring.viabilidad_municipal`")
        st.stop()

    df = load_data()
    top5 = df.head(5).reset_index(drop=True)
    numero1 = top5.iloc[0]

    # -----------------------------------------------------------------------
    # Seccion 1: Top 5 graficas individuales
    # -----------------------------------------------------------------------
    st.header("Top 5 municipios — Score total")
    st.caption("Cada barra muestra el score multidimensional del municipio. La parte gris indica la distancia al maximo posible (1.0).")

    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            row = top5.iloc[i]
            fig = chart_municipio(row, i + 1, TOP5_COLORS[i])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.divider()

    # -----------------------------------------------------------------------
    # Seccion 2: Desglose del municipio #1
    # -----------------------------------------------------------------------
    st.header(f"Desglose del #1: {numero1['municipio']} ({numero1['departamento']})")
    st.caption(
        f"Score total: **{float(numero1['v_i_multidimensional']):.4f}** · "
        f"Clasificacion: **{numero1.get('clasificacion_multidim', '—')}**"
    )

    # Radar
    col_radar, col_info = st.columns([3, 2])
    with col_radar:
        fig_r = chart_radar(numero1)
        st.pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)
    with col_info:
        st.markdown("**Pesos del modelo:**")
        for key, (label, weight, color) in DIMS.items():
            score_val = float(numero1.get(key, 0) or 0)
            aporte = score_val * weight
            st.markdown(
                f"<span style='color:{color}'>●</span> **{label}** — "
                f"score: `{score_val:.4f}` × peso `{weight:.0%}` = **`{aporte:.4f}`**",
                unsafe_allow_html=True,
            )
        total = float(numero1.get("v_i_multidimensional", 0) or 0)
        st.markdown(f"---\n**Score total: `{total:.4f}`**")

    st.divider()

    # -----------------------------------------------------------------------
    # Tablas de desglose
    # -----------------------------------------------------------------------
    st.subheader("Como se forma el score — tablas de variables")

    tab1, tab2, tab3 = st.tabs([
        "Fisico + Electrico",
        "Economico + Agropecuario",
        "Riesgo + Score Total",
    ])

    with tab1:
        st.markdown("#### Score Fisico `(peso: 30%)`")
        st.caption("Variables climaticas y de recursos solares que determinan el potencial energetico del municipio.")
        st.dataframe(
            style_table(build_table(numero1, FISICO_VARS)),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Score Electrico `(peso: 25%)`")
        st.caption("Proximidad y capacidad de la infraestructura de red para evacuar la energia generada.")
        st.dataframe(
            style_table(build_table(numero1, ELECTRICO_VARS)),
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        st.markdown("#### Score Economico `(peso: 20%)`")
        st.caption("Costo de la tierra y del agua como proxy de la viabilidad economica de la inversion.")
        st.dataframe(
            style_table(build_table(numero1, ECONOMICO_VARS)),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Score Agropecuario `(peso: 15%)`")
        st.caption("Carga ganadera (UGG/ha) y uso del suelo como indicador del costo de oportunidad agrivoltaico.")
        st.dataframe(
            style_table(build_table(numero1, AGRO_VARS)),
            use_container_width=True,
            hide_index=True,
        )

    with tab3:
        st.markdown("#### Score Riesgo `(peso: 10%)`")
        st.caption("Indices de riesgo de inundacion y sequia (IMRC DNP) y riesgo de viento.")
        st.dataframe(
            style_table(build_table(numero1, RIESGO_VARS)),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Composicion del Score Total")
        st.caption("Suma ponderada de las 5 dimensiones que produce el score multidimensional final.")
        total_df = build_table(numero1, TOTAL_VARS)
        # Agregar columna de aporte ponderado
        pesos = {
            "score_fisico": 0.30,
            "score_electrico": 0.25,
            "score_economico": 0.20,
            "score_agropecuario": 0.15,
            "score_riesgo": 0.10,
            "v_i_multidimensional": None,
        }
        aportes = []
        for col in pesos:
            val = float(numero1.get(col, 0) or 0)
            peso = pesos[col]
            if peso is not None:
                aportes.append(f"{val:.4f} × {peso:.0%} = {val * peso:.4f}")
            else:
                aportes.append(f"= {val:.4f}")
        total_df["Calculo"] = aportes
        st.dataframe(
            style_table(total_df),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.caption(
        "Datos: NASA Power · IGAC · UPME · RUNAP · UPRA · EVA-ICA · IMRC DNP · SUI · INVIAS. "
        "Modelo: Jhon T — ITM Introduccion a la Inteligencia Artificial."
    )


if __name__ == "__main__":
    main()
