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
# Estilos globales Plotly — fuentes siempre oscuras y legibles
# ---------------------------------------------------------------------------
_FONT_COLOR  = "#111827"   # casi negro
_FONT_FAMILY = "Arial"
_FONT_SIZE   = 14          # base
_TITLE_SIZE  = 16
_TICK_SIZE   = 13
_LEGEND_SIZE = 13
_ANNOT_SIZE  = 13


def _apply_readable_fonts(fig, title_size: int = _TITLE_SIZE) -> None:
    """Aplica fuentes oscuras y grandes a todas las capas del figura Plotly."""
    base = dict(family=_FONT_FAMILY, color=_FONT_COLOR)
    layout_kw: dict = dict(font=dict(**base, size=_FONT_SIZE))
    # Solo aplicar title_font si la figura ya tiene titulo (Plotly 6.x muestra "undefined" si no)
    if getattr(fig.layout.title, "text", None):
        layout_kw["title_font"] = dict(**base, size=title_size)
    fig.update_layout(**layout_kw)
    fig.update_xaxes(
        title_font=dict(**base, size=_FONT_SIZE),
        tickfont=dict(**base, size=_TICK_SIZE),
        title_standoff=12,
    )
    fig.update_yaxes(
        title_font=dict(**base, size=_FONT_SIZE),
        tickfont=dict(**base, size=_TICK_SIZE),
        title_standoff=12,
    )
    fig.update_layout(
        legend=dict(font=dict(**base, size=_LEGEND_SIZE)),
    )
    # Anotaciones existentes
    for ann in fig.layout.annotations:
        if ann.font.color in (None, "") or ann.font.color.startswith("rgba(0"):
            ann.font.color = _FONT_COLOR
        if ann.font.size in (None, 0):
            ann.font.size = _ANNOT_SIZE


# ---------------------------------------------------------------------------
# Graficas
# ---------------------------------------------------------------------------
def chart_top5_stacked(top5: pd.DataFrame):
    """Barras horizontales apiladas interactivas (Plotly). score × peso por dimension."""
    import plotly.graph_objects as go

    keys    = list(DIMS.keys())
    labels  = [DIMS[k][0] for k in keys]
    colors  = [DIMS[k][2] for k in keys]
    weights = [DIMS[k][1] for k in keys]

    y_labels = [
        f"#{i+1} {row['municipio']} ({row['departamento']})"
        for i, (_, row) in enumerate(top5.iterrows())
    ]

    fig = go.Figure()
    for key, label, color, weight in zip(keys, labels, colors, weights):
        scores  = top5[key].fillna(0).values.astype(float)
        aportes = scores * weight
        hover   = [
            f"<b>{label}</b><br>Score: {s:.4f}<br>Peso: {weight:.0%}<br>Aporte: {a:.4f}"
            for s, a in zip(scores, aportes)
        ]
        fig.add_trace(go.Bar(
            name=f"{label} (×{weight:.0%})",
            y=y_labels,
            x=aportes,
            orientation="h",
            marker_color=color,
            hovertext=hover,
            hoverinfo="text",
            text=[f"{s:.2f}" for s in scores],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=13, family="Arial Black"),
        ))

    # Score total como anotaciones
    annotations = []
    for i, (_, row) in enumerate(top5.iterrows()):
        total = float(row.get("v_i_multidimensional", 0) or 0)
        annotations.append(dict(
            x=total + 0.012,
            y=y_labels[i],
            text=f"<b>{total:.4f}</b>",
            showarrow=False,
            xanchor="left",
            font=dict(size=14, color="#111827", family="Arial"),
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(
            text="Top 5 municipios — Composicion del score multidimensional",
            font=dict(size=16, color="#111827", family="Arial"),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(
                text="Aporte ponderado (score × peso)",
                font=dict(size=14, color="#111827", family="Arial"),
            ),
            range=[0, 1.18],
            gridcolor="#E5E7EB",
            tickfont=dict(size=13, color="#374151"),
            showline=True,
            linecolor="#9CA3AF",
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            tickfont=dict(size=13, color="#111827", family="Arial"),
        ),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.28,
            xanchor="center", x=0.5,
            font=dict(size=13, color="#111827", family="Arial"),
        ),
        annotations=annotations,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        margin=dict(l=10, r=100, t=60, b=120),
    )
    _apply_readable_fonts(fig)
    return fig


def chart_radar(row: pd.Series):
    """Radar interactivo Plotly con los 5 scores discriminados."""
    import plotly.graph_objects as go

    keys    = list(DIMS.keys())
    labels  = [f"{DIMS[k][0]}<br>w={DIMS[k][1]:.0%}" for k in keys]
    weights = [DIMS[k][1] for k in keys]
    values  = [float(row.get(k, 0) or 0) for k in keys]

    # Cerrar el polígono
    labels_c = labels + [labels[0]]
    values_c = values + [values[0]]

    score_total = float(row.get("v_i_multidimensional", 0) or 0)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_c,
        theta=labels_c,
        fill="toself",
        fillcolor="rgba(37,99,235,0.15)",
        line=dict(color="#2563EB", width=2.5),
        marker=dict(size=8, color="#2563EB"),
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.4f}<extra></extra>",
        name="Score",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                tickfont=dict(size=11, color="#374151", family="Arial"),
                gridcolor="#D1D5DB",
                linecolor="#6B7280",
            ),
            angularaxis=dict(
                tickfont=dict(size=14, color="#111827", family="Arial Black"),
                gridcolor="#D1D5DB",
            ),
            bgcolor="white",
        ),
        showlegend=False,
        margin=dict(t=65, b=45, l=65, r=65),
        width=468, height=468,
        title=dict(
            text=f"#1 {row['municipio']} ({row['departamento']})<br><sup>Score total: {score_total:.4f}</sup>",
            font=dict(size=14, color="#111827", family="Arial"),
            x=0.5,
        ),
        paper_bgcolor="white",
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(tickfont=dict(family=_FONT_FAMILY, color="#374151", size=12)),
            angularaxis=dict(tickfont=dict(family=_FONT_FAMILY, color=_FONT_COLOR, size=15)),
        )
    )
    _apply_readable_fonts(fig)
    return fig


def radar_values_table(row: pd.Series) -> pd.DataFrame:
    """Tabla lateral con scores y aportes ponderados para el radar."""
    records = []
    total_check = 0.0
    for key, (label, weight, color) in DIMS.items():
        score = float(row.get(key, 0) or 0)
        aporte = score * weight
        total_check += aporte
        records.append({
            "Dimension": label,
            "Peso": f"{weight:.0%}",
            "Score": round(score, 4),
            "Aporte": round(aporte, 4),
        })
    records.append({
        "Dimension": "TOTAL",
        "Peso": "100%",
        "Score": "—",
        "Aporte": round(total_check, 4),
    })
    return pd.DataFrame(records)


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
# Variables crudas por dimension (pre-normalizacion)
# ---------------------------------------------------------------------------

# Leyendas fijas por variable: (fuente, que_mide, como_entra_al_score, alerta)
VAR_LEGENDS: dict[str, dict[str, str]] = {
    "allsky_sfc_sw_dwn_media_kwh_m2_day": {
        "fuente": "NASA Power (2001-2022, promedio multianual)",
        "que_mide": "Energia solar total que llega a la superficie por dia. Es la materia prima del proyecto: sin irradiacion no hay generacion posible.",
        "escala": "En Colombia varia entre ~3.5 kWh/m²/dia (Pacifico nublado) y ~6.5 kWh/m²/dia (La Guajira). La media nacional ronda 4.8.",
        "score": "Mayor irradiacion → mayor score_fisico. Se normaliza min-max sobre el rango nacional. Un panel en La Guajira genera casi el doble que uno en Narino.",
        "alerta": "",
    },
    "pvout_kwh_kwp_day": {
        "fuente": "Solargis PVOUT (modelo satelital)",
        "que_mide": "Energia real que produce 1 kWp instalado, ya descontando perdidas por temperatura, angulo solar y eficiencia del sistema. Es el dato que usan los bancos para financiar proyectos.",
        "escala": "Varia de ~3.0 a ~5.8 kWh/kWp/dia. Siempre menor que la irradiacion bruta porque incorpora las perdidas reales del sistema.",
        "score": "Mayor PVOUT → mayor score_fisico. Es la variable con mayor peso dentro de la dimension fisica porque refleja generacion real, no solo recurso bruto.",
        "alerta": "",
    },
    "t2m_media": {
        "fuente": "NASA Power (temperatura del aire a 2 metros, promedio anual)",
        "que_mide": "Temperatura ambiente. Los paneles solares pierden eficiencia a medida que sube la temperatura: aproximadamente -0.4% por cada grado sobre 25 °C.",
        "escala": "Colombia varia de ~8 °C en paramos andinos a ~30 °C en costa y Llanos. Distribucion bimodal: grupo andino (~17-22 °C) y grupo tierra caliente (~27-30 °C).",
        "score": "Temperatura moderada (18-24 °C) es optima. Muy fria reduce generacion de madrugada; muy caliente degrada eficiencia del panel.",
        "alerta": "",
    },
    "prectotcorr_suma_mm_year": {
        "fuente": "NASA Power (precipitacion total corregida, suma anual)",
        "que_mide": "Milimetros de lluvia acumulada al año. Proxy de nubosidad: mas lluvia generalmente implica mas dias nublados y menos irradiacion directa.",
        "escala": "Colombia es uno de los paises mas lluviosos del mundo. Varia de ~200 mm/año (La Guajira) hasta >8000 mm/año (Choco). Distribucion muy sesgada a la derecha.",
        "score": "Menos precipitacion → mejor score_fisico. Pero el PVOUT ya captura buena parte del efecto de nubosidad, por lo que esta variable tiene peso secundario.",
        "alerta": "",
    },
    "ws10m_media": {
        "fuente": "NASA Power (velocidad del viento a 10 metros, promedio anual)",
        "que_mide": "Viento medio. Cumple dos roles opuestos: enfria los paneles (mejora eficiencia) pero tambien genera cargas estructurales sobre los soportes.",
        "escala": "En Colombia continental: 1-5 m/s. La Guajira supera 8 m/s. Distribucion sesgada con cola derecha en zonas costeras.",
        "score": "Entra tanto en score_fisico (enfriamiento positivo) como en score_riesgo (viento extremo negativo). El score_riesgo_viento_modelo ya es la version normalizada.",
        "alerta": "⚠️ score_riesgo usa solo esta variable porque los indices IMRC de inundacion/sequia no fueron cargados.",
    },
    "dist_subestacion_km": {
        "fuente": "UPME — Red de subestaciones del Sistema Interconectado Nacional",
        "que_mide": "Distancia en linea recta al punto de conexion a la red electrica mas cercano. Es el determinante economico mas critico: cada km de linea de transmision cuesta entre USD 80,000 y 200,000.",
        "escala": "La mayoria de municipios esta dentro de 0-60 km. Cola larga con municipios remotos hasta 700+ km (Amazonia, Pacifico).",
        "score": "Menor distancia → mayor score_electrico (peso 50% dentro de la dimension). Si el Top 5 tiene lineas verticales en 0-20 km, esto explica gran parte del ranking.",
        "alerta": "",
    },
    "tension_mas_cercana": {
        "fuente": "UPME",
        "que_mide": "Tension en kV de la subestacion mas cercana.",
        "escala": "No disponible como numero: el CSV contiene texto mixto como '500/115' para subestaciones de doble devanado.",
        "score": "El scoring usa nivel_tension_mas_cercana (Nivel 1-5) mapeado a 0.0-1.0. Esta columna cruda no es procesable directamente.",
        "alerta": "⚠️ Columna no numerica. El scoring usa la columna nivel_tension_mas_cercana en su lugar.",
    },
    "capacidad_mva_mas_cercana": {
        "fuente": "UPME — capacidad de transformacion instalada",
        "que_mide": "Megavoltamperios disponibles en la subestacion mas cercana. Aunque la subestacion este cerca, si esta saturada no puede absorber energia nueva sin costosas ampliaciones.",
        "escala": "Subestaciones rurales: 10-40 MVA. Subestaciones troncales: 200-900 MVA. Distribucion muy sesgada: la mayoria son pequeñas, unas pocas son enormes.",
        "score": "Mayor capacidad → mayor score_electrico (peso 20%). La linea verde en ~900 MVA del Top 5 indica que uno de los municipios top tiene acceso a una subestacion troncal de gran capacidad.",
        "alerta": "",
    },
    "score_tierra_modelo": {
        "fuente": "UPRA / imputacion departamental",
        "que_mide": "Score ya normalizado (0-1) del precio de la tierra. Mayor score = tierra mas barata = menor costo de instalacion. Los datos brutos (COP/ha) estan en CSV externo.",
        "escala": "0 = tierra mas cara del pais, 1 = tierra mas barata. Si la distribucion se concentra en pocos valores, indica que muchos municipios recibieron el mismo valor imputado.",
        "score": "Entra directamente al score_economico (peso variable). La imputacion usa mediana departamental cuando no hay dato directo del municipio.",
        "alerta": "⚠️ Muchos municipios pueden tener valor imputado, no dato real de mercado.",
    },
    "score_agua_modelo": {
        "fuente": "SUI (Sistema Unico de Informacion de Servicios Publicos) / imputacion",
        "que_mide": "Score normalizado (0-1) de la tarifa de acueducto. Agua barata = menor costo operativo de limpieza de paneles.",
        "escala": "0 = tarifa mas alta, 1 = tarifa mas baja. Distribucion puede tener picos en valores de imputacion.",
        "score": "Entra al score_economico. La limpieza de paneles requiere ~98 litros/MWh generado. En zonas con agua cara, el costo operativo sube.",
        "alerta": "⚠️ Muchos municipios con dato imputado por mediana nacional o departamental.",
    },
    "ugg_ha_proxy": {
        "fuente": "EVA Pecuaria ICA — inventario bovino 2019-2023",
        "que_mide": "Unidades Gran Ganado por hectarea. Mide la presion ganadera: cuantas cabezas equivalentes de ganado hay por hectarea del municipio.",
        "escala": "0-1 extensivo bajo, 1-3 tradicional, >3 tecnificado. La mayoria de municipios colombianos son extensivos.",
        "score": "Mayor UGG/ha → mayor costo de oportunidad → score agropecuario alto en modelo agrivoltaico (la solar convive con el ganado).",
        "alerta": "⚠️ DATO NO DISPONIBLE EN EL CSV FINAL. El archivo fue generado pero no se unio correctamente al pipeline de scoring. El score_agropecuario actual no incluye carga ganadera.",
    },
    "pct_area_protegida_runap": {
        "fuente": "RUNAP — Registro Unico Nacional de Areas Protegidas",
        "que_mide": "Porcentaje del area del municipio bajo alguna figura de proteccion ambiental (parques nacionales, reservas, sitios Ramsar). En area protegida NO se puede instalar solar.",
        "escala": "0% (sin proteccion) a 100% (completamente protegido). Distribucion muy sesgada: municipios andinos/cafeteros con <10%, Amazonia con >70%.",
        "score": "Mayor % protegido → menor score_agropecuario. Se invierte: u_i_no_protegido_runap = 1 - pct. Las lineas del Top 5 en valores bajos (0-5%) confirman que los mejores municipios tienen poca restriccion ambiental.",
        "alerta": "",
    },
    "riesgo_inundacion_idx": {
        "fuente": "IMRC DNP — Indice Municipal de Riesgo de Desastres",
        "que_mide": "Indice compuesto de amenaza + exposicion + vulnerabilidad ante inundaciones. No es solo probabilidad climatica, incorpora la capacidad de respuesta del municipio.",
        "escala": "0 = sin riesgo, 1 = maximo riesgo. Municipios ribereños del Caribe y Pacifico tienen valores altos.",
        "score": "Mayor riesgo → menor score_riesgo. Se invierte en el modelo.",
        "alerta": "⚠️ DATO NO DISPONIBLE. Los datos IMRC nunca fueron descargados ni cargados al pipeline. El score_riesgo se calcula SIN este indice.",
    },
    "riesgo_sequia_idx": {
        "fuente": "IMRC DNP — Indice Municipal de Riesgo de Desastres",
        "que_mide": "Indice compuesto de riesgo ante sequia: amenaza climatica + exposicion de cultivos y poblacion + vulnerabilidad institucional.",
        "escala": "0 = sin riesgo, 1 = maximo riesgo. La Guajira y Caribe seco tienen valores altos.",
        "score": "Mayor riesgo sequia → menor score_riesgo. La sequia afecta el agua disponible para limpieza de paneles.",
        "alerta": "⚠️ DATO NO DISPONIBLE. Los datos IMRC nunca fueron descargados. El score_riesgo se calcula SIN este indice.",
    },
    "score_riesgo_viento_modelo": {
        "fuente": "NASA Power ws10m_media, normalizado e imputado",
        "que_mide": "Score de riesgo estructural por viento. Alto score = poco riesgo de viento. Se construyo a partir del viento medio: viento extremo (>8 m/s) penaliza.",
        "escala": "0-1. Media nacional 0.934, rango 0.93-1.0 para el 95% de municipios. La concentracion cerca de 1.0 indica que casi todo el pais tiene viento moderado.",
        "score": "Es la UNICA variable de riesgo actualmente en el calculo. Dado que casi todos los municipios tienen score ~0.93-0.95, esta dimension apenas diferencia entre municipios.",
        "alerta": "⚠️ PROBLEMA DE DISCRIMINACION: score_riesgo es casi identico para todos los municipios porque solo usa viento (que es homogeneo en Colombia). Sin datos de inundacion y sequia, la dimension Riesgo no aporta informacion real al ranking.",
    },
}

RAW_DIMS: dict[str, tuple[str, str, list[tuple[str, str]]]] = {
    "Fisico": (
        "#2563EB",
        "Variables climaticas y de recurso solar antes de normalizar (0-1).",
        [
            ("allsky_sfc_sw_dwn_media_kwh_m2_day", "Irradiacion solar (kWh/m²/dia)"),
            ("pvout_kwh_kwp_day",                  "Rendimiento PV (kWh/kWp/dia)"),
            ("t2m_media",                          "Temperatura media 2m (°C)"),
            ("prectotcorr_suma_mm_year",           "Precipitacion anual (mm/año)"),
            ("ws10m_media",                        "Viento medio 10m (m/s)"),
        ],
    ),
    "Electrico": (
        "#16A34A",
        "Infraestructura de red: distancia y capacidad antes de normalizar.",
        [
            ("dist_subestacion_km",       "Distancia a subestacion (km)"),
            ("tension_mas_cercana",       "Tension subestacion (kV)"),
            ("capacidad_mva_mas_cercana", "Capacidad (MVA)"),
        ],
    ),
    "Economico": (
        "#D97706",
        "Scores de tierra y agua (ya en 0-1; datos brutos en CSV externos).",
        [
            ("score_tierra_modelo", "Score precio tierra (0-1)"),
            ("score_agua_modelo",   "Score tarifa agua (0-1)"),
        ],
    ),
    "Agropecuario": (
        "#DC2626",
        "Carga ganadera y area protegida antes de normalizar.",
        [
            ("ugg_ha_proxy",             "UGG por hectarea (carga ganadera)"),
            ("pct_area_protegida_runap", "% area protegida RUNAP"),
        ],
    ),
    "Riesgo": (
        "#7C3AED",
        "Indices de riesgo antes de normalizar (mayor = mas riesgo).",
        [
            ("riesgo_inundacion_idx",       "Indice riesgo inundacion"),
            ("riesgo_sequia_idx",           "Indice riesgo sequia"),
            ("score_riesgo_viento_modelo",  "Score riesgo viento (0-1)"),
        ],
    ),
}


def chart_raw_variable(
    df: pd.DataFrame,
    col: str,
    label: str,
    color: str,
    top5: pd.DataFrame,
) -> object:
    """Histograma Plotly de una variable cruda con top 5 marcados."""
    import plotly.graph_objects as go

    serie = pd.to_numeric(df[col], errors="coerce").dropna()
    if serie.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=serie,
        nbinsx=40,
        marker_color=color,
        opacity=0.75,
        name="Todos los municipios",
        hovertemplate="Rango: %{x}<br>Municipios: %{y}<extra></extra>",
    ))

    # Top 5 como lineas verticales
    top5_colors = ["#1D4ED8", "#15803D", "#B45309", "#B91C1C", "#6D28D9"]
    for i, (_, row) in enumerate(top5.iterrows()):
        val = pd.to_numeric(row.get(col, None), errors="coerce")
        if pd.isna(val):
            continue
        val = float(val)
        municipio = f"#{i+1} {row['municipio']}"
        fig.add_vline(
            x=val,
            line=dict(color=top5_colors[i], width=2, dash="dash"),
            annotation_text=municipio,
            annotation_position="top",
            annotation=dict(font=dict(size=10, color=top5_colors[i])),
        )

    p25, p50, p75 = serie.quantile([0.25, 0.50, 0.75])
    fig.update_layout(
        title=dict(text=label, font=dict(size=13, color="#111827"), x=0),
        xaxis=dict(title=label, tickfont=dict(size=11, color="#374151")),
        yaxis=dict(title="Municipios", tickfont=dict(size=11, color="#374151")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=280,
        margin=dict(l=10, r=10, t=40, b=40),
        showlegend=False,
        bargap=0.05,
        annotations=[
            dict(
                x=p50, y=0, yref="paper",
                text=f"Med: {p50:.2f}",
                showarrow=False,
                font=dict(size=9, color="#6B7280"),
                yanchor="bottom",
            )
        ],
    )
    _apply_readable_fonts(fig)
    return fig


def render_ranking_tab(df: pd.DataFrame, top5: pd.DataFrame, numero1: pd.Series) -> None:
    st.header("Top 5 municipios — Score total")
    st.caption("Cada segmento muestra el aporte ponderado (score × peso) de cada dimension al score total.")
    fig_stack = chart_top5_stacked(top5)
    st.plotly_chart(fig_stack, use_container_width=True)

    st.divider()
    st.header(f"Desglose del #1: {numero1['municipio']} ({numero1['departamento']})")
    st.caption(
        f"Score total: **{float(numero1['v_i_multidimensional']):.4f}** · "
        f"Clasificacion: **{numero1.get('clasificacion_multidim', '—')}**"
    )

    col_radar, col_info = st.columns([1, 1])
    with col_radar:
        fig_r = chart_radar(numero1)
        st.plotly_chart(fig_r, use_container_width=False)
    with col_info:
        st.markdown("**Scores por dimension**")
        st.dataframe(radar_values_table(numero1), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Como se forma el score — tablas de variables")

    tab1, tab2, tab3 = st.tabs(["Fisico + Electrico", "Economico + Agropecuario", "Riesgo + Score Total"])

    with tab1:
        st.markdown("#### Score Fisico `(peso: 30%)`")
        st.caption("Variables climaticas y de recursos solares que determinan el potencial energetico del municipio.")
        st.dataframe(style_table(build_table(numero1, FISICO_VARS)), use_container_width=True, hide_index=True)
        st.markdown("#### Score Electrico `(peso: 25%)`")
        st.caption("Proximidad y capacidad de la infraestructura de red para evacuar la energia generada.")
        st.dataframe(style_table(build_table(numero1, ELECTRICO_VARS)), use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("#### Score Economico `(peso: 20%)`")
        st.caption("Costo de la tierra y del agua como proxy de la viabilidad economica de la inversion.")
        st.dataframe(style_table(build_table(numero1, ECONOMICO_VARS)), use_container_width=True, hide_index=True)
        st.markdown("#### Score Agropecuario `(peso: 15%)`")
        st.caption("Carga ganadera (UGG/ha) y uso del suelo como indicador del costo de oportunidad agrivoltaico.")
        st.dataframe(style_table(build_table(numero1, AGRO_VARS)), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### Score Riesgo `(peso: 10%)`")
        st.caption("Indices de riesgo de inundacion y sequia (IMRC DNP) y riesgo de viento.")
        st.dataframe(style_table(build_table(numero1, RIESGO_VARS)), use_container_width=True, hide_index=True)
        st.markdown("#### Composicion del Score Total")
        st.caption("Suma ponderada de las 5 dimensiones que produce el score multidimensional final.")
        total_df = build_table(numero1, TOTAL_VARS)
        pesos_map = {
            "score_fisico": 0.30, "score_electrico": 0.25, "score_economico": 0.20,
            "score_agropecuario": 0.15, "score_riesgo": 0.10, "v_i_multidimensional": None,
        }
        aportes = []
        for col, peso in pesos_map.items():
            val = float(numero1.get(col, 0) or 0)
            aportes.append(f"{val:.4f} × {peso:.0%} = {val*peso:.4f}" if peso else f"= {val:.4f}")
        total_df["Calculo"] = aportes
        st.dataframe(style_table(total_df), use_container_width=True, hide_index=True)


def _legend_card(col: str, df: pd.DataFrame, top5: pd.DataFrame, color: str) -> None:
    """Renderiza la leyenda explicativa de una variable con hallazgos dinamicos."""
    leg = VAR_LEGENDS.get(col, {})
    if not leg:
        return

    serie = pd.to_numeric(df[col], errors="coerce").dropna() if col in df.columns else pd.Series([], dtype=float)

    # Hallazgos dinamicos calculados del CSV real
    hallazgos: list[str] = []
    top5_vals: list[float] = []
    if not serie.empty:
        med   = serie.median()
        mean  = serie.mean()
        p25   = serie.quantile(0.25)
        p75   = serie.quantile(0.75)
        skew  = serie.skew()
        n_tot = len(serie)

        hallazgos.append(f"**{n_tot:,} municipios** con dato. Mediana: **{med:.3g}** · Media: **{mean:.3g}**")
        hallazgos.append(f"Rango intercuartil: {p25:.3g} – {p75:.3g}")

        if abs(skew) > 1.5:
            dir_skew = "derecha" if skew > 0 else "izquierda"
            hallazgos.append(f"Distribucion muy sesgada a la **{dir_skew}** (skew={skew:.1f}): la mayoria de municipios se concentra en un extremo con una cola larga en el otro.")

        # Donde caen los top 5
        top5_vals = [pd.to_numeric(row.get(col, None), errors="coerce") for _, row in top5.iterrows()]
        top5_vals = [v for v in top5_vals if not pd.isna(v)]
        if top5_vals:
            t5_med = float(pd.Series(top5_vals).median())
            pct = (serie < t5_med).mean() * 100
            hallazgos.append(f"Los Top 5 tienen mediana **{t5_med:.3g}** en esta variable — superan al **{pct:.0f}%** de los municipios del pais.")

    top5_colors = ["#1D4ED8", "#15803D", "#B45309", "#B91C1C", "#6D28D9"]
    top5_names  = [f"#{i+1} {row['municipio']}" for i, (_, row) in enumerate(top5.iterrows())]

    st.markdown(f"**Fuente:** {leg.get('fuente', '—')}")
    st.markdown(f"**Que mide:** {leg.get('que_mide', '—')}")
    st.markdown(f"**Escala:** {leg.get('escala', '—')}")
    st.markdown(f"**Como entra al score:** {leg.get('score', '—')}")

    if leg.get("alerta"):
        st.error(leg["alerta"])

    if hallazgos:
        st.markdown("---")
        st.markdown("**Hallazgos en los datos:**")
        for h in hallazgos:
            st.markdown(f"- {h}")

    if top5_vals:
        st.markdown("---")
        st.markdown("**Lineas de colores = Top 5:**")
        for name, val, col_hex in zip(top5_names, top5_vals, top5_colors):
            st.markdown(
                f"<span style='color:{col_hex}'>▏</span> {name}: **{val:.3g}**",
                unsafe_allow_html=True,
            )

    if serie.empty and col not in df.columns:
        st.error("Columna ausente en el CSV. Dato no cargado al pipeline.")


def render_raw_tab(df: pd.DataFrame, top5: pd.DataFrame) -> None:
    """Pestaña de distribuciones de variables crudas por dimension."""
    st.subheader("Distribucion de variables antes de normalizar")
    st.caption(
        "Cada histograma muestra como se distribuyen los ~1100 municipios de Colombia en esa variable cruda. "
        "Las lineas verticales de colores indican donde cae cada uno de los Top 5 del ranking. "
        "A la derecha de cada grafica encontraras la fuente, interpretacion, escala y hallazgos de los datos."
    )

    dim_tabs = st.tabs(list(RAW_DIMS.keys()))
    for dim_tab, (dim_name, (color, caption, variables)) in zip(dim_tabs, RAW_DIMS.items()):
        with dim_tab:
            st.caption(caption)

            missing = [lbl for col, lbl in variables if col not in df.columns]
            if missing:
                st.warning(f"Sin datos en el CSV: {', '.join(missing)}")

            available = [(col, lbl) for col, lbl in variables if col in df.columns]

            # Variables no disponibles — mostrar leyenda aunque no haya grafica
            for col, lbl in variables:
                if col in df.columns:
                    continue
                st.markdown(f"#### {lbl}")
                leg_col, _ = st.columns([1, 1])
                with leg_col:
                    _legend_card(col, df, top5, color)
                st.divider()

            # Variables disponibles — grafica + leyenda lado a lado
            for col, lbl in available:
                st.markdown(f"#### {lbl}")
                chart_col, legend_col = st.columns([3, 2])
                with chart_col:
                    fig = chart_raw_variable(df, col, lbl, color, top5)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Variable presente pero sin valores numericos.")
                with legend_col:
                    _legend_card(col, df, top5, color)
                st.divider()

            # Estadisticas basicas al final
            stat_cols = [c for c, _ in available if c in df.columns]
            if stat_cols:
                with st.expander("Estadisticas descriptivas de todas las variables"):
                    num_cols = [c for c in stat_cols if pd.to_numeric(df[c], errors="coerce").notna().any()]
                    if num_cols:
                        label_map = {col: lbl for col, lbl in available}
                        stats = df[num_cols].apply(pd.to_numeric, errors="coerce").describe().T.round(3)
                        stats.index = stats.index.map(lambda c: label_map.get(c, c))
                        st.dataframe(stats, use_container_width=True)


# ---------------------------------------------------------------------------
# Pestaña agrivoltaica
# ---------------------------------------------------------------------------

# Parametros fijos del modelo agrivoltaico (NREL / AGROSAVIA / FEDEGAN)
_HA_POR_MW        = 2.02    # NREL: 5 acres/MW utility PV
_KG_CARNE_UGG_AÑO = 250.0  # kg carne viva por UGG por año (conservador)
_AGUA_M3_POR_MWH  = 0.098  # NREL: litros de lavado (m³/MWh)

# Defaults economicos compartidos entre pestañas (Agrivoltaico y Viabilidad Financiera)
_DEFAULT_PRECIO_ENERGIA = 200       # COP/kWh — PPA bilateral solar Colombia 2024 realista
_DEFAULT_PRECIO_CARNE   = 10_000    # COP/kg carne viva en pie (novillo gordo Colombia 2024)
_DEFAULT_UGG_AGRO       = 2.0       # UGG/ha bajo paneles (AGROSAVIA silvopastoral)
_DEFAULT_UGG_TRAD       = 1.5       # UGG/ha sistema tradicional (tropico bajo)
_DEFAULT_TARIFA_AGUA    = 1_800     # COP/m³ default (mediana SUI)


def _calcular_agrovoltaico(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Calcula el beneficio agrivoltaico por municipio.

    Modelo:
      ha_viable = area_no_protegida_km2 * 100 * score_pendiente
      MW_solar  = ha_viable / ha_por_mw
      MWh_año   = MW_solar * pvout * 365 * 1000 / 1000
      ingreso_solar (M COP) = MWh * precio_energia / 1e6

      bovinos_agrivoltaico = ha_viable * ugg_agro
      bovinos_puro         = ha_viable * ugg_tradicional
      ingreso_carne_agro (M COP) = bovinos_agrivoltaico * kg_ugg * precio_carne / 1e6
      ingreso_carne_puro (M COP) = bovinos_puro * kg_ugg * precio_carne / 1e6

      ingreso_total_agro = ingreso_solar + ingreso_carne_agro
      ganancia_vs_puro_solar    = ingreso_total_agro - ingreso_solar
      ganancia_vs_puro_ganadero = ingreso_total_agro - ingreso_carne_puro
    """
    out = df[["codigo_dane", "municipio", "departamento", "lat", "lon",
              "pvout_kwh_kwp_day", "score_pendiente",
              "area_no_protegida_km2_runap", "area_km2_igac",
              "pct_area_protegida_runap", "precio_tierra_ha_cop",
              "tarifa_acueducto_m3_cop", "v_i_multidimensional"]].copy()

    for c in ["pvout_kwh_kwp_day", "score_pendiente", "area_no_protegida_km2_runap",
              "area_km2_igac", "pct_area_protegida_runap",
              "precio_tierra_ha_cop", "tarifa_acueducto_m3_cop"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Area viable para instalacion solar (ha)
    # Se descuentan: RUNAP, urbano, vias, hidrico, cultivos y se penaliza por pendiente
    area_no_prot_km2 = out["area_no_protegida_km2_runap"].fillna(
        out["area_km2_igac"] * (1 - out["pct_area_protegida_runap"].fillna(0))
    )
    area_disponible_km2 = area_no_prot_km2 * (1 - _FACTOR_USO_NO_SOLAR)   # descuenta urb+vias+agua+cultivo
    pendiente_factor = out["score_pendiente"].clip(0, 1).fillna(0.5)
    out["area_descontada_km2"] = (area_no_prot_km2 - area_disponible_km2).round(2)
    out["area_disponible_km2"] = area_disponible_km2.round(2)
    out["ha_viable"] = (area_disponible_km2 * 100 * pendiente_factor).round(1)

    # Potencial solar
    pvout = out["pvout_kwh_kwp_day"].fillna(out["pvout_kwh_kwp_day"].median())
    out["mw_solar"]       = (out["ha_viable"] / params["ha_por_mw"]).round(2)
    out["mwh_año"]        = (out["mw_solar"] * pvout * 365).round(0)
    # mwh esta en MWh, precio en COP/kWh -> convertir MWh a kWh (x1000)
    out["ingreso_solar_m_cop"] = (out["mwh_año"] * 1000 * params["precio_energia_cop_kwh"] / 1e6).round(2)

    # Agua de limpieza de paneles
    out["agua_lavado_m3_año"] = (out["mwh_año"] * _AGUA_M3_POR_MWH).round(0)
    out["costo_agua_m_cop"]   = (out["agua_lavado_m3_año"] *
                                  out["tarifa_acueducto_m3_cop"].fillna(params["tarifa_agua_default"]) / 1e6).round(3)

    # Bovinos agrivoltaico
    out["bovinos_agro"]          = (out["ha_viable"] * params["ugg_agro"]).round(0)
    out["ingreso_carne_agro_m_cop"] = (
        out["bovinos_agro"] * _KG_CARNE_UGG_AÑO * params["precio_carne_cop_kg"] / 1e6
    ).round(2)

    # Bovinos puro ganadero (baseline)
    out["bovinos_puro"]          = (out["ha_viable"] * params["ugg_tradicional"]).round(0)
    out["ingreso_carne_puro_m_cop"] = (
        out["bovinos_puro"] * _KG_CARNE_UGG_AÑO * params["precio_carne_cop_kg"] / 1e6
    ).round(2)

    # Totales
    out["ingreso_total_agro_m_cop"] = (
        out["ingreso_solar_m_cop"] + out["ingreso_carne_agro_m_cop"] - out["costo_agua_m_cop"]
    ).round(2)
    out["ingreso_puro_solar_m_cop"] = (out["ingreso_solar_m_cop"] - out["costo_agua_m_cop"]).round(2)

    out["ganancia_vs_puro_solar_m_cop"]    = (out["ingreso_total_agro_m_cop"] - out["ingreso_puro_solar_m_cop"]).round(2)
    out["ganancia_vs_puro_ganadero_m_cop"] = (out["ingreso_total_agro_m_cop"] - out["ingreso_carne_puro_m_cop"]).round(2)

    # Por hectarea (comparacion justa entre municipios de distinto tamaño)
    out["ingreso_agro_m_cop_por_ha"]    = (out["ingreso_total_agro_m_cop"] / out["ha_viable"].replace(0, float("nan"))).round(4)
    out["ingreso_solar_m_cop_por_ha"]   = (out["ingreso_puro_solar_m_cop"] / out["ha_viable"].replace(0, float("nan"))).round(4)
    out["ingreso_ganadero_m_cop_por_ha"]= (out["ingreso_carne_puro_m_cop"] / out["ha_viable"].replace(0, float("nan"))).round(4)

    # Ingreso efectivo ponderado por score multidimensional
    # Esto evita que municipios enormes pero de baja calidad (sin red, sin demanda)
    # dominen el ranking solo por area
    score_mul = pd.to_numeric(out["v_i_multidimensional"], errors="coerce").fillna(0).clip(0, 1)
    out["ingreso_efectivo_por_ha"] = (out["ingreso_agro_m_cop_por_ha"] * score_mul).round(4)
    out["score_x_ingreso_total"]   = (out["ingreso_total_agro_m_cop"] * score_mul).round(2)

    return out.dropna(subset=["ha_viable", "ingreso_total_agro_m_cop"])


def render_agrivoltaico_tab(df: pd.DataFrame, top5: pd.DataFrame) -> None:
    import plotly.graph_objects as go

    st.subheader("Beneficio agrivoltaico — Solar + Bovinos en la misma hectarea")
    st.caption(
        "Cuanto genera un municipio instalando paneles solares y manteniendo ganado bovino "
        "en las mismas hectareas viables. Compara contra solo ganaderia tradicional o solo solar."
    )

    # Parametros fijos del modelo (UNIFICADOS con tab Viabilidad Financiera)
    params = {
        "precio_energia_cop_kwh": _DEFAULT_PRECIO_ENERGIA,
        "precio_carne_cop_kg":    _DEFAULT_PRECIO_CARNE,
        "ugg_agro":               _DEFAULT_UGG_AGRO,
        "ugg_tradicional":        _DEFAULT_UGG_TRAD,
        "ha_por_mw":              _HA_POR_MW,
        "tarifa_agua_default":    _DEFAULT_TARIFA_AGUA,
    }
    # Alias locales para usar en f-strings de la calculadora
    precio_energia      = params["precio_energia_cop_kwh"]
    precio_carne        = params["precio_carne_cop_kg"]
    ugg_agro            = params["ugg_agro"]
    ugg_tradicional     = params["ugg_tradicional"]
    ha_por_mw           = params["ha_por_mw"]
    tarifa_agua_default = params["tarifa_agua_default"]

    agro = _calcular_agrovoltaico(df, params)
    agro_viable = agro[agro["ha_viable"] > 0].copy()
    # Ranking ponderado por score multidimensional (calidad x oportunidad/ha)
    agro_top = agro_viable.sort_values("ingreso_efectivo_por_ha", ascending=False).reset_index(drop=True)

    # ---- Metricas resumen (solo municipios con ha viable > 0) ----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Municipios viables", f"{len(agro_viable):,}")
    if not agro_top.empty:
        lider = agro_top.iloc[0]
        m2.metric(
            "Municipio lider (calidad × ingreso/ha)",
            lider["municipio"],
            f"{lider['ingreso_agro_m_cop_por_ha']:,.2f} M COP/ha/año (score {pd.to_numeric(lider['v_i_multidimensional'], errors='coerce'):.3f})",
            help="Ranking ponderado por score multidimensional para evitar que municipios grandes pero de baja calidad dominen.",
        )
    m3.metric(
        "Hectareas viables (mediana)",
        f"{agro_viable['ha_viable'].median():,.0f} ha" if not agro_viable.empty else "—",
    )
    m4.metric(
        "Ganancia media vs solo ganadero",
        f"+{agro_viable['ganancia_vs_puro_ganadero_m_cop'].median():,.0f} M COP/año" if not agro_viable.empty else "—",
    )

    st.caption(
        f"Parametros (mismos en todas las pestañas): {params['precio_energia_cop_kwh']} COP/kWh · {params['precio_carne_cop_kg']:,} COP/kg carne · "
        f"{params['ugg_agro']} UGG/ha agrivoltaico · {params['ugg_tradicional']} UGG/ha tradicional · "
        f"{params['ha_por_mw']} ha/MW · {_KG_CARNE_UGG_AÑO:.0f} kg/UGG/año · {_AGUA_M3_POR_MWH} m³/MWh. "
        "El ranking se ordena por **ingreso/ha × score multidimensional** — asi un municipio enorme pero sin red electrica no domina solo por area."
    )

    st.divider()

    # ---- Selector de municipio (controla grafica + calculadora) ----
    st.markdown("### Analizar municipio")
    mun_options = agro_viable.sort_values("ingreso_efectivo_por_ha", ascending=False)["municipio"].tolist()
    if not mun_options:
        st.warning("No hay municipios con hectareas viables calculadas.")
        return

    mun_sel = st.selectbox(
        "Selecciona un municipio para ver su comparacion y desglose",
        mun_options,
        index=0,
        key="agro_mun_sel",
    )
    row_sel = agro_viable[agro_viable["municipio"] == mun_sel].iloc[0]

    # ---- Grafica: 3 estrategias para el municipio seleccionado ----
    i_gan  = float(row_sel["ingreso_carne_puro_m_cop"])
    i_sol  = float(row_sel["ingreso_puro_solar_m_cop"])
    i_agro = float(row_sel["ingreso_total_agro_m_cop"])

    fig2 = go.Figure()
    estrategias = [
        ("Solo ganaderia trad.", i_gan,  "#DC2626"),
        ("Solo solar",           i_sol,  "#2563EB"),
        ("Agrivoltaico",         i_agro, "#16A34A"),
    ]
    for nombre, val, color in estrategias:
        fig2.add_trace(go.Bar(
            name=nombre,
            x=[nombre],
            y=[val],
            marker_color=color,
            text=[f"{val:,.0f} M COP/año"],
            textposition="outside",
            textfont=dict(color="#111827", size=14, family="Arial"),
            hovertemplate=f"<b>{nombre}</b><br>%{{y:,.1f}} M COP/año<extra></extra>",
            width=0.45,
        ))
    fig2.update_layout(
        barmode="group", height=400,
        showlegend=False,
        xaxis=dict(tickfont=dict(size=14, color="#111827", family="Arial")),
        yaxis=dict(title="Ingreso anual (Millones COP)", tickfont=dict(size=13, color="#111827")),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=20, t=20, b=40),
    )
    _apply_readable_fonts(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    # ---- Calculadora: desglose paso a paso ----
    st.divider()
    st.markdown("### Por que el agrivoltaico da ese resultado")

    ha   = row_sel["ha_viable"]
    mw   = row_sel["mw_solar"]
    mwh  = row_sel["mwh_año"]
    bov_agro = row_sel["bovinos_agro"]
    bov_puro = row_sel["bovinos_puro"]
    i_sol  = row_sel["ingreso_solar_m_cop"]
    i_carn = row_sel["ingreso_carne_agro_m_cop"]
    c_agua = row_sel["costo_agua_m_cop"]
    i_tot  = row_sel["ingreso_total_agro_m_cop"]
    i_sol_puro  = row_sel["ingreso_puro_solar_m_cop"]
    i_gan_puro  = row_sel["ingreso_carne_puro_m_cop"]
    g_vs_gan    = row_sel["ganancia_vs_puro_ganadero_m_cop"]
    g_vs_sol    = row_sel["ganancia_vs_puro_solar_m_cop"]
    pvout_val   = float(pd.to_numeric(row_sel.get("pvout_kwh_kwp_day", 0), errors="coerce") or 0)
    precio_tierra = float(pd.to_numeric(row_sel.get("precio_tierra_ha_cop", 0), errors="coerce") or 0)
    tarifa_agua_r = float(pd.to_numeric(row_sel.get("tarifa_acueducto_m3_cop", 0), errors="coerce") or tarifa_agua_default)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Paso 1 — Area viable para el proyecto")
        area_total      = float(pd.to_numeric(row_sel.get("area_km2_igac", 0), errors="coerce") or 0)
        area_no_runap   = float(pd.to_numeric(row_sel.get("area_no_protegida_km2_runap", 0), errors="coerce") or 0)
        if area_no_runap == 0 and area_total > 0:
            area_no_runap = area_total * (1 - float(pd.to_numeric(row_sel.get("pct_area_protegida_runap", 0), errors="coerce") or 0))
        area_descontada = area_no_runap * _FACTOR_USO_NO_SOLAR
        area_dispon_km2 = area_no_runap - area_descontada
        score_pend_v    = float(pd.to_numeric(row_sel.get("score_pendiente", 1), errors="coerce") or 0)
        st.markdown(f"""
| Concepto | Valor |
|---|---|
| Area total municipio | `{area_total:,.1f} km²` |
| (−) Area protegida RUNAP | `{area_total - area_no_runap:,.1f} km²` |
| Area no-RUNAP | `{area_no_runap:,.1f} km²` |
| (−) Urbano ({_PCT_URBANO*100:.0f}%) | `−{area_no_runap*_PCT_URBANO:,.1f} km²` |
| (−) Vias + buffer ({_PCT_VIAS*100:.0f}%) | `−{area_no_runap*_PCT_VIAS:,.1f} km²` |
| (−) Hidrico + ronda 30m ({_PCT_HIDRICO*100:.0f}%) | `−{area_no_runap*_PCT_HIDRICO:,.1f} km²` |
| (−) Cultivos activos ({_PCT_CULTIVADO*100:.0f}%) | `−{area_no_runap*_PCT_CULTIVADO:,.1f} km²` |
| Area disponible (km²) | `{area_dispon_km2:,.1f} km²` |
| × Factor pendiente | `× {score_pend_v:.2f}` |
| **Hectareas viables** | **`{ha:,.0f} ha`** |
""")
        st.caption(
            f"Ha viable = area_no_RUNAP × (1 − {_FACTOR_USO_NO_SOLAR:.2f}) × 100 × score_pendiente. "
            f"Descuentos por uso del suelo: IDEAM coberturas + IGAC + UPRA frontera agricola + Dec 2811/74."
        )

        st.markdown("#### Paso 2 — Potencial solar")
        st.markdown(f"""
| Concepto | Calculo | Resultado |
|---|---|---|
| Capacidad instalable | {ha:,.0f} ha ÷ {ha_por_mw} ha/MW | **{mw:,.1f} MW** |
| Rendimiento PVOUT | {pvout_val:.3f} kWh/kWp/dia | — |
| Generacion anual | {mw:,.1f} MW × {pvout_val:.3f} × 365 × 1000 | **{mwh:,.0f} MWh/año** |
| **Ingreso solar** | {mwh:,.0f} MWh × 1,000 kWh/MWh × {precio_energia} COP/kWh | **{i_sol:,.1f} M COP/año** |
""")
        st.caption(f"PVOUT fuente: Solargis. Precio energia: {precio_energia} COP/kWh (ajustable).")

        st.markdown("#### Paso 3 — Costo operativo (agua de lavado)")
        agua_m3 = row_sel["agua_lavado_m3_año"]
        st.markdown(f"""
| Concepto | Calculo | Resultado |
|---|---|---|
| Agua necesaria | {mwh:,.0f} MWh × {_AGUA_M3_POR_MWH} m³/MWh | **{agua_m3:,.0f} m³/año** |
| Tarifa acueducto | Real o default | **{tarifa_agua_r:,.0f} COP/m³** |
| **Costo agua** | {agua_m3:,.0f} m³ × {tarifa_agua_r:,.0f} COP | **{c_agua:.3f} M COP/año** |
""")
        st.caption("Fuente tarifa: SUI. Si no hay dato real se usa el default del slider.")

    with right:
        st.markdown("#### Paso 4 — Bovinos bajo los paneles")
        st.markdown(f"""
| Concepto | Calculo | Resultado |
|---|---|---|
| Carga agrivoltaica | {ugg_agro} UGG/ha | — |
| **Bovinos bajo paneles** | {ha:,.0f} ha × {ugg_agro} UGG/ha | **{bov_agro:,.0f} bovinos** |
| Produccion por animal | {_KG_CARNE_UGG_AÑO:.0f} kg carne viva/UGG/año | — |
| **Ingreso ganadero** | {bov_agro:,.0f} × {_KG_CARNE_UGG_AÑO:.0f} kg × {precio_carne:,} COP | **{i_carn:,.1f} M COP/año** |
""")
        st.caption(
            f"Los paneles generan sombra parcial (40-60%). Segun AGROSAVIA, el pasto bajo sombra "
            f"mantiene {ugg_agro} UGG/ha con suplementacion vs {ugg_tradicional} UGG/ha en pastoreo "
            f"tradicional abierto."
        )

        st.markdown("#### Paso 5 — Total agrivoltaico y comparacion")
        st.markdown(f"""
| Estrategia | Calculo | Ingreso anual |
|---|---|---|
| Solo ganaderia ({ugg_tradicional} UGG/ha) | {bov_puro:,.0f} bovinos × {_KG_CARNE_UGG_AÑO:.0f} kg × {precio_carne:,} COP | **{i_gan_puro:,.1f} M COP** |
| Solo solar | {i_sol:,.1f} M − {c_agua:.3f} M agua | **{i_sol_puro:,.1f} M COP** |
| **Agrivoltaico** | {i_sol:,.1f} M solar + {i_carn:,.1f} M ganado − {c_agua:.3f} M agua | **{i_tot:,.1f} M COP** |
""")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric(
            "Solo ganado",
            f"{i_gan_puro:,.1f} M COP/año",
            help=f"{bov_puro:,.0f} bovinos × {_KG_CARNE_UGG_AÑO:.0f} kg × {precio_carne:,} COP",
        )
        col_b.metric(
            "Solo solar",
            f"{i_sol_puro:,.1f} M COP/año",
            help=f"{mwh:,.0f} MWh × {precio_energia} COP/kWh − {c_agua:.2f} M COP agua",
        )
        col_c.metric(
            "Juntos (agrivoltaico)",
            f"{i_tot:,.1f} M COP/año",
            f"+{max(g_vs_gan, g_vs_sol):,.1f} M COP/año vs mejor estrategia individual",
            delta_color="normal",
        )

        st.markdown("#### Por que funciona esta combinacion")
        razon_solar   = "buena irradiacion solar" if pvout_val >= 4.5 else "irradiacion solar moderada"
        razon_tierra  = f"precio de tierra {'bajo' if precio_tierra < 5_000_000 else 'moderado'} ({precio_tierra:,.0f} COP/ha)" if precio_tierra > 0 else "costo de tierra disponible"
        razon_bovinos = f"alta densidad ganadera posible ({ugg_agro} UGG/ha) gracias a la sombra parcial de los paneles"
        st.info(
            f"**{mun_sel}** combina: **{razon_solar}** (PVOUT {pvout_val:.2f} kWh/kWp/dia), "
            f"**{razon_tierra}**, y **{razon_bovinos}**. "
            f"El agrivoltaico permite generar {mwh:,.0f} MWh de electricidad al año y mantener "
            f"{bov_agro:,.0f} bovinos simultaneamente en las mismas {ha:,.0f} ha — algo imposible "
            f"en un modelo de uso exclusivo de suelo."
        )

    st.caption(
        f"**Supuestos:** {_KG_CARNE_UGG_AÑO:.0f} kg carne viva/UGG/año · "
        f"{_AGUA_M3_POR_MWH} m³ agua/MWh (NREL) · {ha_por_mw} ha/MW (NREL) · "
        "Ha viable = area no protegida RUNAP × score_pendiente. "
        "Fuentes: NREL, AGROSAVIA, FEDEGAN, UPME, UPRA, SUI."
    )


# ---------------------------------------------------------------------------
# Pestaña viabilidad financiera
# ---------------------------------------------------------------------------
# ----- Descuentos de uso del suelo (sobre area no protegida por RUNAP) -----
# Promedios nacionales — IDEAM coberturas terrestres, IGAC, UPRA frontera agricola.
# Estos descuentos se RESTAN del area no-RUNAP para reflejar que no todo el
# territorio "no protegido" es usable para utility-scale solar.
_PCT_URBANO           = 0.02   # 2% asentamientos + casco urbano POT (DANE/IGAC)
_PCT_VIAS             = 0.03   # 3% red vial + buffer ROW 50m (INVIAS)
_PCT_HIDRICO          = 0.08   # 8% cuerpos de agua + ronda hidrica 30m (Decreto 2811/74)
_PCT_CULTIVADO        = 0.15   # 15% frontera agricola activa (UPRA)
_FACTOR_USO_NO_SOLAR  = _PCT_URBANO + _PCT_VIAS + _PCT_HIDRICO + _PCT_CULTIVADO  # 0.28

_PANEL_W             = 600        # Wp por modulo (bifacial Tier-1 2024 utility-scale)
_PANEL_COST_COP      = 420_000    # COP por modulo (USD 105/panel a 4,000 COP/USD; NREL Q1-2024 USD 0.21/W)
_PANEL_LIFE_YEARS    = 25         # vida util de los modulos (garantia fabricante)
_BOS_M_COP_POR_MW    = 2_900.0    # M COP/MW: resto del sistema (inversores, estructura, BOS, instalacion, soft)
_BOS_LIFE_YEARS      = 25         # se amortiza igual que los paneles
_CAPEX_M_COP_POR_MW  = 3_600.0    # paneles + BOS = ~USD 900/kW (suma teorica de 700 paneles + 2,900 BOS)
_VIDA_UTIL_AÑOS      = 25         # vida util default del proyecto
_PRESTACIONES        = 1.52       # factor prestaciones sociales Colombia
_SMMLV_MENSUAL_COP   = 1_300_000  # SMMLV 2024
_DEGRAD_PANEL_AÑO    = 0.005      # 0.5%/año degradacion lineal paneles c-Si (NREL)
_IMPUESTO_RENTA_PCT  = 0.30       # 30% impuesto renta corporativa Colombia (Ley 2277/2022)


def _factor_degradacion_promedio(vida_util: int) -> float:
    """Factor promedio de produccion considerando degradacion lineal de 0.5%/año.

    Año 1: 100%, año 2: 99.5%, ... año 25: 88%. Promedio = 1 - (vida_util-1)/2 * deg_anual.
    """
    return max(0.0, 1.0 - (vida_util - 1) / 2 * _DEGRAD_PANEL_AÑO)


def _paneles_por_ha(ha_por_mw: float) -> float:
    """Paneles necesarios por hectarea, derivado de Wp por panel y densidad NREL."""
    paneles_por_mw = 1_000_000 / _PANEL_W           # 1,667 paneles/MW para 600 Wp
    return paneles_por_mw / max(ha_por_mw, 0.01)    # ~825 paneles/ha para 2.02 ha/MW


def _viabilidad_financiera(row: "pd.Series", params: dict) -> dict:
    """Calcula P&L completo de una granja agrivoltaica en un municipio.

    El tamaño de la planta se controla con `ha_paneles_max`: se instalan paneles
    sobre min(ha_paneles_max, ha_viable). El ganado bovino solo se cuenta dentro
    del area de paneles (proyecto agrivoltaico), no en toda el area viable.
    """
    def _safe(val, default: float = 0.0) -> float:
        """Convierte a float ignorando NaN. Tolera Series (toma primer elemento)."""
        if isinstance(val, pd.Series):
            val = val.iloc[0] if len(val) > 0 else default
        v = pd.to_numeric(val, errors="coerce")
        if isinstance(v, pd.Series):
            v = v.iloc[0] if len(v) > 0 else float("nan")
        return float(v) if pd.notna(v) else float(default)

    ha_viable_total = _safe(row.get("ha_viable", 0), 0.0)
    ha_paneles_max  = float(params.get("ha_paneles_max", ha_viable_total))
    ha = min(ha_paneles_max, ha_viable_total)   # ha efectivas con paneles
    mw   = ha / params["ha_por_mw"] if params["ha_por_mw"] else 0
    pvout = _safe(row.get("pvout_kwh_kwp_day"), 4.5)
    # Generacion teorica (año 1 al 100%); aplicamos degradacion promedio sobre la vida del proyecto
    factor_deg = _factor_degradacion_promedio(int(params["vida_util"]))
    mwh  = mw * pvout * 365 * factor_deg

    precio_tierra = _safe(row.get("precio_tierra_ha_cop"), 0.0)
    tarifa_agua   = _safe(row.get("tarifa_acueducto_m3_cop"), 0.0)
    if tarifa_agua <= 0:
        tarifa_agua = float(params["tarifa_agua_default"])

    # ---------- INGRESOS ----------
    # mwh esta en MWh, precio_energia en COP/kWh -> convertir MWh a kWh (x1000)
    i_solar  = mwh * 1000 * params["precio_energia"] / 1e6
    bovinos  = ha * params["ugg_agro"]
    i_ganado = bovinos * _KG_CARNE_UGG_AÑO * params["precio_carne"] / 1e6
    i_total  = i_solar + i_ganado

    # ---------- COSTOS OPERATIVOS ----------
    agua_m3      = mwh * _AGUA_M3_POR_MWH
    c_agua       = agua_m3 * tarifa_agua / 1e6
    c_oym_solar  = params["oym_pct"] / 100 * _CAPEX_M_COP_POR_MW * mw      # % del CAPEX paneles
    n_emp_om     = max(1.0, params["om_per_mw"] * mw)                        # minimo 1 empleado
    c_empleados  = n_emp_om * params["salario_mensual"] * 12 * _PRESTACIONES / 1e6
    c_ganadero   = params["costo_ganadero_pct"] / 100 * i_ganado             # vet, suplementacion
    c_operativos = c_agua + c_oym_solar + c_empleados + c_ganadero

    ebitda = i_total - c_operativos

    # ---------- AMORTIZACION (CAPEX) ----------
    # Paneles: costo unitario × paneles/ha × ha, amortizado segun vida del modulo
    paneles_por_ha = _paneles_por_ha(params["ha_por_mw"])
    paneles_total  = paneles_por_ha * ha
    capex_paneles  = paneles_total * _PANEL_COST_COP / 1e6                        # solo modulos
    amort_paneles  = capex_paneles / _PANEL_LIFE_YEARS                            # M COP/año
    # BOS (inversores, estructura, BOS electrico, instalacion, soft costs)
    capex_bos      = _BOS_M_COP_POR_MW * mw
    amort_bos      = capex_bos / _BOS_LIFE_YEARS
    # Tierra
    capex_tierra   = precio_tierra * ha / 1e6 if precio_tierra > 0 else 0
    amort_tierra   = capex_tierra / params["vida_util"]
    capex_total    = capex_paneles + capex_bos + capex_tierra
    amort_anual    = amort_paneles + amort_bos + amort_tierra

    ebit  = ebitda - amort_anual
    impuesto_renta = max(ebit, 0) * _IMPUESTO_RENTA_PCT   # solo se paga si EBIT > 0
    utilidad_neta  = ebit - impuesto_renta
    payback = capex_total / utilidad_neta if utilidad_neta > 0 else float("inf")
    roi_pct = utilidad_neta / capex_total * 100 if capex_total > 0 else 0

    # ---------- MINIMO VIABLE ----------
    # Encontrar ha donde EBIT = 0: lineal, despejar
    # i_total(ha) - c_op(ha) - amort(ha) = 0
    # Todos los terminos escalan con ha o mw(ha) → lineal en ha
    # EBIT/ha = (i_total - c_op - amort) / ha  cuando ha > 0
    ebit_por_ha = ebit / ha if ha > 0 else 0
    ha_min_viable = -ebit / ebit_por_ha + ha if ebit_por_ha != 0 and ebit < 0 else (
        ha * 0.1 if ebit > 0 else float("inf")
    )

    # Empleos
    emp_om_directos  = params["om_per_mw"] * mw
    emp_om_total     = params["om_total_per_mw"] * mw
    emp_construccion = 2.028 * mw   # job-years temporales (IRENA)

    return {
        "ha": ha, "ha_viable_total": ha_viable_total,
        "mw": mw, "mwh": mwh, "bovinos": bovinos,
        "i_solar": i_solar, "i_ganado": i_ganado, "i_total": i_total,
        "c_agua": c_agua, "c_oym_solar": c_oym_solar,
        "c_empleados": c_empleados, "c_ganadero": c_ganadero,
        "c_operativos": c_operativos, "ebitda": ebitda,
        "paneles_total": paneles_total, "paneles_por_ha": paneles_por_ha,
        "capex_paneles": capex_paneles, "capex_bos": capex_bos,
        "capex_tierra": capex_tierra, "capex_total": capex_total,
        "amort_paneles": amort_paneles, "amort_bos": amort_bos,
        "amort_tierra": amort_tierra, "amort_anual": amort_anual,
        "ebit": ebit, "impuesto_renta": impuesto_renta,
        "utilidad_neta": utilidad_neta, "factor_degradacion": factor_deg,
        "payback": payback, "roi_pct": roi_pct,
        "n_emp_om": n_emp_om,
        "emp_om_directos": emp_om_directos,
        "emp_om_total": emp_om_total,
        "emp_construccion": emp_construccion,
        "ha_min_viable": ha_min_viable,
        "precio_tierra": precio_tierra,
    }


def render_viabilidad_financiera_tab(df: pd.DataFrame, top5: pd.DataFrame) -> None:
    import plotly.graph_objects as go

    st.subheader("Viabilidad financiera — P&L de la granja agrivoltaica")
    st.caption(
        "Calcula el estado de resultados completo: ingresos solar + ganadero, "
        "costos operativos, amortizacion de paneles y tierra, empleados, payback y ROI."
    )

    # ---- 1. Selector de municipio (PRIMERO — carga los parametros reales automaticamente) ----
    all_muns = sorted(df["municipio"].dropna().astype(str).unique().tolist())
    top5_muns = top5["municipio"].tolist()
    mun_ordered = top5_muns + [m for m in all_muns if m not in top5_muns]

    def _on_mun_change() -> None:
        mun = st.session_state.get("fin_mun_sel", "")
        matches = df[df["municipio"] == mun]
        if matches.empty:
            return
        row = matches.iloc[0]
        tarifa = float(pd.to_numeric(row.get("tarifa_acueducto_m3_cop", 0), errors="coerce") or 0)
        if 500 <= tarifa <= 5000:
            st.session_state["fin_s_tarifa"] = int(tarifa)
        ugg_val = 0.0
        for col in ["carga_bovina_ua_ha", "ugg_ha_proxy"]:
            v = float(pd.to_numeric(row.get(col, 0), errors="coerce") or 0)
            if v > 0:
                ugg_val = v
                break
        if ugg_val > 0:
            st.session_state["fin_s_ugg"] = round(min(4.0, max(1.0, ugg_val * 1.3)), 1)

    mun_sel = st.selectbox(
        "Municipio a analizar",
        mun_ordered,
        index=0,
        key="fin_mun_sel",
        on_change=_on_mun_change,
        help="Selecciona un municipio. La tarifa de agua y la carga ganadera se actualizan automaticamente con los datos reales del municipio.",
    )

    # Mostrar badge con los datos reales cargados
    row_mun_raw = df[df["municipio"] == mun_sel]
    if not row_mun_raw.empty:
        rm = row_mun_raw.iloc[0]
        tarifa_real      = float(pd.to_numeric(rm.get("tarifa_acueducto_m3_cop", 0), errors="coerce") or 0)
        pvout_real       = float(pd.to_numeric(rm.get("pvout_kwh_kwp_day", 0), errors="coerce") or 0)
        precio_tierra_r  = float(pd.to_numeric(rm.get("precio_tierra_ha_cop", 0), errors="coerce") or 0)
        ugg_real = 0.0
        for col in ["carga_bovina_ua_ha", "ugg_ha_proxy"]:
            v = float(pd.to_numeric(rm.get(col, 0), errors="coerce") or 0)
            if v > 0:
                ugg_real = v
                break
        datos_ok = []
        if tarifa_real > 0:
            datos_ok.append(f"tarifa agua: **{tarifa_real:,.0f} COP/m³**")
        if pvout_real > 0:
            datos_ok.append(f"PVOUT: **{pvout_real:.3f} kWh/kWp/dia**")
        if ugg_real > 0:
            ugg_sug = round(min(4.0, max(1.0, ugg_real * 1.3)), 1)
            datos_ok.append(f"carga ganadera: **{ugg_real:.2f} UGG/ha** → agrivoltaico sugerido: **{ugg_sug} UGG/ha**")
        if precio_tierra_r > 0:
            datos_ok.append(f"precio tierra: **{precio_tierra_r:,.0f} COP/ha**")
        if datos_ok:
            st.success("📊 Datos reales cargados de **" + mun_sel + "**: " + " · ".join(datos_ok))
        else:
            st.info(f"ℹ️ {mun_sel}: sin datos especificos disponibles — usando valores de referencia nacional.")

    # ---- 2. Parametros financieros ----
    with st.expander("Parametros financieros (ajustables)", expanded=True):
        st.caption("Los valores marcados con * se inicializan con datos reales del municipio seleccionado.")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        precio_energia  = r1c1.slider(
            "Precio energia (COP/kWh)", 100, 500, _DEFAULT_PRECIO_ENERGIA, 10,
            help=(
                "Precio promedio de venta de energia. "
                "Subastas CREG FNCER 2024: 140-200. PPA bilateral solar: 200-280. "
                "Bolsa (spot): 300-450. Para utility-scale realista usar 180-220."
            ),
            key="fin_s_precio_e",
        )
        precio_carne    = r1c2.slider("Precio carne viva (COP/kg)", 6_000, 14_000, _DEFAULT_PRECIO_CARNE, 500, key="fin_s_carne")
        ugg_agro        = r1c3.slider("Carga agrivoltaica (UGG/ha) *", 1.0, 4.0, _DEFAULT_UGG_AGRO, 0.1, key="fin_s_ugg")
        ha_por_mw       = r1c4.slider("Ha por MW solar", 1.5, 3.0, _HA_POR_MW, 0.01, key="fin_s_hamw")

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        vida_util       = r2c1.slider("Vida util proyecto (años)", 20, 30, 25, 1, key="fin_s_vida")
        oym_pct         = r2c2.slider("O&M anual (% CAPEX paneles)", 0.5, 3.0, 1.5, 0.1,
                                       help="Costo operacion y mantenimiento solar. Tipico: 1-2% del CAPEX/año.",
                                       key="fin_s_oym")
        salario_mensual = r2c3.slider("Salario mensual empleado (COP)", 1_300_000, 6_000_000, 2_600_000, 100_000,
                                       help="Incluye todas las prestaciones (factor x1.52 aplicado automaticamente).",
                                       key="fin_s_salario")
        costo_ganadero_pct = r2c4.slider("Costo operativo ganadero (% ingreso carne)", 20, 60, 35, 5,
                                          help="Veterinaria, suplementacion mineral, jornales ganaderos.",
                                          key="fin_s_cgano")

        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        om_per_mw       = r3c1.slider("Empleados O&M directos / MW", 0.05, 0.5, 0.142, 0.001,
                                       help="SEIA 2025: 0.142 empleados permanentes por MW instalado.",
                                       key="fin_s_ompm")
        om_total_per_mw = r3c2.slider("Empleados O&M totales / MW", 0.1, 0.8, 0.262, 0.001,
                                       help="Directo + indirecto + inducido (IRENA).",
                                       key="fin_s_omtot")
        tarifa_agua_default = r3c3.slider("Tarifa agua default (COP/m³) *", 500, 5_000, _DEFAULT_TARIFA_AGUA, 100,
                                           key="fin_s_tarifa")
        ha_paneles_max  = r3c4.slider(
            "Hectareas con paneles solares", 10, 5_000, 500, 10,
            help=(
                "Tamano del proyecto: cuantas hectareas se cubren con paneles. "
                "El ganado se cuenta solo dentro de esta area. "
                "Referencia: 500 ha ~ 247 MW (utility-scale Colombia tipico 100-500 MW)."
            ),
            key="fin_s_hapan",
        )
        st.caption(
            f"Equivalente del slider de paneles: **{ha_paneles_max:,} ha = ~{ha_paneles_max/ha_por_mw:,.1f} MW**  ·  "
            f"Bovinos esperados bajo paneles: **{ha_paneles_max * ugg_agro:,.0f}** "
            f"({ha_paneles_max:,} ha x {ugg_agro} UGG/ha). "
            f"Si el municipio tiene menos area viable, se usa el tope del municipio."
        )

    params = dict(
        precio_energia=precio_energia, precio_carne=precio_carne,
        ugg_agro=ugg_agro, ha_por_mw=ha_por_mw, vida_util=vida_util,
        oym_pct=oym_pct, salario_mensual=salario_mensual,
        costo_ganadero_pct=costo_ganadero_pct,
        om_per_mw=om_per_mw, om_total_per_mw=om_total_per_mw,
        tarifa_agua_default=tarifa_agua_default,
        ha_paneles_max=ha_paneles_max,
    )

    # Calcular agrivoltaico base para obtener ha_viable por municipio
    agro_base = _calcular_agrovoltaico(df, {
        "precio_energia_cop_kwh": precio_energia,
        "precio_carne_cop_kg": precio_carne,
        "ugg_agro": ugg_agro,
        "ugg_tradicional": 1.5,
        "ha_por_mw": ha_por_mw,
        "tarifa_agua_default": tarifa_agua_default,
    })
    # Dedupe por municipio (puede haber nombres repetidos en distintos departamentos)
    agro_base = agro_base.drop_duplicates(subset=["municipio"]).set_index("municipio")

    if mun_sel not in agro_base.index:
        st.warning("Municipio sin datos suficientes.")
        return

    # Combinar datos del CSV con los del calculo agro — siempre como Series escalar
    row_agro_raw = agro_base.loc[mun_sel]
    row_agro = row_agro_raw.iloc[0] if isinstance(row_agro_raw, pd.DataFrame) else row_agro_raw
    matches_orig = df[df["municipio"] == mun_sel]
    row_orig = matches_orig.iloc[0] if not matches_orig.empty else row_agro
    row_combined = row_orig.copy()
    row_combined["ha_viable"] = float(pd.to_numeric(row_agro["ha_viable"], errors="coerce") or 0)

    f = _viabilidad_financiera(row_combined, params)

    # ---- Metricas principales ----
    st.markdown(f"## {mun_sel} ({row_orig.get('departamento','—')})")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    delta_ha = (
        f"de {f['ha_viable_total']:,.0f} ha viables del municipio"
        if f["ha_viable_total"] > f["ha"]
        else "(usando todo el viable)"
    )
    mc1.metric("Ha del proyecto solar", f"{f['ha']:,.0f} ha", delta_ha, delta_color="off")
    mc2.metric("Capacidad solar", f"{f['mw']:,.1f} MW")
    mc3.metric("Generacion anual", f"{f['mwh']:,.0f} MWh/año")
    mc4.metric("Bovinos bajo paneles", f"{f['bovinos']:,.0f}")
    mc5.metric(
        "Empleados O&M permanentes (total)",
        f"{f['n_emp_om']:.0f}",
        help=f"Total para toda la planta (no por hectarea). {f['n_emp_om']/max(f['mw'],1):.3f} empleados/MW segun SEIA.",
    )

    st.divider()

    col_charts, col_info = st.columns([3, 2])

    # ---- Grafica 1: Waterfall P&L (POR HECTAREA POR AÑO) ----
    with col_charts:
        st.markdown("#### Estado de resultados (M COP / hectarea / año)")
        st.caption(
            f"Valores normalizados por hectarea para comparar municipios. "
            f"Multiplica cualquier valor por **{f['ha']:,.0f} ha** para obtener el total anual del proyecto."
        )
        ha_div = max(f["ha"], 1.0)   # evita division por cero
        items_wf = [
            ("Ingreso solar",        f["i_solar"]        / ha_div, "relative"),
            ("Ingreso ganadero",     f["i_ganado"]       / ha_div, "relative"),
            ("- Costo agua",        -f["c_agua"]         / ha_div, "relative"),
            ("- O&M solar",         -f["c_oym_solar"]    / ha_div, "relative"),
            ("- Empleados",         -f["c_empleados"]    / ha_div, "relative"),
            ("- Costo ganadero",    -f["c_ganadero"]     / ha_div, "relative"),
            ("EBITDA",               f["ebitda"]         / ha_div, "total"),
            ("- Amort. paneles",    -f["amort_paneles"]  / ha_div, "relative"),
            ("- Amort. BOS",        -f["amort_bos"]      / ha_div, "relative"),
            ("- Amort. tierra",     -f["amort_tierra"]   / ha_div, "relative"),
            ("EBIT",                 f["ebit"]           / ha_div, "total"),
            ("- Impuesto renta 30%",-f["impuesto_renta"] / ha_div, "relative"),
            ("Utilidad neta",        f["utilidad_neta"]  / ha_div, "total"),
        ]
        fig_wf = go.Figure(go.Waterfall(
            name="P&L",
            orientation="v",
            measure=[m for _, _, m in items_wf],
            x=[n for n, _, _ in items_wf],
            y=[v for _, v, _ in items_wf],
            text=[f"{v:+,.1f}" for _, v, _ in items_wf],
            textposition="outside",
            textfont=dict(size=12, color="#111827"),
            connector=dict(line=dict(color="#9CA3AF", width=1)),
            increasing=dict(marker_color="#16A34A"),
            decreasing=dict(marker_color="#DC2626"),
            totals=dict(marker_color="#2563EB"),
        ))
        fig_wf.update_layout(
            height=420, plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="M COP / ha / año"),
            margin=dict(l=10, r=10, t=30, b=80),
            xaxis=dict(tickangle=-20),
        )
        _apply_readable_fonts(fig_wf)
        st.plotly_chart(fig_wf, use_container_width=True)

        # ---- Grafica 2: Curva de recuperacion de inversion ----
        st.markdown("#### Recuperacion de la inversion (flujo acumulado)")
        años = list(range(0, int(vida_util) + 1))
        # Flujo de caja anual = utilidad neta + amortizacion (es contable, no salida real de caja)
        flujo_caja_anual = f["utilidad_neta"] + f["amort_anual"]
        flujo_acum = [-f["capex_total"]] + [
            -f["capex_total"] + flujo_caja_anual * a for a in range(1, int(vida_util) + 1)
        ]
        fig_pay = go.Figure()
        fig_pay.add_trace(go.Scatter(
            x=años, y=flujo_acum,
            mode="lines+markers", line=dict(color="#2563EB", width=2.5),
            marker=dict(size=5),
            name="Flujo acumulado",
            hovertemplate="Año %{x}: %{y:,.0f} M COP<extra></extra>",
        ))
        fig_pay.add_hline(y=0, line=dict(color="#DC2626", dash="dash", width=2),
                          annotation_text="Punto de equilibrio", annotation_position="top right",
                          annotation_font=dict(color="#DC2626", size=12))
        if f["payback"] != float("inf") and f["payback"] <= vida_util:
            fig_pay.add_vline(x=f["payback"], line=dict(color="#D97706", dash="dot", width=2),
                              annotation_text=f"Payback: {f['payback']:.1f} años",
                              annotation_position="top left",
                              annotation_font=dict(color="#D97706", size=12))
        fig_pay.update_layout(
            height=300, plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="Año del proyecto"),
            yaxis=dict(title="M COP acumulados"),
            margin=dict(l=10, r=10, t=30, b=50),
            showlegend=False,
        )
        _apply_readable_fonts(fig_pay)
        st.plotly_chart(fig_pay, use_container_width=True)

    # ---- Panel derecho: tablas y empleos ----
    with col_info:
        # Viabilidad minima
        mw_min_viable = 10.0
        ha_min_viable = mw_min_viable * ha_por_mw
        status = "✅ VIABLE" if f["mw"] >= mw_min_viable and f["utilidad_neta"] > 0 else (
            "⚠️ MARGINAL" if f["utilidad_neta"] > 0 else "❌ NO VIABLE con estos parametros"
        )
        color_status = "#16A34A" if "VIABLE" in status and "NO" not in status else (
            "#D97706" if "MARGINAL" in status else "#DC2626"
        )
        st.markdown(
            f"<div style='background:{color_status}22;border-left:4px solid {color_status};"
            f"padding:12px;border-radius:6px;margin-bottom:12px'>"
            f"<b style='color:{color_status};font-size:16px'>{status}</b></div>",
            unsafe_allow_html=True,
        )

        st.markdown("**Inversion (CAPEX) y amortizacion**")
        st.caption(
            f"Calculo de paneles: **{_PANEL_COST_COP:,.0f} COP/panel** ({_PANEL_W} Wp) ÷ **{_PANEL_LIFE_YEARS} años** "
            f"× **{f['paneles_por_ha']:.0f} paneles/ha** × **{f['ha']:,.0f} ha** = "
            f"**{f['amort_paneles']:,.1f} M COP/año**"
        )
        payback_str = f"{f['payback']:.1f} años" if f['payback'] != float('inf') else "No recupera"
        st.markdown(f"""
| Concepto | Valor |
|---|---|
| Paneles instalados | `{f['paneles_total']:,.0f} unidades` ({_PANEL_W} Wp c/u) |
| CAPEX modulos | `{f['capex_paneles']:,.0f} M COP` |
| CAPEX BOS (inversor+estr+inst) | `{f['capex_bos']:,.0f} M COP` |
| CAPEX tierra | `{f['capex_tierra']:,.0f} M COP` |
| **CAPEX total** | **`{f['capex_total']:,.0f} M COP`** |
| — | — |
| Amort. paneles ({_PANEL_LIFE_YEARS} a) | `{f['amort_paneles']:,.1f} M COP/año` |
| Amort. BOS ({_BOS_LIFE_YEARS} a) | `{f['amort_bos']:,.1f} M COP/año` |
| Amort. tierra | `{f['amort_tierra']:,.1f} M COP/año` |
| **Amortizacion total/año** | **`{f['amort_anual']:,.1f} M COP/año`** |
| — | — |
| Ingreso solar (con deg. {(1-f['factor_degradacion'])*100:.1f}%) | `{f['i_solar']:,.1f} M COP/año` |
| Ingreso ganadero | `{f['i_ganado']:,.1f} M COP/año` |
| **Ingreso bruto** | **`{f['i_total']:,.1f} M COP/año`** |
| Total costos oper. | `{f['c_operativos']:,.1f} M COP/año` |
| **EBITDA** | **`{f['ebitda']:,.1f} M COP/año`** |
| **EBIT** | **`{f['ebit']:,.1f} M COP/año`** |
| Impuesto renta (30%) | `{f['impuesto_renta']:,.1f} M COP/año` |
| **Utilidad neta** | **`{f['utilidad_neta']:,.1f} M COP/año`** |
| **Payback (sobre util. neta)** | **`{payback_str}`** |
| **ROI neto** | **`{f['roi_pct']:.1f}%`** |
""")

        st.markdown("**Empleos generados**")
        st.markdown(f"""
| Tipo | Cantidad |
|---|---|
| O&M directos permanentes | `{f['emp_om_directos']:.1f}` |
| O&M totales (dir+ind+ind.) | `{f['emp_om_total']:.1f}` |
| Construccion (temporales) | `{f['emp_construccion']:.1f} job-years` |
| **Salario total anual** | **`{f['c_empleados']:,.1f} M COP/año`** |
""")
        st.caption("Fuente: SEIA 2025 (O&M) e IRENA (construccion). Incluye factor prestaciones ×1.52.")

        st.markdown("**Tamano minimo viable**")
        st.markdown(f"""
| Criterio | Valor |
|---|---|
| MW minimo (economias escala) | `{mw_min_viable:.0f} MW` |
| Hectareas minimas | `{ha_min_viable:.0f} ha` |
| Municipio tiene | `{f['mw']:.1f} MW` en `{f['ha']:,.0f} ha` |
| Cubre minimo? | {'✅ Si' if f['mw'] >= mw_min_viable else '❌ No'} |
""")
        st.caption(
            "El minimo de 10 MW corresponde al umbral tipico de viabilidad economica utility-scale "
            "en Colombia (UPME/CREG). Por debajo, el costo fijo por kW instalado se dispara."
        )

        st.markdown("**Desglose de costos operativos**")
        fig_pie = go.Figure(go.Pie(
            labels=["Agua lavado", "O&M solar", "Empleados", "Operativo ganadero"],
            values=[f["c_agua"], f["c_oym_solar"], f["c_empleados"], f["c_ganadero"]],
            hole=0.4,
            marker_colors=["#2563EB", "#16A34A", "#D97706", "#DC2626"],
            textfont=dict(size=13, color="#111827"),
        ))
        fig_pie.update_layout(
            height=260, margin=dict(l=0, r=0, t=10, b=10),
            legend=dict(font=dict(size=12, color="#111827")),
            paper_bgcolor="white",
        )
        _apply_readable_fonts(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ---- Comparacion Top 5 ----
    st.divider()
    st.markdown("### Comparacion financiera — Top 5 del ranking solar")
    filas_top5 = []
    for _, row_t in top5.iterrows():
        mun = row_t["municipio"]
        if mun not in agro_base.index:
            continue
        rc = row_t.copy()
        rc["ha_viable"] = agro_base.loc[mun, "ha_viable"]
        fv = _viabilidad_financiera(rc, params)
        filas_top5.append({
            "Municipio": f"{mun} ({row_t.get('departamento','—')})",
            "Ha viables": f"{fv['ha']:,.0f}",
            "MW solar": f"{fv['mw']:,.1f}",
            "Ingreso solar (M COP)": f"{fv['i_solar']:,.1f}",
            "Ingreso ganado (M COP)": f"{fv['i_ganado']:,.1f}",
            "EBITDA (M COP)": f"{fv['ebitda']:,.1f}",
            "EBIT (M COP)": f"{fv['ebit']:,.1f}",
            "Payback (años)": f"{fv['payback']:.1f}" if fv['payback'] != float('inf') else "—",
            "ROI (%)": f"{fv['roi_pct']:.1f}",
            "Emp. O&M": f"{fv['n_emp_om']:.1f}",
        })

    if filas_top5:
        df_comp = pd.DataFrame(filas_top5)

        muns_t5 = df_comp["Municipio"].tolist()
        fig_t5 = go.Figure()
        for col_v, color, nombre in [
            ("Ingreso solar (M COP)",  "#2563EB", "Ingreso solar"),
            ("Ingreso ganado (M COP)", "#16A34A", "Ingreso ganadero"),
        ]:
            vals = [float(v.replace(",","")) for v in df_comp[col_v]]
            fig_t5.add_trace(go.Bar(
                name=nombre, x=muns_t5, y=vals,
                marker_color=color,
                text=[f"{v:,.0f}" for v in vals],
                textposition="inside",
                textfont=dict(color="white", size=12),
                hovertemplate=f"<b>%{{x}}</b><br>{nombre}: %{{y:,.1f}} M COP<extra></extra>",
            ))
        # EBIT como linea
        ebit_vals = [float(v["EBIT (M COP)"].replace(",","")) for v in filas_top5]
        fig_t5.add_trace(go.Scatter(
            name="EBIT (util. oper.)", x=muns_t5, y=ebit_vals,
            mode="lines+markers+text",
            line=dict(color="#D97706", width=3, dash="dash"),
            marker=dict(size=10, color="#D97706"),
            text=[f"{v:,.0f}" for v in ebit_vals],
            textposition="top center",
            textfont=dict(size=12, color="#D97706"),
        ))
        fig_t5.update_layout(
            barmode="stack", height=420,
            xaxis=dict(tickfont=dict(size=11)),
            yaxis=dict(title="Millones COP / año"),
            legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=10, t=30, b=110),
        )
        _apply_readable_fonts(fig_t5)
        st.plotly_chart(fig_t5, use_container_width=True)

        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.caption(
        f"**CAPEX paneles asumido:** {_CAPEX_M_COP_POR_MW:,.0f} M COP/MW (~USD 900/kW a 4,000 COP/USD, 2024)  ·  "
        f"**Prestaciones:** factor {_PRESTACIONES}  ·  "
        "Fuentes: SEIA, IRENA, NREL, UPME, CREG, FEDEGAN."
    )


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

    tab_ranking, tab_raw, tab_agro, tab_fin = st.tabs([
        "Ranking",
        "Variables crudas por dimension",
        "Beneficio Agrivoltaico",
        "Viabilidad Financiera",
    ])

    with tab_raw:
        render_raw_tab(df, top5)

    with tab_agro:
        render_agrivoltaico_tab(df, top5)

    with tab_fin:
        render_viabilidad_financiera_tab(df, top5)

    with tab_ranking:
        render_ranking_tab(df, top5, numero1)

    st.caption(
        "Datos: NASA Power · IGAC · UPME · RUNAP · UPRA · EVA-ICA · IMRC DNP · SUI · INVIAS. "
        "Modelo: Jhon T — ITM Introduccion a la Inteligencia Artificial."
    )


if __name__ == "__main__":
    main()
