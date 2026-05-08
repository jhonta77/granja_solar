from __future__ import annotations

from pathlib import Path
from textwrap import shorten
import unicodedata

import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk
import streamlit as st

from src.db.mysql_cli import MySQLSettings, query_dataframe


PROJECT_ROOT = Path(__file__).resolve().parent
MYSQL_DATABASE = MySQLSettings.from_env().database
VIABILITY_PATH = PROJECT_ROOT / "data" / "clean" / "viabilidad_municipal" / "viabilidad_municipal_preliminar.csv"
CLUSTERS_PATH = PROJECT_ROOT / "data" / "clean" / "clusters_municipios" / "municipios_clusters_kmeans.csv"
CLUSTER_PROFILE_PATH = PROJECT_ROOT / "data" / "clean" / "clusters_municipios" / "perfil_clusters_kmeans.csv"
CLUSTER_EVALUATION_PATH = PROJECT_ROOT / "data" / "clean" / "clusters_municipios" / "evaluacion_kmeans_k.csv"
CLUSTER_SUMMARY_PATH = PROJECT_ROOT / "data" / "clean" / "clusters_municipios" / "resumen_seleccion_kmeans.csv"
SOLAR_COSTS_PATH = PROJECT_ROOT / "data" / "clean" / "solar_costs" / "solar_escenarios_por_hectarea.csv"
ENERGY_PRICES_PATH = (
    PROJECT_ROOT / "data" / "clean" / "energy_prices" / "precios_compra_energia_minenergia_caribe.csv"
)
EDA_VISUAL_DIR = PROJECT_ROOT / "data" / "clean" / "eda_visual_mysql"
EDA_MANIFEST_PATH = EDA_VISUAL_DIR / "manifest_eda_visual_mysql.csv"
EDA_VERIFICATION_PATH = EDA_VISUAL_DIR / "verificacion_posterior_limpieza.csv"
EDA_REPORT_PATH = EDA_VISUAL_DIR / "reporte_eda_visual_mysql.md"
EDA_WHATSAPP_DIR = PROJECT_ROOT / "data" / "clean" / "visualizaciones_municipios"
VIABILITY_SQL = """
SELECT
    v.*,
    c.cluster_kmeans,
    c.cluster_kmeans_label,
    c.silhouette_municipio
FROM viabilidad_municipal v
LEFT JOIN cluster_municipal c
    ON c.codigo_dane = v.codigo_dane
"""
CLUSTER_PROFILE_SQL = "SELECT * FROM cluster_perfiles"
CLUSTER_EVALUATION_SQL = "SELECT * FROM cluster_evaluacion_k"
CLUSTER_SUMMARY_SQL = "SELECT * FROM cluster_resumen_seleccion"
SOLAR_COSTS_SQL = "SELECT * FROM solar_escenarios"
ENERGY_PRICES_SQL = "SELECT * FROM precios_energia_departamento"
GRAPH_QUERY_SAMPLES = {
    "Ranking municipal base": """
SELECT *
FROM vw_dashboard_municipal
ORDER BY v_i_modelo_rural DESC
LIMIT 50;
""".strip(),
    "Contexto tecnico municipal": """
SELECT *
FROM vw_municipal_contexto
WHERE departamento = 'Santander'
ORDER BY pvout_kwh_kwp_day DESC;
""".strip(),
    "Relacion solar-red": """
SELECT codigo_dane, municipio, departamento, pvout_kwh_kwp_day, dist_subestacion_km, v_i_modelo_rural
FROM viabilidad_municipal
WHERE r_i_preliminar = 1;
""".strip(),
}
PRICE_COLUMN_CANDIDATES = [
    "precio_compra_cop_kwh",
    "precio_energia_compra_cop_kwh",
    "tarifa_compra_cop_kwh",
    "tarifa_cop_kwh",
]

FIGURE_SCALE = 0.55
FIGURE_DPI = 90
plt.rcParams.update(
    {
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.titlesize": 10,
    }
)

SCORE_COL = "v_i_modelo_rural"
ECONOMIC_CLIMATE_SCORE_COL = "v_i_modelo_rural_economico_climatico"
WEIGHTS = {
    "s_i_solar": 0.35,
    "g_i_red": 0.30,
    "p_i_pendiente_proxy": 0.25,
    "u_i_uso_suelo": 0.10,
}
ECONOMIC_CLIMATE_WEIGHTS = {
    "s_i_solar": 0.30,
    "g_i_red": 0.25,
    "p_i_pendiente_proxy": 0.20,
    "u_i_uso_suelo": 0.10,
    "score_tierra_modelo": 0.08,
    "score_agua_modelo": 0.04,
    "score_riesgo_viento_modelo": 0.03,
}
COMPONENT_LABELS = {
    "s_i_solar": "Solar",
    "g_i_red": "Red",
    "p_i_pendiente_proxy": "Pendiente",
    "u_i_uso_suelo": "Uso rural / restriccion",
}
DISPLAY_COLUMNS = [
    "ranking",
    "municipio",
    "departamento",
    SCORE_COL,
    ECONOMIC_CLIMATE_SCORE_COL,
    "clasificacion_preliminar",
    "pvout_kwh_kwp_day",
    "annual_yield_kwh_kw_year",
    "dist_subestacion_km",
    "pendiente_igac",
    "pct_area_protegida_runap",
    "precio_tierra_ha_cop",
    "tarifa_acueducto_m3_cop",
    "velocidad_viento_max_ms",
    "score_tierra",
    "score_agua",
    "score_riesgo_viento",
    "flag_dato_tierra",
    "flag_dato_agua",
    "flag_dato_viento",
    "tipo_capa_pot",
    "categoria_aptitud_pot",
    "uso_pot",
    "score_rural_con_bono_demanda",
    "bono_demanda_favorable",
    "zona_xm_demanda",
    "tipo_mapeo_demanda",
    "cluster_kmeans_label",
    "por_que_aparece",
]

GLOSSARY_ROWS = [
    {
        "grupo": "Identificacion",
        "variable": "codigo_dane",
        "significado": "Codigo oficial DANE del municipio o distrito.",
        "lectura": "Identificador unico para unir tablas municipales.",
        "uso_modelo": "Llave de datos",
    },
    {
        "grupo": "Identificacion",
        "variable": "municipio",
        "significado": "Nombre del municipio o distrito.",
        "lectura": "Se usa para lectura humana del ranking.",
        "uso_modelo": "Descripcion",
    },
    {
        "grupo": "Identificacion",
        "variable": "departamento",
        "significado": "Departamento al que pertenece el municipio.",
        "lectura": "Permite filtrar resultados por region administrativa.",
        "uso_modelo": "Filtro",
    },
    {
        "grupo": "Identificacion",
        "variable": "lon",
        "significado": "Longitud del punto municipal usado para muestreos.",
        "lectura": "Ubica el municipio en mapas y consultas puntuales.",
        "uso_modelo": "Mapa",
    },
    {
        "grupo": "Identificacion",
        "variable": "lat",
        "significado": "Latitud del punto municipal usado para muestreos.",
        "lectura": "Ubica el municipio en mapas y consultas puntuales.",
        "uso_modelo": "Mapa",
    },
    {
        "grupo": "Identificacion",
        "variable": "area_km2_igac",
        "significado": "Area municipal reportada desde la base IGAC.",
        "lectura": "Da contexto de tamano; no decide por si sola.",
        "uso_modelo": "Contexto",
    },
    {
        "grupo": "Solar",
        "variable": "pvout_kwh_kwp_day",
        "significado": "Produccion solar especifica diaria estimada.",
        "lectura": "Mas alto indica mejor recurso solar preliminar.",
        "uso_modelo": "Entrada directa",
    },
    {
        "grupo": "Solar",
        "variable": "annual_yield_kwh_kw_year",
        "significado": "Generacion anual estimada por kW instalado.",
        "lectura": "Sirve para comparar potencial energetico anual.",
        "uso_modelo": "Contexto tecnico",
    },
    {
        "grupo": "Solar",
        "variable": "s_i_solar",
        "significado": "Score solar normalizado en escala 0-1.",
        "lectura": "1 es el mejor PVOUT relativo dentro de la tabla.",
        "uso_modelo": "Entra a V_i rural",
    },
    {
        "grupo": "Red",
        "variable": "dist_subestacion_km",
        "significado": "Distancia al punto de subestacion UPME mas cercano.",
        "lectura": "Menor distancia suele ser mas favorable.",
        "uso_modelo": "Entrada directa",
    },
    {
        "grupo": "Red",
        "variable": "g_i_red",
        "significado": "Score de cercania a red normalizado en escala 0-1.",
        "lectura": "1 significa muy cerca de subestacion en el conjunto analizado.",
        "uso_modelo": "Entra a V_i rural",
    },
    {
        "grupo": "Pendiente",
        "variable": "pendiente_igac",
        "significado": "Clase de pendiente consultada en IGAC.",
        "lectura": "Plano o baja pendiente favorece granjas solares.",
        "uso_modelo": "Entrada directa",
    },
    {
        "grupo": "Pendiente",
        "variable": "viabilidad_pendiente",
        "significado": "Etiqueta tecnica derivada de la clase de pendiente.",
        "lectura": "Puede ser viable, condicional, no_viable o desconocida.",
        "uso_modelo": "Restriccion",
    },
    {
        "grupo": "Pendiente",
        "variable": "p_i_pendiente_proxy",
        "significado": "Score de pendiente en escala 0-1.",
        "lectura": "1 indica pendiente favorable; 0 indica no favorable.",
        "uso_modelo": "Entra a V_i rural",
    },
    {
        "grupo": "Restricciones",
        "variable": "pct_area_protegida_runap",
        "significado": "Proporcion municipal cubierta por areas RUNAP.",
        "lectura": "Valores altos reducen el espacio preliminarmente apto.",
        "uso_modelo": "Restriccion",
    },
    {
        "grupo": "Restricciones",
        "variable": "u_i_no_protegido_runap",
        "significado": "Proporcion municipal no protegida segun RUNAP.",
        "lectura": "Mas alto indica menor restriccion RUNAP.",
        "uso_modelo": "Componente de U_i",
    },
    {
        "grupo": "Restricciones",
        "variable": "r_i_runap",
        "significado": "Restriccion dura por area protegida RUNAP.",
        "lectura": "0 excluye si RUNAP cubre demasiado el municipio.",
        "uso_modelo": "Restriccion",
    },
    {
        "grupo": "POT",
        "variable": "tipo_capa_pot",
        "significado": "Tipo de capa POT muestreada: rural o urbana.",
        "lectura": "Urbana debe excluirse para granja solar rural.",
        "uso_modelo": "Restriccion",
    },
    {
        "grupo": "POT",
        "variable": "categoria_aptitud_pot",
        "significado": "Clasificacion de aptitud territorial derivada del POT.",
        "lectura": "Resume si el uso es compatible, condicional o restrictivo.",
        "uso_modelo": "Restriccion",
    },
    {
        "grupo": "POT",
        "variable": "uso_pot",
        "significado": "Etiqueta original de uso del suelo POT.",
        "lectura": "Permite auditar por que una zona se considera apta o no.",
        "uso_modelo": "Auditoria",
    },
    {
        "grupo": "POT",
        "variable": "u_i_pot_compatible",
        "significado": "Score proxy de compatibilidad POT en escala 0-1.",
        "lectura": "1 compatible; 0 no compatible o restringido.",
        "uso_modelo": "Componente de U_i",
    },
    {
        "grupo": "POT",
        "variable": "r_i_zona_urbana_pot",
        "significado": "Restriccion por capa urbana POT.",
        "lectura": "0 excluye el municipio/punto por zona urbana.",
        "uso_modelo": "Restriccion",
    },
    {
        "grupo": "POT",
        "variable": "r_i_restriccion_pot",
        "significado": "Restriccion territorial estricta detectada en POT.",
        "lectura": "0 excluye por proteccion, amenaza u otra restriccion fuerte.",
        "uso_modelo": "Restriccion",
    },
    {
        "grupo": "Uso suelo",
        "variable": "u_i_uso_suelo",
        "significado": "Score integrado de uso/restriccion territorial.",
        "lectura": "Combina RUNAP y POT disponible; mas alto es mejor.",
        "uso_modelo": "Entra a V_i rural",
    },
    {
        "grupo": "Demanda",
        "variable": "d_i_demanda",
        "significado": "Score de demanda XM normalizado en escala 0-1.",
        "lectura": "Demanda alta favorece conexion comercial, pero no decide ubicacion.",
        "uso_modelo": "Contexto",
    },
    {
        "grupo": "Demanda",
        "variable": "demanda_xm_proxy_mwh_o_unidad_fuente",
        "significado": "Valor proxy de demanda asignado desde zonas XM.",
        "lectura": "No es medicion municipal directa.",
        "uso_modelo": "Contexto",
    },
    {
        "grupo": "Demanda",
        "variable": "flag_revision_demanda",
        "significado": "Marca si la asignacion de demanda requiere revision.",
        "lectura": "1 indica incertidumbre metodologica.",
        "uso_modelo": "Control de calidad",
    },
    {
        "grupo": "Demanda",
        "variable": "flag_atipico_eda_demanda",
        "significado": "Marca de atipico o caso sensible detectado en EDA.",
        "lectura": "1 advierte que la demanda debe leerse con cuidado.",
        "uso_modelo": "Control de calidad",
    },
    {
        "grupo": "Demanda",
        "variable": "bono_demanda_favorable",
        "significado": "Bono separado por demanda favorable.",
        "lectura": "No entra al V_i rural; solo muestra sensibilidad comercial.",
        "uso_modelo": "Sensibilidad",
    },
    {
        "grupo": "Score",
        "variable": "r_i_preliminar",
        "significado": "Restriccion dura preliminar total.",
        "lectura": "1 permite calcular score; 0 excluye preliminarmente.",
        "uso_modelo": "Restriccion principal",
    },
    {
        "grupo": "Score",
        "variable": "v_i_modelo_rural",
        "significado": "Score principal de viabilidad rural sin demanda.",
        "lectura": "Ranking principal del dashboard.",
        "uso_modelo": "Score principal",
    },
    {
        "grupo": "Score",
        "variable": "v_i_modelo_rural_economico_climatico",
        "significado": "Score adicional que incorpora tierra, agua y viento.",
        "lectura": "Permite comparar el ranking tecnico con una sensibilidad economica y climatica.",
        "uso_modelo": "Score adicional",
    },
    {
        "grupo": "Costos",
        "variable": "precio_tierra_ha_cop",
        "significado": "Precio comercial rural por hectarea en COP.",
        "lectura": "Menor precio favorece el score economico-climatico.",
        "uso_modelo": "Entrada de T_i",
    },
    {
        "grupo": "Costos",
        "variable": "tarifa_acueducto_m3_cop",
        "significado": "Tarifa variable de acueducto aproximada en COP/m3.",
        "lectura": "Menor tarifa favorece el score economico-climatico.",
        "uso_modelo": "Entrada de A_i",
    },
    {
        "grupo": "Clima",
        "variable": "velocidad_viento_max_ms",
        "significado": "Maximo municipal observado de velocidad de viento IDEAM.",
        "lectura": "Menor valor reduce el proxy de riesgo por viento fuerte.",
        "uso_modelo": "Entrada de W_i",
    },
    {
        "grupo": "Score",
        "variable": "score_rural_con_bono_demanda",
        "significado": "Score rural mas bono de demanda favorable.",
        "lectura": "Sirve como sensibilidad; no reemplaza V_i rural.",
        "uso_modelo": "Sensibilidad",
    },
    {
        "grupo": "Score",
        "variable": "clasificacion_preliminar",
        "significado": "Clase por percentiles del score rural.",
        "lectura": "muy_alta, alta, media, baja o excluida preliminar.",
        "uso_modelo": "Etiqueta descriptiva",
    },
    {
        "grupo": "Score",
        "variable": "estado_modelo_oficial",
        "significado": "Estado metodologico del registro.",
        "lectura": "Indica si el municipio tiene POT pendiente, demanda en revision o exclusion.",
        "uso_modelo": "Auditoria",
    },
    {
        "grupo": "K-Means",
        "variable": "cluster_kmeans",
        "significado": "Identificador numerico del cluster K-Means.",
        "lectura": "Agrupa municipios con perfil similar; no es ranking.",
        "uso_modelo": "Agrupamiento",
    },
    {
        "grupo": "K-Means",
        "variable": "cluster_kmeans_label",
        "significado": "Etiqueta legible del cluster K-Means.",
        "lectura": "Facilita leer el grupo en tablas y graficos.",
        "uso_modelo": "Agrupamiento",
    },
    {
        "grupo": "Explicacion",
        "variable": "por_que_aparece",
        "significado": "Resumen textual de razones del ranking.",
        "lectura": "Ayuda a leer rapidamente por que aparece un municipio.",
        "uso_modelo": "Explicacion",
    },
]


st.set_page_config(
    page_title="Granja solar Colombia",
    page_icon="",
    layout="wide",
)


def annotate_frame(
    df: pd.DataFrame,
    source_label: str,
    source_warning: str | None = None,
) -> pd.DataFrame:
    df.attrs["data_source"] = source_label
    if source_warning:
        df.attrs["source_warning"] = source_warning
    return df


def frame_source_label(df: pd.DataFrame) -> str:
    return str(df.attrs.get("data_source", "Origen desconocido"))


def frame_source_warning(df: pd.DataFrame) -> str | None:
    warning = df.attrs.get("source_warning")
    if warning is None:
        return None
    return str(warning)


def mysql_settings() -> MySQLSettings:
    return MySQLSettings.from_env(database_override=MYSQL_DATABASE)


def load_mysql_frame(sql_text: str, label: str) -> pd.DataFrame:
    settings = mysql_settings()
    if not settings.user or not settings.password:
        raise ValueError("Faltan MYSQL_USER y MYSQL_PASSWORD para consultar MySQL.")
    df = query_dataframe(sql_text, settings=settings, database=settings.database)
    return annotate_frame(df, f"MySQL - {label}")


def read_csv_with_source(path: Path, **kwargs: object) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    return annotate_frame(df, f"CSV - {path.relative_to(PROJECT_ROOT)}")


def load_frame_with_fallback(
    *,
    sql_text: str,
    mysql_label: str,
    csv_path: Path,
    csv_read_kwargs: dict[str, object] | None = None,
    required: bool = False,
) -> pd.DataFrame:
    csv_read_kwargs = csv_read_kwargs or {}
    mysql_error: Exception | None = None

    try:
        mysql_df = load_mysql_frame(sql_text, mysql_label)
        if not mysql_df.empty or not csv_path.exists():
            return mysql_df
    except Exception as exc:
        mysql_error = exc

    if csv_path.exists():
        csv_df = read_csv_with_source(csv_path, **csv_read_kwargs)
        if mysql_error is not None:
            return annotate_frame(csv_df, frame_source_label(csv_df), str(mysql_error))
        return csv_df

    if mysql_error is not None:
        raise mysql_error
    if required:
        raise FileNotFoundError(f"No existe {csv_path}")
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_viability() -> pd.DataFrame:
    df = load_frame_with_fallback(
        sql_text=VIABILITY_SQL,
        mysql_label="viabilidad_municipal + cluster_municipal",
        csv_path=VIABILITY_PATH,
        csv_read_kwargs={"dtype": {"codigo_dane": "string"}},
        required=True,
    )
    source_label = frame_source_label(df)
    source_warning = frame_source_warning(df)
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
        ECONOMIC_CLIMATE_SCORE_COL,
        "v_i_modelo_oficial",
        "v_i_modelo_proxy_xm",
        "score_preliminar_solar_red_pendiente_runap",
        "score_rural_con_bono_demanda",
        "bono_demanda_favorable",
        "pvout_kwh_kwp_day",
        "annual_yield_kwh_kw_year",
        "dist_subestacion_km",
        "pct_area_protegida_runap",
        "area_km2_igac",
        "r_i_preliminar",
        "r_i_zona_urbana_pot",
        "r_i_restriccion_pot",
        "u_i_pot_compatible",
        "flag_atipico_eda_demanda",
        "flag_revision_demanda",
        "demanda_xm_proxy_mwh_o_unidad_fuente",
        "d_i_demanda",
        "precio_tierra_ha_cop",
        "score_tierra",
        "score_tierra_modelo",
        "tarifa_acueducto_m3_cop",
        "score_agua",
        "score_agua_modelo",
        "velocidad_viento_ms",
        "velocidad_viento_max_ms",
        "score_riesgo_viento",
        "score_riesgo_viento_modelo",
        "flag_dato_tierra",
        "flag_dato_agua",
        "flag_dato_viento",
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
    return annotate_frame(df, source_label, source_warning)


@st.cache_data(show_spinner=False)
def load_cluster_profile() -> pd.DataFrame:
    return load_frame_with_fallback(
        sql_text=CLUSTER_PROFILE_SQL,
        mysql_label="cluster_perfiles",
        csv_path=CLUSTER_PROFILE_PATH,
    )


@st.cache_data(show_spinner=False)
def load_cluster_evaluation() -> pd.DataFrame:
    return load_frame_with_fallback(
        sql_text=CLUSTER_EVALUATION_SQL,
        mysql_label="cluster_evaluacion_k",
        csv_path=CLUSTER_EVALUATION_PATH,
    )


@st.cache_data(show_spinner=False)
def load_cluster_summary() -> pd.DataFrame:
    return load_frame_with_fallback(
        sql_text=CLUSTER_SUMMARY_SQL,
        mysql_label="cluster_resumen_seleccion",
        csv_path=CLUSTER_SUMMARY_PATH,
    )


@st.cache_data(show_spinner=False)
def load_solar_costs() -> pd.DataFrame:
    return load_frame_with_fallback(
        sql_text=SOLAR_COSTS_SQL,
        mysql_label="solar_escenarios",
        csv_path=SOLAR_COSTS_PATH,
    )


@st.cache_data(show_spinner=False)
def load_energy_prices() -> pd.DataFrame:
    return load_frame_with_fallback(
        sql_text=ENERGY_PRICES_SQL,
        mysql_label="precios_energia_departamento",
        csv_path=ENERGY_PRICES_PATH,
        csv_read_kwargs={"dtype": {"codigo_dane": "string"}},
    )


@st.cache_data(show_spinner=False)
def load_eda_manifest() -> pd.DataFrame:
    if not EDA_MANIFEST_PATH.exists():
        return pd.DataFrame()
    manifest = pd.read_csv(EDA_MANIFEST_PATH)
    if "archivo" in manifest.columns:
        manifest["archivo"] = manifest["archivo"].astype(str)
    return manifest


@st.cache_data(show_spinner=False)
def load_eda_verification() -> pd.DataFrame:
    if not EDA_VERIFICATION_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(EDA_VERIFICATION_PATH)


def clean_eda_variable_name(file_path: Path, chart_type: str) -> str:
    name = file_path.stem
    prefixes = {
        "kde": "kde_",
        "histograma": "histograma_",
        "boxplot": "boxplot_",
    }
    name = name.removeprefix(prefixes.get(chart_type, ""))
    if chart_type == "boxplot" and "_por_" in name:
        name = name.split("_por_", 1)[0]
    return name


def clean_eda_segment_name(file_path: Path, chart_type: str) -> str:
    if chart_type != "boxplot" or "_por_" not in file_path.stem:
        return ""
    return file_path.stem.split("_por_", 1)[1]


@st.cache_data(show_spinner=False)
def load_eda_whatsapp_assets() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = {
        "kde": ("kde", EDA_WHATSAPP_DIR / "kde"),
        "histograma": ("histogramas", EDA_WHATSAPP_DIR / "histogramas"),
        "boxplot": ("boxplots", EDA_WHATSAPP_DIR / "boxplots"),
    }
    for chart_type, (folder_label, folder_path) in groups.items():
        if not folder_path.exists():
            continue
        for image_path in sorted(folder_path.glob("*.png")):
            rows.append(
                {
                    "chart_type": chart_type,
                    "grupo": folder_label,
                    "variable": clean_eda_variable_name(image_path, chart_type),
                    "segmento": clean_eda_segment_name(image_path, chart_type),
                    "archivo": str(image_path),
                    "actualizado": pd.Timestamp.fromtimestamp(image_path.stat().st_mtime),
                }
            )
    return pd.DataFrame(rows)


def normalize_text_key(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.lower().strip().split())


def merge_energy_prices(base_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Une precios reales por DANE, municipio/departamento o departamento."""

    if base_df.empty or prices_df.empty:
        return base_df.copy()

    price_columns = [column for column in PRICE_COLUMN_CANDIDATES if column in prices_df.columns]
    if not price_columns:
        return base_df.copy()

    price_column = price_columns[0]
    prices = prices_df.copy()
    prices["precio_compra_cop_kwh"] = pd.to_numeric(prices[price_column], errors="coerce")
    prices = prices.dropna(subset=["precio_compra_cop_kwh"]).copy()
    if prices.empty:
        return base_df.copy()

    result = base_df.copy()

    if "codigo_dane" in prices.columns and "codigo_dane" in result.columns:
        prices["codigo_dane"] = prices["codigo_dane"].astype("string").str.zfill(5)
        keep_cols = [
            column
            for column in [
                "codigo_dane",
                "precio_compra_cop_kwh",
                "prestador_tarifa",
                "fecha_publicacion",
                "fuente_url",
                "nota",
            ]
            if column in prices.columns
        ]
        return result.merge(prices[keep_cols], on="codigo_dane", how="left")

    result["_departamento_key"] = result["departamento"].map(normalize_text_key)
    prices["_departamento_key"] = prices["departamento"].map(normalize_text_key)

    if "municipio" in prices.columns:
        result["_municipio_key"] = result["municipio"].map(normalize_text_key)
        prices["_municipio_key"] = prices["municipio"].map(normalize_text_key)
        join_cols = ["_departamento_key", "_municipio_key"]
    else:
        join_cols = ["_departamento_key"]

    keep_cols = [
        column
        for column in [
            *join_cols,
            "precio_compra_cop_kwh",
            "prestador_tarifa",
            "fecha_publicacion",
            "fuente_url",
            "nota",
        ]
        if column in prices.columns
    ]
    result = result.merge(prices[keep_cols], on=join_cols, how="left")
    return result.drop(columns=[column for column in ["_departamento_key", "_municipio_key"] if column in result.columns])


def build_glossary(df: pd.DataFrame) -> pd.DataFrame:
    glossary = pd.DataFrame(GLOSSARY_ROWS)
    available_columns = set(df.columns)
    glossary["disponible_en_tabla"] = glossary["variable"].isin(available_columns).map(
        {True: "si", False: "no"}
    )
    return glossary


def filter_glossary(glossary: pd.DataFrame, groups: list[str], query: str) -> pd.DataFrame:
    filtered = glossary.copy()
    if groups:
        filtered = filtered[filtered["grupo"].isin(groups)]
    query = query.strip().lower()
    if query:
        search_text = filtered.astype(str).agg(" ".join, axis=1).str.lower()
        filtered = filtered[search_text.str.contains(query, regex=False)]
    return filtered


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
        reasons.append("baja restriccion territorial")

    if str(row.get("tipo_capa_pot", "")).lower() == "rural":
        reasons.append("POT rural")

    if row.get("r_i_zona_urbana_pot", 1) == 0:
        reasons.append("excluido por zona urbana POT")

    if row.get("d_i_demanda", 0) >= 0.50:
        reasons.append("demanda favorable como contexto")

    if row.get("flag_atipico_eda_demanda", 0) == 1:
        reasons.append("demanda marcada para revision")

    return "; ".join(reasons) if reasons else "score explicado por combinacion de criterios"


def filter_data(
    df: pd.DataFrame,
    departments: list[str],
    hide_demand_outliers: bool,
    only_eligible: bool,
) -> pd.DataFrame:
    filtered = df.copy()
    if departments:
        filtered = filtered[filtered["departamento"].isin(departments)]
    if hide_demand_outliers and "flag_atipico_eda_demanda" in filtered.columns:
        filtered = filtered[filtered["flag_atipico_eda_demanda"].fillna(0).ne(1)]
    if only_eligible and "r_i_preliminar" in filtered.columns:
        filtered = filtered[filtered["r_i_preliminar"].fillna(0).eq(1)]
    return filtered


def top_municipalities(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return top_municipalities_by_score(df, n, SCORE_COL)


def top_municipalities_by_score(df: pd.DataFrame, n: int, score_column: str) -> pd.DataFrame:
    if score_column not in df.columns:
        return pd.DataFrame()
    top = df.dropna(subset=[score_column]).sort_values(score_column, ascending=False).head(n).copy()
    top.insert(0, "ranking", range(1, len(top) + 1))
    return top


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in DISPLAY_COLUMNS if column in df.columns]
    table = df[available].copy()
    rename = {
        SCORE_COL: "score_rural",
        ECONOMIC_CLIMATE_SCORE_COL: "score_economico_climatico",
        "pvout_kwh_kwp_day": "pvout_kwh_kwp_dia",
        "annual_yield_kwh_kw_year": "kwh_kw_anio",
        "dist_subestacion_km": "dist_red_km",
        "pct_area_protegida_runap": "pct_runap",
        "score_rural_con_bono_demanda": "score_con_bono_demanda",
    }
    table = table.rename(columns=rename)
    return table


def scaled_figsize(width: float, height: float) -> tuple[float, float]:
    return max(3.8, width * FIGURE_SCALE), max(2.4, height * FIGURE_SCALE)


def render_chart(fig: plt.Figure) -> None:
    st.pyplot(fig, clear_figure=True, width="content")


def make_top_score_chart(top: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=scaled_figsize(10, max(4, 0.45 * len(top))), dpi=FIGURE_DPI)
    data = top.sort_values(SCORE_COL, ascending=True)
    labels = data["municipio"].astype(str) + " (" + data["departamento"].astype(str) + ")"
    ax.barh(labels, data[SCORE_COL])
    ax.set_xlabel("Score V_i rural")
    ax.set_title("Top municipios por viabilidad rural")
    ax.set_xlim(0, 1)
    fig.tight_layout(pad=0.7)
    return fig


def make_score_chart(top: pd.DataFrame, score_column: str, title: str, xlabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=scaled_figsize(10, max(4, 0.45 * len(top))), dpi=FIGURE_DPI)
    if top.empty or score_column not in top.columns:
        ax.text(0.5, 0.5, "Sin datos suficientes", ha="center", va="center")
        ax.axis("off")
        return fig
    data = top.sort_values(score_column, ascending=True)
    labels = data["municipio"].astype(str) + " (" + data["departamento"].astype(str) + ")"
    ax.barh(labels, data[score_column])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    fig.tight_layout(pad=0.7)
    return fig


def make_ranking_comparison(df: pd.DataFrame, n: int) -> pd.DataFrame:
    required = [SCORE_COL, ECONOMIC_CLIMATE_SCORE_COL]
    if any(column not in df.columns for column in required):
        return pd.DataFrame()
    working = df.dropna(subset=required).copy()
    working["ranking_original"] = working[SCORE_COL].rank(ascending=False, method="min")
    working["ranking_economico_climatico"] = working[ECONOMIC_CLIMATE_SCORE_COL].rank(
        ascending=False,
        method="min",
    )
    working["cambio_ranking"] = working["ranking_original"] - working["ranking_economico_climatico"]
    keep = [
        "codigo_dane",
        "municipio",
        "departamento",
        SCORE_COL,
        ECONOMIC_CLIMATE_SCORE_COL,
        "ranking_original",
        "ranking_economico_climatico",
        "cambio_ranking",
        "precio_tierra_ha_cop",
        "tarifa_acueducto_m3_cop",
        "velocidad_viento_max_ms",
    ]
    return working.sort_values("ranking_economico_climatico").head(n)[
        [column for column in keep if column in working.columns]
    ]


def make_low_land_cost_chart(df: pd.DataFrame, n: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=scaled_figsize(10, max(4, 0.45 * n)), dpi=FIGURE_DPI)
    if "precio_tierra_ha_cop" not in df.columns:
        ax.text(0.5, 0.5, "Sin datos de tierra", ha="center", va="center")
        ax.axis("off")
        return fig
    data = (
        df.dropna(subset=["precio_tierra_ha_cop"])
        .sort_values("precio_tierra_ha_cop", ascending=True)
        .head(n)
        .sort_values("precio_tierra_ha_cop", ascending=False)
    )
    if data.empty:
        ax.text(0.5, 0.5, "Sin datos de tierra", ha="center", va="center")
        ax.axis("off")
        return fig
    labels = data["municipio"].astype(str) + " (" + data["departamento"].astype(str) + ")"
    ax.barh(labels, data["precio_tierra_ha_cop"] / 1_000_000)
    ax.set_xlabel("Millones COP/ha")
    ax.set_title("Menor costo de tierra rural")
    fig.tight_layout(pad=0.7)
    return fig


def make_wind_risk_chart(df: pd.DataFrame, n: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=scaled_figsize(10, max(4, 0.45 * n)), dpi=FIGURE_DPI)
    if "velocidad_viento_max_ms" not in df.columns:
        ax.text(0.5, 0.5, "Sin datos de viento", ha="center", va="center")
        ax.axis("off")
        return fig
    data = (
        df.dropna(subset=["velocidad_viento_max_ms"])
        .sort_values("velocidad_viento_max_ms", ascending=False)
        .head(n)
        .sort_values("velocidad_viento_max_ms", ascending=True)
    )
    if data.empty:
        ax.text(0.5, 0.5, "Sin datos de viento", ha="center", va="center")
        ax.axis("off")
        return fig
    labels = data["municipio"].astype(str) + " (" + data["departamento"].astype(str) + ")"
    ax.barh(labels, data["velocidad_viento_max_ms"])
    ax.set_xlabel("m/s")
    ax.set_title("Mayor velocidad maxima de viento observada")
    fig.tight_layout(pad=0.7)
    return fig


def make_components_chart(top: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=scaled_figsize(11, max(4, 0.48 * len(top))), dpi=FIGURE_DPI)
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
    ax.set_title("Por que aparecen: aportes ponderados sin demanda")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.7)
    return fig


def make_component_heatmap(top: pd.DataFrame) -> plt.Figure:
    component_cols = [column for column in WEIGHTS if column in top.columns]
    matrix = top.set_index("municipio")[component_cols].fillna(0)
    fig, ax = plt.subplots(figsize=scaled_figsize(9, max(4, 0.45 * len(matrix))), dpi=FIGURE_DPI)
    image = ax.imshow(matrix.values, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(component_cols)))
    ax.set_xticklabels([COMPONENT_LABELS.get(column, column) for column in component_cols], rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Componentes normalizados del top")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04, label="Score 0-1")
    fig.tight_layout(pad=0.7)
    return fig


_DIM_LABELS = {
    "score_fisico":       "Fisico\n(solar/viento)",
    "score_electrico":    "Electrico\n(red/subestacion)",
    "score_economico":    "Economico\n(tierra/costos)",
    "score_agropecuario": "Agropecuario\n(UGG/uso suelo)",
    "score_riesgo":       "Riesgo\n(inundacion/sequia)",
}
_DIM_WEIGHTS = {
    "score_fisico": 0.30,
    "score_electrico": 0.25,
    "score_economico": 0.20,
    "score_agropecuario": 0.15,
    "score_riesgo": 0.10,
}
_TOP5_COLORS = ["#2563EB", "#16A34A", "#D97706", "#DC2626", "#7C3AED"]


def make_top5_individual_charts(top5: pd.DataFrame) -> plt.Figure:
    """Una barra por municipio (5 subplots) mostrando score total vs maximo posible."""
    n = min(5, len(top5))
    data = top5.head(n).reset_index(drop=True)
    fig, axes = plt.subplots(1, n, figsize=scaled_figsize(3.2 * n, 3.5), dpi=FIGURE_DPI)
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        row = data.iloc[i]
        score = float(row.get("v_i_multidimensional", row.get(SCORE_COL, 0)) or 0)
        color = _TOP5_COLORS[i]
        label = f"#{i+1}"
        ax.bar([label], [score], color=color, width=0.5, zorder=3)
        ax.bar([label], [1 - score], bottom=[score], color="#E5E7EB", width=0.5, zorder=2)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", ".25", ".50", ".75", "1"] if i == 0 else [])
        ax.set_title(
            f"#{i+1}\n{row['municipio']}\n{row['departamento']}",
            fontsize=8.5,
            fontweight="bold" if i == 0 else "normal",
        )
        ax.text(
            0, score + 0.02, f"{score:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color=color,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
        ax.set_xlabel("Score total")
    fig.suptitle("Top 5 municipios - Score multidimensional", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def make_numero1_radar(row: pd.Series) -> plt.Figure:
    """Radar (spider) chart con los 5 scores discriminados del municipio #1."""
    import numpy as np

    dims = list(_DIM_LABELS.keys())
    labels = list(_DIM_LABELS.values())
    weights = [_DIM_WEIGHTS[d] for d in dims]
    values = [float(row.get(d, 0) or 0) for d in dims]

    N = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=scaled_figsize(6, 6), dpi=FIGURE_DPI, subplot_kw={"polar": True})

    # Area del score real
    ax.fill(angles_plot[:-1], values_plot[:-1], color=_TOP5_COLORS[0], alpha=0.25)
    ax.plot(angles_plot, values_plot, color=_TOP5_COLORS[0], linewidth=2, marker="o", markersize=7)

    # Referencia: score maximo (1.0)
    ax.plot(angles + [angles[0]], [1.0] * (N + 1), color="#9CA3AF", linewidth=1, linestyle="--", alpha=0.6)

    # Etiquetas de dimensiones con peso
    ax.set_xticks(angles)
    ax.set_xticklabels(
        [f"{lbl}\n(w={w:.0%})" for lbl, w in zip(labels, weights)],
        size=8,
    )
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], size=7, color="#6B7280")

    # Score total
    score_total = float(row.get("v_i_multidimensional", row.get(SCORE_COL, 0)) or 0)
    municipio = str(row.get("municipio", ""))
    depto = str(row.get("departamento", ""))
    ax.set_title(
        f"#1 - {municipio} ({depto})\nScore total: {score_total:.4f}",
        fontsize=11, fontweight="bold", pad=20,
    )

    # Anotar valores en cada punta
    for angle, val, dim in zip(angles, values, dims):
        ax.annotate(
            f"{val:.3f}",
            xy=(angle, val),
            xytext=(angle, val + 0.08),
            ha="center", va="center",
            fontsize=8, color=_TOP5_COLORS[0], fontweight="bold",
        )

    fig.tight_layout()
    return fig


def make_scatter_solar_grid(df: pd.DataFrame) -> plt.Figure:
    plot = df.dropna(subset=["pvout_kwh_kwp_day", "dist_subestacion_km", SCORE_COL]).copy()
    fig, ax = plt.subplots(figsize=scaled_figsize(9, 5), dpi=FIGURE_DPI)
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
    fig.colorbar(scatter, ax=ax, label="V_i rural")
    fig.tight_layout(pad=0.7)
    return fig


def make_score_histogram(df: pd.DataFrame) -> plt.Figure:
    scores = df[SCORE_COL].dropna()
    fig, ax = plt.subplots(figsize=scaled_figsize(9, 4.5), dpi=FIGURE_DPI)
    ax.hist(scores, bins=25)
    ax.set_xlabel("Score V_i rural")
    ax.set_ylabel("Municipios")
    ax.set_title("Distribucion del score de viabilidad")
    fig.tight_layout(pad=0.7)
    return fig


def make_classification_chart(df: pd.DataFrame) -> plt.Figure:
    counts = df["clasificacion_preliminar"].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=scaled_figsize(9, 4.5), dpi=FIGURE_DPI)
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_ylabel("Municipios")
    ax.set_title("Conteo por clasificacion preliminar")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(pad=0.7)
    return fig


def build_energy_value_estimate(
    df: pd.DataFrame,
    scenario: pd.Series,
    hectares: float,
    price_column: str,
) -> pd.DataFrame:
    """Estima energia anual por municipio y valor de compra anual."""

    if df.empty or price_column not in df.columns:
        return pd.DataFrame()

    capacity_kw_per_hectare = pd.to_numeric(
        pd.Series([scenario.get("capacidad_kw_por_hectarea")]),
        errors="coerce",
    ).iloc[0]
    scenario_yield = pd.to_numeric(
        pd.Series([scenario.get("annual_yield_kwh_per_kw_year")]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(capacity_kw_per_hectare) or capacity_kw_per_hectare <= 0:
        return pd.DataFrame()

    estimate = df.copy()
    if "annual_yield_kwh_kw_year" in estimate.columns:
        annual_yield = pd.to_numeric(estimate["annual_yield_kwh_kw_year"], errors="coerce")
        annual_yield = annual_yield.fillna(scenario_yield)
    else:
        annual_yield = pd.Series(scenario_yield, index=estimate.index)

    viability = pd.to_numeric(estimate.get(SCORE_COL, 0), errors="coerce").fillna(0).clip(0, 1)
    price_cop_kwh = pd.to_numeric(estimate[price_column], errors="coerce")
    estimate["precio_compra_cop_kwh"] = price_cop_kwh
    estimate = estimate.dropna(subset=["precio_compra_cop_kwh"]).copy()
    if estimate.empty:
        return pd.DataFrame()
    annual_yield = annual_yield.loc[estimate.index]
    viability = viability.loc[estimate.index]
    price_cop_kwh = estimate["precio_compra_cop_kwh"]
    price_source = f"columna {price_column}"

    capacity_kw = capacity_kw_per_hectare * hectares
    generation_kwh = annual_yield * capacity_kw
    purchase_value_cop = generation_kwh * price_cop_kwh
    viability_weighted_value_cop = purchase_value_cop * viability

    estimate["escenario_energia"] = scenario.get("scenario_name", "escenario")
    estimate["hectareas_proyecto"] = hectares
    estimate["fuente_precio_compra"] = price_source
    estimate["capacidad_mw_estimada"] = capacity_kw / 1000.0
    estimate["generacion_kwh_anual_estimada"] = generation_kwh
    estimate["generacion_mwh_anual_estimada"] = generation_kwh / 1000.0
    estimate["valor_compra_cop_anual"] = purchase_value_cop
    estimate["valor_compra_millones_cop_anual"] = purchase_value_cop / 1_000_000.0
    estimate["valor_ponderado_viabilidad_cop_anual"] = viability_weighted_value_cop
    estimate["valor_ponderado_viabilidad_millones_cop_anual"] = (
        viability_weighted_value_cop / 1_000_000.0
    )
    return estimate


def make_energy_value_chart(energy_df: pd.DataFrame, n: int) -> plt.Figure:
    data = (
        energy_df.dropna(subset=["valor_ponderado_viabilidad_millones_cop_anual"])
        .sort_values("valor_ponderado_viabilidad_millones_cop_anual", ascending=False)
        .head(n)
        .sort_values("valor_ponderado_viabilidad_millones_cop_anual", ascending=True)
    )

    fig, ax = plt.subplots(figsize=scaled_figsize(11, max(4.5, 0.5 * len(data))), dpi=FIGURE_DPI)
    if data.empty:
        ax.text(0.5, 0.5, "Sin datos suficientes", ha="center", va="center")
        ax.axis("off")
        return fig

    labels = data["municipio"].astype(str) + " (" + data["departamento"].astype(str) + ")"
    ax.barh(
        labels,
        data["valor_compra_millones_cop_anual"],
        alpha=0.30,
        label="Energia anual * precio compra",
    )
    ax.barh(
        labels,
        data["valor_ponderado_viabilidad_millones_cop_anual"],
        alpha=0.90,
        label="Ponderado por V_i",
    )
    ax.set_xlabel("Millones COP/anio")
    ax.set_title("Valor anual estimado por compra de energia")
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.7)
    return fig


def make_energy_value_scatter(energy_df: pd.DataFrame) -> plt.Figure:
    plot = energy_df.dropna(
        subset=[SCORE_COL, "valor_ponderado_viabilidad_millones_cop_anual"]
    ).copy()
    fig, ax = plt.subplots(figsize=scaled_figsize(9, 5), dpi=FIGURE_DPI)
    if plot.empty:
        ax.text(0.5, 0.5, "Sin datos suficientes", ha="center", va="center")
        ax.axis("off")
        return fig

    scatter = ax.scatter(
        plot[SCORE_COL],
        plot["valor_ponderado_viabilidad_millones_cop_anual"],
        c=plot["generacion_mwh_anual_estimada"],
        alpha=0.72,
    )
    ax.set_xlabel("Viabilidad V_i rural")
    ax.set_ylabel("Valor ponderado, millones COP/anio")
    ax.set_title("Viabilidad vs valor anual ponderado")
    fig.colorbar(scatter, ax=ax, label="MWh/anio")
    fig.tight_layout(pad=0.7)
    return fig


def make_map_scatter(df: pd.DataFrame, top: pd.DataFrame) -> plt.Figure:
    plot = df.dropna(subset=["lon", "lat", SCORE_COL]).copy()
    fig, ax = plt.subplots(figsize=scaled_figsize(8, 8), dpi=FIGURE_DPI)
    scatter = ax.scatter(plot["lon"], plot["lat"], c=plot[SCORE_COL], s=18, alpha=0.65)
    if not top.empty:
        ax.scatter(top["lon"], top["lat"], s=70, facecolors="none", edgecolors="black", linewidths=1.2)
        for _, row in top.head(10).iterrows():
            ax.text(row["lon"], row["lat"], str(int(row["ranking"])), fontsize=8)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Mapa simple: municipios por score y top resaltado")
    fig.colorbar(scatter, ax=ax, label="V_i rural")
    fig.tight_layout(pad=0.7)
    return fig


def color_from_score(score: float | int | None) -> list[int]:
    if score is None or pd.isna(score):
        return [160, 160, 160, 120]
    value = max(0.0, min(1.0, float(score)))
    if value >= 0.85:
        return [35, 132, 67, 190]
    if value >= 0.70:
        return [120, 198, 121, 180]
    if value >= 0.55:
        return [255, 193, 7, 175]
    if value >= 0.35:
        return [245, 124, 0, 165]
    return [198, 40, 40, 155]


def prepare_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    required = ["lon", "lat", SCORE_COL, "municipio", "departamento"]
    columns = [column for column in required if column in df.columns]
    map_df = df[columns].dropna(subset=["lon", "lat", SCORE_COL]).copy()
    if map_df.empty:
        return map_df

    map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
    map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
    map_df[SCORE_COL] = pd.to_numeric(map_df[SCORE_COL], errors="coerce").clip(0, 1)
    map_df = map_df.dropna(subset=["lon", "lat", SCORE_COL]).copy()
    map_df["peso_heatmap"] = 1 + 9 * map_df[SCORE_COL]
    map_df["radio_punto"] = 2500 + 8500 * map_df[SCORE_COL]
    map_df["color_score"] = map_df[SCORE_COL].apply(color_from_score)
    map_df["score_texto"] = map_df[SCORE_COL].map(lambda value: f"{value:.3f}")
    return map_df


def make_colombia_heatmap(df: pd.DataFrame, top: pd.DataFrame) -> pdk.Deck:
    map_df = prepare_heatmap_data(df)
    top_df = prepare_heatmap_data(top)

    layers = [
        pdk.Layer(
            "HeatmapLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_weight="peso_heatmap",
            radiusPixels=45,
            intensity=1.2,
            threshold=0.08,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_fill_color="color_score",
            get_radius="radio_punto",
            pickable=True,
            opacity=0.55,
            stroked=False,
        ),
    ]

    if not top_df.empty:
        top_df["radio_punto"] = 12000
        top_df["color_score"] = [[17, 24, 39, 230] for _ in range(len(top_df))]
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=top_df,
                get_position="[lon, lat]",
                get_fill_color="color_score",
                get_radius="radio_punto",
                pickable=True,
                opacity=0.85,
                stroked=True,
                get_line_color=[255, 255, 255, 230],
                line_width_min_pixels=2,
            )
        )

    return pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(
            latitude=4.6,
            longitude=-74.1,
            zoom=4.45,
            min_zoom=3.2,
            max_zoom=10,
            pitch=0,
        ),
        layers=layers,
        tooltip={
            "html": (
                "<b>{municipio}</b><br/>"
                "{departamento}<br/>"
                "V_i rural: {score_texto}"
            ),
            "style": {
                "backgroundColor": "white",
                "color": "#111827",
                "fontSize": "12px",
                "border": "1px solid #d1d5db",
            },
        },
    )


def make_cluster_profile_chart(profile: pd.DataFrame) -> plt.Figure:
    metrics = [
        "s_i_solar_promedio",
        "g_i_red_promedio",
        "p_i_pendiente_promedio",
        "u_i_uso_suelo_promedio",
    ]
    existing = [column for column in metrics if column in profile.columns]
    matrix = profile.set_index("cluster_kmeans")[existing].fillna(0)
    fig, ax = plt.subplots(figsize=scaled_figsize(9, 5), dpi=FIGURE_DPI)
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
    fig.tight_layout(pad=0.7)
    return fig


def make_k_evaluation_chart(evaluation: pd.DataFrame) -> plt.Figure:
    plot = evaluation.sort_values("k").copy()
    fig, axes = plt.subplots(1, 2, figsize=scaled_figsize(14, 4.8), dpi=FIGURE_DPI)

    axes[0].plot(plot["k"], plot["inertia"], marker="o")
    elbow_flag = (
        pd.to_numeric(plot["recomendado_codo"], errors="coerce").fillna(0)
        if "recomendado_codo" in plot.columns
        else pd.Series(0, index=plot.index)
    )
    elbow = plot[elbow_flag.eq(1)]
    if not elbow.empty:
        row = elbow.iloc[0]
        axes[0].scatter([row["k"]], [row["inertia"]], s=70)
        axes[0].axvline(row["k"], linestyle="--", linewidth=1)
    axes[0].set_title("Metodo del codo")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inercia")
    axes[0].grid(alpha=0.25)

    silhouette = plot.dropna(subset=["silhouette"])
    axes[1].plot(silhouette["k"], silhouette["silhouette"], marker="o")
    best_flag = (
        pd.to_numeric(silhouette["recomendado_silhouette"], errors="coerce").fillna(0)
        if "recomendado_silhouette" in silhouette.columns
        else pd.Series(0, index=silhouette.index)
    )
    best = silhouette[best_flag.eq(1)]
    if not best.empty:
        row = best.iloc[0]
        axes[1].scatter([row["k"]], [row["silhouette"]], s=70)
        axes[1].axvline(row["k"], linestyle="--", linewidth=1)
    axes[1].set_title("Coeficiente silhouette")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette promedio")
    axes[1].set_ylim(-1, 1)
    axes[1].grid(alpha=0.25)

    fig.tight_layout(pad=0.7)
    return fig


def metric_value(value: float | int | None, decimals: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/D"
    return f"{value:.{decimals}f}"


def eda_chart_label(chart_type: object) -> str:
    labels = {
        "kde": "KDE",
        "histograma": "Histograma",
        "boxplot": "Boxplot",
        "scatter": "Scatter",
        "post_limpieza": "EDA posterior a la limpieza",
    }
    return labels.get(str(chart_type), str(chart_type))


def render_eda_manifest_row(row: pd.Series) -> None:
    chart_label = eda_chart_label(row.get("chart_type"))
    st.markdown(f"**{chart_label}**")

    title = row.get("titulo")
    if pd.notna(title):
        st.caption(str(title))

    image_path = Path(str(row.get("archivo", "")))
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.warning(f"No se encontro la imagen: {image_path}")

    interpretation = row.get("interpretacion")
    if pd.notna(interpretation):
        st.markdown(str(interpretation))

    subtitle = row.get("subtitulo")
    if pd.notna(subtitle):
        st.caption(str(subtitle))

    source_note = row.get("fuente_dato")
    if pd.notna(source_note):
        st.caption(f"Fuente del dato: {source_note}")


def render_eda_visual_tab(manifest_df: pd.DataFrame, verification_df: pd.DataFrame) -> None:
    st.subheader("EDA visual generado desde MySQL")
    st.caption(
        "Esta seccion reutiliza las graficas exportadas por `src.visualization.eda_visual_mysql` y las muestra dentro del dashboard."
    )

    controls_left, controls_right = st.columns([1, 1])
    with controls_left:
        st.caption(f"Directorio de salida: `{EDA_VISUAL_DIR}`")
    with controls_right:
        if EDA_REPORT_PATH.exists():
            st.download_button(
                "Descargar reporte EDA en Markdown",
                data=EDA_REPORT_PATH.read_text(encoding="utf-8").encode("utf-8"),
                file_name=EDA_REPORT_PATH.name,
                mime="text/markdown",
            )

    if manifest_df.empty:
        st.warning("No existe el manifest del EDA visual. Primero genera las graficas con el script del EDA.")
        st.code(
            "\n".join(
                [
                    "# Requiere variables definidas en .env",
                    "venv\\Scripts\\python.exe -m src.visualization.eda_visual_mysql",
                ]
            ),
            language="powershell",
        )
        return

    source_rows = manifest_df[manifest_df["seccion"].isin(["fuente", "verificacion"])].copy()
    explanatory_rows = manifest_df[manifest_df["seccion"].eq("explicativa")].copy()
    order = ["kde", "histograma", "boxplot", "scatter", "post_limpieza"]
    source_names = source_rows["fuente"].dropna().astype(str).unique().tolist()

    st.markdown("**Fuentes analizadas**")
    for index, source_name in enumerate(source_names):
        rows = source_rows[source_rows["fuente"].astype(str).eq(source_name)].copy()
        question_values = rows["pregunta"].dropna().astype(str).unique().tolist()
        with st.expander(source_name, expanded=index == 0):
            if question_values:
                st.markdown(f"**Pregunta de analisis:** {question_values[0]}")
            for chart_type in order:
                selected = rows[rows["chart_type"].astype(str).eq(chart_type)]
                if selected.empty:
                    continue
                render_eda_manifest_row(selected.iloc[0])
                if chart_type != order[-1]:
                    st.divider()

    st.markdown("**Verificacion posterior**")
    if verification_df.empty:
        st.info("No existe la tabla de verificacion posterior del EDA visual.")
    else:
        st.dataframe(verification_df, use_container_width=True, hide_index=True)

    st.markdown("**Graficas explicativas**")
    if explanatory_rows.empty:
        st.info("No existen graficas explicativas exportadas.")
    else:
        for _, row in explanatory_rows.iterrows():
            render_eda_manifest_row(row)
            st.divider()


def render_eda_whatsapp_tab(assets_df: pd.DataFrame, mysql_manifest_df: pd.DataFrame) -> None:
    st.subheader("EDA WhatsApp")
    st.caption(
        "Graficas generadas por `src.visualization.eda_visual`: KDE, histogramas y boxplots sobre la tabla municipal integrada."
    )
    st.caption(f"Directorio de salida: `{EDA_WHATSAPP_DIR}`")

    if assets_df.empty:
        st.warning("No hay imagenes del EDA WhatsApp generadas todavia.")
        st.code(
            "venv\\Scripts\\python.exe -m src.visualization.eda_visual",
            language="powershell",
        )
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Graficas", f"{len(assets_df):,}")
    metric_cols[1].metric("Variables", f"{assets_df['variable'].nunique():,}")
    metric_cols[2].metric("Tipos", f"{assets_df['chart_type'].nunique():,}")
    metric_cols[3].metric("Ultima generacion", assets_df["actualizado"].max().strftime("%Y-%m-%d %H:%M"))

    controls_left, controls_mid, controls_right = st.columns([1, 1.2, 1])
    with controls_left:
        chart_type = st.selectbox(
            "Tipo de grafica",
            ["kde", "histograma", "boxplot"],
            format_func=lambda value: {"kde": "KDE", "histograma": "Histograma", "boxplot": "Boxplot"}[value],
        )
    chart_assets = assets_df[assets_df["chart_type"].eq(chart_type)].copy()
    with controls_mid:
        variable_options = sorted(chart_assets["variable"].dropna().astype(str).unique().tolist())
        selected_variable = st.selectbox(
            "Variable",
            variable_options,
            format_func=lambda value: value.replace("_", " "),
        )
    selected_assets = chart_assets[chart_assets["variable"].astype(str).eq(selected_variable)].copy()
    with controls_right:
        compare_mysql = st.checkbox(
            "Comparar con EDA MySQL",
            value=not mysql_manifest_df.empty,
            disabled=mysql_manifest_df.empty,
        )

    if chart_type == "boxplot" and not selected_assets.empty:
        segment_options = ["Todos", *sorted(selected_assets["segmento"].dropna().astype(str).unique().tolist())]
        selected_segment = st.selectbox(
            "Segmentacion",
            segment_options,
            format_func=lambda value: value.replace("_", " "),
        )
        if selected_segment != "Todos":
            selected_assets = selected_assets[selected_assets["segmento"].astype(str).eq(selected_segment)]

    if selected_assets.empty:
        st.info("No hay graficas para la seleccion actual.")
        return

    mysql_matches = pd.DataFrame()
    if compare_mysql:
        mysql_matches = mysql_manifest_df[
            mysql_manifest_df["chart_type"].astype(str).eq(chart_type)
            & mysql_manifest_df["archivo"].astype(str).str.contains(selected_variable, case=False, regex=False)
        ].copy()

    for _, asset in selected_assets.iterrows():
        image_path = Path(str(asset["archivo"]))
        title = selected_variable.replace("_", " ")
        if asset.get("segmento"):
            title = f"{title} por {str(asset['segmento']).replace('_', ' ')}"
        st.markdown(f"**{title}**")

        if compare_mysql and not mysql_matches.empty:
            left_col, right_col = st.columns(2)
            with left_col:
                st.caption("EDA WhatsApp")
                st.image(str(image_path), use_container_width=True)
            with right_col:
                st.caption("EDA MySQL")
                render_eda_manifest_row(mysql_matches.iloc[0])
        else:
            st.image(str(image_path), use_container_width=True)
        st.divider()

    observations_path = EDA_WHATSAPP_DIR / "observaciones_visualizaciones.txt"
    if observations_path.exists():
        with st.expander("Observaciones exportadas"):
            st.text(observations_path.read_text(encoding="utf-8", errors="replace"))

    with st.expander("Inventario de graficas EDA WhatsApp"):
        st.dataframe(
            assets_df[["grupo", "chart_type", "variable", "segmento", "archivo", "actualizado"]],
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    st.title("Dashboard de viabilidad para granja solar en Colombia")
    st.caption(
        "Modelo preliminar rural. El ranking usa V_i sin demanda; la demanda queda solo como contexto favorable."
    )

    try:
        df = load_viability()
    except Exception as exc:
        st.error(f"No se pudo cargar la tabla de viabilidad: {exc}")
        st.stop()

    cluster_profile = load_cluster_profile()
    cluster_evaluation = load_cluster_evaluation()
    cluster_summary = load_cluster_summary()
    solar_costs = load_solar_costs()
    energy_prices = load_energy_prices()
    eda_manifest = load_eda_manifest()
    eda_verification = load_eda_verification()
    eda_whatsapp_assets = load_eda_whatsapp_assets()
    source_summary = pd.DataFrame(
        [
            {"dataset": "Viabilidad municipal", "origen": frame_source_label(df)},
            {"dataset": "Perfil de clusters", "origen": frame_source_label(cluster_profile)},
            {"dataset": "Evaluacion de K", "origen": frame_source_label(cluster_evaluation)},
            {"dataset": "Resumen de clusters", "origen": frame_source_label(cluster_summary)},
            {"dataset": "Escenarios solares", "origen": frame_source_label(solar_costs)},
            {"dataset": "Precios de energia", "origen": frame_source_label(energy_prices)},
            {
                "dataset": "EDA visual exportado",
                "origen": (
                    f"Archivos locales - {EDA_VISUAL_DIR.relative_to(PROJECT_ROOT)}"
                    if not eda_manifest.empty
                    else "No disponible"
                ),
            },
            {
                "dataset": "EDA WhatsApp",
                "origen": (
                    f"Archivos locales - {EDA_WHATSAPP_DIR.relative_to(PROJECT_ROOT)}"
                    if not eda_whatsapp_assets.empty
                    else "No disponible"
                ),
            },
        ]
    )
    source_warnings = [
        warning
        for warning in [
            frame_source_warning(df),
            frame_source_warning(cluster_profile),
            frame_source_warning(cluster_evaluation),
            frame_source_warning(cluster_summary),
            frame_source_warning(solar_costs),
            frame_source_warning(energy_prices),
        ]
        if warning
    ]

    with st.sidebar:
        st.header("Filtros")
        hide_outliers = st.checkbox(
            "Ocultar demanda en revision",
            value=False,
            help="Solo oculta flags de demanda; no cambia el score rural.",
        )
        only_eligible = st.checkbox(
            "Solo municipios no excluidos por R_i",
            value=True,
            help="Filtra R_i_preliminar = 1, incluyendo exclusion por zona urbana POT cuando existe.",
        )
        top_n = st.slider("Numero de municipios", 5, 30, 10)
        departments_available = sorted(df["departamento"].dropna().astype(str).unique().tolist())
        departments = st.multiselect("Departamentos", departments_available)
        with st.expander("Origen de datos", expanded=False):
            st.dataframe(source_summary, use_container_width=True, hide_index=True)
            st.caption(f"Base SQL objetivo: `{MYSQL_DATABASE}`")
            if source_warnings:
                st.warning("Algunos datasets usaron CSV fallback.")
                for warning in sorted(set(source_warnings)):
                    st.caption(warning)

    filtered = filter_data(df, departments, hide_outliers, only_eligible)
    top = top_municipalities(filtered, top_n)
    top_economic_climate = top_municipalities_by_score(
        filtered,
        top_n,
        ECONOMIC_CLIMATE_SCORE_COL,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Municipios base", f"{len(df):,}")
    col2.metric("Municipios filtrados", f"{len(filtered):,}")
    col3.metric("Top mostrado", f"{len(top):,}")
    col4.metric("Max V_i rural", metric_value(filtered[SCORE_COL].max() if not filtered.empty else None))

    if filtered.empty:
        st.warning("No hay municipios despues de aplicar los filtros.")
        st.stop()

    st.info(
        "La demanda XM no entra al ranking rural. Se muestra como bono/contexto para priorizar conexion comercial, "
        "pero no decide donde ubicar la granja."
    )

    (
        tab_top,
        tab_economic_climate,
        tab_explain,
        tab_map,
        tab_clusters,
        tab_costs,
        tab_eda,
        tab_eda_whatsapp,
        tab_glossary,
        tab_data,
    ) = st.tabs(
        [
            "Top municipios",
            "Economico-climatico",
            "Por que aparecen",
            "Mapa",
            "Clusters",
            "Economia",
            "EDA visual",
            "EDA WhatsApp",
            "Glosario",
            "Datos",
        ]
    )

    with tab_top:
        st.subheader(f"Top {len(top)} municipios mas probables")
        render_chart(make_top_score_chart(top))

        # --- Top 5 graficas individuales + radar #1 ---
        multidim_path = PROJECT_ROOT / "data" / "clean" / "viabilidad_municipal" / "viabilidad_municipal_multidimensional.csv"
        if multidim_path.exists():
            dim_df = pd.read_csv(multidim_path, dtype={"codigo_dane": "string"})
            dim_top5 = (
                dim_df
                .sort_values("v_i_multidimensional", ascending=False)
                .head(5)
                .reset_index(drop=True)
            )
            if not dim_top5.empty:
                st.markdown("---")
                st.subheader("Top 5 — Score total por municipio")
                render_chart(make_top5_individual_charts(dim_top5))

                st.markdown("---")
                st.subheader("Municipio #1 — Scores por dimension")
                col_radar, col_table = st.columns([3, 2])
                with col_radar:
                    render_chart(make_numero1_radar(dim_top5.iloc[0]))
                with col_table:
                    st.markdown(f"**{dim_top5.iloc[0]['municipio']} ({dim_top5.iloc[0]['departamento']})**")
                    dim_detail = {
                        "Dimension": list(_DIM_LABELS.values()),
                        "Score": [
                            round(float(dim_top5.iloc[0].get(d, 0) or 0), 4)
                            for d in _DIM_LABELS
                        ],
                        "Peso": [f"{_DIM_WEIGHTS[d]:.0%}" for d in _DIM_LABELS],
                        "Aporte": [
                            round(float(dim_top5.iloc[0].get(d, 0) or 0) * _DIM_WEIGHTS[d], 4)
                            for d in _DIM_LABELS
                        ],
                    }
                    st.dataframe(
                        pd.DataFrame(dim_detail),
                        use_container_width=True,
                        hide_index=True,
                    )
                    score_total = float(dim_top5.iloc[0].get("v_i_multidimensional", 0) or 0)
                    st.metric("Score total multidimensional", f"{score_total:.4f}")

        st.dataframe(
            format_table(top),
            use_container_width=True,
            hide_index=True,
        )

    with tab_economic_climate:
        st.subheader("Ranking economico-climatico")
        st.caption(
            "Este ranking no reemplaza el V_i rural; agrega tierra, agua y viento como sensibilidad adicional."
        )
        st.code(
            "V_i = R_i(0.30*S_i + 0.25*G_i + 0.20*P_i + 0.10*U_i + 0.08*T_i + 0.04*A_i + 0.03*W_i)"
        )
        left, right = st.columns(2)
        with left:
            render_chart(
                make_score_chart(
                    top,
                    SCORE_COL,
                    "Ranking original",
                    "V_i rural",
                )
            )
        with right:
            render_chart(
                make_score_chart(
                    top_economic_climate,
                    ECONOMIC_CLIMATE_SCORE_COL,
                    "Ranking economico-climatico",
                    "V_i economico-climatico",
                )
            )

        comparison = make_ranking_comparison(filtered, top_n)
        if comparison.empty:
            st.warning("No hay datos suficientes para comparar rankings.")
        else:
            st.markdown("**Comparacion entre rankings**")
            st.dataframe(comparison, use_container_width=True, hide_index=True)

        st.markdown("**Tierra, agua y viento**")
        cost_columns = [
            "ranking",
            "municipio",
            "departamento",
            ECONOMIC_CLIMATE_SCORE_COL,
            "precio_tierra_ha_cop",
            "score_tierra",
            "score_tierra_modelo",
            "tarifa_acueducto_m3_cop",
            "score_agua",
            "score_agua_modelo",
            "velocidad_viento_ms",
            "velocidad_viento_max_ms",
            "score_riesgo_viento",
            "score_riesgo_viento_modelo",
            "flag_dato_tierra",
            "flag_dato_agua",
            "flag_dato_viento",
        ]
        st.dataframe(
            top_economic_climate[[column for column in cost_columns if column in top_economic_climate.columns]],
            use_container_width=True,
            hide_index=True,
        )

        chart_left, chart_right = st.columns(2)
        with chart_left:
            render_chart(make_low_land_cost_chart(filtered, top_n))
        with chart_right:
            render_chart(make_wind_risk_chart(filtered, top_n))

    with tab_explain:
        st.subheader("Explicacion del score")
        left, right = st.columns([1.2, 1])
        with left:
            render_chart(make_components_chart(top))
        with right:
            render_chart(make_component_heatmap(top))

        st.markdown("**Formula usada**")
        st.code("V_i = R_i(0.35*S_i + 0.30*G_i + 0.25*P_i + 0.10*U_i)")
        st.caption("D_i demanda no entra a V_i; score_rural_con_bono_demanda es solo sensibilidad.")

        st.markdown("**Lectura rapida del top**")
        for _, row in top.head(10).iterrows():
            st.write(
                f"{int(row['ranking'])}. **{row['municipio']} ({row['departamento']})**: "
                f"{shorten(str(row['por_que_aparece']), width=180, placeholder='...')}"
            )

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            render_chart(make_scatter_solar_grid(filtered))
        with c2:
            render_chart(make_score_histogram(filtered))

    with tab_map:
        st.subheader("Ubicacion de municipios")
        heatmap_df = prepare_heatmap_data(filtered)
        if heatmap_df.empty:
            st.warning("No hay coordenadas y score suficientes para construir el mapa.")
        else:
            st.pydeck_chart(make_colombia_heatmap(filtered, top), use_container_width=True)
            st.caption(
                "La capa de calor pondera cada municipio por V_i rural. "
                "Los puntos oscuros resaltan el top mostrado con los filtros actuales."
            )
            st.dataframe(
                top[
                    [
                        column
                        for column in ["ranking", "municipio", "departamento", SCORE_COL, "lat", "lon"]
                        if column in top.columns
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with tab_clusters:
        st.subheader("Agrupamiento K-Means")
        if not cluster_summary.empty:
            summary = cluster_summary.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("K usado", metric_value(summary.get("k_usado"), decimals=0))
            c2.metric("K por codo", metric_value(summary.get("k_recomendado_codo"), decimals=0))
            c3.metric(
                "K por silhouette",
                metric_value(summary.get("k_recomendado_silhouette"), decimals=0),
            )
            c4.metric("Silhouette K usado", metric_value(summary.get("silhouette_k_usado")))
        if not cluster_evaluation.empty:
            render_chart(make_k_evaluation_chart(cluster_evaluation))
            with st.expander("Tabla de evaluacion de K"):
                st.dataframe(cluster_evaluation, use_container_width=True, hide_index=True)
        if cluster_profile.empty:
            st.warning("No existe perfil de clusters. Ejecuta src/scoring/kmeans_municipios.py.")
        else:
            render_chart(make_cluster_profile_chart(cluster_profile))
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
            fig, ax = plt.subplots(figsize=scaled_figsize(9, 4.5), dpi=FIGURE_DPI)
            ax.bar(solar_costs["scenario_name"], solar_costs["generacion_mwh_por_hectarea_anual"])
            ax.set_ylabel("MWh/ha-anio")
            ax.set_title("Generacion anual estimada por hectarea")
            fig.tight_layout(pad=0.7)
            render_chart(fig)

            st.divider()
            st.subheader("Valor anual por municipio")
            st.caption(
                "Calculo: energia anual generada por municipio * precio de compra COP/kWh. "
                "La barra ponderada multiplica ese valor por V_i rural."
            )

            scenario_names = solar_costs["scenario_name"].astype(str).tolist()
            default_scenario_index = scenario_names.index("base") if "base" in scenario_names else 0
            controls_left, controls_mid, controls_right = st.columns(3)
            with controls_left:
                selected_scenario_name = st.selectbox(
                    "Escenario de planta",
                    scenario_names,
                    index=default_scenario_index,
                    help="Define kW instalables por hectarea segun el escenario economico.",
                )
            with controls_mid:
                project_hectares = st.number_input(
                    "Area del proyecto (ha)",
                    min_value=0.1,
                    max_value=10000.0,
                    value=1.0,
                    step=1.0,
                )
            with controls_right:
                available_price_columns = [
                    column for column in PRICE_COLUMN_CANDIDATES if column in filtered.columns
                ]
                price_source_options = []
                if available_price_columns:
                    price_source_options.append("columna existente")
                if not energy_prices.empty:
                    price_source_options.append("Minenergia Caribe")
                price_source_options.append("cargar CSV real")
                selected_price_mode = st.selectbox(
                    "Fuente precio compra",
                    price_source_options,
                    help="No se usa supuesto: se requiere una columna o archivo real de tarifa.",
                )

            price_base = filtered.copy()
            price_column_option: str | None = None
            if selected_price_mode == "columna existente":
                price_column_option = st.selectbox(
                    "Columna de precio",
                    available_price_columns,
                )
            elif selected_price_mode == "Minenergia Caribe":
                price_base = merge_energy_prices(filtered, energy_prices)
                price_column_option = "precio_compra_cop_kwh"
                covered = int(price_base["precio_compra_cop_kwh"].notna().sum())
                st.caption(
                    f"Fuente oficial regional Minenergia: {covered:,} municipios con precio. "
                    "Solo cubre Caribe Air-e/AFINIA; no es tarifa municipal individual."
                )
            else:
                uploaded_prices = st.file_uploader(
                    "CSV real de precios por municipio",
                    type=["csv"],
                    help=(
                        "Debe incluir precio_compra_cop_kwh o tarifa_cop_kwh y codigo_dane; "
                        "tambien sirve municipio + departamento."
                    ),
                )
                if uploaded_prices is not None:
                    uploaded_df = pd.read_csv(uploaded_prices, dtype={"codigo_dane": "string"})
                    price_base = merge_energy_prices(filtered, uploaded_df)
                    price_column_option = "precio_compra_cop_kwh"

            selected_scenario = solar_costs[
                solar_costs["scenario_name"].astype(str).eq(selected_scenario_name)
            ].iloc[0]
            energy_value = (
                build_energy_value_estimate(
                    price_base,
                    selected_scenario,
                    hectares=project_hectares,
                    price_column=price_column_option,
                )
                if price_column_option
                else pd.DataFrame()
            )

            if energy_value.empty:
                st.warning(
                    "No se calcula valor anual porque falta un precio real de compra COP/kWh "
                    "asociado al municipio, departamento o codigo DANE."
                )
                st.markdown(
                    "- Fuente regional usada si aplica: Minenergia, tarifas Caribe Air-e/AFINIA.\n"
                    "- Para tarifa municipal real, descargar de SUI/Superservicios o del "
                    "comercializador local una tabla con `codigo_dane` y `precio_compra_cop_kwh`."
                )
            else:
                best_energy = energy_value.sort_values(
                    "valor_ponderado_viabilidad_cop_anual",
                    ascending=False,
                ).head(top_n)
                value_cols = st.columns(4)
                value_cols[0].metric(
                    "Capacidad estimada",
                    f"{energy_value['capacidad_mw_estimada'].iloc[0]:,.2f} MW",
                )
                value_cols[1].metric(
                    "Max MWh/anio",
                    f"{energy_value['generacion_mwh_anual_estimada'].max():,.0f}",
                )
                value_cols[2].metric(
                    "Max valor compra",
                    f"{energy_value['valor_compra_millones_cop_anual'].max():,.1f} M COP/anio",
                )
                value_cols[3].metric(
                    "Max ponderado V_i",
                    (
                        f"{energy_value['valor_ponderado_viabilidad_millones_cop_anual'].max():,.1f} "
                        "M COP/anio"
                    ),
                )

                chart_left, chart_right = st.columns([1.2, 1])
                with chart_left:
                    render_chart(make_energy_value_chart(energy_value, top_n))
                with chart_right:
                    render_chart(make_energy_value_scatter(energy_value))

                value_table_cols = [
                    "municipio",
                    "departamento",
                    SCORE_COL,
                    "annual_yield_kwh_kw_year",
                    "capacidad_mw_estimada",
                    "generacion_mwh_anual_estimada",
                    "precio_compra_cop_kwh",
                    "prestador_tarifa",
                    "fecha_publicacion",
                    "valor_compra_millones_cop_anual",
                    "valor_ponderado_viabilidad_millones_cop_anual",
                    "fuente_precio_compra",
                    "nota",
                ]
                st.dataframe(
                    best_energy[
                        [column for column in value_table_cols if column in best_energy.columns]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    with tab_eda:
        render_eda_visual_tab(eda_manifest, eda_verification)

    with tab_eda_whatsapp:
        render_eda_whatsapp_tab(eda_whatsapp_assets, eda_manifest)

    with tab_glossary:
        st.subheader("Glosario de variables")
        glossary = build_glossary(df)
        filter_left, filter_right = st.columns([1, 1])
        with filter_left:
            glossary_groups = st.multiselect(
                "Grupo",
                sorted(glossary["grupo"].unique().tolist()),
            )
        with filter_right:
            glossary_query = st.text_input("Buscar variable o concepto")

        glossary_view = filter_glossary(glossary, glossary_groups, glossary_query)
        st.dataframe(
            glossary_view[
                [
                    "grupo",
                    "variable",
                    "significado",
                    "lectura",
                    "uso_modelo",
                    "disponible_en_tabla",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Formula principal**")
        st.code("V_i = R_i(0.35*S_i + 0.30*G_i + 0.25*P_i + 0.10*U_i)")
        st.markdown("**Pesos del score rural**")
        weights_table = pd.DataFrame(
            [
                {"componente": COMPONENT_LABELS[column], "variable": column, "peso": weight}
                for column, weight in WEIGHTS.items()
            ]
        )
        st.dataframe(weights_table, use_container_width=True, hide_index=True)

    with tab_data:
        st.subheader("Control de calidad y datos")
        render_chart(make_classification_chart(df))
        with st.expander("Consultas SQL base para futuras graficas"):
            st.caption("Estas consultas ya funcionan sobre MySQL y sirven como base para Streamlit, Power BI o notebooks.")
            for title, sql_text in GRAPH_QUERY_SAMPLES.items():
                st.markdown(f"**{title}**")
                st.code(sql_text, language="sql")
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
        "Limitacion: PVOUT y pendiente son proxies puntuales; POT urbano depende de la capa muestreada; demanda es proxy regional XM; "
        "el resultado es priorizacion preliminar, no seleccion final de predios."
    )


if __name__ == "__main__":
    main()
