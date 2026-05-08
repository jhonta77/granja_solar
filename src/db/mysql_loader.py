from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import math
import unicodedata

import pandas as pd

from .mysql_cli import MySQLSettings, execute_sql


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
DEFAULT_SQL_OUT = PROJECT_ROOT / "data" / "interim" / "mysql" / "bootstrap_granja_solar.sql"


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.upper().split())


def normalize_text_lower(value: Any) -> str:
    return normalize_text(value).lower()


def read_csv(path: Path, *, string_columns: list[str] | None = None) -> pd.DataFrame:
    kwargs: dict[str, Any] = {}
    if string_columns:
        kwargs["dtype"] = {column: "string" for column in string_columns}
    df = pd.read_csv(path, **kwargs)
    for column in string_columns or []:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()
    return df


def coerce_bool(series: pd.Series) -> pd.Series:
    mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
    }
    lowered = series.astype("string").str.strip().str.lower()
    values = lowered.map(mapping)
    return values.astype("Int64")


def ensure_string_code(series: pd.Series, width: int = 5) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(width)


@dataclass
class ForeignKeySpec:
    columns: list[str]
    ref_table: str
    ref_columns: list[str]
    name: str


@dataclass
class TableSpec:
    name: str
    df: pd.DataFrame
    primary_key: list[str]
    foreign_keys: list[ForeignKeySpec] = field(default_factory=list)
    unique_keys: list[list[str]] = field(default_factory=list)
    indexes: list[list[str]] = field(default_factory=list)
    type_overrides: dict[str, str] = field(default_factory=dict)


def infer_mysql_type(series: pd.Series, *, override: str | None = None) -> str:
    if override is not None:
        return override
    non_null = series.dropna()
    if non_null.empty:
        return "TEXT"

    if pd.api.types.is_integer_dtype(series.dtype):
        minimum = int(non_null.min())
        maximum = int(non_null.max())
        if -(2**31) <= minimum <= maximum <= (2**31 - 1):
            return "INT"
        return "BIGINT"

    if pd.api.types.is_float_dtype(series.dtype):
        return "DOUBLE"

    if pd.api.types.is_bool_dtype(series.dtype):
        return "TINYINT(1)"

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "DATETIME"

    sample = non_null.astype("string")
    lowered = sample.str.lower()
    if lowered.isin(["0", "1", "true", "false"]).all():
        return "TINYINT(1)"

    max_length = int(sample.str.len().max())
    if max_length <= 5:
        return "VARCHAR(5)"
    if max_length <= 30:
        return "VARCHAR(30)"
    if max_length <= 80:
        return "VARCHAR(80)"
    if max_length <= 120:
        return "VARCHAR(120)"
    if max_length <= 255:
        return "VARCHAR(255)"
    return "TEXT"


def mysql_literal(value: Any) -> str:
    if value is None or value is pd.NA:
        return "NULL"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "NULL"
        return format(value, ".15g")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, bool):
        return "1" if value else "0"
    if pd.isna(value):
        return "NULL"
    text = str(value)
    text = text.replace("\\", "\\\\").replace("'", "''")
    text = text.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
    return f"'{text}'"


def build_insert_sql(table: TableSpec, batch_size: int = 300) -> str:
    if table.df.empty:
        return ""
    columns = list(table.df.columns)
    header = ", ".join(f"`{column}`" for column in columns)
    statements: list[str] = []
    records = table.df.to_dict(orient="records")
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        values_sql = []
        for row in batch:
            values = ", ".join(mysql_literal(row[column]) for column in columns)
            values_sql.append(f"({values})")
        statements.append(
            f"INSERT INTO `{table.name}` ({header}) VALUES\n" + ",\n".join(values_sql) + ";\n"
        )
    return "\n".join(statements)


def create_table_sql(table: TableSpec) -> str:
    column_lines = []
    for column in table.df.columns:
        mysql_type = infer_mysql_type(table.df[column], override=table.type_overrides.get(column))
        nullable = "NOT NULL" if column in table.primary_key else "NULL"
        column_lines.append(f"  `{column}` {mysql_type} {nullable}")

    constraints = []
    if table.primary_key:
        pk = ", ".join(f"`{column}`" for column in table.primary_key)
        constraints.append(f"  PRIMARY KEY ({pk})")

    for index, columns in enumerate(table.unique_keys, start=1):
        unique_cols = ", ".join(f"`{column}`" for column in columns)
        constraints.append(f"  UNIQUE KEY `uk_{table.name}_{index}` ({unique_cols})")

    for index, foreign_key in enumerate(table.foreign_keys, start=1):
        cols = ", ".join(f"`{column}`" for column in foreign_key.columns)
        ref_cols = ", ".join(f"`{column}`" for column in foreign_key.ref_columns)
        constraints.append(
            f"  CONSTRAINT `{foreign_key.name}` FOREIGN KEY ({cols}) "
            f"REFERENCES `{foreign_key.ref_table}` ({ref_cols})"
        )

    definition = ",\n".join(column_lines + constraints)
    return (
        f"CREATE TABLE `{table.name}` (\n"
        f"{definition}\n"
        f") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;\n"
    )


def create_index_sql(table: TableSpec) -> str:
    statements = []
    for index, columns in enumerate(table.indexes, start=1):
        cols = ", ".join(f"`{column}`" for column in columns)
        statements.append(f"CREATE INDEX `idx_{table.name}_{index}` ON `{table.name}` ({cols});")
    return "\n".join(statements) + ("\n" if statements else "")


def build_departments_and_municipalities() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_path = DATA_CLEAN_DIR / "base_municipios" / "municipios_distritos_colombia.csv"
    municipalities = read_csv(base_path, string_columns=["codigo_dane"])
    municipalities["codigo_dane"] = ensure_string_code(municipalities["codigo_dane"])

    departments = (
        municipalities[["departamento", "departamento_normalizado"]]
        .drop_duplicates()
        .sort_values("departamento_normalizado")
        .reset_index(drop=True)
    )
    departments.insert(0, "departamento_id", range(1, len(departments) + 1))
    departments["departamento_id"] = departments["departamento_id"].astype("Int64")

    municipalities = municipalities.merge(
        departments,
        on=["departamento", "departamento_normalizado"],
        how="left",
        validate="many_to_one",
    )
    municipalities["departamento_id"] = pd.to_numeric(
        municipalities["departamento_id"], errors="coerce"
    ).astype("Int64")
    municipality_columns = [
        "codigo_dane",
        "departamento_id",
        "objectid",
        "municipio",
        "municipio_normalizado",
        "categoria",
        "categoria_nombre",
        "normatividad",
        "area_km2_igac",
        "altitud_m",
        "fuente",
        "url_servicio",
        "fecha_descarga_utc",
        "centroide_lon",
        "centroide_lat",
        "lon",
        "lat",
    ]
    return departments, municipalities[municipality_columns].copy()


def build_pvout() -> pd.DataFrame:
    path = DATA_CLEAN_DIR / "pvout_municipios" / "pvout_puntos_extraidos.csv"
    df = read_csv(path, string_columns=["codigo_dane"])
    df["codigo_dane"] = ensure_string_code(df["codigo_dane"])
    return df[["codigo_dane", "pvout_kwh_kwp_day", "annual_yield_kwh_kw_year"]].copy()


def build_pendiente() -> tuple[pd.DataFrame, pd.DataFrame]:
    points_path = DATA_CLEAN_DIR / "pendientes_municipios" / "pendiente_puntos_extraidos.csv"
    criteria_path = DATA_CLEAN_DIR / "pendientes_municipios" / "criterios_viabilidad_pendiente.csv"
    df = read_csv(points_path, string_columns=["codigo_dane"])
    df["codigo_dane"] = ensure_string_code(df["codigo_dane"])
    if "identify_ok" in df.columns:
        df["identify_ok"] = coerce_bool(df["identify_ok"])
    municipality_columns = [
        "codigo_dane",
        "indice_punto",
        "lon_consulta",
        "lat_consulta",
        "pendiente_igac",
        "identify_ok",
        "identify_mensaje",
        "clase_igac",
        "pendiente_min_pct",
        "pendiente_max_pct",
        "max_slope_percent",
        "conditional_max_slope_percent",
        "viabilidad_pendiente",
        "score_pendiente",
        "criterio",
    ]
    criteria = read_csv(criteria_path)
    return df[municipality_columns].copy(), criteria.copy()


def build_subestaciones() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub_path = DATA_CLEAN_DIR / "subestaciones_upme" / "subestaciones_upme.csv"
    distance_path = DATA_CLEAN_DIR / "subestaciones_upme" / "distancia_subestacion_municipios.csv"
    sub = read_csv(sub_path, string_columns=["codigo_dane_municipio"])
    sub["codigo_dane_municipio"] = ensure_string_code(sub["codigo_dane_municipio"])
    if "activo" in sub.columns:
        sub["activo"] = coerce_bool(sub["activo"])
    if "ampliacion" in sub.columns:
        sub["ampliacion"] = coerce_bool(sub["ampliacion"])

    base_columns = [
        "id_subestacion",
        "cod_sub_upme",
        "nombre_subestacion",
        "nombre_normalizado",
        "codigo_dane_municipio",
        "latitud",
        "longitud",
        "altitud_m",
        "fuente",
        "url_servicio",
        "fecha_descarga_utc",
    ]
    substations = (
        sub.sort_values(["id_subestacion", "nombre_normalizado"])
        .drop_duplicates(subset=["id_subestacion"], keep="first")[base_columns]
        .copy()
    )

    config_columns = [
        "id_subestacion",
        "id_nivel_tension",
        "nivel_tension",
        "id_tension_sub",
        "tension_descripcion",
        "id_estado_sub",
        "estado_subestacion",
        "porcentaje_carga",
        "capacidad_mva",
        "activo",
        "ampliacion",
        "observacion",
    ]
    configs = (
        sub[config_columns]
        .drop_duplicates()
        .sort_values(["id_subestacion", "id_tension_sub", "tension_descripcion"])
        .reset_index(drop=True)
    )
    configs.insert(0, "subestacion_config_id", range(1, len(configs) + 1))
    configs["subestacion_config_id"] = configs["subestacion_config_id"].astype("Int64")

    distance = read_csv(distance_path, string_columns=["codigo_dane"])
    distance["codigo_dane"] = ensure_string_code(distance["codigo_dane"])
    distance["id_subestacion_mas_cercana"] = pd.to_numeric(
        distance["id_subestacion_mas_cercana"], errors="coerce"
    ).astype("Int64")
    distance_columns = [
        "codigo_dane",
        "dist_subestacion_km",
        "subestacion_mas_cercana",
        "id_subestacion_mas_cercana",
        "cod_sub_upme_mas_cercana",
        "nivel_tension_mas_cercana",
        "tension_mas_cercana",
        "estado_subestacion_mas_cercana",
        "capacidad_mva_mas_cercana",
        "subestacion_lon",
        "subestacion_lat",
        "criterio_red",
    ]
    return substations, configs, distance[distance_columns].copy()


def build_runap() -> tuple[pd.DataFrame, pd.DataFrame]:
    restrictions_path = DATA_CLEAN_DIR / "runap_protegidas" / "runap_restricciones_municipios.csv"
    categories_path = DATA_CLEAN_DIR / "runap_protegidas" / "runap_intersecciones_categoria_municipio.csv"
    restrictions = read_csv(restrictions_path, string_columns=["codigo_dane"])
    restrictions["codigo_dane"] = ensure_string_code(restrictions["codigo_dane"])
    restriction_columns = [
        "codigo_dane",
        "area_municipio_km2_calc",
        "area_protegida_km2_runap",
        "pct_area_protegida_runap_raw",
        "pct_area_protegida_runap",
        "area_no_protegida_km2_runap",
        "u_i_no_protegido_runap",
        "r_i_runap",
        "umbral_exclusion_runap",
        "clasificacion_restriccion_runap",
        "criterio_runap",
    ]
    restrictions = restrictions[restriction_columns].copy()
    restrictions["r_i_runap"] = pd.to_numeric(restrictions["r_i_runap"], errors="coerce").astype("Int64")

    categories = read_csv(categories_path, string_columns=["codigo_dane"])
    categories["codigo_dane"] = ensure_string_code(categories["codigo_dane"])
    return restrictions, categories.copy()


def build_pot() -> tuple[pd.DataFrame, pd.DataFrame]:
    classifications_path = DATA_CLEAN_DIR / "usos_suelo_pot" / "usos_pot_clasificacion_leyenda.csv"
    points_path = DATA_CLEAN_DIR / "usos_suelo_pot" / "usos_pot_puntos_extraidos.csv"
    classifications = read_csv(classifications_path)
    points = read_csv(points_path, string_columns=["codigo_dane"])
    points["codigo_dane"] = ensure_string_code(points["codigo_dane"])

    bool_columns = ["apto_doble_uso_pastoreo", "restriccion_territorial_proxy", "pot_identify_ok"]
    for column in bool_columns:
        if column in classifications.columns:
            classifications[column] = coerce_bool(classifications[column])
        if column in points.columns:
            points[column] = coerce_bool(points[column])

    classifications["uso_pot_normalizado_loader"] = classifications["uso_pot_normalizado"].map(
        normalize_text_lower
    )
    dedupe_columns = [
        "layer_id",
        "layer_name",
        "tipo_capa",
        "uso_pot_normalizado_loader",
        "categoria_aptitud_pot",
        "u_i_uso_suelo_proxy",
        "apto_doble_uso_pastoreo",
        "restriccion_territorial_proxy",
        "criterio_clasificacion",
    ]
    available_dedupe = [column for column in dedupe_columns if column in classifications.columns]
    pot_classifications = (
        classifications[available_dedupe]
        .drop_duplicates()
        .sort_values(["layer_id", "tipo_capa", "uso_pot_normalizado_loader"])
        .reset_index(drop=True)
    )
    pot_classifications.insert(0, "pot_clasificacion_id", range(1, len(pot_classifications) + 1))
    pot_classifications["pot_clasificacion_id"] = pot_classifications["pot_clasificacion_id"].astype(
        "Int64"
    )

    points["uso_pot_normalizado_loader"] = points["uso_pot"].map(normalize_text_lower)
    match_columns = [
        "layer_id",
        "tipo_capa",
        "uso_pot_normalizado_loader",
        "categoria_aptitud_pot",
        "u_i_uso_suelo_proxy",
        "apto_doble_uso_pastoreo",
        "restriccion_territorial_proxy",
        "criterio_clasificacion",
    ]
    source_points = points.rename(
        columns={
            "layer_id_pot": "layer_id",
            "tipo_capa_pot": "tipo_capa",
        }
    )
    lookup_columns = ["pot_clasificacion_id"] + [column for column in match_columns if column in pot_classifications.columns]
    source_match_columns = [column for column in match_columns if column in source_points.columns and column in pot_classifications.columns]
    if source_match_columns:
        source_points = source_points.merge(
            pot_classifications[["pot_clasificacion_id"] + source_match_columns],
            on=source_match_columns,
            how="left",
        )
    source_points["pot_clasificacion_id"] = pd.to_numeric(
        source_points["pot_clasificacion_id"], errors="coerce"
    ).astype("Int64")

    municipality_columns = [
        "codigo_dane",
        "pot_clasificacion_id",
        "municipio_pot",
        "layer_id",
        "layer_name_pot",
        "tipo_capa",
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
    ]
    municipality_points = source_points[municipality_columns].rename(
        columns={
            "layer_id": "layer_id_pot",
            "tipo_capa": "tipo_capa_pot",
        }
    )
    return pot_classifications.drop(columns=["uso_pot_normalizado_loader"]), municipality_points


def build_xm() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    zones_path = DATA_CLEAN_DIR / "xm_demanda_municipal" / "xm_zonas_eda_etiquetadas.csv"
    mapping_path = DATA_CLEAN_DIR / "xm_demanda_municipal" / "mapeo_departamento_zona_xm.csv"
    demand_path = DATA_CLEAN_DIR / "xm_demanda_municipal" / "xm_demanda_municipal_proxy.csv"

    zones = read_csv(zones_path)
    zones = zones.sort_values(["tipo_archivo", "zona"]).reset_index(drop=True)
    zones.insert(0, "zona_xm_id", range(1, len(zones) + 1))
    zones["zona_xm_id"] = zones["zona_xm_id"].astype("Int64")

    mapping = read_csv(mapping_path)
    mapping["departamento_normalizado"] = mapping["departamento_normalizado"].astype("string").str.strip()
    mapping = mapping.merge(
        zones[["zona_xm_id", "tipo_archivo", "zona"]],
        left_on=["zona_xm"],
        right_on=["zona"],
        how="left",
    )
    mapping = mapping.drop(columns=["zona"]).rename(columns={"zona_xm": "zona_xm_nombre"})
    mapping["zona_xm_id"] = pd.to_numeric(mapping["zona_xm_id"], errors="coerce").astype("Int64")
    if "revision" in mapping.columns:
        mapping["revision"] = coerce_bool(mapping["revision"])
    if "zona_xm_existe_en_fuente" in mapping.columns:
        mapping["zona_xm_existe_en_fuente"] = coerce_bool(mapping["zona_xm_existe_en_fuente"])

    demand = read_csv(demand_path, string_columns=["codigo_dane"])
    demand["codigo_dane"] = ensure_string_code(demand["codigo_dane"])
    demand = demand.merge(
        zones[["zona_xm_id", "zona"]],
        left_on="zona_xm_demanda",
        right_on="zona",
        how="left",
    ).drop(columns=["zona"])
    demand["zona_xm_id"] = pd.to_numeric(demand["zona_xm_id"], errors="coerce").astype("Int64")
    for column in ["flag_revision_demanda", "flag_atipico_eda_demanda", "demanda_xm_disponible"]:
        if column in demand.columns:
            demand[column] = coerce_bool(demand[column])
    demand_columns = [
        "codigo_dane",
        "zona_xm_id",
        "zona_xm_demanda",
        "tipo_mapeo_demanda",
        "confianza_mapeo_demanda",
        "flag_revision_demanda",
        "flag_atipico_eda_demanda",
        "nota_mapeo_demanda",
        "demanda_xm_valor_promedio",
        "demanda_xm_valor_maximo",
        "demanda_xm_valor_p95",
        "demanda_xm_desviacion",
        "demanda_xm_dias_cubiertos",
        "demanda_xm_score_relativo_fuente",
        "demanda_xm_probabilidad_relativa_fuente",
        "demanda_xm_proxy_mwh_o_unidad_fuente",
        "d_i_demanda",
        "demanda_xm_disponible",
        "decision_eda_demanda",
    ]
    return zones, mapping, demand[demand_columns].copy()


def build_nasa() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_path = DATA_CLEAN_DIR / "nasa_power" / "nasa_power_puntos_diario.csv"
    summary_path = DATA_CLEAN_DIR / "nasa_power" / "nasa_power_resumen_puntos.csv"
    daily = read_csv(daily_path)
    summary = read_csv(summary_path)
    points = (
        daily[["punto_id", "lon", "lat", "pvout_kwh_kwp_day"]]
        .drop_duplicates(subset=["punto_id"])
        .sort_values("punto_id")
        .reset_index(drop=True)
    )
    return points, daily.copy(), summary.copy()


def build_solar_and_prices(departments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    solar_path = DATA_CLEAN_DIR / "solar_costs" / "solar_escenarios_por_hectarea.csv"
    prices_path = DATA_CLEAN_DIR / "energy_prices" / "precios_compra_energia_minenergia_caribe.csv"
    solar = read_csv(solar_path)
    prices = read_csv(prices_path)
    prices["departamento_normalizado"] = prices["departamento"].map(normalize_text)
    prices = prices.merge(
        departments[["departamento_id", "departamento_normalizado"]],
        on="departamento_normalizado",
        how="left",
    )
    prices.insert(0, "precio_id", range(1, len(prices) + 1))
    prices["precio_id"] = prices["precio_id"].astype("Int64")
    prices["departamento_id"] = pd.to_numeric(prices["departamento_id"], errors="coerce").astype(
        "Int64"
    )
    ordered_columns = [
        "precio_id",
        "departamento_id",
        "departamento",
        "departamento_normalizado",
        "prestador_tarifa",
        "precio_compra_cop_kwh",
        "fecha_publicacion",
        "fuente_url",
        "nota",
    ]
    return solar.copy(), prices[ordered_columns].copy()


def build_simem() -> dict[str, pd.DataFrame]:
    base = DATA_CLEAN_DIR / "simem"
    result = {
        "simem_datasets_descargados": read_csv(base / "resumen_datasets_descargados.csv"),
        "simem_catalogo_hidroelectricas": read_csv(base / "catalogo_hidroelectricas_simem.csv"),
        "simem_catalogo_publico": read_csv(base / "catalogo_publico_simem.csv"),
        "simem_a0cf2a_embalses": read_csv(base / "A0CF2A" / "registros.csv"),
        "simem_ba1c55_aportes_hidricos": read_csv(base / "BA1C55" / "registros.csv"),
        "simem_b0e933_reservas_hidraulicas": read_csv(base / "B0E933" / "registros.csv"),
        "simem_f99e13_subestaciones": read_csv(base / "F99E13" / "registros.csv"),
    }
    result["simem_a0cf2a_embalses"].insert(0, "dataset_id", "A0CF2A")
    result["simem_ba1c55_aportes_hidricos"].insert(0, "dataset_id", "BA1C55")
    result["simem_b0e933_reservas_hidraulicas"].insert(0, "dataset_id", "B0E933")
    result["simem_f99e13_subestaciones"].insert(0, "dataset_id", "F99E13")
    return result


def _valid_dane_codes() -> set[str]:
    """Retorna el conjunto de codigo_dane que existen en municipios_distritos_colombia.csv."""
    path = DATA_CLEAN_DIR / "base_municipios" / "municipios_distritos_colombia.csv"
    if not path.exists():
        return set()
    df = read_csv(path, string_columns=["codigo_dane"])
    return set(ensure_string_code(df["codigo_dane"]))


def _filter_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas cuyo codigo_dane no existe en la tabla municipios."""
    valid = _valid_dane_codes()
    if not valid:
        return df
    mask = df["codigo_dane"].isin(valid)
    dropped = (~mask).sum()
    if dropped:
        print(f"  Filtrados {dropped} registros sin codigo_dane en municipios")
    return df[mask].copy()


def build_nasa_municipios() -> pd.DataFrame | None:
    path = DATA_CLEAN_DIR / "nasa_power" / "nasa_power_resumen_municipios.csv"
    if not path.exists():
        return None
    df = read_csv(path, string_columns=["codigo_dane"])
    df["codigo_dane"] = ensure_string_code(df["codigo_dane"])
    return _filter_valid(df)


def build_upra_tierra() -> pd.DataFrame | None:
    path = DATA_CLEAN_DIR / "upra_tierra" / "upra_precio_tierra_municipal.csv"
    if not path.exists():
        return None
    df = read_csv(path, string_columns=["codigo_dane"])
    df["codigo_dane"] = ensure_string_code(df["codigo_dane"])
    return _filter_valid(df)


def build_upra_agropecuario() -> pd.DataFrame | None:
    path = DATA_CLEAN_DIR / "upra_agropecuario" / "upra_agropecuario_municipal.csv"
    if not path.exists():
        return None
    df = read_csv(path, string_columns=["codigo_dane"])
    df["codigo_dane"] = ensure_string_code(df["codigo_dane"])
    return _filter_valid(df)


def build_invias_vias() -> pd.DataFrame | None:
    path = DATA_CLEAN_DIR / "invias_vias" / "distancia_vias_municipios.csv"
    if not path.exists():
        return None
    df = read_csv(path, string_columns=["codigo_dane"])
    df["codigo_dane"] = ensure_string_code(df["codigo_dane"])
    return _filter_valid(df)


def build_ideam_riesgo() -> pd.DataFrame | None:
    path = DATA_CLEAN_DIR / "ideam_riesgo" / "ideam_riesgo_municipal.csv"
    if not path.exists():
        return None
    df = read_csv(path, string_columns=["codigo_dane"])
    df["codigo_dane"] = ensure_string_code(df["codigo_dane"])
    return _filter_valid(df)


def build_sui_agua() -> pd.DataFrame | None:
    path = DATA_CLEAN_DIR / "sui_agua" / "sui_costo_agua_municipal.csv"
    if not path.exists():
        return None
    df = read_csv(path, string_columns=["codigo_dane"])
    df["codigo_dane"] = ensure_string_code(df["codigo_dane"])
    return _filter_valid(df)


def build_imrc_dnp() -> pd.DataFrame | None:
    path = DATA_CLEAN_DIR / "ideam_riesgo" / "ideam_riesgo_municipal.csv"
    if not path.exists():
        return None
    df = read_csv(path, string_columns=["codigo_dane"])
    df["codigo_dane"] = ensure_string_code(df["codigo_dane"])
    return _filter_valid(df)


def build_sui_aseo() -> pd.DataFrame | None:
    path = DATA_CLEAN_DIR / "sui_aseo" / "sui_aseo_municipal.csv"
    if not path.exists():
        return None
    df = read_csv(path, string_columns=["codigo_dane"])
    df["codigo_dane"] = ensure_string_code(df["codigo_dane"])
    return _filter_valid(df)


def build_ideam_bart() -> pd.DataFrame | None:
    path = DATA_CLEAN_DIR / "ideam_bart" / "ideam_clima_municipal.csv"
    if not path.exists():
        return None
    df = read_csv(path, string_columns=["codigo_dane"])
    df["codigo_dane"] = ensure_string_code(df["codigo_dane"])
    return _filter_valid(df)


def build_era5() -> pd.DataFrame | None:
    path = DATA_CLEAN_DIR / "copernicus_era5" / "era5_resumen_municipios.csv"
    if not path.exists():
        return None
    df = read_csv(path, string_columns=["codigo_dane"])
    df["codigo_dane"] = ensure_string_code(df["codigo_dane"])
    return _filter_valid(df)


def build_upra_conflicto() -> pd.DataFrame | None:
    path = DATA_CLEAN_DIR / "upra_conflicto" / "upra_conflicto_municipal.csv"
    if not path.exists():
        return None
    df = read_csv(path, string_columns=["codigo_dane"])
    df["codigo_dane"] = ensure_string_code(df["codigo_dane"])
    return _filter_valid(df)


def build_viabilidad_multidimensional() -> pd.DataFrame | None:
    path = DATA_CLEAN_DIR / "viabilidad_municipal" / "viabilidad_municipal_multidimensional.csv"
    if not path.exists():
        return None
    df = read_csv(path, string_columns=["codigo_dane"])
    df["codigo_dane"] = ensure_string_code(df["codigo_dane"])
    dim_cols = [
        "codigo_dane",
        "score_fisico", "score_electrico", "score_economico",
        "score_agropecuario", "score_riesgo",
        "v_i_multidimensional", "clasificacion_multidim",
        "dims_disponibles", "dims_faltantes",
    ]
    available = [c for c in dim_cols if c in df.columns]
    return _filter_valid(df[available].copy())


def build_viabilidad_and_clusters() -> dict[str, pd.DataFrame]:
    viability_path = DATA_CLEAN_DIR / "viabilidad_municipal" / "viabilidad_municipal_preliminar.csv"
    cluster_dir = DATA_CLEAN_DIR / "clusters_municipios"
    viability = read_csv(viability_path, string_columns=["codigo_dane"])
    viability["codigo_dane"] = ensure_string_code(viability["codigo_dane"])

    cluster_municipal = read_csv(cluster_dir / "municipios_clusters_kmeans.csv", string_columns=["codigo_dane"])
    cluster_municipal["codigo_dane"] = ensure_string_code(cluster_municipal["codigo_dane"])

    cluster_excluded = read_csv(cluster_dir / "municipios_excluidos_kmeans.csv", string_columns=["codigo_dane"])
    cluster_excluded["codigo_dane"] = ensure_string_code(cluster_excluded["codigo_dane"])

    return {
        "viabilidad_municipal": viability,
        "cluster_municipal": cluster_municipal,
        "cluster_perfiles": read_csv(cluster_dir / "perfil_clusters_kmeans.csv"),
        "cluster_centroides": read_csv(cluster_dir / "centroides_clusters_kmeans.csv"),
        "cluster_evaluacion_k": read_csv(cluster_dir / "evaluacion_kmeans_k.csv"),
        "cluster_resumen_seleccion": read_csv(cluster_dir / "resumen_seleccion_kmeans.csv"),
        "cluster_municipios_excluidos": cluster_excluded,
    }


def build_tables() -> list[TableSpec]:
    departments, municipalities = build_departments_and_municipalities()
    pvout = build_pvout()
    pendiente, pendiente_criterios = build_pendiente()
    substations, sub_configs, municipality_grid = build_subestaciones()
    runap, runap_category = build_runap()
    pot_classifications, municipality_pot = build_pot()
    xm_zones, xm_mapping, municipality_demand = build_xm()
    nasa_points, nasa_daily, nasa_summary = build_nasa()
    solar_scenarios, energy_prices = build_solar_and_prices(departments)
    simem = build_simem()
    model_data = build_viabilidad_and_clusters()

    nasa_municipios = build_nasa_municipios()
    upra_tierra = build_upra_tierra()
    upra_agro = build_upra_agropecuario()
    invias = build_invias_vias()
    ideam = build_ideam_riesgo()
    sui_agua = build_sui_agua()
    imrc_dnp = build_imrc_dnp()
    sui_aseo = build_sui_aseo()
    ideam_bart = build_ideam_bart()
    era5 = build_era5()
    upra_conflicto = build_upra_conflicto()
    viabilidad_multidim = build_viabilidad_multidimensional()

    tables = [
        TableSpec(
            name="departamentos",
            df=departments,
            primary_key=["departamento_id"],
            unique_keys=[["departamento_normalizado"]],
            indexes=[["departamento"]],
        ),
        TableSpec(
            name="municipios",
            df=municipalities,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(
                    columns=["departamento_id"],
                    ref_table="departamentos",
                    ref_columns=["departamento_id"],
                    name="fk_municipios_departamento",
                )
            ],
            indexes=[["municipio_normalizado"], ["departamento_id"]],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ),
        TableSpec(
            name="municipio_pvout",
            df=pvout,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_municipio_pvout_municipio")
            ],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ),
        TableSpec(
            name="pendiente_criterios",
            df=pendiente_criterios,
            primary_key=["clase_igac"],
        ),
        TableSpec(
            name="municipio_pendiente",
            df=pendiente,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_municipio_pendiente_municipio")
            ],
            indexes=[["viabilidad_pendiente"]],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ),
        TableSpec(
            name="subestaciones",
            df=substations,
            primary_key=["id_subestacion"],
            unique_keys=[["cod_sub_upme"]],
            indexes=[["nombre_normalizado"], ["codigo_dane_municipio"]],
            type_overrides={"codigo_dane_municipio": "CHAR(5)"},
        ),
        TableSpec(
            name="subestacion_configuraciones",
            df=sub_configs,
            primary_key=["subestacion_config_id"],
            foreign_keys=[
                ForeignKeySpec(
                    ["id_subestacion"],
                    "subestaciones",
                    ["id_subestacion"],
                    "fk_subestacion_config_subestacion",
                )
            ],
            indexes=[["id_subestacion"], ["estado_subestacion"]],
        ),
        TableSpec(
            name="municipio_red",
            df=municipality_grid,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_municipio_red_municipio"),
                ForeignKeySpec(
                    ["id_subestacion_mas_cercana"],
                    "subestaciones",
                    ["id_subestacion"],
                    "fk_municipio_red_subestacion",
                ),
            ],
            indexes=[["id_subestacion_mas_cercana"]],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ),
        TableSpec(
            name="municipio_runap",
            df=runap,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_municipio_runap_municipio")
            ],
            indexes=[["clasificacion_restriccion_runap"]],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ),
        TableSpec(
            name="municipio_runap_categoria",
            df=runap_category,
            primary_key=["codigo_dane", "ap_categoria"],
            foreign_keys=[
                ForeignKeySpec(
                    ["codigo_dane"],
                    "municipios",
                    ["codigo_dane"],
                    "fk_municipio_runap_categoria_municipio",
                )
            ],
            indexes=[["ap_categoria"]],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ),
        TableSpec(
            name="pot_clasificaciones",
            df=pot_classifications,
            primary_key=["pot_clasificacion_id"],
            indexes=[["layer_id"], ["tipo_capa"], ["categoria_aptitud_pot"]],
        ),
        TableSpec(
            name="municipio_pot",
            df=municipality_pot,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_municipio_pot_municipio"),
                ForeignKeySpec(
                    ["pot_clasificacion_id"],
                    "pot_clasificaciones",
                    ["pot_clasificacion_id"],
                    "fk_municipio_pot_clasificacion",
                ),
            ],
            indexes=[["pot_clasificacion_id"], ["tipo_capa_pot"]],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ),
        TableSpec(
            name="xm_zonas",
            df=xm_zones,
            primary_key=["zona_xm_id"],
            unique_keys=[["tipo_archivo", "zona"]],
            indexes=[["zona_grupo"], ["zona"]],
        ),
        TableSpec(
            name="xm_mapeo_departamento_zona",
            df=xm_mapping,
            primary_key=["departamento_normalizado"],
            foreign_keys=[
                ForeignKeySpec(
                    ["zona_xm_id"],
                    "xm_zonas",
                    ["zona_xm_id"],
                    "fk_xm_mapeo_departamento_zona",
                )
            ],
            indexes=[["zona_xm_id"]],
        ),
        TableSpec(
            name="municipio_demanda_xm",
            df=municipality_demand,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_municipio_demanda_municipio"),
                ForeignKeySpec(["zona_xm_id"], "xm_zonas", ["zona_xm_id"], "fk_municipio_demanda_zona"),
            ],
            indexes=[["zona_xm_id"], ["flag_revision_demanda"]],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ),
        TableSpec(
            name="nasa_puntos",
            df=nasa_points,
            primary_key=["punto_id"],
        ),
        TableSpec(
            name="nasa_power_diaria",
            df=nasa_daily,
            primary_key=["punto_id", "fecha"],
            foreign_keys=[
                ForeignKeySpec(["punto_id"], "nasa_puntos", ["punto_id"], "fk_nasa_diaria_punto")
            ],
            indexes=[["fecha"]],
        ),
        TableSpec(
            name="nasa_power_resumen",
            df=nasa_summary,
            primary_key=["punto_id"],
            foreign_keys=[
                ForeignKeySpec(["punto_id"], "nasa_puntos", ["punto_id"], "fk_nasa_resumen_punto")
            ],
        ),
        TableSpec(
            name="solar_escenarios",
            df=solar_scenarios,
            primary_key=["scenario_name"],
        ),
        TableSpec(
            name="precios_energia_departamento",
            df=energy_prices,
            primary_key=["precio_id"],
            foreign_keys=[
                ForeignKeySpec(
                    ["departamento_id"],
                    "departamentos",
                    ["departamento_id"],
                    "fk_precios_departamento",
                )
            ],
            indexes=[["departamento_id"], ["prestador_tarifa"]],
        ),
        TableSpec(
            name="simem_datasets_descargados",
            df=simem["simem_datasets_descargados"],
            primary_key=["dataset_id"],
        ),
        TableSpec(
            name="simem_catalogo_hidroelectricas",
            df=simem["simem_catalogo_hidroelectricas"],
            primary_key=["idConfiguracionGeneracionArchivos"],
            indexes=[["idDataset"]],
        ),
        TableSpec(
            name="simem_catalogo_publico",
            df=simem["simem_catalogo_publico"],
            primary_key=["idConfiguracionGeneracionArchivos"],
            indexes=[["idDataset"]],
        ),
        TableSpec(
            name="simem_a0cf2a_embalses",
            df=simem["simem_a0cf2a_embalses"],
            primary_key=["dataset_id", "fecha", "codigoembalse", "fechaejecucion"],
            foreign_keys=[
                ForeignKeySpec(
                    ["dataset_id"],
                    "simem_datasets_descargados",
                    ["dataset_id"],
                    "fk_simem_a0cf2a_dataset",
                )
            ],
        ),
        TableSpec(
            name="simem_ba1c55_aportes_hidricos",
            df=simem["simem_ba1c55_aportes_hidricos"],
            primary_key=["dataset_id", "fecha", "codigoseriehidrologica"],
            foreign_keys=[
                ForeignKeySpec(
                    ["dataset_id"],
                    "simem_datasets_descargados",
                    ["dataset_id"],
                    "fk_simem_ba1c55_dataset",
                )
            ],
        ),
        TableSpec(
            name="simem_b0e933_reservas_hidraulicas",
            df=simem["simem_b0e933_reservas_hidraulicas"],
            primary_key=["dataset_id", "fecha", "codigoembalse"],
            foreign_keys=[
                ForeignKeySpec(
                    ["dataset_id"],
                    "simem_datasets_descargados",
                    ["dataset_id"],
                    "fk_simem_b0e933_dataset",
                )
            ],
        ),
        TableSpec(
            name="simem_f99e13_subestaciones",
            df=simem["simem_f99e13_subestaciones"],
            primary_key=["dataset_id", "fecha", "codigosubestacion"],
            foreign_keys=[
                ForeignKeySpec(
                    ["dataset_id"],
                    "simem_datasets_descargados",
                    ["dataset_id"],
                    "fk_simem_f99e13_dataset",
                )
            ],
        ),
        TableSpec(
            name="viabilidad_municipal",
            df=model_data["viabilidad_municipal"],
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_viabilidad_municipio")
            ],
            indexes=[
                [column]
                for column in [
                    "clasificacion_preliminar",
                    "v_i_modelo_rural",
                    "v_i_modelo_rural_economico_climatico",
                ]
                if column in model_data["viabilidad_municipal"].columns
            ],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ),
        TableSpec(
            name="cluster_municipal",
            df=model_data["cluster_municipal"],
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_cluster_municipio")
            ],
            indexes=[["cluster_kmeans"], ["cluster_kmeans_label"]],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ),
        TableSpec(
            name="cluster_perfiles",
            df=model_data["cluster_perfiles"],
            primary_key=["cluster_kmeans"],
        ),
        TableSpec(
            name="cluster_centroides",
            df=model_data["cluster_centroides"],
            primary_key=["cluster_kmeans"],
        ),
        TableSpec(
            name="cluster_evaluacion_k",
            df=model_data["cluster_evaluacion_k"],
            primary_key=["k"],
        ),
        TableSpec(
            name="cluster_resumen_seleccion",
            df=model_data["cluster_resumen_seleccion"],
            primary_key=["k_usado"],
        ),
        TableSpec(
            name="cluster_municipios_excluidos",
            df=model_data["cluster_municipios_excluidos"],
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(
                    ["codigo_dane"],
                    "municipios",
                    ["codigo_dane"],
                    "fk_cluster_excluido_municipio",
                )
            ],
            indexes=[["cluster_exclusion_reason"]],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ),
    ]

    optional_tables = []

    if nasa_municipios is not None:
        optional_tables.append(TableSpec(
            name="municipio_nasa_climatico",
            df=nasa_municipios,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_nasa_climatico_municipio")
            ],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ))

    if upra_tierra is not None:
        optional_tables.append(TableSpec(
            name="municipio_upra_tierra",
            df=upra_tierra,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_upra_tierra_municipio")
            ],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ))

    if upra_agro is not None:
        optional_tables.append(TableSpec(
            name="municipio_upra_agropecuario",
            df=upra_agro,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_upra_agro_municipio")
            ],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ))

    if invias is not None:
        optional_tables.append(TableSpec(
            name="municipio_invias_vias",
            df=invias,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_invias_municipio")
            ],
            indexes=[["dist_via_primaria_km"]],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ))

    if ideam is not None:
        optional_tables.append(TableSpec(
            name="municipio_ideam_riesgo",
            df=ideam,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_ideam_municipio")
            ],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ))

    if sui_agua is not None:
        optional_tables.append(TableSpec(
            name="municipio_sui_agua",
            df=sui_agua,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_sui_agua_municipio")
            ],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ))

    if imrc_dnp is not None:
        optional_tables.append(TableSpec(
            name="municipio_imrc_riesgo",
            df=imrc_dnp,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_imrc_municipio")
            ],
            indexes=[["riesgo_inundacion_idx"], ["imrc_exceso"]],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ))

    if sui_aseo is not None:
        optional_tables.append(TableSpec(
            name="municipio_sui_aseo",
            df=sui_aseo,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_sui_aseo_municipio")
            ],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ))

    if ideam_bart is not None:
        optional_tables.append(TableSpec(
            name="municipio_ideam_bart_clima",
            df=ideam_bart,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_ideam_bart_municipio")
            ],
            indexes=[["t_media_c"]],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ))

    if era5 is not None:
        optional_tables.append(TableSpec(
            name="municipio_era5_clima",
            df=era5,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_era5_municipio")
            ],
            indexes=[["t2m_media_c"]],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ))

    if upra_conflicto is not None:
        optional_tables.append(TableSpec(
            name="municipio_upra_conflicto",
            df=upra_conflicto,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(["codigo_dane"], "municipios", ["codigo_dane"], "fk_upra_conflicto_municipio")
            ],
            indexes=[["conflicto_dominante"], ["score_conflicto"]],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ))

    if viabilidad_multidim is not None:
        optional_tables.append(TableSpec(
            name="viabilidad_multidimensional",
            df=viabilidad_multidim,
            primary_key=["codigo_dane"],
            foreign_keys=[
                ForeignKeySpec(
                    ["codigo_dane"], "municipios", ["codigo_dane"], "fk_viabilidad_multidim_municipio"
                )
            ],
            indexes=[["v_i_multidimensional"], ["clasificacion_multidim"]],
            type_overrides={"codigo_dane": "CHAR(5)"},
        ))

    return tables + optional_tables


def build_views_sql() -> str:
    return """
CREATE OR REPLACE VIEW `vw_dashboard_municipal` AS
SELECT
  m.codigo_dane,
  m.municipio,
  d.departamento,
  m.categoria_nombre,
  m.lon,
  m.lat,
  v.v_i_modelo_rural,
  v.score_rural_con_bono_demanda,
  v.clasificacion_preliminar,
  v.pvout_kwh_kwp_day,
  v.annual_yield_kwh_kw_year,
  v.dist_subestacion_km,
  v.pendiente_igac,
  v.pct_area_protegida_runap,
  v.tipo_capa_pot,
  v.categoria_aptitud_pot,
  v.zona_xm_demanda,
  v.d_i_demanda,
  c.cluster_kmeans,
  c.cluster_kmeans_label,
  c.silhouette_municipio
FROM municipios m
JOIN departamentos d ON d.departamento_id = m.departamento_id
LEFT JOIN viabilidad_municipal v ON v.codigo_dane = m.codigo_dane
LEFT JOIN cluster_municipal c ON c.codigo_dane = m.codigo_dane;

CREATE OR REPLACE VIEW `vw_municipal_contexto` AS
SELECT
  m.codigo_dane,
  m.municipio,
  d.departamento,
  pv.pvout_kwh_kwp_day,
  pv.annual_yield_kwh_kw_year,
  pe.pendiente_igac,
  pe.viabilidad_pendiente,
  pe.score_pendiente,
  mr.dist_subestacion_km,
  mr.subestacion_mas_cercana,
  ru.pct_area_protegida_runap,
  ru.u_i_no_protegido_runap,
  pot.tipo_capa_pot,
  pot.uso_pot,
  pot.categoria_aptitud_pot,
  dem.zona_xm_demanda,
  dem.d_i_demanda
FROM municipios m
JOIN departamentos d ON d.departamento_id = m.departamento_id
LEFT JOIN municipio_pvout pv ON pv.codigo_dane = m.codigo_dane
LEFT JOIN municipio_pendiente pe ON pe.codigo_dane = m.codigo_dane
LEFT JOIN municipio_red mr ON mr.codigo_dane = m.codigo_dane
LEFT JOIN municipio_runap ru ON ru.codigo_dane = m.codigo_dane
LEFT JOIN municipio_pot pot ON pot.codigo_dane = m.codigo_dane
LEFT JOIN municipio_demanda_xm dem ON dem.codigo_dane = m.codigo_dane;

CREATE OR REPLACE VIEW `vw_subestaciones_municipios` AS
SELECT
  m.codigo_dane,
  m.municipio,
  d.departamento,
  mr.dist_subestacion_km,
  s.id_subestacion,
  s.cod_sub_upme,
  s.nombre_subestacion,
  s.latitud,
  s.longitud
FROM municipio_red mr
JOIN municipios m ON m.codigo_dane = mr.codigo_dane
JOIN departamentos d ON d.departamento_id = m.departamento_id
LEFT JOIN subestaciones s ON s.id_subestacion = mr.id_subestacion_mas_cercana;
""".strip() + "\n"


def build_optional_views_sql(tables: list[TableSpec]) -> str:
    table_names = {t.name for t in tables}
    views = []

    if "viabilidad_multidimensional" in table_names:
        views.append("""
CREATE OR REPLACE VIEW `vw_comparacion_scores` AS
SELECT
  m.codigo_dane,
  m.municipio,
  d.departamento,
  v.v_i_modelo_rural,
  vm.v_i_multidimensional,
  vm.score_fisico,
  vm.score_electrico,
  vm.score_economico,
  vm.score_agropecuario,
  vm.score_riesgo,
  vm.clasificacion_multidim,
  vm.dims_disponibles,
  vm.dims_faltantes
FROM municipios m
JOIN departamentos d ON d.departamento_id = m.departamento_id
LEFT JOIN viabilidad_municipal v ON v.codigo_dane = m.codigo_dane
LEFT JOIN viabilidad_multidimensional vm ON vm.codigo_dane = m.codigo_dane
ORDER BY vm.v_i_multidimensional DESC;
""".strip())

    return "\n".join(views) + ("\n" if views else "")


def build_bootstrap_sql(database: str) -> str:
    tables = build_tables()
    ordered_table_names = [table.name for table in tables]
    statements = [
        f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;",
        f"USE `{database}`;",
        "SET NAMES utf8mb4;",
        "SET FOREIGN_KEY_CHECKS = 0;",
    ]
    for view_name in ["vw_subestaciones_municipios", "vw_municipal_contexto", "vw_dashboard_municipal"]:
        statements.append(f"DROP VIEW IF EXISTS `{view_name}`;")
    for table_name in reversed(ordered_table_names):
        statements.append(f"DROP TABLE IF EXISTS `{table_name}`;")
    statements.append("SET FOREIGN_KEY_CHECKS = 1;\n")

    for table in tables:
        statements.append(create_table_sql(table))
    for table in tables:
        index_sql = create_index_sql(table)
        if index_sql:
            statements.append(index_sql)
    for table in tables:
        insert_sql = build_insert_sql(table)
        if insert_sql:
            statements.append(insert_sql)
    statements.append(build_views_sql())
    statements.append(build_optional_views_sql(tables))
    return "\n".join(statements)


def reset_database(database: str, settings: MySQLSettings, tables: list[TableSpec]) -> None:
    ordered_table_names = [table.name for table in tables]
    statements = [
        f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;",
        f"USE `{database}`;",
        "SET FOREIGN_KEY_CHECKS = 0;",
    ]
    for view_name in ["vw_subestaciones_municipios", "vw_municipal_contexto", "vw_dashboard_municipal"]:
        statements.append(f"DROP VIEW IF EXISTS `{view_name}`;")
    for table_name in reversed(ordered_table_names):
        statements.append(f"DROP TABLE IF EXISTS `{table_name}`;")
    statements.append("SET FOREIGN_KEY_CHECKS = 1;")
    execute_sql("\n".join(statements), settings=settings)


def apply_bootstrap(database: str, settings: MySQLSettings) -> None:
    tables = build_tables()
    reset_database(database, settings, tables)

    for table in tables:
        ddl = create_table_sql(table)
        index_sql = create_index_sql(table)
        execute_sql(ddl + ("\n" + index_sql if index_sql else ""), settings=settings, database=database)

    for table in tables:
        insert_sql = build_insert_sql(table)
        if not insert_sql.strip():
            continue
        try:
            execute_sql(insert_sql, settings=settings, database=database)
        except Exception as error:
            raise RuntimeError(f"Fallo cargando datos en la tabla '{table.name}'.") from error

    execute_sql(build_views_sql(), settings=settings, database=database)
    optional_views = build_optional_views_sql(tables)
    if optional_views.strip():
        execute_sql(optional_views, settings=settings, database=database)


def parse_args() -> argparse.Namespace:
    default_database = MySQLSettings.from_env().database
    parser = argparse.ArgumentParser(
        description="Construye y opcionalmente aplica el esquema MySQL del proyecto de granja solar."
    )
    parser.add_argument(
        "--database",
        default=default_database,
        help="Nombre de la base de datos MySQL destino.",
    )
    parser.add_argument(
        "--sql-out",
        default=str(DEFAULT_SQL_OUT),
        help="Ruta donde se escribira el SQL generado.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Ejecuta el SQL generado contra MySQL usando MYSQL_HOST, MYSQL_PORT, MYSQL_USER y MYSQL_PASSWORD.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sql_text = build_bootstrap_sql(args.database)
    output_path = Path(args.sql_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(sql_text, encoding="utf-8")
    print(f"SQL MySQL generado en: {output_path}")

    if args.apply:
        settings = MySQLSettings.from_env(database_override=args.database)
        apply_bootstrap(args.database, settings=settings)
        print(
            f"Base MySQL '{args.database}' creada y poblada. "
            "Quedaron disponibles las vistas vw_dashboard_municipal, "
            "vw_municipal_contexto y vw_subestaciones_municipios."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
