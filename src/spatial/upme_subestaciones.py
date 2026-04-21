from __future__ import annotations

import argparse
import json
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import urllib3


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "upme_subestaciones"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "subestaciones_upme"
DEFAULT_MUNICIPALITIES_PATH = (
    PROJECT_ROOT / "data" / "clean" / "base_municipios" / "municipios_distritos_colombia.csv"
)

SERVICE_URL = (
    "https://geo.upme.gov.co/server/rest/services/"
    "SUBESTACIONES/UPME_EN_DI_SUBESTACION_consulta/MapServer"
)
SUBSTATIONS_LAYER = 0
TENSION_TABLE = 1
NIVEL_TENSION_TABLE = 2
ESTADO_TABLE = 3

OUTPUT_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:3116"


def normalize_text(value: Any) -> str:
    """Normaliza texto para cruces y busquedas."""

    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.upper().split())


def request_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = 120,
    retries: int = 3,
    verify_ssl: bool = False,
) -> dict[str, Any]:
    """Consulta JSON de ArcGIS REST con reintentos.

    El portal geográfico UPME puede fallar con la validación de certificado en
    algunos entornos locales. Se conserva `verify_ssl` configurable y se deja
    advertencia en las observaciones cuando se usa `False`.
    """

    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    last_error: Exception | None = None
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json,*/*"}
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                verify=verify_ssl,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(data["error"])
            return data
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"No fue posible consultar {url}. Ultimo error: {last_error}")


def write_json(path: Path, data: Any) -> None:
    """Guarda JSON local para trazabilidad."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Lee JSON cacheado."""

    return json.loads(path.read_text(encoding="utf-8"))


def fetch_metadata(raw_dir: Path, force: bool = False, verify_ssl: bool = False) -> dict[str, Any]:
    """Descarga metadatos del servicio UPME."""

    raw_path = raw_dir / "mapserver_metadata.json"
    if raw_path.exists() and not force:
        return read_json(raw_path)

    data = request_json(SERVICE_URL, params={"f": "pjson"}, verify_ssl=verify_ssl)
    write_json(raw_path, data)
    return data


def fetch_layer_metadata(
    raw_dir: Path,
    layer_id: int,
    force: bool = False,
    verify_ssl: bool = False,
) -> dict[str, Any]:
    """Descarga metadatos de una capa/tabla."""

    raw_path = raw_dir / f"layer_{layer_id}_metadata.json"
    if raw_path.exists() and not force:
        return read_json(raw_path)

    data = request_json(f"{SERVICE_URL}/{layer_id}", params={"f": "pjson"}, verify_ssl=verify_ssl)
    write_json(raw_path, data)
    return data


def fetch_geojson(
    raw_dir: Path,
    force: bool = False,
    verify_ssl: bool = False,
) -> dict[str, Any]:
    """Descarga subestaciones como GeoJSON EPSG:4326."""

    raw_path = raw_dir / "subestaciones_upme.geojson"
    if raw_path.exists() and not force:
        return read_json(raw_path)

    data = request_json(
        f"{SERVICE_URL}/{SUBSTATIONS_LAYER}/query",
        params={
            "f": "geojson",
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": 4326,
        },
        timeout=180,
        verify_ssl=verify_ssl,
    )
    write_json(raw_path, data)
    return data


def fetch_table(
    raw_dir: Path,
    table_id: int,
    name: str,
    force: bool = False,
    verify_ssl: bool = False,
) -> pd.DataFrame:
    """Descarga una tabla auxiliar del servicio."""

    raw_path = raw_dir / f"{name}.json"
    if raw_path.exists() and not force:
        data = read_json(raw_path)
    else:
        data = request_json(
            f"{SERVICE_URL}/{table_id}/query",
            params={
                "f": "json",
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "false",
            },
            verify_ssl=verify_ssl,
        )
        write_json(raw_path, data)

    rows = [feature.get("attributes", {}) for feature in data.get("features", [])]
    return pd.DataFrame(rows)


def clean_substations(
    feature_collection: dict[str, Any],
    tension_df: pd.DataFrame,
    nivel_df: pd.DataFrame,
    estado_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Limpia y estandariza subestaciones UPME."""

    gdf = gpd.GeoDataFrame.from_features(feature_collection["features"], crs=OUTPUT_CRS)
    rename_map = {
        "OBJECTID": "objectid",
        "ID_SUBESTACION": "id_subestacion",
        "COD_SUB_UPME": "cod_sub_upme",
        "NOM_SUBESTACION": "nombre_subestacion",
        "ID_NIVEL_TENSION": "id_nivel_tension",
        "ID_TENSION_SUB": "id_tension_sub",
        "ID_ESTADO_SUB": "id_estado_sub",
        "LATITUD": "latitud",
        "LONGITUD": "longitud",
        "ALTITUD": "altitud_m",
        "PORCENTAJE_CARGA": "porcentaje_carga",
        "CAPACIDAD_MVA": "capacidad_mva",
        "COD_DPTO": "cod_departamento",
        "COD_MPIO": "cod_municipio",
        "ACTIVO": "activo",
        "AMPLIACION": "ampliacion",
        "OBSERVACION": "observacion",
        "FECHA_OPERACION": "fecha_operacion",
        "FECHA_CORTE_UPME": "fecha_corte_upme",
    }
    gdf = gdf.rename(columns=rename_map)

    for column in [
        "objectid",
        "id_subestacion",
        "cod_sub_upme",
        "nombre_subestacion",
        "id_nivel_tension",
        "id_tension_sub",
        "id_estado_sub",
        "latitud",
        "longitud",
        "altitud_m",
        "porcentaje_carga",
        "capacidad_mva",
        "cod_departamento",
        "cod_municipio",
        "activo",
        "ampliacion",
        "observacion",
        "geometry",
    ]:
        if column not in gdf.columns:
            gdf[column] = pd.NA

    gdf["nombre_subestacion"] = gdf["nombre_subestacion"].astype("string").str.strip()
    gdf["nombre_normalizado"] = gdf["nombre_subestacion"].map(normalize_text)
    gdf["codigo_dane_municipio"] = (
        gdf["cod_departamento"].astype("string").str.zfill(2)
        + gdf["cod_municipio"].astype("string").str.zfill(3)
    )

    numeric_columns = [
        "id_nivel_tension",
        "id_tension_sub",
        "id_estado_sub",
        "latitud",
        "longitud",
        "altitud_m",
        "porcentaje_carga",
        "capacidad_mva",
    ]
    for column in numeric_columns:
        gdf[column] = pd.to_numeric(gdf[column], errors="coerce")

    if not nivel_df.empty:
        nivel = nivel_df.rename(
            columns={"ID_NIVEL_TENSION": "id_nivel_tension", "DESCRIPCION": "nivel_tension"}
        )
        gdf = gdf.merge(nivel[["id_nivel_tension", "nivel_tension"]], on="id_nivel_tension", how="left")
    else:
        gdf["nivel_tension"] = pd.NA

    if not tension_df.empty:
        tension = tension_df.rename(
            columns={
                "ID_TENSION": "id_tension_sub",
                "DESCRIPCION": "tension_descripcion",
            }
        )
        gdf = gdf.merge(
            tension[["id_tension_sub", "tension_descripcion"]],
            on="id_tension_sub",
            how="left",
        )
    else:
        gdf["tension_descripcion"] = pd.NA

    if not estado_df.empty:
        estado = estado_df.rename(
            columns={
                "ID_ESTADO_SUB": "id_estado_sub",
                "NOM_ESTADO_SUB": "estado_subestacion",
            }
        )
        gdf = gdf.merge(
            estado[["id_estado_sub", "estado_subestacion"]],
            on="id_estado_sub",
            how="left",
        )
    else:
        gdf["estado_subestacion"] = pd.NA

    gdf["fuente"] = "UPME - SUBESTACIONES/UPME_EN_DI_SUBESTACION_consulta"
    gdf["url_servicio"] = SERVICE_URL
    gdf["fecha_descarga_utc"] = datetime.now(timezone.utc).isoformat()

    columns = [
        "objectid",
        "id_subestacion",
        "cod_sub_upme",
        "nombre_subestacion",
        "nombre_normalizado",
        "id_nivel_tension",
        "nivel_tension",
        "id_tension_sub",
        "tension_descripcion",
        "id_estado_sub",
        "estado_subestacion",
        "latitud",
        "longitud",
        "altitud_m",
        "porcentaje_carga",
        "capacidad_mva",
        "codigo_dane_municipio",
        "activo",
        "ampliacion",
        "observacion",
        "fuente",
        "url_servicio",
        "fecha_descarga_utc",
        "geometry",
    ]
    return gdf[columns].sort_values(["nombre_normalizado", "id_subestacion"])


def calculate_nearest_substations(
    municipalities_path: Path,
    substations_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    voltage_levels: list[int] | None = None,
    only_in_service: bool = True,
) -> pd.DataFrame:
    """Calcula distancia minima desde cada municipio a subestacion UPME."""

    if not municipalities_path.exists():
        raise FileNotFoundError(f"No existe base municipal: {municipalities_path}")

    municipalities = pd.read_csv(municipalities_path, dtype={"codigo_dane": "string"})
    municipalities["codigo_dane"] = municipalities["codigo_dane"].astype("string").str.zfill(5)
    required = {"lon", "lat", "codigo_dane", "municipio", "departamento"}
    missing = required - set(municipalities.columns)
    if missing:
        raise ValueError(f"Faltan columnas en municipios: {sorted(missing)}")

    candidates = substations_gdf.copy()
    if voltage_levels:
        candidates = candidates[candidates["id_nivel_tension"].isin(voltage_levels)].copy()
    if only_in_service:
        candidates = candidates[candidates["estado_subestacion"].eq("En Servicio")].copy()

    candidates = candidates.dropna(subset=["longitud", "latitud"])
    if candidates.empty:
        raise ValueError("No hay subestaciones candidatas con coordenadas para calcular distancias.")

    muni_gdf = gpd.GeoDataFrame(
        municipalities.copy(),
        geometry=gpd.points_from_xy(municipalities["lon"], municipalities["lat"]),
        crs=OUTPUT_CRS,
    ).to_crs(PROJECTED_CRS)
    sub_gdf = candidates.to_crs(PROJECTED_CRS)

    muni_xy = np.column_stack([muni_gdf.geometry.x.to_numpy(), muni_gdf.geometry.y.to_numpy()])
    sub_xy = np.column_stack([sub_gdf.geometry.x.to_numpy(), sub_gdf.geometry.y.to_numpy()])

    nearest_indices: list[int] = []
    nearest_distances_m: list[float] = []
    chunk_size = 100
    for start in range(0, len(muni_xy), chunk_size):
        chunk = muni_xy[start : start + chunk_size]
        distances = np.sqrt(((chunk[:, None, :] - sub_xy[None, :, :]) ** 2).sum(axis=2))
        nearest = distances.argmin(axis=1)
        nearest_indices.extend(nearest.tolist())
        nearest_distances_m.extend(distances[np.arange(len(chunk)), nearest].tolist())

    nearest_sub = candidates.reset_index(drop=True).iloc[nearest_indices].reset_index(drop=True)
    result = municipalities.reset_index(drop=True).copy()
    result["dist_subestacion_km"] = np.array(nearest_distances_m) / 1000.0
    result["subestacion_mas_cercana"] = nearest_sub["nombre_subestacion"].values
    result["id_subestacion_mas_cercana"] = nearest_sub["id_subestacion"].values
    result["cod_sub_upme_mas_cercana"] = nearest_sub["cod_sub_upme"].values
    result["nivel_tension_mas_cercana"] = nearest_sub["nivel_tension"].values
    result["tension_mas_cercana"] = nearest_sub["tension_descripcion"].values
    result["estado_subestacion_mas_cercana"] = nearest_sub["estado_subestacion"].values
    result["capacidad_mva_mas_cercana"] = nearest_sub["capacidad_mva"].values
    result["subestacion_lon"] = nearest_sub["longitud"].values
    result["subestacion_lat"] = nearest_sub["latitud"].values
    result["criterio_red"] = (
        "Distancia euclidiana en EPSG:3116 desde punto interno municipal a la subestacion "
        "UPME mas cercana. Candidatas: en servicio y nivel de tension 4/5 por defecto."
    )

    output_path = output_dir / "distancia_subestacion_municipios.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def build_metadata_table(metadata: dict[str, Any], layer_metadata: dict[str, Any]) -> pd.DataFrame:
    """Resume metadatos de fuente."""

    return pd.DataFrame(
        [
            {"clave": "fuente", "valor": "UPME Geoportal"},
            {"clave": "url_servicio", "valor": SERVICE_URL},
            {"clave": "descripcion", "valor": metadata.get("serviceDescription", "")},
            {"clave": "capa", "valor": layer_metadata.get("name", "")},
            {"clave": "tipo_geometria", "valor": layer_metadata.get("geometryType", "")},
            {"clave": "crs_origen", "valor": metadata.get("spatialReference", {}).get("wkid", "")},
            {"clave": "crs_salida", "valor": OUTPUT_CRS},
            {"clave": "max_record_count", "valor": metadata.get("maxRecordCount", "")},
        ]
    )


def build_quality_table(gdf: gpd.GeoDataFrame, distance_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Controles basicos de calidad."""

    rows = [
        {"indicador": "subestaciones", "valor": int(len(gdf))},
        {"indicador": "con_coordenadas", "valor": int(gdf[["latitud", "longitud"]].notna().all(axis=1).sum())},
        {"indicador": "sin_coordenadas", "valor": int(gdf[["latitud", "longitud"]].isna().any(axis=1).sum())},
        {"indicador": "nombres_unicos", "valor": int(gdf["nombre_normalizado"].nunique(dropna=True))},
    ]
    for level, count in gdf["nivel_tension"].value_counts(dropna=False).items():
        rows.append({"indicador": f"nivel_tension_{level}", "valor": int(count)})
    for state, count in gdf["estado_subestacion"].value_counts(dropna=False).items():
        rows.append({"indicador": f"estado_{state}", "valor": int(count)})
    if distance_df is not None and not distance_df.empty:
        rows.extend(
            [
                {"indicador": "municipios_con_distancia_red", "valor": int(len(distance_df))},
                {
                    "indicador": "distancia_red_min_km",
                    "valor": float(distance_df["dist_subestacion_km"].min()),
                },
                {
                    "indicador": "distancia_red_media_km",
                    "valor": float(distance_df["dist_subestacion_km"].mean()),
                },
                {
                    "indicador": "distancia_red_max_km",
                    "valor": float(distance_df["dist_subestacion_km"].max()),
                },
            ]
        )
    return pd.DataFrame(rows)


def write_observations(
    path: Path,
    gdf: gpd.GeoDataFrame,
    distance_df: pd.DataFrame | None,
    verify_ssl: bool,
) -> None:
    """Documenta decisiones y limitaciones."""

    lines = [
        "Subestaciones UPME",
        "==================",
        "",
        "Fuente: UPME Geoportal, servicio SUBESTACIONES/UPME_EN_DI_SUBESTACION_consulta.",
        f"URL: {SERVICE_URL}",
        f"Registros de subestaciones descargados: {len(gdf)}",
        "",
        "Diferencia con SIMEM F99E13:",
        "- SIMEM F99E13 entrega listado/codigo/nombre de subestaciones.",
        "- UPME entrega geometria puntual, latitud, longitud, tension, estado y capacidad.",
        "- Para calcular G_i se usa UPME porque la distancia requiere coordenadas.",
        "",
        "Criterio adoptado para distancia a red:",
        "- Se calcula distancia minima desde lon/lat municipal a subestaciones UPME.",
        "- Por defecto se priorizan subestaciones en servicio y niveles de tension 4/5.",
        "- Nivel 4 corresponde a tension nominal >=57,5 kV y <220 kV.",
        "- Nivel 5 corresponde a subestacion de transmision nacional.",
        "",
        "Limitaciones:",
        "- La distancia es euclidiana en CRS proyectado EPSG:3116; no representa costo real de conexion.",
        "- No incluye trazado de lineas, servidumbres, capacidad disponible real ni restricciones prediales.",
        "- La cercania a una subestacion no garantiza factibilidad tecnica de conexion.",
    ]
    if not verify_ssl:
        lines.append(
            "- Advertencia tecnica: se consulto UPME con verificacion SSL desactivada porque el entorno local no valido el certificado."
        )
    if distance_df is not None and not distance_df.empty:
        lines.extend(
            [
                "",
                f"Municipios con distancia calculada: {len(distance_df)}",
                f"Distancia minima: {distance_df['dist_subestacion_km'].min():.2f} km",
                f"Distancia media: {distance_df['dist_subestacion_km'].mean():.2f} km",
                f"Distancia maxima: {distance_df['dist_subestacion_km'].max():.2f} km",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_upme_substations(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    municipalities_path: Path = DEFAULT_MUNICIPALITIES_PATH,
    force: bool = False,
    verify_ssl: bool = False,
    skip_distances: bool = False,
    voltage_levels: list[int] | None = None,
) -> dict[str, Path]:
    """Ejecuta extraccion UPME y distancia municipal a red."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if voltage_levels is None:
        voltage_levels = [4, 5]

    metadata = fetch_metadata(raw_dir, force=force, verify_ssl=verify_ssl)
    layer_metadata = fetch_layer_metadata(raw_dir, SUBSTATIONS_LAYER, force=force, verify_ssl=verify_ssl)
    feature_collection = fetch_geojson(raw_dir, force=force, verify_ssl=verify_ssl)
    tension_df = fetch_table(raw_dir, TENSION_TABLE, "vista_se_tension", force=force, verify_ssl=verify_ssl)
    nivel_df = fetch_table(
        raw_dir,
        NIVEL_TENSION_TABLE,
        "vista_se_nivel_tension",
        force=force,
        verify_ssl=verify_ssl,
    )
    estado_df = fetch_table(raw_dir, ESTADO_TABLE, "vista_se_estado_sub", force=force, verify_ssl=verify_ssl)

    substations = clean_substations(feature_collection, tension_df, nivel_df, estado_df)
    csv_path = output_dir / "subestaciones_upme.csv"
    geojson_path = output_dir / "subestaciones_upme.geojson"
    metadata_path = output_dir / "subestaciones_metadata_fuente.csv"
    quality_path = output_dir / "subestaciones_calidad.csv"
    observations_path = output_dir / "subestaciones_observaciones.txt"

    substations.drop(columns="geometry").to_csv(csv_path, index=False, encoding="utf-8-sig")
    substations.to_file(geojson_path, driver="GeoJSON")

    distance_df = None
    outputs = {
        "subestaciones": csv_path,
        "geojson": geojson_path,
        "metadata": metadata_path,
        "calidad": quality_path,
        "observaciones": observations_path,
    }
    if not skip_distances:
        distance_df = calculate_nearest_substations(
            municipalities_path=municipalities_path,
            substations_gdf=substations,
            output_dir=output_dir,
            voltage_levels=voltage_levels,
            only_in_service=True,
        )
        outputs["distancias_municipios"] = output_dir / "distancia_subestacion_municipios.csv"

    build_metadata_table(metadata, layer_metadata).to_csv(
        metadata_path,
        index=False,
        encoding="utf-8-sig",
    )
    build_quality_table(substations, distance_df).to_csv(
        quality_path,
        index=False,
        encoding="utf-8-sig",
    )
    write_observations(observations_path, substations, distance_df, verify_ssl=verify_ssl)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga subestaciones UPME y calcula distancia municipal a red electrica."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--municipalities-path", type=Path, default=DEFAULT_MUNICIPALITIES_PATH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-ssl", action="store_true", help="Activa validacion SSL estricta.")
    parser.add_argument("--skip-distances", action="store_true")
    parser.add_argument(
        "--voltage-levels",
        default="4,5",
        help="Niveles de tension UPME para distancia a red. Por defecto 4,5.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    voltage_levels = [
        int(part.strip())
        for part in str(args.voltage_levels).split(",")
        if part.strip()
    ]
    outputs = run_upme_substations(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        municipalities_path=args.municipalities_path,
        force=args.force,
        verify_ssl=args.verify_ssl,
        skip_distances=args.skip_distances,
        voltage_levels=voltage_levels,
    )
    print("Subestaciones UPME procesadas.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print("G_i puede calcularse desde distancia_subestacion_municipios.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
