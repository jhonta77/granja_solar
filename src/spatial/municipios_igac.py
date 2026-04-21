from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

try:
    import geopandas as gpd
except ImportError as error:  # pragma: no cover - validacion en runtime
    raise SystemExit(
        "Falta geopandas. Instala dependencias geoespaciales antes de ejecutar "
        "este script: pip install geopandas pyogrio shapely pyproj"
    ) from error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "igac_municipios"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "base_municipios"

SERVICE_URL = "https://mapas2.igac.gov.co/server/rest/services/limites/limites/MapServer"
LAYER_ID = 1
LAYER_URL = f"{SERVICE_URL}/{LAYER_ID}"
QUERY_URL = f"{LAYER_URL}/query"

SOURCE_NAME = "IGAC - Servicio limites/limites, capa municipios"
SOURCE_CRS = "EPSG:9377"
OUTPUT_CRS = "EPSG:4326"

FIELDS = [
    "OBJECTID",
    "MpCodigo",
    "MpNombre",
    "Depto",
    "MpCategor",
    "MpNorma",
    "MpArea",
    "MpAltitud",
]

CATEGORY_NAMES = {
    1: "Municipio",
    2: "Distrito",
    3: "Area no municipalizada",
    4: "Sin categoria",
}


def normalize_text(value: Any) -> str:
    """Normaliza texto para llaves de cruce robustas."""

    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().upper()


def request_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = 120,
    retries: int = 3,
) -> dict[str, Any]:
    """Consulta JSON con reintentos para servicios ArcGIS REST."""

    last_error: Exception | None = None
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
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
    """Guarda JSON local para trazabilidad y ejecucion reproducible."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Lee JSON cacheado."""

    return json.loads(path.read_text(encoding="utf-8"))


def chunked(values: list[int], size: int) -> Iterable[list[int]]:
    """Divide una lista en bloques para evitar limites del servicio."""

    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_layer_metadata(raw_dir: Path, force: bool = False) -> dict[str, Any]:
    """Descarga metadatos oficiales de la capa municipal IGAC."""

    raw_path = raw_dir / "layer_1_metadata.json"
    if raw_path.exists() and not force:
        return read_json(raw_path)

    metadata = request_json(LAYER_URL, params={"f": "pjson"})
    write_json(raw_path, metadata)
    return metadata


def fetch_object_ids(raw_dir: Path, force: bool = False) -> list[int]:
    """Obtiene los OBJECTID disponibles en la capa."""

    raw_path = raw_dir / "object_ids.json"
    if raw_path.exists() and not force:
        return [int(value) for value in read_json(raw_path)["objectIds"]]

    data = request_json(
        QUERY_URL,
        params={
            "f": "json",
            "where": "1=1",
            "returnIdsOnly": "true",
        },
    )
    object_ids = sorted(int(value) for value in data.get("objectIds", []))
    if not object_ids:
        raise ValueError("La capa IGAC no devolvio OBJECTID para municipios.")

    write_json(raw_path, {"objectIds": object_ids})
    return object_ids


def fetch_geojson_chunk(
    object_ids: list[int],
    raw_dir: Path,
    chunk_index: int,
    force: bool = False,
) -> dict[str, Any]:
    """Descarga un bloque de municipios en GeoJSON EPSG:4326."""

    raw_path = raw_dir / "chunks" / f"municipios_chunk_{chunk_index:03d}.geojson"
    if raw_path.exists() and not force:
        return read_json(raw_path)

    data = request_json(
        QUERY_URL,
        params={
            "f": "geojson",
            "objectIds": ",".join(str(value) for value in object_ids),
            "outFields": ",".join(FIELDS),
            "returnGeometry": "true",
            "outSR": 4326,
        },
    )
    write_json(raw_path, data)
    return data


def fetch_municipios_geojson(
    raw_dir: Path,
    chunk_size: int = 250,
    force: bool = False,
) -> dict[str, Any]:
    """Descarga y combina todos los municipios/entidades municipales IGAC."""

    object_ids = fetch_object_ids(raw_dir, force=force)
    features: list[dict[str, Any]] = []
    for index, ids_chunk in enumerate(chunked(object_ids, chunk_size), start=1):
        data = fetch_geojson_chunk(ids_chunk, raw_dir, index, force=force)
        features.extend(data.get("features", []))

    if not features:
        raise ValueError("No se descargaron geometrias municipales desde IGAC.")

    combined = {
        "type": "FeatureCollection",
        "name": "municipios_igac",
        "crs": {"type": "name", "properties": {"name": OUTPUT_CRS}},
        "features": features,
    }
    write_json(raw_dir / "municipios_igac.geojson", combined)
    return combined


def clean_municipios_gdf(feature_collection: dict[str, Any]) -> gpd.GeoDataFrame:
    """Estandariza nombres y campos relevantes para el modelo municipal."""

    gdf = gpd.GeoDataFrame.from_features(feature_collection["features"], crs=OUTPUT_CRS)
    rename_map = {
        "OBJECTID": "objectid",
        "MpCodigo": "codigo_dane",
        "MpNombre": "municipio",
        "Depto": "departamento",
        "MpCategor": "categoria",
        "MpNorma": "normatividad",
        "MpArea": "area_km2_igac",
        "MpAltitud": "altitud_m",
    }
    gdf = gdf.rename(columns=rename_map)

    expected = [
        "objectid",
        "codigo_dane",
        "municipio",
        "departamento",
        "categoria",
        "normatividad",
        "area_km2_igac",
        "altitud_m",
        "geometry",
    ]
    for column in expected:
        if column not in gdf.columns:
            gdf[column] = pd.NA

    gdf["codigo_dane"] = (
        gdf["codigo_dane"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    )
    gdf["codigo_dane"] = gdf["codigo_dane"].str.zfill(5)
    gdf["municipio"] = gdf["municipio"].astype("string").str.strip()
    gdf["departamento"] = gdf["departamento"].astype("string").str.strip()
    gdf["categoria"] = pd.to_numeric(gdf["categoria"], errors="coerce").astype("Int64")
    gdf["categoria_nombre"] = gdf["categoria"].map(CATEGORY_NAMES).fillna("Sin clasificar")
    gdf["area_km2_igac"] = pd.to_numeric(gdf["area_km2_igac"], errors="coerce")
    gdf["altitud_m"] = pd.to_numeric(gdf["altitud_m"], errors="coerce")
    gdf["municipio_normalizado"] = gdf["municipio"].map(normalize_text)
    gdf["departamento_normalizado"] = gdf["departamento"].map(normalize_text)
    gdf["fuente"] = SOURCE_NAME
    gdf["url_servicio"] = LAYER_URL
    gdf["fecha_descarga_utc"] = datetime.now(timezone.utc).isoformat()

    return gdf[
        [
            "objectid",
            "codigo_dane",
            "municipio",
            "municipio_normalizado",
            "departamento",
            "departamento_normalizado",
            "categoria",
            "categoria_nombre",
            "normatividad",
            "area_km2_igac",
            "altitud_m",
            "fuente",
            "url_servicio",
            "fecha_descarga_utc",
            "geometry",
        ]
    ].sort_values(["departamento_normalizado", "municipio_normalizado", "codigo_dane"])


def add_sampling_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Agrega centroides y puntos internos para muestrear rasters/APIs por municipio."""

    projected = gdf.to_crs(SOURCE_CRS)
    centroid_points = gpd.GeoSeries(projected.geometry.centroid, crs=SOURCE_CRS).to_crs(OUTPUT_CRS)
    representative_points = gpd.GeoSeries(
        projected.geometry.representative_point(),
        crs=SOURCE_CRS,
    ).to_crs(OUTPUT_CRS)

    result = gdf.copy()
    result["centroide_lon"] = centroid_points.x
    result["centroide_lat"] = centroid_points.y
    # lon/lat se dejan como punto interno para que NASA, PVOUT e IGAC no muestreen
    # accidentalmente fuera de poligonos concavos o insulares.
    result["lon"] = representative_points.x
    result["lat"] = representative_points.y
    return result


def build_metadata_table(layer_metadata: dict[str, Any]) -> pd.DataFrame:
    """Resume metadatos principales de la capa."""

    fields = layer_metadata.get("fields", [])
    field_names = ", ".join(field.get("name", "") for field in fields)
    return pd.DataFrame(
        [
            {"clave": "fuente", "valor": SOURCE_NAME},
            {"clave": "url_servicio", "valor": LAYER_URL},
            {"clave": "nombre_capa", "valor": layer_metadata.get("name", "")},
            {"clave": "tipo_geometria", "valor": layer_metadata.get("geometryType", "")},
            {"clave": "crs_origen", "valor": layer_metadata.get("sourceSpatialReference", {}).get("wkid", "")},
            {"clave": "crs_salida", "valor": OUTPUT_CRS},
            {"clave": "max_record_count", "valor": layer_metadata.get("maxRecordCount", "")},
            {"clave": "campos", "valor": field_names},
        ]
    )


def build_quality_table(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Genera controles basicos de calidad para la base municipal."""

    rows = [
        {"indicador": "filas", "valor": int(len(gdf))},
        {"indicador": "codigo_dane_unicos", "valor": int(gdf["codigo_dane"].nunique(dropna=True))},
        {"indicador": "codigo_dane_nulos", "valor": int(gdf["codigo_dane"].isna().sum())},
        {"indicador": "municipio_nulos", "valor": int(gdf["municipio"].isna().sum())},
        {"indicador": "departamento_nulos", "valor": int(gdf["departamento"].isna().sum())},
        {"indicador": "geometrias_nulas", "valor": int(gdf.geometry.isna().sum())},
        {"indicador": "geometrias_invalidas", "valor": int((~gdf.geometry.is_valid).sum())},
    ]
    for category, count in gdf["categoria_nombre"].value_counts(dropna=False).items():
        rows.append({"indicador": f"categoria_{category}", "valor": int(count)})
    return pd.DataFrame(rows)


def write_observations(
    output_path: Path,
    gdf: gpd.GeoDataFrame,
    filtered_gdf: gpd.GeoDataFrame,
) -> None:
    """Documenta decisiones metodologicas y limitaciones."""

    category_counts = gdf["categoria_nombre"].value_counts(dropna=False)
    lines = [
        "Base municipal IGAC",
        "",
        f"Fuente: {SOURCE_NAME}",
        f"URL: {LAYER_URL}",
        f"Registros totales descargados: {len(gdf)}",
        f"Registros municipio/distrito para scoring estricto: {len(filtered_gdf)}",
        "",
        "Conteo por categoria:",
    ]
    for category, count in category_counts.items():
        lines.append(f"- {category}: {count}")

    lines.extend(
        [
            "",
            "Campos principales:",
            "- codigo_dane: tomado de MpCodigo, alias Codigo DANE.",
            "- municipio: tomado de MpNombre.",
            "- departamento: tomado de Depto.",
            "- area_km2_igac: tomado de MpArea.",
            "- altitud_m: tomado de MpAltitud.",
            "- lon/lat: punto interno del poligono, util como punto de muestreo para APIs.",
            "- centroide_lon/centroide_lat: centroide geometrico calculado en EPSG:9377 y convertido a EPSG:4326.",
            "",
            "Uso recomendado en el modelo:",
            "- Usar municipios_distritos_colombia.csv para el score municipal estricto.",
            "- Usar municipios_colombia.geojson para agregaciones espaciales por poligono.",
            "- Para una primera version se puede muestrear PVOUT, NASA e IGAC en lon/lat.",
            "- Para una version mas rigurosa, calcular promedios/proporciones por poligono municipal.",
            "",
            "Limitaciones:",
            "- La capa incluye areas no municipalizadas y una entidad sin categoria; no deben mezclarse sin declarar el criterio.",
            "- El punto lon/lat no reemplaza el promedio espacial del municipio.",
            "- La distancia a red aun requiere coordenadas de subestaciones o lineas electricas.",
            "- El uso del suelo y restricciones territoriales aun requieren capas adicionales.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_outputs(
    gdf: gpd.GeoDataFrame,
    layer_metadata: dict[str, Any],
    output_dir: Path,
    export_geojson: bool = True,
) -> dict[str, Path]:
    """Exporta CSV, GeoJSON y documentacion de la base municipal."""

    output_dir.mkdir(parents=True, exist_ok=True)
    gdf = add_sampling_points(gdf)
    filtered = gdf[gdf["categoria"].isin([1, 2])].copy()

    csv_columns = [column for column in gdf.columns if column != "geometry"]
    all_csv = output_dir / "municipios_colombia.csv"
    filtered_csv = output_dir / "municipios_distritos_colombia.csv"
    quality_csv = output_dir / "municipios_calidad.csv"
    metadata_csv = output_dir / "municipios_metadata_fuente.csv"
    observations_txt = output_dir / "municipios_observaciones.txt"

    gdf[csv_columns].to_csv(all_csv, index=False, encoding="utf-8-sig")
    filtered[csv_columns].to_csv(filtered_csv, index=False, encoding="utf-8-sig")
    build_quality_table(gdf).to_csv(quality_csv, index=False, encoding="utf-8-sig")
    build_metadata_table(layer_metadata).to_csv(metadata_csv, index=False, encoding="utf-8-sig")
    write_observations(observations_txt, gdf, filtered)

    outputs = {
        "municipios_csv": all_csv,
        "municipios_distritos_csv": filtered_csv,
        "calidad": quality_csv,
        "metadata": metadata_csv,
        "observaciones": observations_txt,
    }

    if export_geojson:
        geojson_path = output_dir / "municipios_colombia.geojson"
        gdf.to_file(geojson_path, driver="GeoJSON")
        outputs["geojson"] = geojson_path

    return outputs


def run_municipios_igac(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    chunk_size: int = 50,
    force: bool = False,
    export_geojson: bool = True,
) -> dict[str, Path]:
    """Ejecuta el pipeline completo de base municipal IGAC."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    layer_metadata = fetch_layer_metadata(raw_dir, force=force)
    feature_collection = fetch_municipios_geojson(raw_dir, chunk_size=chunk_size, force=force)
    gdf = clean_municipios_gdf(feature_collection)
    return export_outputs(gdf, layer_metadata, output_dir, export_geojson=export_geojson)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga y limpia la base municipal oficial desde IGAC limites/limites."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Cantidad de entidades por consulta. El servicio IGAC puede fallar con bloques grandes.",
    )
    parser.add_argument("--force", action="store_true", help="Ignora cache local y vuelve a descargar.")
    parser.add_argument("--no-geojson", action="store_true", help="No exporta geometria GeoJSON limpia.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_municipios_igac(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        force=args.force,
        export_geojson=not args.no_geojson,
    )

    print("Base municipal IGAC generada.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print("Usa municipios_distritos_colombia.csv como unidad municipal estricta para el score.")


if __name__ == "__main__":
    main()
