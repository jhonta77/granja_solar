from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "runap"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "runap_protegidas"
DEFAULT_MUNICIPALITIES_GEOJSON = (
    PROJECT_ROOT / "data" / "clean" / "base_municipios" / "municipios_colombia.geojson"
)

SERVICE_URL = "https://mapas.parquesnacionales.gov.co/arcgis/rest/services/pnn/runap/MapServer"
DOWNLOAD_URL = "https://storage.googleapis.com/pnn_geodatabase/runap/latest.zip"
PROJECTED_CRS = "EPSG:3116"
OUTPUT_CRS = "EPSG:4326"
DEFAULT_VALID_CONDITIONS = {"REGISTRADA", "REGISTRADO", "INSCRITA"}


def normalize_text(value: Any) -> str:
    """Normaliza texto para filtros robustos."""

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
    """Consulta JSON con reintentos."""

    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json,*/*"}
    last_error: Exception | None = None
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def download_runap_zip(raw_dir: Path, force: bool = False) -> Path:
    """Descarga el ZIP oficial RUNAP publicado por Parques Nacionales."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "latest.zip"
    if zip_path.exists() and not force:
        return zip_path

    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(DOWNLOAD_URL, stream=True, timeout=600, headers=headers) as response:
        response.raise_for_status()
        with zip_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    return zip_path


def fetch_metadata(raw_dir: Path, force: bool = False) -> dict[str, Any]:
    """Descarga metadatos del servicio RUNAP."""

    raw_path = raw_dir / "mapserver_metadata.json"
    if raw_path.exists() and not force:
        return read_json(raw_path)

    metadata = request_json(SERVICE_URL, params={"f": "pjson"})
    write_json(raw_path, metadata)
    return metadata


def zip_shapefile_uri(zip_path: Path) -> str:
    """Construye URI zip para geopandas."""

    with zipfile.ZipFile(zip_path) as archive:
        shp_names = [name for name in archive.namelist() if name.lower().endswith(".shp")]
    if not shp_names:
        raise FileNotFoundError(f"El ZIP RUNAP no contiene archivo .shp: {zip_path}")
    return f"zip://{zip_path.resolve()}!{shp_names[0]}"


def read_runap(zip_path: Path) -> gpd.GeoDataFrame:
    """Lee shapefile RUNAP desde ZIP oficial."""

    return gpd.read_file(zip_shapefile_uri(zip_path))


def make_valid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Corrige geometrías inválidas sin cambiar el CRS."""

    result = gdf.copy()
    try:
        result["geometry"] = result.geometry.make_valid()
    except Exception:  # noqa: BLE001
        result["geometry"] = result.geometry.buffer(0)
    return result


def clean_runap(
    gdf: gpd.GeoDataFrame,
    valid_conditions: set[str] = DEFAULT_VALID_CONDITIONS,
) -> gpd.GeoDataFrame:
    """Estandariza campos RUNAP y filtra condiciones vigentes."""

    rename_map = {
        "objectid": "objectid",
        "ap_id": "ap_id",
        "condicion": "condicion",
        "ap_nombre": "ap_nombre",
        "ap_categor": "ap_categoria",
        "fecha_insc": "fecha_inscripcion",
        "fecha_regi": "fecha_registro",
        "organizaci": "organizacion",
        "nit": "nit",
        "area_ha_to": "area_ha_total",
        "area_ha_ma": "area_ha_marina",
        "area_ha_te": "area_ha_terrestre",
        "centroide_": "centroide_lon",
        "centroid_1": "centroide_lat",
        "territoria": "territorial",
        "sirap": "sirap",
        "url": "url_runap",
        "wkid": "wkid_fuente",
        "territor_1": "territorial_codigo",
    }
    result = gdf.rename(columns=rename_map).copy()
    for column in [
        "objectid",
        "ap_id",
        "condicion",
        "ap_nombre",
        "ap_categoria",
        "fecha_inscripcion",
        "fecha_registro",
        "organizacion",
        "area_ha_total",
        "area_ha_marina",
        "area_ha_terrestre",
        "centroide_lon",
        "centroide_lat",
        "territorial",
        "sirap",
        "url_runap",
        "geometry",
    ]:
        if column not in result.columns:
            result[column] = pd.NA

    result["condicion_normalizada"] = result["condicion"].map(normalize_text)
    result["ap_nombre"] = result["ap_nombre"].astype("string").str.strip()
    result["ap_categoria"] = result["ap_categoria"].astype("string").str.strip()
    result["ap_categoria_normalizada"] = result["ap_categoria"].map(normalize_text)
    numeric_columns = [
        "objectid",
        "ap_id",
        "area_ha_total",
        "area_ha_marina",
        "area_ha_terrestre",
        "centroide_lon",
        "centroide_lat",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if valid_conditions:
        result = result[result["condicion_normalizada"].isin(valid_conditions)].copy()

    result["fuente"] = "Parques Nacionales Naturales de Colombia - RUNAP"
    result["url_servicio"] = SERVICE_URL
    result["url_descarga"] = DOWNLOAD_URL
    result["fecha_descarga_utc"] = datetime.now(timezone.utc).isoformat()

    columns = [
        "objectid",
        "ap_id",
        "condicion",
        "condicion_normalizada",
        "ap_nombre",
        "ap_categoria",
        "ap_categoria_normalizada",
        "fecha_inscripcion",
        "fecha_registro",
        "organizacion",
        "area_ha_total",
        "area_ha_marina",
        "area_ha_terrestre",
        "centroide_lon",
        "centroide_lat",
        "territorial",
        "sirap",
        "url_runap",
        "fuente",
        "url_servicio",
        "url_descarga",
        "fecha_descarga_utc",
        "geometry",
    ]
    return make_valid_geometries(result[columns])


def load_municipalities(municipalities_geojson: Path) -> gpd.GeoDataFrame:
    """Carga municipios/distritos IGAC."""

    if not municipalities_geojson.exists():
        raise FileNotFoundError(f"No existe GeoJSON municipal: {municipalities_geojson}")
    municipalities = gpd.read_file(municipalities_geojson)
    municipalities["categoria"] = pd.to_numeric(municipalities["categoria"], errors="coerce")
    municipalities = municipalities[municipalities["categoria"].isin([1, 2])].copy()
    municipalities["codigo_dane"] = municipalities["codigo_dane"].astype("string").str.zfill(5)
    return make_valid_geometries(municipalities)


def calculate_municipal_restrictions(
    runap: gpd.GeoDataFrame,
    municipalities: gpd.GeoDataFrame,
    hard_threshold: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula área protegida y restricción RUNAP por municipio."""

    runap_proj = make_valid_geometries(runap.to_crs(PROJECTED_CRS))
    municipalities_proj = make_valid_geometries(municipalities.to_crs(PROJECTED_CRS))
    municipalities_proj["area_municipio_km2_calc"] = municipalities_proj.geometry.area / 1_000_000

    overlay_columns_muni = [
        "codigo_dane",
        "municipio",
        "departamento",
        "area_km2_igac",
        "area_municipio_km2_calc",
        "geometry",
    ]
    overlay_columns_runap = [
        "ap_id",
        "ap_nombre",
        "ap_categoria",
        "condicion_normalizada",
        "geometry",
    ]
    intersections = gpd.overlay(
        municipalities_proj[overlay_columns_muni],
        runap_proj[overlay_columns_runap],
        how="intersection",
        keep_geom_type=True,
    )
    if intersections.empty:
        base = municipalities_proj.drop(columns="geometry").copy()
        base["area_protegida_km2_runap"] = 0.0
        base["pct_area_protegida_runap"] = 0.0
        base["u_i_no_protegido_runap"] = 1.0
        base["r_i_runap"] = 1.0
        base["clasificacion_restriccion_runap"] = "sin_area_protegida"
        return base, pd.DataFrame()

    intersections["area_interseccion_km2"] = intersections.geometry.area / 1_000_000
    category_summary = (
        intersections.drop(columns="geometry")
        .groupby(["codigo_dane", "municipio", "departamento", "ap_categoria"], dropna=False)
        .agg(
            area_interseccion_km2=("area_interseccion_km2", "sum"),
            numero_intersecciones=("ap_id", "count"),
        )
        .reset_index()
        .sort_values(["codigo_dane", "area_interseccion_km2"], ascending=[True, False])
    )

    # Se disuelve por municipio para evitar doble conteo cuando figuras RUNAP se traslapan.
    protected_by_municipality = intersections[["codigo_dane", "geometry"]].dissolve(by="codigo_dane")
    protected_by_municipality["area_protegida_km2_runap"] = (
        protected_by_municipality.geometry.area / 1_000_000
    )
    protected_area = protected_by_municipality.drop(columns="geometry").reset_index()

    base = municipalities_proj.drop(columns="geometry").copy()
    result = base.merge(protected_area, on="codigo_dane", how="left")
    result["area_protegida_km2_runap"] = result["area_protegida_km2_runap"].fillna(0.0)
    result["pct_area_protegida_runap_raw"] = (
        result["area_protegida_km2_runap"] / result["area_municipio_km2_calc"]
    )
    result["pct_area_protegida_runap"] = result["pct_area_protegida_runap_raw"].clip(0, 1)
    result["area_no_protegida_km2_runap"] = (
        result["area_municipio_km2_calc"] - result["area_protegida_km2_runap"]
    ).clip(lower=0)
    result["u_i_no_protegido_runap"] = 1 - result["pct_area_protegida_runap"]
    result["r_i_runap"] = (result["pct_area_protegida_runap"] < hard_threshold).astype(float)
    result["umbral_exclusion_runap"] = hard_threshold
    result["clasificacion_restriccion_runap"] = pd.cut(
        result["pct_area_protegida_runap"],
        bins=[-0.001, 0.0, 0.30, 0.80, 1.001],
        labels=[
            "sin_area_protegida",
            "restriccion_baja_media",
            "restriccion_alta",
            "exclusion_preliminar",
        ],
    ).astype("string")
    result["criterio_runap"] = (
        "R_i_RUNAP=0 si el area protegida RUNAP cubre al menos "
        f"{hard_threshold:.0%} del municipio; U_i proxy = 1 - proporcion protegida."
    )
    return result.sort_values("pct_area_protegida_runap", ascending=False), category_summary


def metadata_to_table(metadata: dict[str, Any], zip_path: Path) -> pd.DataFrame:
    """Resume metadatos de fuente."""

    return pd.DataFrame(
        [
            {"clave": "fuente", "valor": "Parques Nacionales Naturales de Colombia - RUNAP"},
            {"clave": "url_servicio", "valor": SERVICE_URL},
            {"clave": "url_descarga", "valor": DOWNLOAD_URL},
            {"clave": "zip_local", "valor": str(zip_path)},
            {"clave": "descripcion", "valor": metadata.get("serviceDescription", "")},
            {"clave": "copyright", "valor": metadata.get("copyrightText", "")},
            {"clave": "capabilities", "valor": metadata.get("capabilities", "")},
            {"clave": "supported_query_formats", "valor": metadata.get("supportedQueryFormats", "")},
            {"clave": "max_record_count", "valor": metadata.get("maxRecordCount", "")},
        ]
    )


def quality_table(
    raw_runap: gpd.GeoDataFrame,
    cleaned_runap: gpd.GeoDataFrame,
    restrictions: pd.DataFrame,
) -> pd.DataFrame:
    """Controles de calidad RUNAP."""

    rows = [
        {"indicador": "runap_registros_crudos", "valor": int(len(raw_runap))},
        {"indicador": "runap_registros_filtrados", "valor": int(len(cleaned_runap))},
        {"indicador": "runap_geometrias_invalidas_filtradas", "valor": int((~cleaned_runap.geometry.is_valid).sum())},
        {"indicador": "municipios_evaluados", "valor": int(len(restrictions))},
        {
            "indicador": "municipios_con_area_protegida",
            "valor": int((restrictions["area_protegida_km2_runap"] > 0).sum()),
        },
        {
            "indicador": "municipios_exclusion_preliminar",
            "valor": int((restrictions["r_i_runap"] == 0).sum()),
        },
    ]
    for category, count in cleaned_runap["ap_categoria"].value_counts(dropna=False).items():
        rows.append({"indicador": f"categoria_{category}", "valor": int(count)})
    return pd.DataFrame(rows)


def write_observations(
    output_path: Path,
    cleaned_runap: gpd.GeoDataFrame,
    restrictions: pd.DataFrame,
    hard_threshold: float,
) -> None:
    """Documenta metodologia y limitaciones."""

    lines = [
        "Restricciones territoriales RUNAP",
        "================================",
        "",
        "Fuente: Parques Nacionales Naturales de Colombia - RUNAP.",
        f"Servicio: {SERVICE_URL}",
        f"Descarga oficial: {DOWNLOAD_URL}",
        f"Areas protegidas usadas: {len(cleaned_runap)}",
        "",
        "Uso en el modelo:",
        "- RUNAP se usa como restriccion territorial.",
        "- u_i_no_protegido_runap = 1 - proporcion del municipio cubierta por areas protegidas RUNAP.",
        f"- r_i_runap = 0 si la proporcion protegida es >= {hard_threshold:.0%}.",
        "- El area protegida municipal se calcula disolviendo intersecciones para evitar doble conteo por traslapes RUNAP.",
        "",
        "Limitaciones:",
        "- Esto no reemplaza estudio predial, ambiental ni licenciamiento.",
        "- RUNAP indica areas protegidas registradas; otras restricciones territoriales pueden no estar incluidas.",
        "- La proporcion protegida se calcula contra geometria municipal IGAC reproyectada a EPSG:3116.",
        "",
        "Resumen municipal:",
        f"- Municipios evaluados: {len(restrictions)}",
        f"- Municipios con alguna area protegida: {(restrictions['area_protegida_km2_runap'] > 0).sum()}",
        f"- Municipios con exclusion preliminar RUNAP: {(restrictions['r_i_runap'] == 0).sum()}",
        "",
        "Categorias RUNAP usadas:",
    ]
    for category, count in cleaned_runap["ap_categoria"].value_counts(dropna=False).items():
        lines.append(f"- {category}: {count}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_runap_processing(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    municipalities_geojson: Path = DEFAULT_MUNICIPALITIES_GEOJSON,
    force: bool = False,
    hard_threshold: float = 0.80,
    export_geojson: bool = False,
    include_construction: bool = False,
) -> dict[str, Path]:
    """Ejecuta pipeline RUNAP completo."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_conditions = set(DEFAULT_VALID_CONDITIONS)
    if include_construction:
        valid_conditions.add("CONSTRUCCION")

    metadata = fetch_metadata(raw_dir, force=force)
    zip_path = download_runap_zip(raw_dir, force=force)
    raw_runap = read_runap(zip_path)
    cleaned_runap = clean_runap(raw_runap, valid_conditions=valid_conditions)
    municipalities = load_municipalities(municipalities_geojson)
    restrictions, category_summary = calculate_municipal_restrictions(
        cleaned_runap,
        municipalities,
        hard_threshold=hard_threshold,
    )

    areas_csv = output_dir / "runap_areas_protegidas.csv"
    restrictions_csv = output_dir / "runap_restricciones_municipios.csv"
    category_csv = output_dir / "runap_intersecciones_categoria_municipio.csv"
    metadata_csv = output_dir / "runap_metadata_fuente.csv"
    quality_csv = output_dir / "runap_calidad.csv"
    observations_txt = output_dir / "runap_observaciones.txt"

    cleaned_runap.drop(columns="geometry").to_csv(areas_csv, index=False, encoding="utf-8-sig")
    restrictions.to_csv(restrictions_csv, index=False, encoding="utf-8-sig")
    category_summary.to_csv(category_csv, index=False, encoding="utf-8-sig")
    metadata_to_table(metadata, zip_path).to_csv(metadata_csv, index=False, encoding="utf-8-sig")
    quality_table(raw_runap, cleaned_runap, restrictions).to_csv(
        quality_csv,
        index=False,
        encoding="utf-8-sig",
    )
    write_observations(observations_txt, cleaned_runap, restrictions, hard_threshold)

    outputs = {
        "areas": areas_csv,
        "restrictions": restrictions_csv,
        "category_summary": category_csv,
        "metadata": metadata_csv,
        "quality": quality_csv,
        "observations": observations_txt,
    }

    if export_geojson:
        geojson_path = output_dir / "runap_areas_protegidas.geojson"
        cleaned_runap.to_crs(OUTPUT_CRS).to_file(geojson_path, driver="GeoJSON")
        outputs["geojson"] = geojson_path

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Procesa RUNAP de Parques Nacionales para restricciones territoriales municipales."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--municipalities-geojson", type=Path, default=DEFAULT_MUNICIPALITIES_GEOJSON)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hard-threshold", type=float, default=0.80)
    parser.add_argument("--export-geojson", action="store_true")
    parser.add_argument(
        "--include-construction",
        action="store_true",
        help="Incluye registros RUNAP en condicion CONSTRUCCION.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_runap_processing(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        municipalities_geojson=args.municipalities_geojson,
        force=args.force,
        hard_threshold=args.hard_threshold,
        export_geojson=args.export_geojson,
        include_construction=args.include_construction,
    )
    print("Restricciones RUNAP procesadas.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print("RUNAP queda listo como U_i no protegido y R_i de restriccion territorial.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
