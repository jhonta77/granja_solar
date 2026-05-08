"""Calcula distancia desde cada municipio a la red vial primaria de Colombia.

Fuente de datos: OpenStreetMap via Overpass API (publica, sin autenticacion).
Se descargan tramos de carreteras con highway=motorway|trunk|primary en Colombia
en una sola consulta, se convierten a GeoDataFrame y se calcula la distancia
euclidiana en EPSG:3116 desde cada centroide municipal al tramo mas cercano.

Por que OSM y no INVIAS
------------------------
INVIAS publica la red vial en datos.gov.co como shapefile descargable manualmente
(no tiene API REST estable de geometrias). OSM tiene cobertura similar para vias
primarias y nacionales de Colombia, y es accesible via Overpass API sin descarga
manual.

Si prefieres usar el shapefile oficial INVIAS, descargalo desde:
    https://www.datos.gov.co/Transporte/Red-Vial-Nacional-INVIAS/...
y ejecuta con --input-geojson ruta/al/archivo.geojson

Salida principal
----------------
    data/clean/invias_vias/distancia_vias_municipios.csv

Columnas:
    codigo_dane          CHAR(5)
    dist_via_primaria_km DOUBLE   distancia euclidiana en EPSG:3116 (km)
    nombre_via           VARCHAR  nombre de la via mas cercana (OSM name=)
    tipo_via             VARCHAR  motorway / trunk / primary
    fuente_vias          VARCHAR  OSM o INVIAS

Uso
---
    python -m src.extract.invias_vias
    python -m src.extract.invias_vias --force
    python -m src.extract.invias_vias --input-geojson data/raw/invias/red_vial.geojson
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import urllib3


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUNICIPALITIES_CSV = (
    PROJECT_ROOT / "data" / "clean" / "base_municipios" / "municipios_distritos_colombia.csv"
)
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "invias_vias"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "invias_vias"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

COLOMBIA_BBOX = "-4.23,-81.73,13.40,-66.87"

OVERPASS_QUERY = f"""
[out:json][timeout:300][bbox:{COLOMBIA_BBOX}];
way["highway"~"^(motorway|trunk|primary)$"];
out geom;
""".strip()

OUTPUT_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:3116"

HIGHWAY_ORDER = {"motorway": 0, "trunk": 1, "primary": 2}


def fetch_roads_overpass(raw_dir: Path, force: bool = False) -> dict[str, Any]:
    raw_path = raw_dir / "colombia_roads_osm.json"
    if raw_path.exists() and not force:
        print("  Cargando red vial OSM desde cache...")
        return json.loads(raw_path.read_text(encoding="utf-8"))

    print("  Descargando red vial OSM (motorway/trunk/primary) desde Overpass API...")
    print(f"  URL: {OVERPASS_URL}")
    print("  Esto puede tardar 1-3 minutos segun la velocidad de la conexion.")

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    for attempt in range(1, 4):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": OVERPASS_QUERY},
                timeout=360,
                headers={"User-Agent": "granja-solar-colombia/1.0"},
            )
            response.raise_for_status()
            data = response.json()
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            n_elements = len(data.get("elements", []))
            print(f"  Descargados {n_elements} elementos de red vial OSM.")
            return data
        except Exception as error:  # noqa: BLE001
            if attempt < 3:
                print(f"  Intento {attempt} fallo ({error}), reintentando en {attempt * 5}s...")
                time.sleep(attempt * 5)
            else:
                raise RuntimeError(
                    f"No fue posible descargar la red vial de Overpass API: {error}\n"
                    "Alternativa: descarga el shapefile INVIAS desde datos.gov.co y usa --input-geojson."
                ) from error
    return {}  # never reached


def osm_json_to_geodataframe(data: dict[str, Any]) -> gpd.GeoDataFrame:
    """Convierte respuesta Overpass JSON a GeoDataFrame de lineas."""

    from shapely.geometry import LineString

    rows = []
    for element in data.get("elements", []):
        if element.get("type") != "way":
            continue
        geometry_nodes = element.get("geometry", [])
        if len(geometry_nodes) < 2:
            continue
        coords = [(node["lon"], node["lat"]) for node in geometry_nodes]
        tags = element.get("tags", {})
        rows.append(
            {
                "osm_id": element.get("id"),
                "highway": tags.get("highway", ""),
                "name": tags.get("name", tags.get("ref", "")),
                "ref": tags.get("ref", ""),
                "geometry": LineString(coords),
            }
        )

    if not rows:
        raise ValueError("No se encontraron tramos de carretera en la respuesta OSM.")

    gdf = gpd.GeoDataFrame(rows, crs=OUTPUT_CRS)
    gdf["highway_order"] = gdf["highway"].map(HIGHWAY_ORDER).fillna(9).astype(int)
    return gdf.sort_values("highway_order").reset_index(drop=True)


def load_roads_from_geojson(path: Path) -> gpd.GeoDataFrame:
    """Carga red vial desde archivo GeoJSON o shapefile local (INVIAS u otro)."""

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(OUTPUT_CRS)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(OUTPUT_CRS)
    gdf["fuente_vias"] = "INVIAS"
    for col in ["highway", "name", "ref"]:
        if col not in gdf.columns:
            gdf[col] = ""
    return gdf


def calculate_road_distances(
    municipalities_csv: Path,
    roads_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    fuente: str = "OSM",
) -> pd.DataFrame:
    """Calcula distancia euclidiana desde cada municipio al tramo vial mas cercano."""

    municipalities = pd.read_csv(municipalities_csv, dtype={"codigo_dane": "string"})
    municipalities["codigo_dane"] = municipalities["codigo_dane"].astype("string").str.zfill(5)

    lon_col = next((c for c in municipalities.columns if c in {"lon", "longitud", "longitude"}), None)
    lat_col = next((c for c in municipalities.columns if c in {"lat", "latitud", "latitude"}), None)
    if lon_col is None or lat_col is None:
        raise ValueError("No se detectaron columnas lon/lat en el CSV de municipios.")

    muni_gdf = gpd.GeoDataFrame(
        municipalities.copy(),
        geometry=gpd.points_from_xy(municipalities[lon_col], municipalities[lat_col]),
        crs=OUTPUT_CRS,
    ).to_crs(PROJECTED_CRS)

    roads_proj = roads_gdf.to_crs(PROJECTED_CRS)

    print(f"  Calculando distancias para {len(muni_gdf)} municipios a {len(roads_proj)} tramos viales...")

    nearest_dist_km: list[float] = []
    nearest_name: list[str] = []
    nearest_type: list[str] = []

    chunk_size = 50
    for start in range(0, len(muni_gdf), chunk_size):
        chunk = muni_gdf.iloc[start : start + chunk_size]
        for _, muni_row in chunk.iterrows():
            point = muni_row.geometry
            distances = roads_proj.geometry.distance(point)
            idx_min = int(distances.idxmin())
            nearest_dist_km.append(float(distances.iloc[idx_min]) / 1000.0)
            nearest_name.append(str(roads_proj.iloc[idx_min].get("name", "")))
            nearest_type.append(str(roads_proj.iloc[idx_min].get("highway", "")))

    result = municipalities[["codigo_dane", "municipio", "departamento"]].copy()
    result["dist_via_primaria_km"] = nearest_dist_km
    result["nombre_via"] = nearest_name
    result["tipo_via"] = nearest_type
    result["fuente_vias"] = fuente
    result["fecha_calculo_utc"] = datetime.now(timezone.utc).isoformat()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "distancia_vias_municipios.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def write_observations(path: Path, n_municipios: int, n_tramos: int, fuente: str) -> None:
    lines = [
        "Vias primarias - distancia municipal",
        "=====================================",
        "",
        f"Fuente geometrias: {fuente}",
        f"API Overpass: {OVERPASS_URL}" if fuente == "OSM" else "  Archivo local.",
        f"Tipos de via incluidos: motorway, trunk, primary",
        f"Tramos de carretera procesados: {n_tramos}",
        f"Municipios con distancia calculada: {n_municipios}",
        "",
        "Metodologia:",
        "  - CRS de calculo: EPSG:3116 (Colombia, metros).",
        "  - Distancia euclidiana desde centroide municipal al tramo mas cercano.",
        "  - No representa distancia real por carretera ni tiempo de viaje.",
        "",
        "Uso en scoring multidimensional:",
        "  score_economico: dist_via_primaria_km (a mayor distancia, menor score).",
        "  Municipios alejados de vias primarias tienen mayor costo de transporte",
        "  para materiales e insumos de construccion y operacion.",
        "",
        "Limitaciones:",
        "  - OSM puede tener omisiones en vias secundarias o carreteables.",
        "    Para vias terciarias usar highway=secondary o tertiary.",
        "  - INVIAS oficial: https://www.datos.gov.co/ buscar 'Red Vial Nacional'.",
        "    Descarga el shapefile y usa --input-geojson para mayor precision oficial.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_invias_vias(
    municipalities_csv: Path = DEFAULT_MUNICIPALITIES_CSV,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    input_geojson: Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Descarga red vial y calcula distancia municipal."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_geojson is not None:
        roads_gdf = load_roads_from_geojson(input_geojson)
        fuente = "INVIAS"
        print(f"  Red vial cargada desde archivo local: {len(roads_gdf)} tramos.")
    else:
        raw_data = fetch_roads_overpass(raw_dir, force=force)
        roads_gdf = osm_json_to_geodataframe(raw_data)
        fuente = "OSM"

    distance_df = calculate_road_distances(municipalities_csv, roads_gdf, output_dir, fuente=fuente)

    observations_path = output_dir / "vias_observaciones.txt"
    write_observations(observations_path, len(distance_df), len(roads_gdf), fuente)

    return {
        "distancias": output_dir / "distancia_vias_municipios.csv",
        "observaciones": observations_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula distancia desde municipios a red vial primaria (OSM o INVIAS)."
    )
    parser.add_argument("--municipalities-csv", type=Path, default=DEFAULT_MUNICIPALITIES_CSV)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--input-geojson",
        type=Path,
        default=None,
        help="Ruta a GeoJSON o shapefile INVIAS local. Si se omite, usa Overpass API.",
    )
    parser.add_argument("--force", action="store_true", help="Re-descargar aunque exista cache.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_invias_vias(
            municipalities_csv=args.municipalities_csv.resolve(),
            raw_dir=args.raw_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            input_geojson=args.input_geojson,
            force=args.force,
        )
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {error}")
        return 1

    print("Distancia a red vial calculada.")
    for label, path in outputs.items():
        print(f"  - {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
