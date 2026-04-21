from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RASTER = PROJECT_ROOT / "data" / "PVOUT.tif"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "pvout"

# Caja aproximada de Colombia en EPSG:4326. No reemplaza un limite oficial.
COLOMBIA_BBOX = (-79.5, -4.5, -66.5, 13.7)


def inspect_raster(raster_path: Path) -> dict[str, Any]:
    """Extrae metadatos basicos del raster sin cargarlo completo."""

    with rasterio.open(raster_path) as src:
        return {
            "raster_path": str(raster_path),
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "crs": str(src.crs),
            "bounds_left": src.bounds.left,
            "bounds_bottom": src.bounds.bottom,
            "bounds_right": src.bounds.right,
            "bounds_top": src.bounds.top,
            "pixel_width": src.transform.a,
            "pixel_height": src.transform.e,
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
            "description": src.tags().get("TIFFTAG_IMAGEDESCRIPTION", ""),
            "source_note": (
                "Global Solar Atlas / World Bank Group, ESMAP, Solargis. "
                "PVOUT en kWh/kWp/dia segun metadatos del producto."
            ),
        }


def safe_window_from_bbox(src: rasterio.DatasetReader, bbox: tuple[float, float, float, float]) -> Window:
    """Construye una ventana raster limitada a los bounds del dataset."""

    min_lon, min_lat, max_lon, max_lat = bbox
    bounded = (
        max(min_lon, src.bounds.left),
        max(min_lat, src.bounds.bottom),
        min(max_lon, src.bounds.right),
        min(max_lat, src.bounds.top),
    )
    return from_bounds(*bounded, transform=src.transform).round_offsets().round_lengths()


def read_bbox_array(
    raster_path: Path,
    bbox: tuple[float, float, float, float] = COLOMBIA_BBOX,
) -> tuple[np.ma.MaskedArray, Any]:
    """Lee solo la ventana de la caja geografica indicada."""

    with rasterio.open(raster_path) as src:
        window = safe_window_from_bbox(src, bbox)
        data = src.read(1, window=window, masked=True)
        transform = src.window_transform(window)

    valid = np.ma.masked_invalid(data)
    valid = np.ma.masked_where(valid <= 0, valid)
    return valid, transform


def summarize_array(data: np.ma.MaskedArray, label: str) -> pd.DataFrame:
    """Calcula estadisticas descriptivas de PVOUT."""

    values = data.compressed()
    if values.size == 0:
        raise ValueError(f"No hay valores validos para resumir en {label}.")

    summary = {
        "area_resumen": label,
        "celdas_validas": int(values.size),
        "pvout_min_kwh_kwp_day": float(np.nanmin(values)),
        "pvout_p05_kwh_kwp_day": float(np.nanpercentile(values, 5)),
        "pvout_p25_kwh_kwp_day": float(np.nanpercentile(values, 25)),
        "pvout_media_kwh_kwp_day": float(np.nanmean(values)),
        "pvout_mediana_kwh_kwp_day": float(np.nanpercentile(values, 50)),
        "pvout_p75_kwh_kwp_day": float(np.nanpercentile(values, 75)),
        "pvout_p95_kwh_kwp_day": float(np.nanpercentile(values, 95)),
        "pvout_max_kwh_kwp_day": float(np.nanmax(values)),
    }
    summary["annual_yield_media_kwh_kw_year"] = (
        summary["pvout_media_kwh_kwp_day"] * 365.0
    )
    summary["annual_yield_p95_kwh_kw_year"] = (
        summary["pvout_p95_kwh_kwp_day"] * 365.0
    )
    return pd.DataFrame([summary])


def top_cells_from_array(
    data: np.ma.MaskedArray,
    transform: Any,
    top_n: int,
) -> pd.DataFrame:
    """Obtiene las celdas con mayor PVOUT dentro de la ventana leida."""

    values = data.filled(np.nan)
    flat = values.ravel()
    valid_indices = np.where(~np.isnan(flat))[0]
    if valid_indices.size == 0:
        return pd.DataFrame()

    top_n = min(top_n, valid_indices.size)
    candidate_values = flat[valid_indices]
    top_positions = np.argpartition(candidate_values, -top_n)[-top_n:]
    selected_flat = valid_indices[top_positions]
    selected_values = flat[selected_flat]

    rows: list[dict[str, float]] = []
    for flat_index, value in zip(selected_flat, selected_values):
        row_index, col_index = np.unravel_index(flat_index, values.shape)
        lon, lat = rasterio.transform.xy(transform, row_index, col_index, offset="center")
        rows.append(
            {
                "lon": float(lon),
                "lat": float(lat),
                "pvout_kwh_kwp_day": float(value),
                "annual_yield_kwh_kw_year": float(value) * 365.0,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "pvout_kwh_kwp_day", ascending=False
    ).reset_index(drop=True)


def sample_points(raster_path: Path, points_csv: Path) -> pd.DataFrame:
    """Extrae PVOUT para puntos con columnas lon y lat."""

    points = pd.read_csv(points_csv)
    required = {"lon", "lat"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(
            "El CSV de puntos debe contener columnas lon y lat. Faltan: "
            + ", ".join(sorted(missing))
        )

    coordinates = list(zip(points["lon"], points["lat"]))
    with rasterio.open(raster_path) as src:
        sampled = [value[0] for value in src.sample(coordinates)]

    result = points.copy()
    result["pvout_kwh_kwp_day"] = pd.to_numeric(sampled, errors="coerce")
    result["annual_yield_kwh_kw_year"] = result["pvout_kwh_kwp_day"] * 365.0
    return result


def build_observations(
    metadata: dict[str, Any],
    summary: pd.DataFrame,
    top_n: int,
    used_points: bool,
) -> list[str]:
    """Construye observaciones metodologicas del procesamiento PVOUT."""

    row = summary.iloc[0].to_dict()
    lines = [
        "Procesamiento de PVOUT/radiacion solar",
        "",
        "Fuente:",
        "- World Bank Group, ESMAP, Solargis / Global Solar Atlas.",
        "- Archivo raster local: " + metadata["raster_path"],
        "- Unidad interpretada: kWh/kWp/dia.",
        "",
        "Metadatos:",
        f"- CRS: {metadata['crs']}",
        f"- Dimensiones: {metadata['width']} x {metadata['height']} celdas.",
        f"- Bounds: {metadata['bounds_left']}, {metadata['bounds_bottom']}, {metadata['bounds_right']}, {metadata['bounds_top']}.",
        "",
        "Resumen aproximado para caja geografica de Colombia:",
        f"- Celdas validas: {row['celdas_validas']}",
        f"- PVOUT medio: {row['pvout_media_kwh_kwp_day']:.3f} kWh/kWp/dia.",
        f"- PVOUT p95: {row['pvout_p95_kwh_kwp_day']:.3f} kWh/kWp/dia.",
        f"- Rendimiento anual medio estimado: {row['annual_yield_media_kwh_kw_year']:.2f} kWh/kW-anio.",
        "",
        "Salidas:",
        "- Metadatos del raster.",
        "- Resumen PVOUT para bbox aproximado Colombia.",
        f"- Top {top_n} celdas por PVOUT dentro del bbox aproximado.",
    ]

    if used_points:
        lines.append("- Extraccion puntual de PVOUT para CSV de coordenadas.")

    lines.extend(
        [
            "",
            "Advertencias:",
            "- La caja geografica de Colombia no es un limite politico oficial; puede incluir zonas fuera del pais.",
            "- Para resultados finales se debe recortar con limite oficial de Colombia, municipios o poligonos candidatos.",
            "- PVOUT no mide demanda; es una variable de recurso solar para el score multicriterio.",
            "- No se generaron diagramas; las salidas son tablas reproducibles.",
        ]
    )
    return lines


def run_pvout_processing(
    raster_path: Path,
    output_dir: Path,
    bbox: tuple[float, float, float, float] = COLOMBIA_BBOX,
    top_n: int = 10,
    points_csv: Path | None = None,
) -> dict[str, Path]:
    """Ejecuta procesamiento tabular del raster PVOUT."""

    if not raster_path.exists():
        raise FileNotFoundError(f"No existe el raster PVOUT: {raster_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = inspect_raster(raster_path)
    data, transform = read_bbox_array(raster_path, bbox=bbox)
    summary = summarize_array(data, "bbox_aproximado_colombia")
    top_cells = top_cells_from_array(data, transform, top_n=top_n)

    metadata_path = output_dir / "pvout_raster_metadata.csv"
    summary_path = output_dir / "pvout_colombia_bbox_summary.csv"
    top_cells_path = output_dir / "pvout_top_celdas_bbox_colombia.csv"
    observations_path = output_dir / "pvout_observaciones.txt"

    pd.DataFrame([metadata]).to_csv(metadata_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    top_cells.to_csv(top_cells_path, index=False, encoding="utf-8-sig")

    exported = {
        "metadata": metadata_path,
        "summary": summary_path,
        "top_cells": top_cells_path,
        "observations": observations_path,
    }

    used_points = points_csv is not None
    if points_csv is not None:
        sampled_points = sample_points(raster_path, points_csv)
        points_path = output_dir / "pvout_puntos_extraidos.csv"
        sampled_points.to_csv(points_path, index=False, encoding="utf-8-sig")
        exported["points"] = points_path

    observations_path.write_text(
        "\n".join(build_observations(metadata, summary, top_n, used_points)),
        encoding="utf-8",
    )
    return exported


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    """Parsea bbox como min_lon,min_lat,max_lon,max_lat."""

    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "El bbox debe tener formato min_lon,min_lat,max_lon,max_lat."
        )
    return parts[0], parts[1], parts[2], parts[3]


def build_argument_parser() -> argparse.ArgumentParser:
    """Configura argumentos CLI."""

    parser = argparse.ArgumentParser(
        description="Procesa raster PVOUT de Global Solar Atlas sin generar diagramas."
    )
    parser.add_argument(
        "--raster",
        type=str,
        default=str(DEFAULT_RASTER),
        help="Ruta del GeoTIFF PVOUT.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Carpeta de salida.",
    )
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        default=COLOMBIA_BBOX,
        help="BBox min_lon,min_lat,max_lon,max_lat. Por defecto usa Colombia aproximado.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Numero de celdas con mayor PVOUT a exportar.",
    )
    parser.add_argument(
        "--points-csv",
        type=str,
        default=None,
        help="CSV opcional con columnas lon,lat para extraer PVOUT puntual.",
    )
    return parser


def main() -> int:
    """Punto de entrada CLI."""

    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        exported = run_pvout_processing(
            raster_path=Path(args.raster).resolve(),
            output_dir=Path(args.output_dir).resolve(),
            bbox=args.bbox,
            top_n=args.top_n,
            points_csv=Path(args.points_csv).resolve() if args.points_csv else None,
        )
    except Exception as error:
        print(f"ERROR: {error}")
        return 1

    print("Procesamiento PVOUT finalizado.")
    for label, path in exported.items():
        print(f"- {label}: {path}")
    print("No se generaron diagramas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
