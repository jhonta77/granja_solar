from __future__ import annotations

import argparse
import html
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "igac_pendientes"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "pendientes_igac"

SERVICE_URL = (
    "https://mapas.igac.gov.co/server/rest/services/"
    "ordenamientoterritorial/pendientescolombia/MapServer"
)

# El servicio IGAC clasifica pendiente en rangos discretos. El umbral por defecto
# usa el limite superior de la clase "Plano" para evitar inventar cortes nuevos.
DEFAULT_MAX_SLOPE_PERCENT = 7.0
DEFAULT_CONDITIONAL_MAX_SLOPE_PERCENT = 14.0


def normalize_text(value: Any) -> str:
    """Normaliza texto para busquedas robustas."""

    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.lower().strip()


def normalize_column_name(column_name: Any) -> str:
    """Convierte nombres de columnas a snake_case."""

    normalized = normalize_text(column_name)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return normalized or "columna_sin_nombre"


def request_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = 90,
    retries: int = 3,
) -> dict[str, Any]:
    """Consulta JSON con reintentos para servicios ArcGIS REST."""

    last_error: Exception | None = None
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"No fue posible consultar {url}. Ultimo error: {last_error}")


def write_json(path: Path, data: Any) -> None:
    """Guarda JSON para trazabilidad."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Lee un JSON local cacheado."""

    return json.loads(path.read_text(encoding="utf-8"))


def fetch_metadata(raw_dir: Path, force: bool = False) -> dict[str, Any]:
    """Descarga metadatos del MapServer de pendientes IGAC."""

    raw_path = raw_dir / "mapserver_metadata.json"
    if raw_path.exists() and not force:
        return read_json(raw_path)

    metadata = request_json(SERVICE_URL, params={"f": "pjson"})
    write_json(raw_path, metadata)
    return metadata


def fetch_layer_metadata(raw_dir: Path, force: bool = False, layer_id: int = 0) -> dict[str, Any]:
    """Descarga metadatos de la capa raster de pendientes."""

    raw_path = raw_dir / f"layer_{layer_id}_metadata.json"
    if raw_path.exists() and not force:
        return read_json(raw_path)

    metadata = request_json(f"{SERVICE_URL}/{layer_id}", params={"f": "pjson"})
    write_json(raw_path, metadata)
    return metadata


def fetch_legend(raw_dir: Path, force: bool = False) -> dict[str, Any]:
    """Descarga la leyenda oficial de clases de pendiente."""

    raw_path = raw_dir / "legend.json"
    if raw_path.exists() and not force:
        return read_json(raw_path)

    legend = request_json(f"{SERVICE_URL}/legend", params={"f": "pjson"})
    write_json(raw_path, legend)
    return legend


def metadata_to_table(metadata: dict[str, Any], layer_metadata: dict[str, Any]) -> pd.DataFrame:
    """Convierte metadatos relevantes a tabla llave-valor."""

    layers = metadata.get("layers") or []
    layer = layers[0] if layers else {}
    full_extent = metadata.get("fullExtent") or {}
    spatial_reference = metadata.get("spatialReference") or {}
    rows = [
        {"campo": "fuente", "valor": "Instituto Geografico Agustin Codazzi - IGAC"},
        {"campo": "url_servicio", "valor": SERVICE_URL},
        {"campo": "map_name", "valor": metadata.get("mapName")},
        {"campo": "service_description", "valor": metadata.get("serviceDescription")},
        {"campo": "copyright", "valor": metadata.get("copyrightText")},
        {"campo": "layer_id", "valor": layer.get("id")},
        {"campo": "layer_name", "valor": layer.get("name")},
        {"campo": "layer_type", "valor": layer.get("type")},
        {"campo": "spatial_reference_wkid", "valor": spatial_reference.get("wkid")},
        {"campo": "spatial_reference_latest_wkid", "valor": spatial_reference.get("latestWkid")},
        {"campo": "full_extent_xmin", "valor": full_extent.get("xmin")},
        {"campo": "full_extent_ymin", "valor": full_extent.get("ymin")},
        {"campo": "full_extent_xmax", "valor": full_extent.get("xmax")},
        {"campo": "full_extent_ymax", "valor": full_extent.get("ymax")},
        {"campo": "max_record_count", "valor": metadata.get("maxRecordCount")},
        {"campo": "layer_fields", "valor": json.dumps(layer_metadata.get("fields", []), ensure_ascii=False)},
    ]
    return pd.DataFrame(rows)


def legend_to_table(legend: dict[str, Any]) -> pd.DataFrame:
    """Convierte la leyenda ArcGIS a tabla plana."""

    rows: list[dict[str, Any]] = []
    for layer in legend.get("layers", []):
        for item in layer.get("legend", []):
            label = item.get("label")
            min_slope, max_slope = parse_slope_range(label)
            rows.append(
                {
                    "layer_id": layer.get("layerId"),
                    "layer_name": layer.get("layerName"),
                    "layer_type": layer.get("layerType"),
                    "clase_igac": label,
                    "pendiente_min_pct": min_slope,
                    "pendiente_max_pct": max_slope,
                }
            )
    return pd.DataFrame(rows)


def parse_slope_range(label: Any) -> tuple[float | None, float | None]:
    """Extrae rango aproximado de pendiente desde etiquetas IGAC."""

    text = normalize_text(label)
    numbers = [float(match.replace(",", ".")) for match in re.findall(r"\d+(?:[\.,]\d+)?", text)]

    if "plano" in text and len(numbers) >= 2:
        return numbers[0], numbers[1]
    if "inclinado" in text and len(numbers) >= 2:
        return numbers[0], numbers[1]
    if "empinado" in text and numbers:
        return numbers[0], None
    return None, None


def classify_slope_class(
    class_label: Any,
    max_slope_percent: float,
    conditional_max_slope_percent: float,
) -> dict[str, Any]:
    """Clasifica una clase IGAC segun umbrales de viabilidad."""

    min_slope, max_slope = parse_slope_range(class_label)
    label = "" if class_label is None else str(class_label)

    if max_slope is not None and max_slope <= max_slope_percent:
        viability = "viable"
        score = 1.0
        reason = f"Clase completa por debajo o igual a {max_slope_percent}%."
    elif max_slope is not None and min_slope is not None and min_slope >= max_slope_percent:
        if max_slope <= conditional_max_slope_percent:
            viability = "condicional"
            score = 0.5
            reason = (
                f"Clase por encima de {max_slope_percent}% pero dentro de "
                f"{conditional_max_slope_percent}%."
            )
        else:
            viability = "no_viable"
            score = 0.0
            reason = f"Clase supera {conditional_max_slope_percent}%."
    elif max_slope is None and min_slope is not None:
        viability = "no_viable"
        score = 0.0
        reason = f"Clase abierta desde {min_slope}%; se excluye por precaucion."
    else:
        viability = "desconocida"
        score = None
        reason = "No fue posible interpretar la etiqueta de pendiente."

    return {
        "clase_igac": label,
        "pendiente_min_pct": min_slope,
        "pendiente_max_pct": max_slope,
        "max_slope_percent": max_slope_percent,
        "conditional_max_slope_percent": conditional_max_slope_percent,
        "viabilidad_pendiente": viability,
        "score_pendiente": score,
        "criterio": reason,
    }


def clean_html_text(value: Any) -> str:
    """Elimina etiquetas HTML simples de textos descriptivos del servicio."""

    text = html.unescape("" if value is None else str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_criteria_table(
    legend_df: pd.DataFrame,
    max_slope_percent: float,
    conditional_max_slope_percent: float,
) -> pd.DataFrame:
    """Construye tabla de criterios de viabilidad por clase IGAC."""

    if legend_df.empty:
        return pd.DataFrame(
            columns=[
                "clase_igac",
                "pendiente_min_pct",
                "pendiente_max_pct",
                "max_slope_percent",
                "conditional_max_slope_percent",
                "viabilidad_pendiente",
                "score_pendiente",
                "criterio",
            ]
        )

    rows = [
        classify_slope_class(row["clase_igac"], max_slope_percent, conditional_max_slope_percent)
        for _, row in legend_df.iterrows()
    ]
    return pd.DataFrame(rows)


def detect_coordinate_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Detecta columnas de longitud y latitud en un CSV de puntos."""

    normalized = {column: normalize_column_name(column) for column in df.columns}
    lon_candidates = [
        column
        for column, normalized_name in normalized.items()
        if normalized_name in {"lon", "longitud", "longitude", "x"} or "longitud" in normalized_name
    ]
    lat_candidates = [
        column
        for column, normalized_name in normalized.items()
        if normalized_name in {"lat", "latitud", "latitude", "y"} or "latitud" in normalized_name
    ]
    if not lon_candidates or not lat_candidates:
        raise ValueError(
            "No se detectaron columnas de coordenadas. Use nombres como lon/lat o longitud/latitud."
        )
    return lon_candidates[0], lat_candidates[0]


def identify_slope_at_point(lon: float, lat: float, tolerance: int = 3) -> dict[str, Any]:
    """Consulta la clase de pendiente IGAC para un punto WGS84."""

    params = {
        "f": "json",
        "geometry": json.dumps(
            {"x": lon, "y": lat, "spatialReference": {"wkid": 4326}},
            ensure_ascii=False,
        ),
        "geometryType": "esriGeometryPoint",
        "sr": 4326,
        "layers": "all:0",
        "tolerance": tolerance,
        "mapExtent": json.dumps(
            {
                "xmin": -79.5,
                "ymin": -4.5,
                "xmax": -66.5,
                "ymax": 13.7,
                "spatialReference": {"wkid": 4326},
            }
        ),
        "imageDisplay": "800,600,96",
        "returnGeometry": "false",
    }
    data = request_json(f"{SERVICE_URL}/identify", params=params, timeout=90)
    results = data.get("results") or []
    if not results:
        return {
            "pendiente_igac": None,
            "identify_ok": False,
            "identify_mensaje": "Sin resultado en IGAC para el punto.",
        }

    attributes = results[0].get("attributes") or {}
    return {
        "pendiente_igac": attributes.get("Raster.Pendiente") or attributes.get("Pendiente"),
        "identify_ok": True,
        "identify_mensaje": "",
    }


def identify_slope_at_points(
    coordinates: list[tuple[float, float]],
    tolerance: int = 3,
) -> list[dict[str, Any]]:
    """Consulta la clase de pendiente IGAC para varios puntos WGS84 en una sola llamada."""

    if not coordinates:
        return []

    xs = [coord[0] for coord in coordinates]
    ys = [coord[1] for coord in coordinates]
    padding = 0.1
    params = {
        "f": "json",
        "geometry": json.dumps(
            {
                "points": [[lon, lat] for lon, lat in coordinates],
                "spatialReference": {"wkid": 4326},
            },
            ensure_ascii=False,
        ),
        "geometryType": "esriGeometryMultipoint",
        "sr": 4326,
        "layers": "all:0",
        "tolerance": tolerance,
        "mapExtent": json.dumps(
            {
                "xmin": min(xs) - padding,
                "ymin": min(ys) - padding,
                "xmax": max(xs) + padding,
                "ymax": max(ys) + padding,
                "spatialReference": {"wkid": 4326},
            }
        ),
        "imageDisplay": "1200,900,96",
        "returnGeometry": "false",
    }
    data = request_json(f"{SERVICE_URL}/identify", params=params, timeout=120)
    results = data.get("results") or []

    if len(results) != len(coordinates):
        # ArcGIS no devuelve identificador por punto. Si el tamano no coincide,
        # se usa consulta individual para no asignar clases en orden incorrecto.
        return [identify_slope_at_point(lon, lat, tolerance=tolerance) for lon, lat in coordinates]

    output: list[dict[str, Any]] = []
    for result in results:
        attributes = result.get("attributes") or {}
        output.append(
            {
                "pendiente_igac": attributes.get("Raster.Pendiente") or attributes.get("Pendiente"),
                "identify_ok": True,
                "identify_mensaje": "",
            }
        )
    return output


def sample_points(
    points_csv: Path,
    max_slope_percent: float,
    conditional_max_slope_percent: float,
    tolerance: int,
    output_path: Path | None = None,
    resume: bool = True,
    flush_every: int = 10,
    limit: int | None = None,
    batch_size: int = 50,
) -> pd.DataFrame:
    """Consulta pendiente IGAC para un CSV de puntos WGS84."""

    points = pd.read_csv(points_csv)
    if points.empty:
        return points

    lon_col, lat_col = detect_coordinate_columns(points)
    points = points.copy()
    points["indice_punto"] = points.index
    if limit is not None:
        points = points.head(limit)

    key_column = "codigo_dane" if "codigo_dane" in points.columns else "indice_punto"
    existing = pd.DataFrame()
    completed_keys: set[str] = set()
    if resume and output_path is not None and output_path.exists():
        existing = pd.read_csv(output_path)
        if key_column in existing.columns:
            completed_keys = set(existing[key_column].astype(str))

    rows: list[dict[str, Any]] = []
    pending = points[~points[key_column].astype(str).isin(completed_keys)]
    total = len(points)
    pending_total = len(pending)
    if completed_keys:
        print(f"Resume activo: {total - pending_total}/{total} puntos ya estaban procesados.")

    def flush_rows() -> None:
        if output_path is None or not rows:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=[key_column], keep="last")
        combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    batch_size = max(1, int(batch_size))
    flush_every = max(1, int(flush_every))
    for start in range(0, pending_total, batch_size):
        batch = pending.iloc[start : start + batch_size]
        valid_positions: list[int] = []
        valid_coordinates: list[tuple[float, float]] = []
        batch_results: list[dict[str, Any]] = []

        for position, (_, row) in enumerate(batch.iterrows()):
            lon = pd.to_numeric(row[lon_col], errors="coerce")
            lat = pd.to_numeric(row[lat_col], errors="coerce")
            result = row.to_dict()
            result["lon_consulta"] = lon
            result["lat_consulta"] = lat
            if pd.isna(lon) or pd.isna(lat):
                result.update(
                    {
                        "pendiente_igac": None,
                        "identify_ok": False,
                        "identify_mensaje": "Coordenadas invalidas.",
                    }
                )
            else:
                valid_positions.append(position)
                valid_coordinates.append((float(lon), float(lat)))
            batch_results.append(result)

        valid_results = identify_slope_at_points(valid_coordinates, tolerance=tolerance)
        for position, slope_result in zip(valid_positions, valid_results):
            batch_results[position].update(slope_result)

        for result in batch_results:
            if "identify_ok" not in result:
                result.update(
                    {
                        "pendiente_igac": None,
                        "identify_ok": False,
                        "identify_mensaje": "Sin resultado asignado por el servicio IGAC.",
                    }
                )
            result.update(
                classify_slope_class(
                    result.get("pendiente_igac"),
                    max_slope_percent=max_slope_percent,
                    conditional_max_slope_percent=conditional_max_slope_percent,
                )
            )
            rows.append(result)

        processed_count = min(start + len(batch), pending_total)
        if output_path is not None and (
            len(rows) % flush_every == 0 or processed_count == pending_total
        ):
            flush_rows()
            print(f"Pendiente IGAC: {len(existing) + len(rows)}/{total} puntos guardados.")

    if output_path is not None and output_path.exists():
        return pd.read_csv(output_path)
    return pd.DataFrame(rows)


def build_observations(
    metadata: dict[str, Any],
    max_slope_percent: float,
    conditional_max_slope_percent: float,
    sampled_points: int | None,
) -> str:
    """Redacta observaciones metodologicas para el informe."""

    description = clean_html_text(
        metadata.get("serviceDescription") or metadata.get("description") or ""
    )
    lines = [
        "Observaciones pendiente IGAC",
        "============================",
        f"Fuente: Instituto Geografico Agustin Codazzi - IGAC.",
        f"URL servicio: {SERVICE_URL}",
        "Producto: Pendientes de Colombia.",
        "Clasificacion oficial del servicio: plana (0-7%), inclinada (>7%-14%) y empinada (>14%).",
        "Insumo reportado por IGAC: MDE SRTM 30 m, 2019, reducido con modelo geoidal GEOCOL2004.",
        "",
        "Criterio adoptado para viabilidad solar preliminar:",
        f"- Viable: pendiente <= {max_slope_percent}%.",
        f"- Condicional: pendiente > {max_slope_percent}% y <= {conditional_max_slope_percent}%.",
        f"- No viable: pendiente > {conditional_max_slope_percent}%.",
        "",
        "Limitaciones:",
        "- La capa IGAC es una clasificacion raster por rangos; no entrega en este flujo el valor continuo exacto de pendiente.",
        "- Si se requiere diseno de ingenieria, se debe usar topografia de mayor resolucion y validar cortes/rellenos, accesos y drenaje.",
        "- Este criterio es un filtro fisico preliminar; no reemplaza restricciones ambientales, prediales, red electrica ni PVOUT.",
        "",
        f"Puntos muestreados: {sampled_points if sampled_points is not None else 'No se proporciono CSV de puntos.'}",
        "",
        "Descripcion original del servicio:",
        str(description),
    ]
    return "\n".join(lines) + "\n"


def run_processing(
    raw_dir: Path,
    output_dir: Path,
    max_slope_percent: float,
    conditional_max_slope_percent: float,
    force: bool,
    points_csv: Path | None = None,
    tolerance: int = 3,
    resume: bool = True,
    flush_every: int = 10,
    limit: int | None = None,
    batch_size: int = 50,
) -> dict[str, Path]:
    """Ejecuta la extraccion y clasificacion de pendiente."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = fetch_metadata(raw_dir, force=force)
    layer_metadata = fetch_layer_metadata(raw_dir, force=force)
    legend = fetch_legend(raw_dir, force=force)

    metadata_df = metadata_to_table(metadata, layer_metadata)
    legend_df = legend_to_table(legend)
    criteria_df = build_criteria_table(
        legend_df,
        max_slope_percent=max_slope_percent,
        conditional_max_slope_percent=conditional_max_slope_percent,
    )

    metadata_path = output_dir / "igac_pendientes_metadata.csv"
    legend_path = output_dir / "igac_pendientes_leyenda.csv"
    criteria_path = output_dir / "criterios_viabilidad_pendiente.csv"
    observations_path = output_dir / "pendientes_observaciones.txt"

    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    legend_df.to_csv(legend_path, index=False, encoding="utf-8-sig")
    criteria_df.to_csv(criteria_path, index=False, encoding="utf-8-sig")

    sampled_points_count: int | None = None
    points_path: Path | None = None
    if points_csv:
        points_path = output_dir / "pendiente_puntos_extraidos.csv"
        if force and points_path.exists():
            points_path.unlink()
        sampled = sample_points(
            points_csv=points_csv,
            max_slope_percent=max_slope_percent,
            conditional_max_slope_percent=conditional_max_slope_percent,
            tolerance=tolerance,
            output_path=points_path,
            resume=resume,
            flush_every=flush_every,
            limit=limit,
            batch_size=batch_size,
        )
        sampled.to_csv(points_path, index=False, encoding="utf-8-sig")
        sampled_points_count = len(sampled)

    observations_path.write_text(
        build_observations(
            metadata=metadata,
            max_slope_percent=max_slope_percent,
            conditional_max_slope_percent=conditional_max_slope_percent,
            sampled_points=sampled_points_count,
        ),
        encoding="utf-8-sig",
    )

    outputs = {
        "metadata": metadata_path,
        "legend": legend_path,
        "criteria": criteria_path,
        "observations": observations_path,
    }
    if points_path:
        outputs["points"] = points_path
    return outputs


def parse_args() -> argparse.Namespace:
    """Argumentos de consola."""

    parser = argparse.ArgumentParser(
        description="Procesa el servicio IGAC de pendientes para criterios de viabilidad solar."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-slope-percent",
        type=float,
        default=DEFAULT_MAX_SLOPE_PERCENT,
        help="Pendiente maxima viable. Por defecto 7, alineado con la clase Plano IGAC.",
    )
    parser.add_argument(
        "--conditional-max-slope-percent",
        type=float,
        default=DEFAULT_CONDITIONAL_MAX_SLOPE_PERCENT,
        help="Pendiente maxima condicional. Por defecto 14, limite superior de la clase Inclinado IGAC.",
    )
    parser.add_argument(
        "--points-csv",
        type=Path,
        help="CSV opcional con columnas lon/lat o longitud/latitud para consultar pendiente.",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=3,
        help="Tolerancia del identify ArcGIS en pixeles. Por defecto 3.",
    )
    parser.add_argument("--force", action="store_true", help="Vuelve a consultar el servicio.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="No reutiliza pendiente_puntos_extraidos.csv existente.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="Guarda avance cada N puntos. Por defecto 10.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Procesa solo los primeros N puntos. Util para pruebas rapidas.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Cantidad de puntos por llamada multipunto al servicio IGAC. Por defecto 50.",
    )
    return parser.parse_args()


def main() -> int:
    """Punto de entrada."""

    args = parse_args()
    if args.conditional_max_slope_percent < args.max_slope_percent:
        raise ValueError("conditional-max-slope-percent no puede ser menor que max-slope-percent.")

    outputs = run_processing(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        max_slope_percent=args.max_slope_percent,
        conditional_max_slope_percent=args.conditional_max_slope_percent,
        force=args.force,
        points_csv=args.points_csv,
        tolerance=args.tolerance,
        resume=not args.no_resume,
        flush_every=args.flush_every,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    print("Procesamiento de pendiente IGAC finalizado.")
    print(f"Umbral viable: <= {args.max_slope_percent}%")
    print(f"Umbral condicional: <= {args.conditional_max_slope_percent}%")
    print(f"Salidas: {args.output_dir}")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
