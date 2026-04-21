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
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "igac_usos_pot"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "usos_suelo_pot"
DEFAULT_POINTS_CSV = (
    PROJECT_ROOT / "data" / "clean" / "base_municipios" / "municipios_distritos_colombia.csv"
)

SERVICE_URL = (
    "https://mapas.igac.gov.co/server/rest/services/"
    "ordenamientoterritorial/zonificacionusossegunpot/MapServer"
)
URBAN_LAYER_ID = 0
RURAL_LAYER_ID = 1


def normalize_text(value: Any) -> str:
    """Normaliza texto para clasificar etiquetas POT."""

    text = "" if value is None else str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_column_name(column_name: Any) -> str:
    normalized = normalize_text(column_name)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return normalized or "columna_sin_nombre"


def request_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = 120,
    retries: int = 3,
) -> dict[str, Any]:
    """Consulta JSON con reintentos para servicios ArcGIS."""

    last_error: Exception | None = None
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json,*/*"}
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


def fetch_metadata(raw_dir: Path, force: bool = False) -> dict[str, Any]:
    """Descarga metadatos generales del MapServer."""

    raw_path = raw_dir / "mapserver_metadata.json"
    if raw_path.exists() and not force:
        return read_json(raw_path)

    metadata = request_json(SERVICE_URL, params={"f": "pjson"}, timeout=180)
    write_json(raw_path, metadata)
    return metadata


def fetch_legend(raw_dir: Path, force: bool = False) -> dict[str, Any]:
    """Descarga leyenda oficial de usos POT."""

    raw_path = raw_dir / "legend.json"
    if raw_path.exists() and not force:
        return read_json(raw_path)

    legend = request_json(f"{SERVICE_URL}/legend", params={"f": "pjson"}, timeout=180)
    write_json(raw_path, legend)
    return legend


def legend_to_table(legend: dict[str, Any]) -> pd.DataFrame:
    """Convierte leyenda ArcGIS a tabla plana."""

    rows: list[dict[str, Any]] = []
    for layer in legend.get("layers", []):
        layer_id = layer.get("layerId")
        layer_name = layer.get("layerName")
        for item in layer.get("legend", []):
            label = item.get("label")
            rows.append(
                {
                    "layer_id": layer_id,
                    "layer_name": layer_name,
                    "tipo_capa": "rural" if layer_id == RURAL_LAYER_ID else "urbana",
                    "uso_pot": label,
                    "uso_pot_normalizado": normalize_text(label),
                }
            )
    return pd.DataFrame(rows)


def classify_pot_use(label: Any, layer_id: int | None = None) -> dict[str, Any]:
    """Clasifica una etiqueta POT para el criterio U_i preliminar."""

    text = normalize_text(label)
    if not text:
        return {
            "categoria_aptitud_pot": "sin_datos",
            "u_i_uso_suelo_proxy": pd.NA,
            "apto_doble_uso_pastoreo": pd.NA,
            "restriccion_territorial_proxy": 0,
            "criterio_clasificacion": "Etiqueta vacia o no disponible.",
        }

    hard_protection_terms = [
        "proteccion",
        "reserva",
        "recurso natural",
        "recursos naturales",
        "humedal",
        "ronda",
        "hidric",
        "bosque protector",
        "amenaza",
        "riesgo",
        "recuperacion",
        "especial significancia",
        "acantilado",
        "inderena",
    ]
    conditional_terms = ["conservacion", "ambiental", "restringido", "restriccion"]
    pasture_terms = [
        "pecuaria",
        "pecuario",
        "ganader",
        "pasto",
        "agropastoril",
        "silvopastoril",
        "agrosilvopastoril",
    ]
    agro_terms = [
        "agricultura",
        "agricola",
        "agropecu",
        "agroindustrial",
        "agroforestal",
        "agroforesteria",
        "cultivo",
    ]
    incompatible_terms = [
        "minera",
        "minero",
        "industrial",
        "residencial",
        "comercial",
        "servicios",
        "institucional",
        "dotacional",
        "mixto",
        "urbano",
    ]
    forest_terms = ["forestal", "bosque"]

    has_pasture = any(term in text for term in pasture_terms)
    has_agro = any(term in text for term in agro_terms)
    has_hard_protection = any(term in text for term in hard_protection_terms)
    has_conditional = any(term in text for term in conditional_terms)

    if has_hard_protection and not (has_pasture or has_agro):
        return {
            "categoria_aptitud_pot": "restriccion_ambiental",
            "u_i_uso_suelo_proxy": 0.0,
            "apto_doble_uso_pastoreo": 0.0,
            "restriccion_territorial_proxy": 1,
            "criterio_clasificacion": "Uso POT asociado a proteccion, conservacion, reserva, amenaza o recurso natural.",
        }
    if (has_hard_protection or has_conditional) and (has_pasture or has_agro):
        return {
            "categoria_aptitud_pot": "condicional_agroambiental",
            "u_i_uso_suelo_proxy": 0.5,
            "apto_doble_uso_pastoreo": 0.5 if has_pasture else 0.2,
            "restriccion_territorial_proxy": 0,
            "criterio_clasificacion": "Uso mixto agro/pastoreo con condicion ambiental o restriccion; requiere revision normativa.",
        }
    if has_hard_protection:
        return {
            "categoria_aptitud_pot": "restriccion_ambiental",
            "u_i_uso_suelo_proxy": 0.0,
            "apto_doble_uso_pastoreo": 0.0,
            "restriccion_territorial_proxy": 1,
            "criterio_clasificacion": "Uso POT asociado a proteccion, reserva, amenaza o recurso natural.",
        }
    if has_pasture:
        return {
            "categoria_aptitud_pot": "compatible_pastoreo",
            "u_i_uso_suelo_proxy": 1.0,
            "apto_doble_uso_pastoreo": 1.0,
            "restriccion_territorial_proxy": 0,
            "criterio_clasificacion": "Uso POT menciona actividad pecuaria, ganadera, pastos o sistemas pastoriles.",
        }
    if has_agro:
        return {
            "categoria_aptitud_pot": "compatible_agropecuario",
            "u_i_uso_suelo_proxy": 0.8,
            "apto_doble_uso_pastoreo": 0.5,
            "restriccion_territorial_proxy": 0,
            "criterio_clasificacion": "Uso POT agropecuario/agricola; compatible preliminar, requiere revision predial.",
        }
    if any(term in text for term in forest_terms):
        return {
            "categoria_aptitud_pot": "condicional_forestal",
            "u_i_uso_suelo_proxy": 0.4,
            "apto_doble_uso_pastoreo": 0.0,
            "restriccion_territorial_proxy": 0,
            "criterio_clasificacion": "Uso forestal; puede ser productivo o protector. Requiere revision antes de usarlo como apto.",
        }
    if any(term in text for term in incompatible_terms) or layer_id == URBAN_LAYER_ID:
        return {
            "categoria_aptitud_pot": "no_preferente",
            "u_i_uso_suelo_proxy": 0.2,
            "apto_doble_uso_pastoreo": 0.0,
            "restriccion_territorial_proxy": 0,
            "criterio_clasificacion": "Uso urbano, residencial, industrial, minero o dotacional; no preferente para granja solar rural.",
        }
    return {
        "categoria_aptitud_pot": "revision_manual",
        "u_i_uso_suelo_proxy": pd.NA,
        "apto_doble_uso_pastoreo": pd.NA,
        "restriccion_territorial_proxy": 0,
        "criterio_clasificacion": "No se infiere aptitud con seguridad; requiere revision manual.",
    }


def classify_legend(legend_df: pd.DataFrame) -> pd.DataFrame:
    """Aplica clasificacion de aptitud a la leyenda completa."""

    rows: list[dict[str, Any]] = []
    for _, row in legend_df.iterrows():
        classification = classify_pot_use(row["uso_pot"], layer_id=int(row["layer_id"]))
        rows.append({**row.to_dict(), **classification})
    return pd.DataFrame(rows)


def detect_coordinate_columns(df: pd.DataFrame) -> tuple[str, str]:
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
        raise ValueError("No se detectaron columnas lon/lat para muestreo POT.")
    return lon_candidates[0], lat_candidates[0]


def identify_pot_batch(
    coordinates: list[tuple[float, float]],
    tolerance: int = 3,
) -> list[dict[str, Any]]:
    """Consulta usos POT en varios puntos; devuelve resultados crudos por poligono."""

    if not coordinates:
        return []

    xs = [lon for lon, _ in coordinates]
    ys = [lat for _, lat in coordinates]
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
        "layers": f"all:{URBAN_LAYER_ID},{RURAL_LAYER_ID}",
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
    data = request_json(f"{SERVICE_URL}/identify", params=params, timeout=180, retries=2)
    return data.get("results") or []


def result_to_record(result: dict[str, Any]) -> dict[str, Any]:
    """Convierte un resultado identify a registro tabular."""

    attrs = result.get("attributes") or {}
    layer_id = result.get("layerId")
    uso = attrs.get("URural") if layer_id == RURAL_LAYER_ID else attrs.get("UUrbano")
    uso = uso or attrs.get("URural") or attrs.get("UUrbano")
    classification = classify_pot_use(uso, layer_id=layer_id)
    return {
        "codigo_dane": str(attrs.get("MpCodigo") or result.get("value") or "").zfill(5),
        "municipio_pot": attrs.get("MpNombre"),
        "layer_id_pot": layer_id,
        "layer_name_pot": result.get("layerName"),
        "tipo_capa_pot": "rural" if layer_id == RURAL_LAYER_ID else "urbana",
        "uso_pot": uso,
        "tipo_uso_pot": attrs.get("URTipoUsoRural") or attrs.get("UUTipoUsoUrbano"),
        "observacion_pot": attrs.get("Observacio"),
        **classification,
    }


def sample_points(
    points_csv: Path,
    output_path: Path,
    batch_size: int = 5,
    flush_every: int = 25,
    resume: bool = True,
    limit: int | None = None,
    tolerance: int = 3,
) -> pd.DataFrame:
    """Muestrea usos POT en puntos municipales con checkpoint."""

    points = pd.read_csv(points_csv, dtype={"codigo_dane": "string"})
    if points.empty:
        return points
    lon_col, lat_col = detect_coordinate_columns(points)
    if "codigo_dane" not in points.columns:
        points["codigo_dane"] = points.index.astype(str)
    points["codigo_dane"] = points["codigo_dane"].astype("string").str.zfill(5)
    if limit is not None:
        points = points.head(limit)

    existing = pd.DataFrame()
    completed: set[str] = set()
    if resume and output_path.exists():
        existing = pd.read_csv(output_path, dtype={"codigo_dane": "string"})
        if "codigo_dane" in existing.columns:
            existing["codigo_dane"] = existing["codigo_dane"].astype("string").str.zfill(5)
            completed = set(existing["codigo_dane"].astype(str))

    pending = points[~points["codigo_dane"].astype(str).isin(completed)].copy()
    if completed:
        print(f"Resume POT: {len(points) - len(pending)}/{len(points)} puntos ya procesados.")

    rows: list[dict[str, Any]] = []

    def flush() -> None:
        if not rows:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        combined["codigo_dane"] = combined["codigo_dane"].astype("string").str.zfill(5)
        combined = combined.drop_duplicates(subset=["codigo_dane"], keep="last")
        combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    for start in range(0, len(pending), batch_size):
        batch = pending.iloc[start : start + batch_size].copy()
        coordinates: list[tuple[float, float]] = []
        valid_codes: set[str] = set()
        for _, row in batch.iterrows():
            lon = pd.to_numeric(row[lon_col], errors="coerce")
            lat = pd.to_numeric(row[lat_col], errors="coerce")
            if pd.notna(lon) and pd.notna(lat):
                coordinates.append((float(lon), float(lat)))
                valid_codes.add(str(row["codigo_dane"]).zfill(5))

        records_by_code: dict[str, dict[str, Any]] = {}
        if coordinates:
            try:
                raw_results = identify_pot_batch(coordinates, tolerance=tolerance)
                for raw in raw_results:
                    record = result_to_record(raw)
                    code = str(record.get("codigo_dane", "")).zfill(5)
                    # Se prioriza capa rural para U_i; si no hay rural, se conserva urbana.
                    if code not in records_by_code or record.get("layer_id_pot") == RURAL_LAYER_ID:
                        records_by_code[code] = record
            except Exception as error:  # noqa: BLE001
                print(f"Advertencia POT lote {start}: {error}")

        for _, row in batch.iterrows():
            code = str(row["codigo_dane"]).zfill(5)
            base = row.to_dict()
            base["codigo_dane"] = code
            record = records_by_code.get(code)
            if record:
                rows.append({**base, **record, "pot_identify_ok": True, "pot_mensaje": ""})
            else:
                classification = classify_pot_use(None)
                rows.append(
                    {
                        **base,
                        "layer_id_pot": pd.NA,
                        "layer_name_pot": pd.NA,
                        "tipo_capa_pot": pd.NA,
                        "uso_pot": pd.NA,
                        "tipo_uso_pot": pd.NA,
                        "observacion_pot": pd.NA,
                        **classification,
                        "pot_identify_ok": False,
                        "pot_mensaje": "Sin resultado POT para el punto o lote no consultado.",
                    }
                )

        processed = min(start + len(batch), len(pending))
        if len(rows) % flush_every == 0 or processed == len(pending):
            flush()
            print(f"Usos POT: {len(existing) + len(rows)}/{len(points)} puntos guardados.")

    if output_path.exists():
        return pd.read_csv(output_path, dtype={"codigo_dane": "string"})
    return pd.DataFrame(rows)


def metadata_to_table(metadata: dict[str, Any]) -> pd.DataFrame:
    layers = metadata.get("layers") or []
    return pd.DataFrame(
        [
            {"clave": "fuente", "valor": "Instituto Geografico Agustin Codazzi - IGAC"},
            {"clave": "url_servicio", "valor": SERVICE_URL},
            {"clave": "map_name", "valor": metadata.get("mapName")},
            {"clave": "descripcion", "valor": re.sub(r"<[^>]+>", " ", html.unescape(str(metadata.get("description", ""))))},
            {"clave": "copyright", "valor": metadata.get("copyrightText")},
            {"clave": "capabilities", "valor": metadata.get("capabilities")},
            {"clave": "max_record_count", "valor": metadata.get("maxRecordCount")},
            {"clave": "layers", "valor": json.dumps(layers, ensure_ascii=False)},
        ]
    )


def write_observations(
    output_path: Path,
    classified_legend: pd.DataFrame,
    sampled: pd.DataFrame | None,
) -> None:
    """Documenta alcance y limitaciones para informe."""

    lines = [
        "Usos del suelo segun POT - IGAC",
        "================================",
        "",
        f"Fuente: {SERVICE_URL}",
        "Capas: Zonificacion de usos urbanos y Zonificacion de usos rurales.",
        "Uso metodologico: proxy territorial U_i y criterio cualitativo de doble uso con pastoreo.",
        "",
        "Interpretacion:",
        "- La capa POT representa zonificacion normativa municipal, no cobertura real observada del suelo.",
        "- Para granja solar rural, se priorizan etiquetas agropecuarias, pecuarias, ganaderas, pastos y sistemas pastoriles.",
        "- Etiquetas de proteccion estricta, reserva, amenaza o recursos naturales se tratan como restriccion proxy.",
        "- Etiquetas mixtas agro/pastoreo con condicion ambiental se tratan como condicionales, no como aptas automaticas.",
        "- Etiquetas urbanas, industriales, mineras o residenciales se tratan como no preferentes.",
        "",
        "Limitacion critica:",
        "- U_i oficial deberia calcularse como area compatible / area total municipal mediante interseccion poligonal.",
        "- En esta version, si se muestrean puntos, U_i es un proxy puntual y no reemplaza el calculo por area.",
        "- El servicio presento lentitud/bloqueos para consultas masivas de poligonos; por eso se documenta la decision.",
        "",
        "Conteo de clases en leyenda clasificada:",
    ]
    for category, count in classified_legend["categoria_aptitud_pot"].value_counts(dropna=False).items():
        lines.append(f"- {category}: {count}")

    if sampled is not None and not sampled.empty:
        lines.extend(["", "Conteo de puntos muestreados:"])
        for category, count in sampled["categoria_aptitud_pot"].value_counts(dropna=False).items():
            lines.append(f"- {category}: {count}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_pot_processing(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    points_csv: Path | None = None,
    force: bool = False,
    batch_size: int = 5,
    flush_every: int = 25,
    resume: bool = True,
    limit: int | None = None,
) -> dict[str, Path]:
    """Ejecuta procesamiento de fuente POT IGAC."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = fetch_metadata(raw_dir, force=force)
    legend = fetch_legend(raw_dir, force=force)
    legend_df = legend_to_table(legend)
    classified = classify_legend(legend_df)

    metadata_path = output_dir / "usos_pot_metadata.csv"
    legend_path = output_dir / "usos_pot_leyenda.csv"
    classified_path = output_dir / "usos_pot_clasificacion_leyenda.csv"
    observations_path = output_dir / "usos_pot_observaciones.txt"
    points_path = output_dir / "usos_pot_puntos_extraidos.csv"

    metadata_to_table(metadata).to_csv(metadata_path, index=False, encoding="utf-8-sig")
    legend_df.to_csv(legend_path, index=False, encoding="utf-8-sig")
    classified.to_csv(classified_path, index=False, encoding="utf-8-sig")

    sampled: pd.DataFrame | None = None
    outputs = {
        "metadata": metadata_path,
        "legend": legend_path,
        "classified_legend": classified_path,
        "observations": observations_path,
    }
    if points_csv is not None:
        if force and points_path.exists():
            points_path.unlink()
        sampled = sample_points(
            points_csv=points_csv,
            output_path=points_path,
            batch_size=batch_size,
            flush_every=flush_every,
            resume=resume,
            limit=limit,
        )
        sampled.to_csv(points_path, index=False, encoding="utf-8-sig")
        outputs["points"] = points_path

    write_observations(observations_path, classified, sampled)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Procesa zonificacion de usos segun POT IGAC para U_i preliminar."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--points-csv", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_pot_processing(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        points_csv=args.points_csv,
        force=args.force,
        batch_size=args.batch_size,
        flush_every=args.flush_every,
        resume=not args.no_resume,
        limit=args.limit,
    )
    print("Procesamiento de usos POT IGAC finalizado.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print("Advertencia: U_i por POT es proxy; el calculo robusto requiere area compatible por municipio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
