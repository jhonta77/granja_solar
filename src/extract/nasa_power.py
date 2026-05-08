from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POINTS_CSV = PROJECT_ROOT / "data" / "clean" / "pvout" / "pvout_top_celdas_bbox_colombia.csv"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nasa_power"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "nasa_power"

NASA_POWER_DAILY_POINT_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_DOCS_URL = "https://power.larc.nasa.gov/docs/services/api/temporal/daily/"

DEFAULT_PARAMETERS = ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN"]
DEFAULT_COMMUNITY = "RE"


def default_year_range() -> tuple[str, str]:
    """Usa el ultimo anio calendario completo para evitar anios parciales."""

    year = date.today().year - 1
    return f"{year}0101", f"{year}1231"


def normalize_text(value: Any) -> str:
    """Normaliza texto para deteccion robusta de columnas."""

    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.lower().strip()


def normalize_column_name(value: Any) -> str:
    """Convierte nombres de columnas a snake_case."""

    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "columna_sin_nombre"


def detect_coordinate_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Detecta columnas lon/lat en un CSV de puntos."""

    normalized = {column: normalize_column_name(column) for column in df.columns}
    lon_candidates = [
        column
        for column, name in normalized.items()
        if name in {"lon", "longitud", "longitude", "x"} or "longitud" in name
    ]
    lat_candidates = [
        column
        for column, name in normalized.items()
        if name in {"lat", "latitud", "latitude", "y"} or "latitud" in name
    ]
    if not lon_candidates or not lat_candidates:
        raise ValueError("No se detectaron columnas de coordenadas. Use lon/lat o longitud/latitud.")
    return lon_candidates[0], lat_candidates[0]


def point_key(lon: float, lat: float, parameters: list[str], start: str, end: str) -> str:
    """Crea clave estable para cache local de una consulta NASA POWER."""

    raw = f"{lon:.6f}_{lat:.6f}_{','.join(parameters)}_{start}_{end}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"lon_{lon:.4f}_lat_{lat:.4f}_{digest}".replace("-", "m").replace(".", "p")


def request_power_json(
    lon: float,
    lat: float,
    parameters: list[str],
    start: str,
    end: str,
    community: str,
    timeout: int = 60,
    retries: int = 3,
) -> dict[str, Any]:
    """Consulta NASA POWER Daily Point API."""

    params = {
        "parameters": ",".join(parameters),
        "community": community,
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": "LST",
    }
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                NASA_POWER_DAILY_POINT_URL,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"No fue posible consultar NASA POWER para lon={lon}, lat={lat}: {last_error}")


def write_json(path: Path, data: Any) -> None:
    """Guarda JSON crudo."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Lee JSON crudo."""

    return json.loads(path.read_text(encoding="utf-8"))


def fetch_or_load_point(
    lon: float,
    lat: float,
    parameters: list[str],
    start: str,
    end: str,
    community: str,
    raw_dir: Path,
    force: bool,
) -> dict[str, Any]:
    """Obtiene una consulta NASA POWER usando cache local."""

    key = point_key(lon, lat, parameters, start, end)
    path = raw_dir / f"{key}.json"
    if path.exists() and not force:
        return read_json(path)

    data = request_power_json(
        lon=lon,
        lat=lat,
        parameters=parameters,
        start=start,
        end=end,
        community=community,
    )
    write_json(path, data)
    return data


def power_json_to_daily_df(
    data: dict[str, Any],
    point_metadata: dict[str, Any],
    parameters: list[str],
) -> pd.DataFrame:
    """Convierte respuesta NASA POWER a DataFrame diario ancho."""

    parameter_data = ((data.get("properties") or {}).get("parameter") or {})
    dates: set[str] = set()
    for parameter in parameters:
        values = parameter_data.get(parameter) or {}
        dates.update(values.keys())

    rows: list[dict[str, Any]] = []
    for date_key in sorted(dates):
        row = {**point_metadata}
        row["fecha"] = pd.to_datetime(date_key, format="%Y%m%d", errors="coerce")
        for parameter in parameters:
            value = (parameter_data.get(parameter) or {}).get(date_key)
            numeric_value = pd.to_numeric(value, errors="coerce")
            if pd.notna(numeric_value) and float(numeric_value) <= -900:
                numeric_value = pd.NA
            row[normalize_column_name(parameter)] = numeric_value
        rows.append(row)

    return pd.DataFrame(rows)


_RADIATION_COLUMNS = {"allsky_sfc_sw_dwn", "clrsky_sfc_sw_dwn"}
_METADATA_COLUMNS = {"punto_id", "codigo_dane", "municipio", "departamento", "lon", "lat", "fecha"}

_PARAM_AGGREGATIONS: dict[str, list[str]] = {
    "allsky_sfc_sw_dwn": ["dias_validos", "media_kwh_m2_day", "mediana_kwh_m2_day", "p95_kwh_m2_day", "suma_kwh_m2_year"],
    "clrsky_sfc_sw_dwn": ["dias_validos", "media_kwh_m2_day", "mediana_kwh_m2_day", "p95_kwh_m2_day", "suma_kwh_m2_year"],
    "t2m":              ["dias_validos", "media", "min", "max"],
    "cloud_amt":        ["dias_validos", "media", "mediana"],
    "ws10m":            ["dias_validos", "media", "max"],
    "prectotcorr":      ["dias_validos", "media_mm_day", "suma_mm_year"],
}


def _aggregate_column(values: pd.Series, column: str) -> dict[str, Any]:
    """Aplica las agregaciones correspondientes a cada parametro NASA POWER."""

    aggs = _PARAM_AGGREGATIONS.get(column, ["dias_validos", "media"])
    row: dict[str, Any] = {}
    for agg in aggs:
        if agg == "dias_validos":
            row[f"{column}_dias_validos"] = int(values.shape[0])
        elif agg.startswith("media"):
            suffix = agg[len("media"):]
            row[f"{column}_media{suffix}"] = values.mean()
        elif agg == "mediana":
            row[f"{column}_mediana"] = values.median()
        elif agg.startswith("mediana"):
            suffix = agg[len("mediana"):]
            row[f"{column}_mediana{suffix}"] = values.median()
        elif agg.startswith("p"):
            try:
                pct = int(agg[1:]) / 100
                suffix = agg[len("p"):] if not agg[1:].isdigit() else ""
                label = f"{column}_{agg}"
                if suffix:
                    label = f"{column}_p{agg[1:]}"
                row[label] = values.quantile(pct)
            except ValueError:
                pass
        elif agg.startswith("suma"):
            suffix = agg[len("suma"):]
            row[f"{column}_suma{suffix}"] = values.sum()
        elif agg == "min":
            row[f"{column}_min"] = values.min()
        elif agg == "max":
            row[f"{column}_max"] = values.max()
    return row


def summarize_points(daily_df: pd.DataFrame, group_key: str = "punto_id") -> pd.DataFrame:
    """Resume variables NASA POWER diarias por punto o por municipio.

    Acepta cualquier parametro presente en el DataFrame: detecta automaticamente
    las columnas de valor (numericas no-metadata) y aplica las agregaciones
    definidas en _PARAM_AGGREGATIONS. Los parametros sin entrada especifica usan
    media como unica agregacion.
    """

    if daily_df.empty:
        return pd.DataFrame()

    id_columns = [
        column
        for column in [group_key, "municipio", "departamento", "codigo_dane", "lon", "lat"]
        if column in daily_df.columns
    ]
    if group_key not in id_columns:
        id_columns.insert(0, group_key)

    value_columns = [
        column
        for column in daily_df.columns
        if column not in _METADATA_COLUMNS and pd.api.types.is_numeric_dtype(daily_df[column])
    ]

    summaries: list[pd.DataFrame] = []
    for point_id, group in daily_df.groupby(group_key, dropna=False):
        row: dict[str, Any] = {group_key: point_id, "dias_validos": len(group)}
        for column in id_columns:
            if column in group.columns and column != group_key:
                row[column] = group[column].iloc[0]
        for column in value_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row.update(_aggregate_column(values, column))
        summaries.append(pd.DataFrame([row]))

    result = pd.concat(summaries, ignore_index=True)
    sort_col = "allsky_sfc_sw_dwn_media_kwh_m2_day"
    if sort_col in result.columns:
        result = result.sort_values(sort_col, ascending=False)
    return result


def build_source_table(parameters: list[str], start: str, end: str, community: str) -> pd.DataFrame:
    """Documenta fuente y parametros consultados."""

    return pd.DataFrame(
        [
            {
                "fuente": "NASA POWER Daily Point API",
                "url_base": NASA_POWER_DAILY_POINT_URL,
                "documentacion": NASA_POWER_DOCS_URL,
                "community": community,
                "parameters": ",".join(parameters),
                "start": start,
                "end": end,
                "time_standard": "LST",
                "nota": (
                    "ALLSKY_SFC_SW_DWN representa irradiancia solar global diaria "
                    "en superficie bajo todo cielo, util como proxy de recurso solar."
                ),
            }
        ]
    )


def build_observations(points_csv: Path, parameters: list[str], start: str, end: str, rows: int) -> str:
    """Genera observaciones metodologicas."""

    return "\n".join(
        [
            "Observaciones NASA POWER",
            "========================",
            "Fuente agregada: NASA POWER Daily Point API.",
            f"Documentacion: {NASA_POWER_DOCS_URL}",
            f"Archivo de puntos usado: {points_csv}",
            f"Rango consultado: {start} a {end}.",
            f"Parametros: {', '.join(parameters)}.",
            f"Registros diarios exportados: {rows}.",
            "",
            "Uso metodologico:",
            "- Esta fuente aporta radiacion/irradiancia solar por punto.",
            "- Sirve para contrastar o complementar PVOUT de Global Solar Atlas.",
            "- Para ranking municipal serio se debe consultar con centroides municipales oficiales o poligonos agregados; las celdas PVOUT no equivalen automaticamente a municipios.",
            "- No se generan graficos, no se imputan datos y no se integra aun al score final.",
            "- Valores de relleno <= -900 se convierten a nulo antes de resumir.",
            "",
            "Limitaciones:",
            "- NASA POWER tiene resolucion global relativamente gruesa; no reemplaza medicion local ni modelacion solar de detalle.",
            "- Si se usan muchos puntos cercanos, pueden repetir la misma celda de la grilla NASA POWER.",
            "- La radiacion por si sola no define viabilidad: debe cruzarse con pendiente, red electrica, restricciones territoriales, costos y demanda.",
        ]
    ) + "\n"


def run_nasa_power(
    points_csv: Path,
    raw_dir: Path,
    output_dir: Path,
    parameters: list[str],
    start: str,
    end: str,
    community: str,
    limit: int | None,
    force: bool,
    sleep_seconds: float,
) -> dict[str, Path]:
    """Ejecuta extraccion NASA POWER para puntos."""

    if not points_csv.exists():
        raise FileNotFoundError(f"No existe el CSV de puntos: {points_csv}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = pd.read_csv(points_csv)
    if points.empty:
        raise ValueError("El CSV de puntos esta vacio.")

    lon_col, lat_col = detect_coordinate_columns(points)
    if limit is not None:
        points = points.head(limit).copy()

    daily_frames: list[pd.DataFrame] = []
    for index, row in points.iterrows():
        lon = pd.to_numeric(row[lon_col], errors="coerce")
        lat = pd.to_numeric(row[lat_col], errors="coerce")
        if pd.isna(lon) or pd.isna(lat):
            continue

        point_metadata = {
            "punto_id": row.get("punto_id", row.get("id", index)),
            "lon": float(lon),
            "lat": float(lat),
        }
        for optional in ["municipio", "departamento", "zona", "pvout_kwh_kwp_day"]:
            if optional in points.columns:
                point_metadata[optional] = row.get(optional)

        data = fetch_or_load_point(
            lon=float(lon),
            lat=float(lat),
            parameters=parameters,
            start=start,
            end=end,
            community=community,
            raw_dir=raw_dir,
            force=force,
        )
        daily = power_json_to_daily_df(data, point_metadata, parameters)
        daily_frames.append(daily)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not daily_frames:
        raise ValueError("No se pudo consultar ningun punto valido.")

    daily_df = pd.concat(daily_frames, ignore_index=True)
    summary_df = summarize_points(daily_df)
    source_df = build_source_table(parameters, start=start, end=end, community=community)

    daily_path = output_dir / "nasa_power_puntos_diario.csv"
    summary_path = output_dir / "nasa_power_resumen_puntos.csv"
    source_path = output_dir / "nasa_power_fuente.csv"
    observations_path = output_dir / "nasa_power_observaciones.txt"

    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    source_df.to_csv(source_path, index=False, encoding="utf-8-sig")
    observations_path.write_text(
        build_observations(points_csv, parameters, start=start, end=end, rows=len(daily_df)),
        encoding="utf-8-sig",
    )

    return {
        "daily": daily_path,
        "summary": summary_path,
        "source": source_path,
        "observations": observations_path,
    }


def parse_args() -> argparse.Namespace:
    """Argumentos CLI."""

    start_default, end_default = default_year_range()
    parser = argparse.ArgumentParser(
        description="Extrae radiacion solar diaria NASA POWER para puntos candidatos."
    )
    parser.add_argument("--points-csv", type=Path, default=DEFAULT_POINTS_CSV)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", default=start_default, help="Fecha inicial YYYYMMDD.")
    parser.add_argument("--end", default=end_default, help="Fecha final YYYYMMDD.")
    parser.add_argument(
        "--parameters",
        default=",".join(DEFAULT_PARAMETERS),
        help="Parametros NASA POWER separados por coma.",
    )
    parser.add_argument("--community", default=DEFAULT_COMMUNITY)
    parser.add_argument("--limit", type=int, default=10, help="Limite de puntos a consultar.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Punto de entrada."""

    args = parse_args()
    parameters = [parameter.strip() for parameter in args.parameters.split(",") if parameter.strip()]
    try:
        outputs = run_nasa_power(
            points_csv=args.points_csv.resolve(),
            raw_dir=args.raw_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            parameters=parameters,
            start=args.start,
            end=args.end,
            community=args.community,
            limit=args.limit,
            force=args.force,
            sleep_seconds=args.sleep_seconds,
        )
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {error}")
        return 1

    print("Extraccion NASA POWER finalizada.")
    print(f"Fuente: {NASA_POWER_DAILY_POINT_URL}")
    for label, path in outputs.items():
        print(f"- {label}: {path}")
    print("No se generaron diagramas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
