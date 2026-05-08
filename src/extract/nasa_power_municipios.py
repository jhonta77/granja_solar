"""Extrae variables climaticas NASA POWER para todos los municipios de Colombia.

Diferencia con nasa_power.py
-----------------------------
nasa_power.py trabaja con celdas PVOUT (punto_id numerico) para contrastar
irradiancia solar. Este modulo usa los centroides municipales oficiales IGAC
(codigo_dane como llave) y solicita ademas parametros climaticos que alimentan
el score_fisico y score_economico del modelo multidimensional:

    T2M        : Temperatura media a 2 metros (°C)
    CLOUD_AMT  : Fraccion de cobertura nubosa (0-1)
    WS10M      : Velocidad del viento a 10 metros (m/s)
    PRECTOTCORR: Precipitacion total corregida (mm/dia)
    ALLSKY_SFC_SW_DWN: Irradiancia global bajo todo cielo (kWh/m2/dia)
    CLRSKY_SFC_SW_DWN: Irradiancia bajo cielo despejado (kWh/m2/dia)

Salida principal
----------------
    data/clean/nasa_power/nasa_power_resumen_municipios.csv

Con columna clave: codigo_dane (CHAR 5, zfill).

Uso
---
    python -m src.extract.nasa_power_municipios
    python -m src.extract.nasa_power_municipios --limit 50 --force
    python -m src.extract.nasa_power_municipios --start 20230101 --end 20231231
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .nasa_power import (
    DEFAULT_COMMUNITY,
    DEFAULT_RAW_DIR,
    DEFAULT_OUTPUT_DIR,
    fetch_or_load_point,
    power_json_to_daily_df,
    summarize_points,
    default_year_range,
    detect_coordinate_columns,
    normalize_column_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUNICIPALITIES_CSV = (
    PROJECT_ROOT / "data" / "clean" / "base_municipios" / "municipios_distritos_colombia.csv"
)
DEFAULT_MUNICIPAL_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nasa_power_municipios"
DEFAULT_MUNICIPAL_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "nasa_power"

MUNICIPAL_PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "T2M",
    "CLOUD_AMT",
    "WS10M",
    "PRECTOTCORR",
]


def _load_municipalities(municipalities_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(municipalities_csv, dtype={"codigo_dane": "string"})
    df["codigo_dane"] = df["codigo_dane"].astype("string").str.strip().str.zfill(5)
    required = {"codigo_dane", "municipio", "departamento"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en municipios: {sorted(missing)}")
    lon_col, lat_col = detect_coordinate_columns(df)
    df = df.rename(columns={lon_col: "lon", lat_col: "lat"})
    return df.dropna(subset=["lon", "lat"]).copy()


def run_nasa_power_municipios(
    municipalities_csv: Path = DEFAULT_MUNICIPALITIES_CSV,
    raw_dir: Path = DEFAULT_MUNICIPAL_RAW_DIR,
    output_dir: Path = DEFAULT_MUNICIPAL_OUTPUT_DIR,
    parameters: list[str] = MUNICIPAL_PARAMETERS,
    start: str | None = None,
    end: str | None = None,
    community: str = DEFAULT_COMMUNITY,
    limit: int | None = None,
    force: bool = False,
    sleep_seconds: float = 0.3,
) -> dict[str, Path]:
    """Extrae variables climaticas NASA POWER para centroides municipales.

    Usa codigo_dane como llave de agrupacion en lugar de punto_id numerico,
    para integrarse directamente al pipeline de scoring multidimensional.
    """

    if start is None or end is None:
        start, end = default_year_range()

    if not municipalities_csv.exists():
        raise FileNotFoundError(f"No existe el CSV de municipios: {municipalities_csv}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    municipalities = _load_municipalities(municipalities_csv)
    if limit is not None:
        municipalities = municipalities.head(limit).copy()

    daily_frames: list[pd.DataFrame] = []
    total = len(municipalities)
    for idx, (_, row) in enumerate(municipalities.iterrows(), start=1):
        lon = float(row["lon"])
        lat = float(row["lat"])
        codigo_dane = str(row["codigo_dane"])

        if idx % 50 == 0 or idx == total:
            print(f"  NASA POWER municipios: {idx}/{total} ({codigo_dane} - {row['municipio']})")

        data = fetch_or_load_point(
            lon=lon,
            lat=lat,
            parameters=parameters,
            start=start,
            end=end,
            community=community,
            raw_dir=raw_dir,
            force=force,
        )
        point_metadata: dict[str, Any] = {
            "codigo_dane": codigo_dane,
            "municipio": row["municipio"],
            "departamento": row["departamento"],
            "lon": lon,
            "lat": lat,
        }
        daily = power_json_to_daily_df(data, point_metadata, parameters)
        daily["punto_id"] = codigo_dane
        daily_frames.append(daily)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not daily_frames:
        raise ValueError("No se pudo consultar ningun municipio valido.")

    daily_df = pd.concat(daily_frames, ignore_index=True)

    summary_df = summarize_points(daily_df, group_key="codigo_dane")
    summary_df["codigo_dane"] = summary_df["codigo_dane"].astype("string").str.zfill(5)

    summary_path = output_dir / "nasa_power_resumen_municipios.csv"
    observations_path = output_dir / "nasa_power_municipios_observaciones.txt"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    _write_observations(observations_path, municipalities_csv, parameters, start, end, len(summary_df))

    print(f"NASA POWER municipios completado: {len(summary_df)} municipios.")
    return {
        "summary": summary_path,
        "observations": observations_path,
    }


def _write_observations(
    path: Path,
    municipalities_csv: Path,
    parameters: list[str],
    start: str,
    end: str,
    n_municipios: int,
) -> None:
    lines = [
        "NASA POWER - Resumen climatico municipal",
        "=========================================",
        "",
        "Fuente: NASA POWER Daily Point API (https://power.larc.nasa.gov/)",
        f"Archivo de municipios: {municipalities_csv}",
        f"Rango consultado: {start} a {end}",
        f"Parametros consultados: {', '.join(parameters)}",
        f"Municipios procesados: {n_municipios}",
        "",
        "Columnas de salida principales:",
        "  allsky_sfc_sw_dwn_media_kwh_m2_day : Irradiancia global media diaria",
        "  t2m_media                          : Temperatura media anual (°C)",
        "  t2m_min / t2m_max                  : Temperatura minima/maxima anual",
        "  cloud_amt_media                    : Fraccion nubosa media anual (0-1)",
        "  ws10m_media / ws10m_max            : Viento medio y maximo (m/s)",
        "  prectotcorr_media_mm_day           : Precipitacion media diaria (mm/dia)",
        "  prectotcorr_suma_mm_year           : Precipitacion total anual (mm)",
        "",
        "Uso en scoring multidimensional:",
        "  score_fisico : usa allsky, t2m, cloud_amt, ws10m, prectotcorr",
        "  score_economico: usa prectotcorr como proxy disponibilidad agua",
        "",
        "Limitaciones:",
        "  - NASA POWER tiene resolucion espacial de ~0.5 grados (~50 km).",
        "    Varios municipios pequenos pueden compartir la misma celda.",
        "  - Los valores son promedios del anio consultado; no representan",
        "    extremos multianuales ni variabilidad interanual.",
        "  - Valores <= -900 en la respuesta NASA se convierten a nulo.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    start_default, end_default = default_year_range()
    parser = argparse.ArgumentParser(
        description="Extrae variables climaticas NASA POWER para municipios colombianos."
    )
    parser.add_argument("--municipalities-csv", type=Path, default=DEFAULT_MUNICIPALITIES_CSV)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_MUNICIPAL_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MUNICIPAL_OUTPUT_DIR)
    parser.add_argument("--start", default=start_default)
    parser.add_argument("--end", default=end_default)
    parser.add_argument(
        "--parameters",
        default=",".join(MUNICIPAL_PARAMETERS),
        help="Parametros NASA POWER separados por coma.",
    )
    parser.add_argument("--community", default=DEFAULT_COMMUNITY)
    parser.add_argument("--limit", type=int, default=None, help="Limitar a N municipios (debug).")
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    parser.add_argument("--force", action="store_true", help="Ignorar cache y re-consultar.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parameters = [p.strip() for p in args.parameters.split(",") if p.strip()]
    try:
        outputs = run_nasa_power_municipios(
            municipalities_csv=args.municipalities_csv.resolve(),
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

    print("Extraccion NASA POWER municipal completada.")
    for label, path in outputs.items():
        print(f"  - {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
