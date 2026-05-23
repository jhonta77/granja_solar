"""Extrae variables climaticas ERA5 (Copernicus CDS) para municipios de Colombia.

Por que ERA5 en vez de NASA POWER para clima
---------------------------------------------
ERA5 (ECMWF) tiene resolucion de ~0.1 grados (~11 km) vs ~0.5 grados de
NASA POWER. Para Colombia con municipios pequenos, ERA5 da valores mas
precisos de temperatura, precipitacion, viento y cobertura nubosa.

Variables que extrae
---------------------
    2m_temperature             → t2m_media_c, t2m_min_c, t2m_max_c
    total_precipitation        → prec_media_mm_day, prec_suma_mm_year
    10m_u_component_of_wind    → (componente U del viento)
    10m_v_component_of_wind    → (componente V del viento → ws10m m/s)
    total_cloud_cover          → cloud_cover_media (fraccion 0-1)

Credenciales Copernicus CDS
----------------------------
Agrega en el archivo .env del proyecto UNA sola variable:

    CDS_API_KEY=el-valor-de-key

Donde encontrar tu key:
    1. Ir a https://cds.climate.copernicus.eu/profile
    2. Seccion "API key" → ver el bloque .cdsapirc
    3. El valor de la linea "key:" es lo que va en CDS_API_KEY
    4. Pegar en el .env del proyecto

Instalacion de dependencias
----------------------------
    pip install cdsapi python-dotenv

Dataset usado
-------------
ERA5-Land Monthly Averages: resolucion 0.1 grados, datos desde 1950.
Dataset ID: reanalysis-era5-land-monthly-means

Nota sobre area de Colombia
---------------------------
Bounding box Colombia: [13.4, -81.7, -4.2, -66.9]  (N, W, S, E)

Salida principal
----------------
    data/clean/copernicus_era5/era5_resumen_municipios.csv

Columnas:
    codigo_dane             CHAR(5)
    t2m_media_c             DOUBLE   temperatura media anual (°C)
    t2m_min_c               DOUBLE   temperatura minima mensual media
    t2m_max_c               DOUBLE   temperatura maxima mensual media
    prec_media_mm_day       DOUBLE   precipitacion media diaria (mm)
    prec_suma_mm_year       DOUBLE   precipitacion total anual estimada (mm)
    ws10m_media_m_s         DOUBLE   velocidad del viento media (m/s)
    cloud_cover_media       DOUBLE   fraccion nubosa media (0-1)
    fuente                  VARCHAR  ERA5-Land Monthly Means, Copernicus CDS
    anio_referencia         INT

Uso
---
    python -m src.extract.copernicus_era5
    python -m src.extract.copernicus_era5 --year 2023
    python -m src.extract.copernicus_era5 --year 2022 --force
"""

from __future__ import annotations

import argparse
import calendar
import os
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUNICIPALITIES_CSV = (
    PROJECT_ROOT / "data" / "clean" / "base_municipios" / "municipios_distritos_colombia.csv"
)
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "copernicus_era5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "copernicus_era5"

CDS_DATASET = "reanalysis-era5-land-monthly-means"
COLOMBIA_AREA = [13.4, -81.7, -4.2, -66.9]

ERA5_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_cloud_cover",
]

KELVIN_OFFSET = 273.15
SECONDS_PER_DAY = 86400.0


def _load_credentials() -> str:
    """Lee CDS_API_KEY desde .env o variables de entorno.

    La nueva API CDS v2 usa una sola key (no UID separado).
    En tu perfil https://cds.climate.copernicus.eu/profile, seccion 'API key',
    el valor que aparece en la linea 'key:' del bloque .cdsapirc es el que va aqui.
    """
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

    api_key = os.environ.get("CDS_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "No se encontro CDS_API_KEY en el .env.\n"
            "Agrega en el archivo .env del proyecto:\n\n"
            "    CDS_API_KEY=el-valor-de-la-linea-key-de-tu-perfil\n\n"
            "Perfil: https://cds.climate.copernicus.eu/profile  →  seccion 'API key'"
        )
    return api_key


def _download_era5(raw_dir: Path, year: int, force: bool) -> Path:
    """Descarga ERA5-Land monthly means para Colombia en formato NetCDF."""
    try:
        import cdsapi
    except ImportError:
        raise ImportError(
            "El paquete 'cdsapi' no esta instalado.\n"
            "Instala con: pip install cdsapi"
        )

    output_path = raw_dir / f"era5_land_colombia_{year}.nc"
    if output_path.exists() and not force:
        print(f"  ERA5 {year}: usando cache {output_path}")
        return _ensure_netcdf_from_download(output_path)

    api_key = _load_credentials()
    client = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key=api_key,
        quiet=False,
    )

    print(f"  Descargando ERA5-Land monthly means para {year} (Colombia)...")
    client.retrieve(
        CDS_DATASET,
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": ERA5_VARIABLES,
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": "00:00",
            "area": COLOMBIA_AREA,
            "format": "netcdf",
        },
        str(output_path),
    )
    print(f"  Descargado: {output_path}")
    return _ensure_netcdf_from_download(output_path)


def _ensure_netcdf_from_download(path: Path) -> Path:
    """CDS puede entregar un ZIP aunque el destino se llame .nc."""
    if not zipfile.is_zipfile(path):
        return path

    extract_dir = path.with_suffix("")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        nc_members = [m for m in zf.namelist() if m.lower().endswith((".nc", ".netcdf"))]
        if not nc_members:
            raise ValueError(f"El ZIP descargado no contiene NetCDF: {path}")
        member = nc_members[0]
        extracted = extract_dir / Path(member).name
        if not extracted.exists() or extracted.stat().st_size == 0:
            zf.extract(member, extract_dir)
    print(f"  ERA5 ZIP extraido: {extracted}")
    return extracted


def _extract_municipal_values(nc_path: Path, municipalities_csv: Path) -> pd.DataFrame:
    """Extrae valores ERA5 en cada centroide municipal por interpolacion bilineal."""
    try:
        import xarray as xr
        import numpy as np
    except ImportError:
        raise ImportError(
            "Los paquetes 'xarray' y 'netcdf4' son necesarios para leer ERA5.\n"
            "Instala con: pip install xarray netcdf4"
        )
    import numpy as np

    municipalities = pd.read_csv(municipalities_csv, dtype={"codigo_dane": "string"})
    municipalities["codigo_dane"] = municipalities["codigo_dane"].astype("string").str.zfill(5)

    lon_col = next((c for c in municipalities.columns if c in {"lon", "longitud"}), None)
    lat_col = next((c for c in municipalities.columns if c in {"lat", "latitud"}), None)
    if not lon_col or not lat_col:
        raise ValueError("No se encontraron columnas lon/lat en el CSV de municipios.")

    ds = xr.open_dataset(nc_path)

    var_map = {
        "t2m": "t2m",
        "tp": "tp",
        "u10": "u10",
        "v10": "v10",
        "tcc": "tcc",
    }
    available_vars = {k: v for k, v in var_map.items() if k in ds.data_vars}

    rows = []
    for _, muni_row in municipalities.iterrows():
        lon = float(muni_row[lon_col])
        lat = float(muni_row[lat_col])

        point_data: dict = {
            "codigo_dane": str(muni_row["codigo_dane"]),
            "municipio": str(muni_row.get("municipio", "")),
            "departamento": str(muni_row.get("departamento", "")),
        }

        for short_var, ds_var in available_vars.items():
            try:
                monthly_values = ds[ds_var].sel(
                    latitude=lat, longitude=lon, method="nearest"
                ).values.flatten()
                monthly_values = monthly_values.astype(float)
                monthly_values[monthly_values < -9000] = float("nan")

                if short_var == "t2m":
                    monthly_c = monthly_values - KELVIN_OFFSET
                    point_data["t2m_media_c"] = float(np.nanmean(monthly_c))
                    point_data["t2m_min_c"] = float(np.nanmin(monthly_c))
                    point_data["t2m_max_c"] = float(np.nanmax(monthly_c))

                elif short_var == "tp":
                    # ERA5-Land monthly means stores total_precipitation in metres
                    # of water equivalent as a monthly mean of daily totals.
                    # Convert each monthly daily mean to mm/day, then annualize
                    # with the number of days in each month.
                    prec_mm_day = monthly_values * 1000
                    days = _days_for_monthly_values(ds, len(prec_mm_day))
                    point_data["prec_media_mm_day"] = float(np.nanmean(prec_mm_day))
                    point_data["prec_suma_mm_year"] = (
                        float(np.nansum(prec_mm_day * days)) if np.isfinite(prec_mm_day).any() else float("nan")
                    )

                elif short_var in ("u10", "v10"):
                    pass

                elif short_var == "tcc":
                    point_data["cloud_cover_media"] = float(np.nanmean(monthly_values))

            except Exception:  # noqa: BLE001
                continue

        if "u10" in available_vars and "v10" in available_vars:
            try:
                u = ds["u10"].sel(latitude=lat, longitude=lon, method="nearest").values.flatten().astype(float)
                v = ds["v10"].sel(latitude=lat, longitude=lon, method="nearest").values.flatten().astype(float)
                ws = np.sqrt(u ** 2 + v ** 2)
                ws[ws < -9000] = float("nan")
                point_data["ws10m_media_m_s"] = float(np.nanmean(ws))
            except Exception:  # noqa: BLE001
                pass

        rows.append(point_data)

    ds.close()
    return pd.DataFrame(rows)


def _days_for_monthly_values(ds: Any, n_values: int) -> Any:
    """Retorna dias por mes alineados al eje temporal mensual de ERA5."""
    import numpy as np

    time_coord = "valid_time" if "valid_time" in ds.coords else "time"
    if time_coord not in ds.coords:
        return np.full(n_values, 30.44)

    values = ds[time_coord].values[:n_values]
    days: list[int] = []
    for value in values:
        ts = pd.Timestamp(value)
        days.append(calendar.monthrange(ts.year, ts.month)[1])
    return np.asarray(days, dtype=float)


def write_observations(path: Path, n: int, year: int) -> None:
    lines = [
        "Copernicus ERA5-Land - Variables climaticas municipales",
        "=========================================================",
        "",
        f"Dataset: ERA5-Land Monthly Averages (reanalysis-era5-land-monthly-means)",
        f"Fuente: Copernicus Climate Data Store (https://cds.climate.copernicus.eu/)",
        f"Anio de referencia: {year}",
        f"Municipios procesados: {n}",
        f"Resolucion: ~0.1 grados (~11 km)",
        f"Area: Colombia ({COLOMBIA_AREA})",
        "",
        "Variables extraidas:",
        "  t2m_media_c      : Temperatura media anual (°C, conversion desde Kelvin)",
        "  t2m_min_c        : Temperatura minima mensual media del anio",
        "  t2m_max_c        : Temperatura maxima mensual media del anio",
        "  prec_media_mm_day: Precipitacion diaria media (mm/dia)",
        "  prec_suma_mm_year: Precipitacion total anual estimada (mm)",
        "  ws10m_media_m_s  : Velocidad del viento a 10m (m/s, vector magnitude)",
        "  cloud_cover_media: Fraccion de cobertura nubosa media (0-1)",
        "",
        "Uso en scoring multidimensional:",
        "  score_fisico: t2m, ws10m, prec_suma como componentes",
        "  score_economico: prec_suma como proxy disponibilidad agua",
        "  score_riesgo: prec_suma > 3000 mm/year como proxy lluvias extremas",
        "",
        "Notas de conversion:",
        "  - ERA5 temperatura en Kelvin → se resta 273.15 para obtener Celsius.",
        "  - ERA5 precipitacion en m/hora → se multiplica por 86400*1000 para mm/dia.",
        "  - ERA5 viento: magnitud del vector (sqrt(u^2 + v^2)).",
        "",
        "Credenciales requeridas en .env:",
        "  CDS_API_KEY=tu-api-key",
        "  Ver: https://cds.climate.copernicus.eu/profile",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_copernicus_era5(
    municipalities_csv: Path = DEFAULT_MUNICIPALITIES_CSV,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    year: int | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Descarga ERA5 y extrae variables climaticas por municipio."""

    if year is None:
        from datetime import date
        year = date.today().year - 1

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    nc_path = _download_era5(raw_dir, year, force)

    print(f"  Extrayendo valores ERA5 para municipios...")
    df = _extract_municipal_values(nc_path, municipalities_csv)
    df["fuente"] = "ERA5-Land Monthly Means, Copernicus CDS"
    df["anio_referencia"] = year

    output_path = output_dir / "era5_resumen_municipios.csv"
    observations_path = output_dir / "era5_observaciones.txt"

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    write_observations(observations_path, len(df), year)

    print(f"  ERA5 completado: {len(df)} municipios -> {output_path}")
    return {
        "summary": output_path,
        "observations": observations_path,
    }


def parse_args() -> argparse.Namespace:
    from datetime import date
    parser = argparse.ArgumentParser(
        description="Extrae variables climaticas ERA5 (Copernicus) para municipios colombianos."
    )
    parser.add_argument("--municipalities-csv", type=Path, default=DEFAULT_MUNICIPALITIES_CSV)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--year",
        type=int,
        default=date.today().year - 1,
        help="Anio de referencia ERA5 (por defecto el anio anterior).",
    )
    parser.add_argument("--force", action="store_true", help="Re-descargar aunque exista cache.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_copernicus_era5(
            municipalities_csv=args.municipalities_csv.resolve(),
            raw_dir=args.raw_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            year=args.year,
            force=args.force,
        )
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {error}")
        return 1

    print("ERA5 Copernicus completado.")
    for label, path in outputs.items():
        print(f"  - {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
