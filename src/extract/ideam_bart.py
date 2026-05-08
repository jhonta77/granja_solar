"""Extrae variables climaticas de estaciones IDEAM via datos.gov.co (Socrata API).

Fuente de datos
---------------
IDEAM publica sus datos de estaciones meteorologicas en el portal de datos
abiertos del gobierno colombiano (datos.gov.co), que usa la API Socrata.
No requiere autenticacion ni credenciales.

URL base Socrata: https://www.datos.gov.co/resource/{dataset_id}.json

Datasets IDEAM usados
----------------------
1. Catalogo de estaciones IDEAM:
   https://www.datos.gov.co/resource/s54a-sgyg.json
   Campos: codigoestacion, nombreestacion, latitud, longitud,
           altitud, departamento, municipio, estado (Activo/Inactivo)

2. Series de temperatura (media mensual):
   https://www.datos.gov.co/resource/sbwg-7ju4.json
   Campos: codigoestacion, anio, mes, valor (°C)

3. Series de precipitacion (total mensual):
   https://www.datos.gov.co/resource/s6ay-4pjm.json
   Campos: codigoestacion, anio, mes, valor (mm)

4. Series de velocidad del viento (media mensual):
   https://www.datos.gov.co/resource/sgfv-3ytu.json
   Campos: codigoestacion, anio, mes, valor (m/s)

Metodologia
-----------
1. Descarga el catalogo de estaciones con coordenadas.
2. Hace join espacial: cada estacion → municipio mas cercano (EPSG:3116).
3. Descarga series de las estaciones activas.
4. Calcula normales climatologicas (promedio multianual) por estacion.
5. Agrega al nivel municipal (promedio de estaciones en el municipio).

Salida principal
----------------
    data/clean/ideam_bart/ideam_clima_municipal.csv

Columnas:
    codigo_dane          CHAR(5)
    t_media_c            DOUBLE   temperatura media (°C)
    t_min_c              DOUBLE   temperatura minima media
    t_max_c              DOUBLE   temperatura maxima media
    prec_media_mm_mes    DOUBLE   precipitacion media mensual (mm)
    prec_suma_mm_year    DOUBLE   precipitacion total anual (mm)
    viento_media_m_s     DOUBLE   velocidad viento media (m/s)
    n_estaciones         INT      estaciones usadas para el municipio
    fuente               VARCHAR  IDEAM - datos.gov.co Socrata
    anio_inicio          INT
    anio_fin             INT

Uso
---
    python -m src.extract.ideam_bart
    python -m src.extract.ideam_bart --force
    python -m src.extract.ideam_bart --anio-inicio 2010 --anio-fin 2023
    python -m src.extract.ideam_bart --limit-estaciones 100   # modo debug
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUNICIPALITIES_CSV = (
    PROJECT_ROOT / "data" / "clean" / "base_municipios" / "municipios_distritos_colombia.csv"
)
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ideam_bart"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "ideam_bart"

SOCRATA_BASE = "https://www.datos.gov.co/resource"

# IDs de datasets IDEAM en datos.gov.co
DATASET_ESTACIONES = "s54a-sgyg"
DATASET_TEMPERATURA = "sbwg-7ju4"
DATASET_PRECIPITACION = "s6ay-4pjm"
DATASET_VIENTO = "sgfv-3ytu"

OUTPUT_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:3116"

SOCRATA_PAGE_SIZE = 10000


def _socrata_request(
    dataset_id: str,
    params: dict[str, Any],
    timeout: int = 60,
    retries: int = 3,
) -> list[dict[str, Any]]:
    """Consulta Socrata API con paginacion automatica y reintentos."""

    url = f"{SOCRATA_BASE}/{dataset_id}.json"
    headers = {
        "User-Agent": "granja-solar-colombia/1.0",
        "Accept": "application/json",
    }
    all_rows: list[dict[str, Any]] = []
    offset = 0

    while True:
        query_params = {**params, "$limit": SOCRATA_PAGE_SIZE, "$offset": offset}
        last_error: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                response = requests.get(url, params=query_params, headers=headers, timeout=timeout)
                response.raise_for_status()
                batch = response.json()
                break
            except Exception as error:  # noqa: BLE001
                last_error = error
                if attempt < retries:
                    time.sleep(2 * attempt)
        else:
            raise RuntimeError(f"No fue posible consultar {url}: {last_error}")

        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < SOCRATA_PAGE_SIZE:
            break
        offset += SOCRATA_PAGE_SIZE

    return all_rows


def _cache_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _load_cache(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_estaciones(raw_dir: Path, force: bool = False) -> pd.DataFrame:
    """Descarga catalogo de estaciones IDEAM."""

    cache = raw_dir / "estaciones_ideam.json"
    if cache.exists() and not force:
        print("  Estaciones IDEAM: cargando desde cache...")
        rows = _load_cache(cache)
    else:
        print("  Descargando catalogo estaciones IDEAM desde datos.gov.co...")
        rows = _socrata_request(
            DATASET_ESTACIONES,
            {"$where": "estado='Activo'"},
        )
        _cache_json(cache, rows)
        print(f"  Estaciones descargadas: {len(rows)}")

    if not rows:
        raise ValueError(
            "No se encontraron estaciones IDEAM en datos.gov.co.\n"
            f"Verifica que el dataset {DATASET_ESTACIONES} siga disponible en:\n"
            f"  https://www.datos.gov.co/resource/{DATASET_ESTACIONES}.json"
        )

    df = pd.DataFrame(rows)
    for col in ["latitud", "longitud", "altitud"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["latitud", "longitud"])
    df = df[df["latitud"].between(-5.0, 14.0) & df["longitud"].between(-82.0, -66.0)]
    return df.reset_index(drop=True)


def fetch_series(
    dataset_id: str,
    variable_name: str,
    raw_dir: Path,
    anio_inicio: int,
    anio_fin: int,
    force: bool = False,
) -> pd.DataFrame:
    """Descarga serie mensual de una variable climatica."""

    cache = raw_dir / f"serie_{variable_name}_{anio_inicio}_{anio_fin}.json"
    if cache.exists() and not force:
        print(f"  Serie {variable_name}: cargando desde cache...")
        rows = _load_cache(cache)
    else:
        print(f"  Descargando serie {variable_name} ({anio_inicio}-{anio_fin})...")
        rows = _socrata_request(
            dataset_id,
            {"$where": f"anio >= {anio_inicio} AND anio <= {anio_fin}"},
        )
        _cache_json(cache, rows)
        print(f"  Registros {variable_name}: {len(rows)}")

    if not rows:
        print(f"  Advertencia: no hay datos para {variable_name} en el rango solicitado.")
        return pd.DataFrame(columns=["codigoestacion", "anio", "mes", "valor"])

    df = pd.DataFrame(rows)
    for col in ["anio", "mes", "valor"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["codigoestacion", "valor"]).reset_index(drop=True)


def assign_stations_to_municipalities(
    estaciones: pd.DataFrame,
    municipalities_csv: Path,
) -> pd.DataFrame:
    """Asigna cada estacion al municipio mas cercano (EPSG:3116)."""

    municipalities = pd.read_csv(municipalities_csv, dtype={"codigo_dane": "string"})
    municipalities["codigo_dane"] = municipalities["codigo_dane"].astype("string").str.zfill(5)

    lon_col = next((c for c in municipalities.columns if c in {"lon", "longitud"}), None)
    lat_col = next((c for c in municipalities.columns if c in {"lat", "latitud"}), None)

    muni_gdf = gpd.GeoDataFrame(
        municipalities,
        geometry=gpd.points_from_xy(municipalities[lon_col], municipalities[lat_col]),
        crs=OUTPUT_CRS,
    ).to_crs(PROJECTED_CRS)

    est_gdf = gpd.GeoDataFrame(
        estaciones,
        geometry=gpd.points_from_xy(estaciones["longitud"], estaciones["latitud"]),
        crs=OUTPUT_CRS,
    ).to_crs(PROJECTED_CRS)

    muni_xy = np.column_stack([muni_gdf.geometry.x.to_numpy(), muni_gdf.geometry.y.to_numpy()])
    est_xy = np.column_stack([est_gdf.geometry.x.to_numpy(), est_gdf.geometry.y.to_numpy()])

    nearest_indices: list[int] = []
    chunk = 200
    for start in range(0, len(est_xy), chunk):
        batch = est_xy[start : start + chunk]
        dists = np.sqrt(((batch[:, None, :] - muni_xy[None, :, :]) ** 2).sum(axis=2))
        nearest_indices.extend(dists.argmin(axis=1).tolist())

    muni_reset = municipalities.reset_index(drop=True)
    assigned = estaciones.copy().reset_index(drop=True)
    assigned["codigo_dane"] = muni_reset.iloc[nearest_indices]["codigo_dane"].values
    assigned["municipio_asignado"] = muni_reset.iloc[nearest_indices]["municipio"].values

    return assigned


def compute_normals(series: pd.DataFrame, id_col: str = "codigoestacion") -> pd.DataFrame:
    """Calcula normales climatologicas (media multianual) por estacion."""

    if series.empty:
        return pd.DataFrame(columns=[id_col, "media", "min", "max", "suma_anual"])

    monthly_means = (
        series.groupby([id_col, "mes"])["valor"]
        .mean()
        .reset_index()
        .rename(columns={"valor": "media_mensual"})
    )
    normals = monthly_means.groupby(id_col)["media_mensual"].agg(
        media="mean",
        min_val="min",
        max_val="max",
    ).reset_index()
    normals["suma_anual"] = monthly_means.groupby(id_col)["media_mensual"].sum().values
    return normals


def aggregate_to_municipalities(
    estaciones_con_dane: pd.DataFrame,
    t_normals: pd.DataFrame,
    prec_normals: pd.DataFrame,
    viento_normals: pd.DataFrame,
) -> pd.DataFrame:
    """Agrega normales de estaciones al nivel municipal."""

    id_col = "codigoestacion"
    base = estaciones_con_dane[["codigoestacion", "codigo_dane"]].copy()

    frames = []
    for normals, prefix, suma_label in [
        (t_normals, "t", "t_suma_anual"),
        (prec_normals, "prec", "prec_suma_mm_year"),
        (viento_normals, "viento", None),
    ]:
        if normals.empty:
            continue
        merged = base.merge(normals, on=id_col, how="left")
        frames.append((merged, prefix, suma_label))

    if not frames:
        return pd.DataFrame()

    result_df = base[["codigo_dane"]].drop_duplicates().copy()

    for merged, prefix, suma_label in frames:
        agg = (
            merged.groupby("codigo_dane")
            .agg(
                media=("media", "mean"),
                min_val=("min_val", "mean"),
                max_val=("max_val", "mean"),
                suma=("suma_anual", "mean") if suma_label else ("media", "mean"),
                n=("media", "count"),
            )
            .reset_index()
        )
        rename = {
            "media": f"{prefix}_media",
            "min_val": f"{prefix}_min",
            "max_val": f"{prefix}_max",
            "suma": suma_label or f"{prefix}_suma",
            "n": f"n_{prefix}",
        }
        agg = agg.rename(columns=rename)
        result_df = result_df.merge(agg, on="codigo_dane", how="left")

    col_map = {
        "t_media": "t_media_c",
        "t_min": "t_min_c",
        "t_max": "t_max_c",
        "prec_media": "prec_media_mm_mes",
        "prec_suma": "prec_suma_mm_year",
        "viento_media": "viento_media_m_s",
        "n_t": "n_estaciones",
    }
    result_df = result_df.rename(columns={k: v for k, v in col_map.items() if k in result_df.columns})

    if "n_estaciones" not in result_df.columns:
        for col in ["n_prec", "n_viento"]:
            if col in result_df.columns:
                result_df["n_estaciones"] = result_df[col]
                break

    return result_df


def run_ideam_bart(
    municipalities_csv: Path = DEFAULT_MUNICIPALITIES_CSV,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    anio_inicio: int = 2010,
    anio_fin: int = 2023,
    limit_estaciones: int | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Descarga y procesa datos climaticos IDEAM para todos los municipios."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    estaciones = fetch_estaciones(raw_dir, force=force)
    if limit_estaciones:
        estaciones = estaciones.head(limit_estaciones).copy()
    print(f"  Estaciones activas con coordenadas: {len(estaciones)}")

    print("  Asignando estaciones a municipios...")
    estaciones_con_dane = assign_stations_to_municipalities(estaciones, municipalities_csv)
    estaciones_path = raw_dir / "estaciones_con_municipio.csv"
    estaciones_con_dane.to_csv(estaciones_path, index=False, encoding="utf-8-sig")

    t_series = fetch_series(DATASET_TEMPERATURA, "temperatura", raw_dir, anio_inicio, anio_fin, force)
    prec_series = fetch_series(DATASET_PRECIPITACION, "precipitacion", raw_dir, anio_inicio, anio_fin, force)
    viento_series = fetch_series(DATASET_VIENTO, "viento", raw_dir, anio_inicio, anio_fin, force)

    print("  Calculando normales climatologicas...")
    t_normals = compute_normals(t_series)
    prec_normals = compute_normals(prec_series)
    viento_normals = compute_normals(viento_series)

    print("  Agregando al nivel municipal...")
    municipal = aggregate_to_municipalities(estaciones_con_dane, t_normals, prec_normals, viento_normals)
    municipal["fuente"] = "IDEAM - datos.gov.co Socrata"
    municipal["anio_inicio"] = anio_inicio
    municipal["anio_fin"] = anio_fin
    municipal["fecha_descarga_utc"] = datetime.now(timezone.utc).isoformat()

    output_path = output_dir / "ideam_clima_municipal.csv"
    obs_path = output_dir / "ideam_bart_observaciones.txt"

    municipal.to_csv(output_path, index=False, encoding="utf-8-sig")
    _write_observations(obs_path, estaciones, municipal, anio_inicio, anio_fin)

    return {
        "clima_municipal": output_path,
        "estaciones": estaciones_path,
        "observaciones": obs_path,
    }


def _write_observations(
    path: Path,
    estaciones: pd.DataFrame,
    municipal: pd.DataFrame,
    anio_inicio: int,
    anio_fin: int,
) -> None:
    n_con_dato = municipal["t_media_c"].notna().sum() if "t_media_c" in municipal.columns else 0
    lines = [
        "IDEAM - Climatologia municipal desde datos.gov.co",
        "==================================================",
        "",
        "Fuente: IDEAM via Portal Datos Abiertos Colombia (Socrata API)",
        f"URL base: {SOCRATA_BASE}",
        f"Dataset estaciones: {DATASET_ESTACIONES}",
        f"Dataset temperatura: {DATASET_TEMPERATURA}",
        f"Dataset precipitacion: {DATASET_PRECIPITACION}",
        f"Dataset viento: {DATASET_VIENTO}",
        f"Periodo: {anio_inicio} - {anio_fin}",
        f"Estaciones activas descargadas: {len(estaciones)}",
        f"Municipios con dato de temperatura: {n_con_dato}",
        "",
        "Metodologia:",
        "  1. Se descarga el catalogo de estaciones activas con coordenadas.",
        "  2. Cada estacion se asigna al municipio mas cercano (EPSG:3116).",
        "  3. Se calculan normales climatologicas (media multianual) por estacion.",
        "  4. Se promedia entre estaciones del mismo municipio.",
        "",
        "Ventaja sobre NASA POWER y ERA5:",
        "  Datos de estaciones terrestres reales (no reanálisis satelital).",
        "  Mejor representacion de microclimas y valles colombianos.",
        "",
        "Uso en scoring multidimensional:",
        "  score_fisico: t_media_c (temperatura optima paneles ~20-25°C)",
        "                prec_suma_mm_year (lluvia para limpieza paneles)",
        "                viento_media_m_s (carga mecanica en estructuras)",
        "  score_economico: prec_suma_mm_year (proxy disponibilidad agua)",
        "  score_riesgo: prec_suma_mm_year > 3000 mm (proxy lluvia extrema)",
        "",
        "Limitaciones:",
        "  - No todos los municipios tienen estacion cercana.",
        "    Municipios sin estacion usaran datos de la estacion mas proxima.",
        "  - La asignacion es puntual (centroide); no considera relieve.",
        "  - Dataset IDs de datos.gov.co pueden cambiar si IDEAM los actualiza.",
        "    Si falla, verificar en: https://www.datos.gov.co/browse?q=IDEAM+clima",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae climatologia IDEAM por municipio via Socrata API (datos.gov.co)."
    )
    parser.add_argument("--municipalities-csv", type=Path, default=DEFAULT_MUNICIPALITIES_CSV)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anio-inicio", type=int, default=2010)
    parser.add_argument("--anio-fin", type=int, default=2023)
    parser.add_argument(
        "--limit-estaciones",
        type=int,
        default=None,
        help="Limitar a N estaciones (modo debug).",
    )
    parser.add_argument("--force", action="store_true", help="Ignorar cache y re-descargar.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_ideam_bart(
            municipalities_csv=args.municipalities_csv.resolve(),
            raw_dir=args.raw_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            anio_inicio=args.anio_inicio,
            anio_fin=args.anio_fin,
            limit_estaciones=args.limit_estaciones,
            force=args.force,
        )
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {error}")
        return 1

    print("IDEAM climatologia municipal completada.")
    for label, path in outputs.items():
        print(f"  - {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
