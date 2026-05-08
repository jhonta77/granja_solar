"""Extrae conflicto de uso del suelo por municipio via ArcGIS REST (UPRA/IGAC).

Fuente de datos
---------------
UPRA publica el analisis de conflicto de uso del suelo a traves de su
geoservicio ArcGIS REST. El dataset combina capacidad del suelo (IGAC) con
uso actual (IDEAM) para clasificar cada poligono en:

    Sin conflicto / Subutilizacion leve, moderada, severa /
    Sobreutilizacion leve, moderada, severa / Areas de uso exclusivo

Endpoint ArcGIS principal (UPRA):
    https://geoservicios.upra.gov.co/arcgis/rest/services/
    → Ordenamiento_Productivo/conflicto_uso_tierra/MapServer/0
    → Consulta: /query?where=1=1&outFields=*&f=json&resultOffset=N

Fallback Socrata (datos.gov.co):
    Dataset ID: rcf6-ftgv
    URL: https://www.datos.gov.co/resource/rcf6-ftgv.json
    Campos: codigo_municipio, nombre_municipio, conflicto (categoria texto)

Metodologia
-----------
1. Intenta descargar poligonos UPRA via ArcGIS REST (paginado).
2. Si falla → intenta Socrata datos.gov.co (rcf6-ftgv).
3. Hace join con centroides municipales (codigo_dane).
4. Calcula fraccion de area en conflicto severo por municipio.
5. Asigna score_conflicto 0-1 (1 = sin conflicto → apto para solar).

Logica del score de conflicto
------------------------------
Las instalaciones solares en suelo con sobreutilizacion severa generan
menor tension social y regulatoria. El score recompensa municipios donde
el suelo ya esta degradado o subutilizado, y penaliza los de alto valor
agropecuario sin conflicto (pues ahi hay oportunidad productiva ya activa).

    score_conflicto = fraccion_conflicto_severo * 0.4
                    + fraccion_subutilizacion   * 0.3
                    + fraccion_sin_conflicto_no_productivo * 0.1
                    + 0.2  (base)

Clipped to [0, 1]. Se usa en score_agropecuario y score_riesgo.

Salida principal
----------------
    data/clean/upra_conflicto/upra_conflicto_municipal.csv

Columnas:
    codigo_dane                 CHAR(5)
    municipio                   VARCHAR
    departamento                VARCHAR
    pct_sin_conflicto           DOUBLE   % area sin conflicto (0-100)
    pct_sobreutilizacion        DOUBLE   % area sobreutilizacion (leve+mod+sev)
    pct_sobreutilizacion_severa DOUBLE   % area sobreutilizacion severa
    pct_subutilizacion          DOUBLE   % area subutilizacion (leve+mod+sev)
    pct_exclusion               DOUBLE   % areas de uso exclusivo/protegido
    conflicto_dominante         VARCHAR  categoria con mayor % de area
    score_conflicto             DOUBLE   score 0-1 para scoring multidim
    fuente                      VARCHAR
    metodo_fuente               VARCHAR  arcgis_upra / socrata_datos_gov / manual_csv

Uso
---
    python -m src.extract.upra_conflicto_suelo
    python -m src.extract.upra_conflicto_suelo --force
    python -m src.extract.upra_conflicto_suelo --input-csv mi_conflicto.csv
    python -m src.extract.upra_conflicto_suelo --solo-socrata
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUNICIPALITIES_CSV = (
    PROJECT_ROOT / "data" / "clean" / "base_municipios" / "municipios_distritos_colombia.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "upra_conflicto"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "upra_conflicto"

# ArcGIS REST endpoint UPRA - conflicto uso tierra
UPRA_ARCGIS_BASE = (
    "https://geoservicios.upra.gov.co/arcgis/rest/services"
    "/Ordenamiento_Productivo/conflicto_uso_tierra/MapServer/0"
)
UPRA_QUERY_URL = f"{UPRA_ARCGIS_BASE}/query"

# Fallback Socrata datos.gov.co
SOCRATA_BASE = "https://www.datos.gov.co/resource"
SOCRATA_DATASET_CONFLICTO = "rcf6-ftgv"

ARCGIS_PAGE_SIZE = 1000
SOCRATA_PAGE_SIZE = 5000
HTTP_TIMEOUT = 60
HTTP_RETRY_DELAY = 3
MAX_RETRIES = 3

# Palabras clave para clasificar categorias de conflicto
_KEYWORDS_SOBREUTILIZACION = {"sobreutilizacion", "sobreuso", "sobre uso"}
_KEYWORDS_SOBREUTILIZACION_SEVERA = {"severa", "severo"}
_KEYWORDS_SUBUTILIZACION = {"subutilizacion", "subuso", "sub uso"}
_KEYWORDS_SIN_CONFLICTO = {"sin conflicto", "adecuado", "uso adecuado"}
_KEYWORDS_EXCLUSION = {"exclusion", "excluida", "no agropecuario", "protegida", "protegido", "cuerpo de agua"}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_json(url: str, params: dict, timeout: int = HTTP_TIMEOUT) -> Any:
    """GET con reintentos."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"    Reintento {attempt}/{MAX_RETRIES} para {url}: {exc}")
            time.sleep(HTTP_RETRY_DELAY * attempt)


# ---------------------------------------------------------------------------
# ArcGIS REST download
# ---------------------------------------------------------------------------

def _arcgis_count(url: str) -> int:
    """Cuenta total de features disponibles."""
    data = _get_json(url, {"where": "1=1", "returnCountOnly": "true", "f": "json"})
    return int(data.get("count", 0))


def _arcgis_fetch_page(url: str, offset: int, fields: str = "*") -> list[dict]:
    """Descarga una pagina de features ArcGIS."""
    params = {
        "where": "1=1",
        "outFields": fields,
        "resultOffset": offset,
        "resultRecordCount": ARCGIS_PAGE_SIZE,
        "f": "json",
        "geometryType": "esriGeometryPolygon",
        "returnGeometry": "false",
    }
    data = _get_json(url, params)
    features = data.get("features", [])
    return [f.get("attributes", {}) for f in features]


def _download_arcgis(cache_dir: Path, force: bool) -> pd.DataFrame | None:
    """Descarga el dataset completo UPRA via ArcGIS REST."""
    cache_path = cache_dir / "upra_conflicto_arcgis.json"
    if cache_path.exists() and not force:
        print(f"  UPRA ArcGIS: usando cache {cache_path}")
        with cache_path.open(encoding="utf-8") as fh:
            return pd.DataFrame(json.load(fh))

    print("  Intentando ArcGIS REST UPRA...")
    try:
        total = _arcgis_count(UPRA_QUERY_URL)
        if total == 0:
            print("  ArcGIS UPRA devolvio 0 features.")
            return None
        print(f"  Total features ArcGIS: {total}")

        rows: list[dict] = []
        offset = 0
        while offset < total:
            page = _arcgis_fetch_page(UPRA_QUERY_URL, offset)
            if not page:
                break
            rows.extend(page)
            offset += len(page)
            print(f"    {offset}/{total} features descargados...", end="\r")

        print()
        if not rows:
            return None

        cache_dir.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False)
        print(f"  Descargado ArcGIS: {len(rows)} features -> {cache_path}")
        return pd.DataFrame(rows)

    except Exception as exc:  # noqa: BLE001
        print(f"  ArcGIS REST fallo: {exc}")
        return None


# ---------------------------------------------------------------------------
# Socrata fallback
# ---------------------------------------------------------------------------

def _socrata_fetch_page(dataset_id: str, offset: int, limit: int = SOCRATA_PAGE_SIZE) -> list[dict]:
    """Descarga una pagina Socrata."""
    url = f"{SOCRATA_BASE}/{dataset_id}.json"
    params = {"$limit": limit, "$offset": offset}
    return _get_json(url, params) or []


def _download_socrata(cache_dir: Path, force: bool) -> pd.DataFrame | None:
    """Descarga conflicto uso suelo via Socrata datos.gov.co."""
    cache_path = cache_dir / "upra_conflicto_socrata.json"
    if cache_path.exists() and not force:
        print(f"  UPRA Socrata: usando cache {cache_path}")
        with cache_path.open(encoding="utf-8") as fh:
            return pd.DataFrame(json.load(fh))

    print("  Intentando Socrata datos.gov.co (rcf6-ftgv)...")
    try:
        rows: list[dict] = []
        offset = 0
        while True:
            page = _socrata_fetch_page(SOCRATA_DATASET_CONFLICTO, offset)
            if not page:
                break
            rows.extend(page)
            offset += len(page)
            print(f"    {offset} registros Socrata...", end="\r")
            if len(page) < SOCRATA_PAGE_SIZE:
                break

        print()
        if not rows:
            print("  Socrata devolvio 0 registros.")
            return None

        cache_dir.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False)
        print(f"  Descargado Socrata: {len(rows)} registros -> {cache_path}")
        return pd.DataFrame(rows)

    except Exception as exc:  # noqa: BLE001
        print(f"  Socrata fallo: {exc}")
        return None


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------

def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _classify_conflicto(value: str) -> str:
    """Clasifica una categoria de conflicto en grupo normalizado."""
    val = str(value).lower().strip()
    if any(k in val for k in _KEYWORDS_EXCLUSION):
        return "exclusion"
    if any(k in val for k in _KEYWORDS_SIN_CONFLICTO):
        return "sin_conflicto"
    if any(k in val for k in _KEYWORDS_SOBREUTILIZACION):
        severa = any(k in val for k in _KEYWORDS_SOBREUTILIZACION_SEVERA)
        return "sobreutilizacion_severa" if severa else "sobreutilizacion"
    if any(k in val for k in _KEYWORDS_SUBUTILIZACION):
        return "subutilizacion"
    return "otro"


# ---------------------------------------------------------------------------
# Process ArcGIS data
# ---------------------------------------------------------------------------

def _process_arcgis(df: pd.DataFrame, municipalities: pd.DataFrame) -> pd.DataFrame:
    """Transforma datos ArcGIS a tabla municipal de conflicto."""
    # Detectar columnas clave
    conflicto_col = _detect_column(df, [
        "conflicto", "tipo_conflicto", "CONFLICTO", "TIPO_CONFLICTO",
        "categoria", "CATEGORIA", "clasificacion", "CLASIFICACION",
    ])
    area_col = _detect_column(df, [
        "area_ha", "AREA_HA", "shape_area", "SHAPE_AREA", "area", "AREA",
    ])
    dane_col = _detect_column(df, [
        "cod_municipio", "COD_MUNICIPIO", "codigo_dane", "CODIGO_DANE",
        "cod_dane", "COD_DANE", "mpio_ccdgo", "MPIO_CCDGO",
    ])

    if not conflicto_col:
        print("  ArcGIS: no se encontro columna de conflicto.")
        return pd.DataFrame()

    df = df.copy()
    df["_grupo"] = df[conflicto_col].fillna("otro").apply(_classify_conflicto)

    # Si no hay codigo DANE, no se puede hacer agregacion municipal
    if not dane_col:
        print("  ArcGIS: sin columna codigo_dane, no se puede agregar por municipio.")
        return pd.DataFrame()

    df["_codigo_dane"] = df[dane_col].astype(str).str.zfill(5)

    if area_col:
        df["_area"] = pd.to_numeric(df[area_col], errors="coerce").fillna(0.0)
    else:
        df["_area"] = 1.0  # cada fila cuenta igual si no hay area

    return _aggregate_to_municipal(df, municipalities)


def _aggregate_to_municipal(df: pd.DataFrame, municipalities: pd.DataFrame) -> pd.DataFrame:
    """Agrega tabla con _codigo_dane, _grupo, _area a nivel municipal."""
    grouped = df.groupby(["_codigo_dane", "_grupo"])["_area"].sum().reset_index()
    total_area = df.groupby("_codigo_dane")["_area"].sum().rename("_total")

    pivot = grouped.pivot(index="_codigo_dane", columns="_grupo", values="_area").fillna(0.0)
    pivot = pivot.join(total_area)

    for col in ["sin_conflicto", "sobreutilizacion", "sobreutilizacion_severa",
                "subutilizacion", "exclusion", "otro"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    pct = pivot.div(pivot["_total"].replace(0, float("nan")), axis=0) * 100

    result = pd.DataFrame()
    result["codigo_dane"] = pivot.index

    result["pct_sin_conflicto"] = pct["sin_conflicto"].values
    result["pct_sobreutilizacion"] = (pct["sobreutilizacion"] + pct["sobreutilizacion_severa"]).values
    result["pct_sobreutilizacion_severa"] = pct["sobreutilizacion_severa"].values
    result["pct_subutilizacion"] = pct["subutilizacion"].values
    result["pct_exclusion"] = pct["exclusion"].values

    # Categoria dominante
    cat_cols = ["sin_conflicto", "sobreutilizacion_severa", "sobreutilizacion",
                "subutilizacion", "exclusion", "otro"]
    present = [c for c in cat_cols if c in pct.columns]
    result["conflicto_dominante"] = pct[present].idxmax(axis=1).values

    result = result.merge(
        municipalities[["codigo_dane", "municipio", "departamento"]],
        on="codigo_dane",
        how="left",
    )
    return result


# ---------------------------------------------------------------------------
# Process Socrata data
# ---------------------------------------------------------------------------

def _process_socrata(df: pd.DataFrame, municipalities: pd.DataFrame) -> pd.DataFrame:
    """Transforma datos Socrata a tabla municipal de conflicto."""
    dane_col = _detect_column(df, [
        "codigo_municipio", "cod_municipio", "codigo_dane", "cod_dane",
        "divipola", "codmpio",
    ])
    conflicto_col = _detect_column(df, [
        "conflicto", "tipo_conflicto", "categoria", "clasificacion",
        "tipo_conflicto_uso_tierra",
    ])
    area_col = _detect_column(df, ["area_ha", "area", "hectareas"])

    if not dane_col or not conflicto_col:
        print(f"  Socrata: columnas detectadas={list(df.columns)[:10]}")
        print("  Socrata: no se encontraron columnas dane o conflicto.")
        return pd.DataFrame()

    df = df.copy()
    df["_codigo_dane"] = df[dane_col].astype(str).str.zfill(5)
    df["_grupo"] = df[conflicto_col].fillna("otro").apply(_classify_conflicto)
    df["_area"] = pd.to_numeric(df[area_col], errors="coerce").fillna(1.0) if area_col else 1.0

    return _aggregate_to_municipal(df, municipalities)


# ---------------------------------------------------------------------------
# Process manual CSV
# ---------------------------------------------------------------------------

def _process_manual_csv(csv_path: Path, municipalities: pd.DataFrame) -> pd.DataFrame:
    """Lee CSV manual exportado por el usuario."""
    print(f"  Leyendo CSV manual: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)

    # Si ya tiene columnas procesadas, usarlas directamente
    pct_cols = [c for c in df.columns if c.startswith("pct_")]
    if pct_cols and "codigo_dane" in df.columns:
        print("  CSV ya tiene columnas pct_*: usando directamente.")
        for col in pct_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df

    # Si tiene columna de conflicto texto, procesar
    dane_col = _detect_column(df, [
        "codigo_dane", "cod_dane", "codigo_municipio", "cod_municipio", "divipola",
    ])
    conflicto_col = _detect_column(df, [
        "conflicto", "tipo_conflicto", "categoria", "clasificacion",
    ])
    area_col = _detect_column(df, ["area_ha", "area", "hectareas"])

    if not dane_col or not conflicto_col:
        raise ValueError(
            f"El CSV '{csv_path}' debe tener columnas de codigo municipal y conflicto.\n"
            "Columnas encontradas: " + ", ".join(df.columns.tolist())
        )

    df["_codigo_dane"] = df[dane_col].astype(str).str.zfill(5)
    df["_grupo"] = df[conflicto_col].fillna("otro").apply(_classify_conflicto)
    df["_area"] = pd.to_numeric(df[area_col], errors="coerce").fillna(1.0) if area_col else 1.0

    return _aggregate_to_municipal(df, municipalities)


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------

def _compute_score_conflicto(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula score_conflicto 0-1.

    Escala 0-100 (pct_*) → fracciones 0-1 para el calculo.
    Un municipio con mucha sobreutilizacion severa tiene score alto
    (el solar puede instaurarse en suelo ya degradado sin conflicto).
    Un municipio con mucho suelo sin conflicto activo productivo
    tiene score bajo (ese suelo vale mas para su uso actual).
    """
    df = df.copy()
    f_sobre_severa = df["pct_sobreutilizacion_severa"].fillna(0.0) / 100
    f_subutil = df["pct_subutilizacion"].fillna(0.0) / 100
    f_excl = df["pct_exclusion"].fillna(0.0) / 100

    score = (
        f_sobre_severa * 0.45
        + f_subutil * 0.25
        + f_excl * 0.10
        + 0.20
    ).clip(0.0, 1.0)

    df["score_conflicto"] = score.round(4)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_upra_conflicto(
    municipalities_csv: Path = DEFAULT_MUNICIPALITIES_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    input_csv: Path | None = None,
    force: bool = False,
    solo_socrata: bool = False,
) -> dict[str, Path]:
    """Descarga conflicto de uso del suelo y agrega por municipio."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "upra_conflicto_municipal.csv"
    if output_path.exists() and not force and input_csv is None:
        print(f"  UPRA Conflicto: usando resultado previo {output_path}")
        return {"summary": output_path}

    municipalities = pd.read_csv(municipalities_csv, dtype={"codigo_dane": "string"})
    municipalities["codigo_dane"] = municipalities["codigo_dane"].str.zfill(5)

    result = pd.DataFrame()
    metodo = "sin_datos"

    if input_csv is not None:
        result = _process_manual_csv(input_csv, municipalities)
        metodo = "manual_csv"

    if result.empty and not solo_socrata:
        raw = _download_arcgis(cache_dir, force)
        if raw is not None and not raw.empty:
            result = _process_arcgis(raw, municipalities)
            metodo = "arcgis_upra"

    if result.empty:
        raw = _download_socrata(cache_dir, force)
        if raw is not None and not raw.empty:
            result = _process_socrata(raw, municipalities)
            metodo = "socrata_datos_gov"

    if result.empty:
        print(
            "\n  UPRA Conflicto: ninguna fuente devolvio datos.\n"
            "  Opciones manuales:\n"
            "    - Descargar shapefile de conflicto desde:\n"
            "      https://visor.upra.gov.co/  → Zonificacion de conflicto de uso del suelo\n"
            "    - Exportar como CSV con columnas: codigo_municipio, conflicto, area_ha\n"
            "    - Pasar con: --input-csv ruta/al/archivo.csv\n"
        )
        return {}

    # Calcular score
    result = _compute_score_conflicto(result)

    # Asegurar columnas minimas
    for col in ["pct_sin_conflicto", "pct_sobreutilizacion",
                "pct_sobreutilizacion_severa", "pct_subutilizacion",
                "pct_exclusion", "conflicto_dominante"]:
        if col not in result.columns:
            result[col] = float("nan") if col.startswith("pct_") else ""

    result["fuente"] = "UPRA - Conflicto de uso del suelo"
    result["metodo_fuente"] = metodo

    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  UPRA Conflicto completado: {len(result)} municipios -> {output_path}")
    return {"summary": output_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae conflicto de uso del suelo (UPRA/IGAC) por municipio."
    )
    parser.add_argument("--municipalities-csv", type=Path, default=DEFAULT_MUNICIPALITIES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="CSV manual exportado desde visor.upra.gov.co o SIAC.",
    )
    parser.add_argument("--force", action="store_true", help="Re-descargar aunque exista cache.")
    parser.add_argument(
        "--solo-socrata",
        action="store_true",
        help="Saltar ArcGIS UPRA e ir directo a Socrata datos.gov.co.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_upra_conflicto(
            municipalities_csv=args.municipalities_csv.resolve(),
            output_dir=args.output_dir.resolve(),
            cache_dir=args.cache_dir.resolve(),
            input_csv=args.input_csv.resolve() if args.input_csv else None,
            force=args.force,
            solo_socrata=args.solo_socrata,
        )
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {error}")
        return 1

    if not outputs:
        print("UPRA Conflicto: sin resultados. Ver instrucciones de descarga manual arriba.")
        return 1

    print("UPRA Conflicto completado.")
    for label, path in outputs.items():
        print(f"  - {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
