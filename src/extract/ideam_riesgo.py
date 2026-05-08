"""Extrae indices de riesgo climatico extremo por municipio desde IDEAM.

Estado de acceso a la API
--------------------------
IDEAM tiene servicios ArcGIS REST pero su disponibilidad es irregular.
El servicio mas util para riesgo de inundacion es:

    https://visor.inundaciones.ideam.gov.co/
    https://www.ideam.gov.co/web/tiempo-y-clima/cambio-climatico

Sin embargo, los servicios MapServer de IDEAM no tienen una ruta estandar
estable para consulta masiva por municipio.

COMO OBTENER LOS DATOS MANUALMENTE
-------------------------------------
Opcion 1 - SIAC (Sistema de Informacion Ambiental de Colombia):
    URL: http://www.siac.gov.co/
    Seccion: Clima → Riesgos climaticos → Inundaciones

Opcion 2 - UNGRD (Unidad Nacional de Gestion del Riesgo):
    URL: https://portal.gestiondelriesgo.gov.co/
    Dataset: Indice de Riesgo Municipal → descarga CSV con codigo_dane

Opcion 3 - IDEAM - Tercera Comunicacion Nacional de Cambio Climatico:
    URL: https://www.ideam.gov.co/web/tiempo-y-clima/cambio-climatico
    Contiene indices de vulnerabilidad por municipio

Opcion 4 - DNP - Indice Municipal de Riesgo de Desastres (IMRD):
    URL: https://www.dnp.gov.co/Paginas/Indice-Municipal-de-Riesgo-de-Desastres.aspx
    Este es el mas granular y actualizado a nivel municipal.
    Columnas: codigo_dane, imrd_inundacion, imrd_sequia, imrd_total

Modo raster
-----------
Si tienes un archivo .tif de IDEAM (p.ej. amenaza_inundacion_colombia.tif),
usa --input-raster para extraer el valor en cada centroide municipal.
El raster debe estar en EPSG:4326 o EPSG:3116.

Estructura esperada del CSV de entrada
---------------------------------------
    codigo_dane          : codigo DANE 5 digitos
    riesgo_inundacion    : indice 0-1 o categorico (Alto/Medio/Bajo)
    riesgo_sequia        : indice 0-1 o categorico (opcional)

Salida
------
    data/clean/ideam_riesgo/ideam_riesgo_municipal.csv

Columnas:
    codigo_dane           CHAR(5)
    riesgo_inundacion_idx DOUBLE  0-1, normalizado
    riesgo_sequia_idx     DOUBLE  0-1, normalizado (si disponible)
    fuente_riesgo         VARCHAR
    fecha_carga_utc       VARCHAR

Uso
---
    python -m src.extract.ideam_riesgo --input-csv data/raw/dnp/imrd_municipal.csv
    python -m src.extract.ideam_riesgo --input-raster data/raw/ideam/amenaza_inundacion.tif
"""

from __future__ import annotations

import argparse
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUNICIPALITIES_CSV = (
    PROJECT_ROOT / "data" / "clean" / "base_municipios" / "municipios_distritos_colombia.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "ideam_riesgo"

_DANE_NAMES = {"codigo_dane", "codigo_municipio", "cod_dane", "dane", "cod_mpio", "codmpio"}
_INUNDACION_NAMES = {
    "riesgo_inundacion", "riesgo_inundacion_idx", "inundacion", "flood_risk",
    "amenaza_inundacion", "indice_inundacion", "imrd_inundacion", "idx_inundacion",
}
_SEQUIA_NAMES = {
    "riesgo_sequia", "riesgo_sequia_idx", "sequia", "drought_risk",
    "amenaza_sequia", "indice_sequia", "imrd_sequia", "idx_sequia",
}

_CATEGORIA_MAP = {
    "muy alto": 1.0, "muy_alto": 1.0,
    "alto": 0.75,
    "medio": 0.50, "moderado": 0.50,
    "bajo": 0.25,
    "muy bajo": 0.0, "muy_bajo": 0.0,
    "sin riesgo": 0.0, "sin_riesgo": 0.0,
}


def normalize_col(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip().replace(" ", "_").replace("-", "_")


def detect_column(df: pd.DataFrame, candidates: set[str], label: str, required: bool = True) -> str | None:
    normalized = {col: normalize_col(col) for col in df.columns}
    for col, norm in normalized.items():
        if norm in candidates:
            return col
    if required:
        raise ValueError(
            f"No se encontro columna '{label}'. "
            f"Columnas disponibles: {list(df.columns)}. "
            f"Nombres aceptados: {sorted(candidates)}"
        )
    return None


def parse_risk_column(series: pd.Series) -> pd.Series:
    """Convierte columna de riesgo a escala 0-1.

    Acepta valores numericos directos o categoricos (Alto, Medio, Bajo).
    Aplica minmax si los valores son numericos y estan fuera del rango 0-1.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    has_numeric = numeric.notna().any()

    if has_numeric:
        min_val = numeric.min(skipna=True)
        max_val = numeric.max(skipna=True)
        if max_val > 1.0 or min_val < 0.0:
            if max_val != min_val:
                numeric = (numeric - min_val) / (max_val - min_val)
        return numeric.clip(0.0, 1.0)

    lowered = series.astype("string").str.lower().str.strip()
    return lowered.map(_CATEGORIA_MAP)


def load_input_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    dane_col = detect_column(df, _DANE_NAMES, "codigo_dane", required=True)
    inundacion_col = detect_column(df, _INUNDACION_NAMES, "riesgo_inundacion", required=True)
    sequia_col = detect_column(df, _SEQUIA_NAMES, "riesgo_sequia", required=False)

    result = pd.DataFrame()
    result["codigo_dane"] = df[dane_col].astype("string").str.strip().str.zfill(5)
    result["riesgo_inundacion_idx"] = parse_risk_column(df[inundacion_col])
    result["riesgo_sequia_idx"] = (
        parse_risk_column(df[sequia_col]) if sequia_col else pd.Series(pd.NA, index=df.index)
    )

    fuente_col = detect_column(df, {"fuente", "source"}, "fuente", required=False)
    result["fuente_riesgo"] = (
        df[fuente_col].astype("string").str.strip() if fuente_col else "IDEAM/DNP"
    )

    result = result.dropna(subset=["codigo_dane", "riesgo_inundacion_idx"])
    result = result.drop_duplicates(subset=["codigo_dane"], keep="first")
    result["fecha_carga_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def load_input_raster(raster_path: Path, municipalities_csv: Path) -> pd.DataFrame:
    """Extrae valor de riesgo desde raster IDEAM en cada centroide municipal."""
    try:
        import rasterio
        from rasterio.transform import rowcol
    except ImportError:
        raise ImportError("rasterio es necesario para --input-raster. Instala con: pip install rasterio")

    municipalities = pd.read_csv(municipalities_csv, dtype={"codigo_dane": "string"})
    municipalities["codigo_dane"] = municipalities["codigo_dane"].astype("string").str.zfill(5)

    lon_col = next((c for c in municipalities.columns if c in {"lon", "longitud"}), None)
    lat_col = next((c for c in municipalities.columns if c in {"lat", "latitud"}), None)
    if not lon_col or not lat_col:
        raise ValueError("CSV de municipios sin columnas lon/lat.")

    rows_data = []
    with rasterio.open(raster_path) as src:
        for _, muni_row in municipalities.iterrows():
            try:
                row_idx, col_idx = rowcol(src.transform, muni_row[lon_col], muni_row[lat_col])
                value = src.read(1)[row_idx, col_idx]
                nodata = src.nodata
                if nodata is not None and float(value) == float(nodata):
                    value = None
                else:
                    value = float(value)
            except Exception:  # noqa: BLE001
                value = None
            rows_data.append({"codigo_dane": str(muni_row["codigo_dane"]), "riesgo_inundacion_raw": value})

    result = pd.DataFrame(rows_data)
    result["riesgo_inundacion_idx"] = parse_risk_column(result["riesgo_inundacion_raw"])
    result["riesgo_sequia_idx"] = pd.NA
    result["fuente_riesgo"] = f"IDEAM raster: {raster_path.name}"
    result["fecha_carga_utc"] = datetime.now(timezone.utc).isoformat()
    return result[["codigo_dane", "riesgo_inundacion_idx", "riesgo_sequia_idx", "fuente_riesgo", "fecha_carga_utc"]]


def write_observations(path: Path, n: int, fuente_path: str) -> None:
    lines = [
        "IDEAM/DNP - Riesgo climatico extremo municipal",
        "================================================",
        "",
        f"Registros cargados: {n}",
        f"Archivo fuente: {fuente_path}",
        "",
        "Fuentes de datos requeridas (descarga manual):",
        "  DNP - Indice Municipal de Riesgo de Desastres (IMRD) [RECOMENDADA]:",
        "    https://www.dnp.gov.co/Paginas/Indice-Municipal-de-Riesgo-de-Desastres.aspx",
        "    Columnas: codigo_dane, imrd_inundacion, imrd_sequia, imrd_total",
        "",
        "  SIAC - Sistema de Informacion Ambiental:",
        "    http://www.siac.gov.co/  →  Clima → Riesgos",
        "",
        "  UNGRD - Unidad Nacional de Gestion del Riesgo:",
        "    https://portal.gestiondelriesgo.gov.co/",
        "",
        "  IDEAM - Cambio Climatico (rasters amenaza):",
        "    https://www.ideam.gov.co/web/tiempo-y-clima/cambio-climatico",
        "    Usa --input-raster si tienes el .tif de amenaza.",
        "",
        "Uso en scoring multidimensional:",
        "  score_riesgo: riesgo_inundacion_idx (inverso; mayor riesgo = menor score).",
        "  Municipios con alta amenaza de inundacion no son adecuados para infraestructura solar.",
        "",
        "Limitaciones:",
        "  - Los indices IDEAM/DNP son amenaza, no riesgo final (no incluyen exposicion/vulnerabilidad).",
        "  - La escala de tiempo es climatica; no captura eventos puntuales.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_ideam_riesgo(
    input_csv: Path | None = None,
    input_raster: Path | None = None,
    municipalities_csv: Path = DEFAULT_MUNICIPALITIES_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    if input_csv is None and input_raster is None:
        raise ValueError(
            "Se requiere --input-csv o --input-raster.\n"
            "Descarga el IMRD desde:\n"
            "  https://www.dnp.gov.co/Paginas/Indice-Municipal-de-Riesgo-de-Desastres.aspx\n"
            "y ejecuta: python -m src.extract.ideam_riesgo --input-csv ruta/imrd.csv"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    if input_raster is not None:
        df = load_input_raster(input_raster, municipalities_csv)
        fuente_str = str(input_raster)
    else:
        if not input_csv.exists():  # type: ignore[union-attr]
            raise FileNotFoundError(f"No existe: {input_csv}")
        df = load_input_csv(input_csv)  # type: ignore[arg-type]
        fuente_str = str(input_csv)

    output_path = output_dir / "ideam_riesgo_municipal.csv"
    observations_path = output_dir / "ideam_riesgo_observaciones.txt"

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    write_observations(observations_path, len(df), fuente_str)

    return {
        "riesgo": output_path,
        "observaciones": observations_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Carga indices de riesgo climatico IDEAM/DNP desde CSV o raster."
    )
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--input-raster", type=Path, default=None)
    parser.add_argument("--municipalities-csv", type=Path, default=DEFAULT_MUNICIPALITIES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_ideam_riesgo(
            input_csv=args.input_csv,
            input_raster=args.input_raster,
            municipalities_csv=args.municipalities_csv.resolve(),
            output_dir=args.output_dir.resolve(),
        )
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {error}")
        return 1

    print("Riesgo climatico IDEAM/DNP cargado.")
    for label, path in outputs.items():
        print(f"  - {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
