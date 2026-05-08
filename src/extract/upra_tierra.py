"""Extrae precio de tierra rural por municipio desde UPRA.

Estado de acceso a la API
--------------------------
UPRA (Unidad de Planificacion Rural Agropecuaria) tiene un geoportal con datos
de valor de tierras rurales, PERO no expone una API REST publica estable de
consulta masiva por municipio. El acceso programatico es inestable.

COMO OBTENER LOS DATOS MANUALMENTE
-------------------------------------
Opcion 1 - Visor UPRA (recomendada):
    URL: https://visor.upra.gov.co/
    Seccion: "Mercado de Tierras" → "Valor de la Tierra"
    Descarga: tabla municipalizada con precio promedio COP/ha
    Formato esperado: Excel o CSV con columnas municipio / codigo_dane / precio

Opcion 2 - SIPRA (Sistema de Informacion para la Planificacion Rural):
    URL: https://sipra.upra.gov.co/
    Dataset: "Valor comercial de la tierra rural"

Opcion 3 - Datos Abiertos Colombia:
    URL: https://www.datos.gov.co/
    Buscar: "valor tierra rural UPRA" o "precio tierra municipio"

Estructura esperada del CSV de entrada
---------------------------------------
El archivo debe tener al menos estas columnas (nombres flexibles):
    codigo_dane   : codigo DANE 5 digitos del municipio
    precio_cop_ha : precio promedio tierra rural (COP por hectarea)

Columnas opcionales reconocidas:
    fuente, anio, departamento, municipio

Salida
------
    data/clean/upra_tierra/upra_precio_tierra_municipal.csv

Columnas:
    codigo_dane              CHAR(5)
    precio_tierra_cop_ha     DOUBLE   precio promedio COP/ha
    fuente_precio_tierra     VARCHAR  UPRA, proxy, manual
    anio_referencia_precio   INT
    fecha_carga_utc          VARCHAR

Uso
---
    python -m src.extract.upra_tierra --input-csv ruta/al/archivo.csv
    python -m src.extract.upra_tierra --input-csv data/raw/upra/valor_tierra.csv --anio 2023
"""

from __future__ import annotations

import argparse
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "upra_tierra"

_LON_NAMES = {"codigo_dane", "codigo_municipio", "cod_dane", "dane", "cod_mpio", "codmpio"}
_PRECIO_NAMES = {
    "precio_cop_ha", "precio_tierra_cop_ha", "precio_cop", "valor_cop_ha",
    "valor_tierra", "precio_ha", "valor_ha", "precio_promedio_cop_ha",
    "vlr_ha", "valor_comercial_cop_ha",
}


def normalize_col(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip().replace(" ", "_").replace("-", "_")


def detect_column(df: pd.DataFrame, candidates: set[str], label: str) -> str:
    normalized = {col: normalize_col(col) for col in df.columns}
    for col, norm in normalized.items():
        if norm in candidates:
            return col
    raise ValueError(
        f"No se encontro columna '{label}' en el CSV. "
        f"Columnas disponibles: {list(df.columns)}. "
        f"Nombres aceptados: {sorted(candidates)}"
    )


def load_input_csv(path: Path, anio: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    dane_col = detect_column(df, _LON_NAMES, "codigo_dane")
    precio_col = detect_column(df, _PRECIO_NAMES, "precio_cop_ha")

    result = pd.DataFrame()
    result["codigo_dane"] = df[dane_col].astype("string").str.strip().str.zfill(5)
    result["precio_tierra_cop_ha"] = pd.to_numeric(df[precio_col], errors="coerce")

    anio_col_candidates = {"anio", "year", "año", "ano", "periodo", "period"}
    try:
        anio_col = detect_column(df, anio_col_candidates, "anio")
        result["anio_referencia_precio"] = pd.to_numeric(df[anio_col], errors="coerce").astype("Int64")
    except ValueError:
        result["anio_referencia_precio"] = anio if anio is not None else pd.NA

    fuente_candidates = {"fuente", "source", "origen"}
    try:
        fuente_col = detect_column(df, fuente_candidates, "fuente")
        result["fuente_precio_tierra"] = df[fuente_col].astype("string").str.strip()
    except ValueError:
        result["fuente_precio_tierra"] = "UPRA"

    result = result.dropna(subset=["codigo_dane", "precio_tierra_cop_ha"])
    result = result.drop_duplicates(subset=["codigo_dane"], keep="first")
    result["fecha_carga_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def write_observations(path: Path, n: int, fuente_path: str) -> None:
    lines = [
        "UPRA - Precio tierra rural municipal",
        "=====================================",
        "",
        f"Registros cargados: {n}",
        f"Archivo fuente: {fuente_path}",
        "",
        "Fuente de datos requerida (descarga manual):",
        "  Opcion 1: https://visor.upra.gov.co/  →  Mercado de Tierras → Valor de la Tierra",
        "  Opcion 2: https://sipra.upra.gov.co/  →  Valor comercial tierra rural",
        "  Opcion 3: https://www.datos.gov.co/  →  buscar 'valor tierra rural UPRA'",
        "",
        "Uso en scoring multidimensional:",
        "  score_economico: precio_tierra_cop_ha (inverso; tierra barata = mayor score).",
        "  score_agropecuario: costo_oportunidad = precio_tierra x productividad ganadera.",
        "",
        "Limitaciones:",
        "  - El precio de tierra rural varia dentro del municipio; este es un promedio.",
        "  - Los datos UPRA tienen rezago de 1-3 anos tipicamente.",
        "  - No incluye costos de servidumbre ni negociacion predial.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_upra_tierra(
    input_csv: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    anio: int | None = None,
) -> dict[str, Path]:
    if not input_csv.exists():
        raise FileNotFoundError(
            f"No existe el archivo de entrada: {input_csv}\n"
            "Descarga los datos de precio de tierra desde:\n"
            "  https://visor.upra.gov.co/  (Mercado de Tierras → Valor de la Tierra)\n"
            "y ejecuta: python -m src.extract.upra_tierra --input-csv ruta/archivo.csv"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_input_csv(input_csv, anio)

    output_path = output_dir / "upra_precio_tierra_municipal.csv"
    observations_path = output_dir / "upra_tierra_observaciones.txt"

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    write_observations(observations_path, len(df), str(input_csv))

    return {
        "precio_tierra": output_path,
        "observaciones": observations_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Carga precio de tierra rural UPRA desde CSV descargado manualmente."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help=(
            "CSV con columnas codigo_dane y precio_cop_ha. "
            "Descarga desde https://visor.upra.gov.co/ o https://sipra.upra.gov.co/"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anio", type=int, default=None, help="Anio de referencia del dato.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_upra_tierra(
            input_csv=args.input_csv.resolve(),
            output_dir=args.output_dir.resolve(),
            anio=args.anio,
        )
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {error}")
        return 1

    print("UPRA precio tierra cargado.")
    for label, path in outputs.items():
        print(f"  - {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
