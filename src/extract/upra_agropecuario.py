"""Carga productividad agropecuaria municipal (carga bovina y pasto).

Estado de acceso a la API
--------------------------
No existe una API REST publica y estable para datos de carga bovina o
productividad de pastos a nivel municipal en Colombia. Las fuentes disponibles
requieren descarga manual:

COMO OBTENER LOS DATOS MANUALMENTE
-------------------------------------
Opcion 1 - EVA (Evaluaciones Agropecuarias Municipales) - MADR/AGRONET:
    URL: https://www.agronet.gov.co/estadistica/Paginas/home.aspx?cod=1
    Dataset: "Produccion agricola y pecuaria municipal"
    Seccion: "Pecuaria" → "Bovinos" → por municipio
    Columnas utiles: municipio, codigo_dane, inventario_bovinos, area_pasturas_ha
    Carga bovina = inventario_bovinos / area_pasturas_ha

Opcion 2 - UPRA Zonificacion Agropecuaria:
    URL: https://visor.upra.gov.co/
    Seccion: "Uso adecuado del suelo" → "Ganaderia"
    Exportar tabla municipal con capacidad de carga animal

Opcion 3 - FEDEGAN (datos agregados por departamento, menos granulares):
    URL: https://www.fedegan.org.co/estadisticas/inventario-ganadero

Opcion 4 - DANE Encuesta Nacional Agropecuaria (ENA):
    URL: https://www.dane.gov.co/index.php/estadisticas-por-tema/agropecuario/
    Nota: la ENA no siempre tiene nivel municipal.

Estructura esperada del CSV de entrada
---------------------------------------
El archivo debe tener al menos:
    codigo_dane       : codigo DANE 5 digitos
    carga_bovina      : unidades animales por hectarea (UA/ha o cabezas/ha)

Columnas opcionales reconocidas:
    productividad_pasto_ton_ha, inventario_bovinos, area_pasturas_ha,
    fuente, anio, departamento, municipio

Si solo tienes inventario_bovinos y area_pasturas_ha (EVA), el script
calcula la carga bovina automaticamente.

Salida
------
    data/clean/upra_agropecuario/upra_agropecuario_municipal.csv

Columnas:
    codigo_dane                 CHAR(5)
    carga_bovina_ua_ha          DOUBLE  unidades animales / ha de pastura
    productividad_pasto_ton_ha  DOUBLE  rendimiento de pasto (si disponible)
    inventario_bovinos          INT     cabezas de ganado (si disponible)
    area_pasturas_ha            DOUBLE  area en pasturas (si disponible)
    fuente_agropecuaria         VARCHAR
    anio_referencia_agro        INT
    fecha_carga_utc             VARCHAR

Uso
---
    python -m src.extract.upra_agropecuario --input-csv data/raw/eva/pecuario_municipal.csv
    python -m src.extract.upra_agropecuario --input-csv ruta.csv --anio 2022
"""

from __future__ import annotations

import argparse
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "upra_agropecuario"

_DANE_NAMES = {"codigo_dane", "codigo_municipio", "cod_dane", "dane", "cod_mpio", "codmpio"}
_CARGA_NAMES = {
    "carga_bovina", "carga_bovina_ua_ha", "ua_ha", "cabezas_ha",
    "unidades_animales_ha", "carga_animal_ua_ha", "capacidad_carga",
}
_INVENTARIO_NAMES = {
    "inventario_bovinos", "bovinos", "cabezas_ganado", "total_bovinos",
    "inventario_ganado", "num_bovinos",
}
_AREA_PASTO_NAMES = {
    "area_pasturas_ha", "area_pasto_ha", "hectareas_pasto", "pasturas_ha",
    "area_pastizales_ha", "ha_pasto",
}
_PRODUCTIVIDAD_NAMES = {
    "productividad_pasto_ton_ha", "rendimiento_pasto", "ton_ha_pasto",
    "produccion_pasto_ton_ha",
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
            f"No se encontro columna '{label}' en el CSV. "
            f"Columnas disponibles: {list(df.columns)}. "
            f"Nombres aceptados: {sorted(candidates)}"
        )
    return None


def load_input_csv(path: Path, anio: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    dane_col = detect_column(df, _DANE_NAMES, "codigo_dane", required=True)

    result = pd.DataFrame()
    result["codigo_dane"] = df[dane_col].astype("string").str.strip().str.zfill(5)

    carga_col = detect_column(df, _CARGA_NAMES, "carga_bovina", required=False)
    inventario_col = detect_column(df, _INVENTARIO_NAMES, "inventario_bovinos", required=False)
    area_col = detect_column(df, _AREA_PASTO_NAMES, "area_pasturas_ha", required=False)

    if carga_col is not None:
        result["carga_bovina_ua_ha"] = pd.to_numeric(df[carga_col], errors="coerce")
    elif inventario_col is not None and area_col is not None:
        inventario = pd.to_numeric(df[inventario_col], errors="coerce")
        area = pd.to_numeric(df[area_col], errors="coerce")
        result["carga_bovina_ua_ha"] = (inventario / area).where(area > 0)
        result["inventario_bovinos"] = inventario.astype("Int64")
        result["area_pasturas_ha"] = area
    else:
        raise ValueError(
            "El CSV debe tener columna 'carga_bovina' (UA/ha) o bien "
            "'inventario_bovinos' + 'area_pasturas_ha' para calcularla."
        )

    if inventario_col is not None and "inventario_bovinos" not in result.columns:
        result["inventario_bovinos"] = pd.to_numeric(df[inventario_col], errors="coerce").astype("Int64")
    if area_col is not None and "area_pasturas_ha" not in result.columns:
        result["area_pasturas_ha"] = pd.to_numeric(df[area_col], errors="coerce")

    prod_col = detect_column(df, _PRODUCTIVIDAD_NAMES, "productividad_pasto", required=False)
    result["productividad_pasto_ton_ha"] = (
        pd.to_numeric(df[prod_col], errors="coerce") if prod_col else pd.NA
    )

    anio_col = detect_column(df, {"anio", "year", "ano", "periodo"}, "anio", required=False)
    result["anio_referencia_agro"] = (
        pd.to_numeric(df[anio_col], errors="coerce").astype("Int64")
        if anio_col else (anio if anio is not None else pd.NA)
    )

    fuente_col = detect_column(df, {"fuente", "source", "origen"}, "fuente", required=False)
    result["fuente_agropecuaria"] = (
        df[fuente_col].astype("string").str.strip() if fuente_col else "EVA-MADR"
    )

    result = result.dropna(subset=["codigo_dane", "carga_bovina_ua_ha"])
    result = result.drop_duplicates(subset=["codigo_dane"], keep="first")
    result["fecha_carga_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def write_observations(path: Path, n: int, fuente_path: str) -> None:
    lines = [
        "UPRA/EVA - Productividad agropecuaria municipal",
        "================================================",
        "",
        f"Registros cargados: {n}",
        f"Archivo fuente: {fuente_path}",
        "",
        "Fuentes de datos requeridas (descarga manual):",
        "  EVA-MADR: https://www.agronet.gov.co/estadistica/Paginas/home.aspx?cod=1",
        "    Seccion: Pecuaria → Bovinos → por municipio",
        "    Columnas necesarias: codigo_dane, inventario_bovinos, area_pasturas_ha",
        "",
        "  UPRA ganaderia: https://visor.upra.gov.co/",
        "    Seccion: Uso adecuado del suelo → Ganaderia",
        "",
        "  FEDEGAN (departamental): https://www.fedegan.org.co/estadisticas/inventario-ganadero",
        "",
        "Uso en scoring multidimensional:",
        "  score_agropecuario:",
        "    carga_bovina_ua_ha alta → mayor costo de oportunidad → mas defensible agrovoltaico.",
        "    La variable se usa INVERSAMENTE: municipio con alta carga ganadera tiene mayor",
        "    oportunidad de combinar ganaderia bajo paneles (agrovoltaica).",
        "",
        "  costo_oportunidad_agro = precio_tierra_cop_ha × carga_bovina_ua_ha",
        "    (calcula cuanto vale producir carne vs. producir energia en ese municipio).",
        "",
        "Limitaciones:",
        "  - EVA tiene cobertura variable; no todos los municipios tienen datos en todos los anios.",
        "  - La carga bovina calculada (inventario/area) asume pasturas homogeneas.",
        "  - No distingue entre ganaderia extensiva, semi-intensiva e intensiva.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_upra_agropecuario(
    input_csv: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    anio: int | None = None,
) -> dict[str, Path]:
    if not input_csv.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {input_csv}\n"
            "Descarga los datos EVA desde:\n"
            "  https://www.agronet.gov.co/estadistica/Paginas/home.aspx?cod=1\n"
            "  Seccion: Pecuaria → Bovinos\n"
            "y ejecuta: python -m src.extract.upra_agropecuario --input-csv ruta/archivo.csv"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_input_csv(input_csv, anio)

    output_path = output_dir / "upra_agropecuario_municipal.csv"
    observations_path = output_dir / "upra_agropecuario_observaciones.txt"

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    write_observations(observations_path, len(df), str(input_csv))

    return {
        "agropecuario": output_path,
        "observaciones": observations_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Carga productividad agropecuaria municipal desde EVA/UPRA."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV con codigo_dane y carga_bovina (o inventario_bovinos + area_pasturas_ha).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anio", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_upra_agropecuario(
            input_csv=args.input_csv.resolve(),
            output_dir=args.output_dir.resolve(),
            anio=args.anio,
        )
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {error}")
        return 1

    print("Productividad agropecuaria cargada.")
    for label, path in outputs.items():
        print(f"  - {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
