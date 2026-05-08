"""Carga costo del agua por municipio desde SUI (Superintendencia de Servicios Publicos).

Estado de acceso a la API
--------------------------
SUI NO tiene una API REST publica para consulta de tarifas por municipio.
Los datos solo son accesibles via descarga manual desde el portal web.

COMO OBTENER LOS DATOS MANUALMENTE
-------------------------------------
Paso 1 - Ingresar al portal SUI:
    URL: https://sui.superservicios.gov.co/

Paso 2 - Seccion de informacion de acueducto:
    Navegar a: Acueducto → Tarifas → Reporte de Tarifas
    O buscar: "Tarifas acueducto municipio"

Paso 3 - Exportar CSV o Excel con:
    - Municipio / Codigo DANE
    - Estrato 1-6
    - Costo fijo y costo variable (COP/m3)
    - Operador del servicio

Paso 4 - Calcular promedio municipal:
    El script acepta el CSV bruto de SUI y calcula el promedio ponderado
    de tarifas variables (COP/m3) por municipio, usando todos los estratos.

Alternativa simplificada
-------------------------
Si solo tienes datos de una muestra de municipios, el scoring usara esos
datos donde esten disponibles y usara la mediana nacional como proxy
donde no haya datos.

Estructura esperada del CSV de entrada
---------------------------------------
    codigo_dane     : codigo DANE 5 digitos
    tarifa_cop_m3   : precio variable del agua (COP por m3)

Columnas opcionales:
    estrato, operador, municipio, departamento, anio

Si el CSV tiene datos por estrato, el script calcula el promedio por municipio.

Salida
------
    data/clean/sui_agua/sui_costo_agua_municipal.csv

Columnas:
    codigo_dane         CHAR(5)
    costo_agua_cop_m3   DOUBLE   promedio tarifa variable COP/m3
    fuente_agua         VARCHAR
    anio_referencia     INT
    cobertura_estratos  INT      cuantos estratos se promediaron
    fecha_carga_utc     VARCHAR

Uso
---
    python -m src.extract.sui_agua --input-csv data/raw/sui/tarifas_acueducto.csv
    python -m src.extract.sui_agua --input-csv ruta.csv --anio 2023
"""

from __future__ import annotations

import argparse
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "sui_agua"

_DANE_NAMES = {"codigo_dane", "cod_dane", "codigo_municipio", "cod_mpio", "dane", "codmpio"}
_TARIFA_NAMES = {
    "tarifa_cop_m3", "costo_cop_m3", "precio_m3", "cargo_variable_cop_m3",
    "tarifa_variable", "precio_agua_m3", "valor_m3", "cvu_cop_m3",
    "cargo_variable", "tarifa_variable_cop_m3",
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


def load_input_csv(path: Path, anio: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    dane_col = detect_column(df, _DANE_NAMES, "codigo_dane", required=True)
    tarifa_col = detect_column(df, _TARIFA_NAMES, "tarifa_cop_m3", required=True)

    df_work = pd.DataFrame()
    df_work["codigo_dane"] = df[dane_col].astype("string").str.strip().str.zfill(5)
    df_work["tarifa_cop_m3"] = pd.to_numeric(df[tarifa_col], errors="coerce")

    estrato_col = detect_column(df, {"estrato", "estrato_socioeconomico", "stratum"}, "estrato", required=False)
    anio_col = detect_column(df, {"anio", "year", "ano", "periodo"}, "anio", required=False)
    fuente_col = detect_column(df, {"fuente", "operador", "prestador"}, "fuente", required=False)

    if estrato_col:
        df_work["estrato"] = pd.to_numeric(df[estrato_col], errors="coerce").astype("Int64")

    if anio_col:
        df_work["anio_referencia"] = pd.to_numeric(df[anio_col], errors="coerce").astype("Int64")
    else:
        df_work["anio_referencia"] = anio if anio is not None else pd.NA

    if fuente_col:
        df_work["fuente"] = df[fuente_col].astype("string").str.strip()

    df_work = df_work.dropna(subset=["codigo_dane", "tarifa_cop_m3"])

    agg_dict: dict[str, Any] = {
        "tarifa_cop_m3": "mean",
        "anio_referencia": "first",
    }
    if "estrato" in df_work.columns:
        agg_dict["estrato"] = "count"

    result = df_work.groupby("codigo_dane").agg(agg_dict).reset_index()
    result = result.rename(columns={
        "tarifa_cop_m3": "costo_agua_cop_m3",
        "estrato": "cobertura_estratos",
    })

    if "cobertura_estratos" not in result.columns:
        result["cobertura_estratos"] = df_work.groupby("codigo_dane").size().reset_index(name="n")["n"]

    result["fuente_agua"] = "SUI"
    result["fecha_carga_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def write_observations(path: Path, n: int, fuente_path: str) -> None:
    lines = [
        "SUI - Costo del agua municipal",
        "================================",
        "",
        f"Registros cargados: {n}",
        f"Archivo fuente: {fuente_path}",
        "",
        "Fuente de datos (descarga manual obligatoria):",
        "  Portal SUI - Superintendencia de Servicios Publicos Domiciliarios:",
        "    URL: https://sui.superservicios.gov.co/",
        "    Seccion: Acueducto → Tarifas → Reporte de Tarifas",
        "",
        "  Campos requeridos del reporte SUI:",
        "    - Codigo DANE o nombre del municipio",
        "    - Cargo variable (COP/m3) por estrato",
        "",
        "Uso en scoring multidimensional:",
        "  score_economico: costo_agua_cop_m3 (inverso; agua mas cara = menor score).",
        "  Variable de menor impacto que precio tierra o distancia a red.",
        "  Los paneles solares consumen agua principalmente para limpieza:",
        "    - Zonas lluviosas: 1-3 m3/ha/mes",
        "    - Zonas aridas: 5-15 m3/ha/mes",
        "",
        "Limitaciones:",
        "  - SUI reporta tarifas por estrato; el promedio puede no ser representativo",
        "    del costo real de un proyecto industrial (que usa tarifa no residencial).",
        "  - La cobertura municipal de SUI no es del 100%.",
        "  - Para proyectos con captacion propia (pozo, quebrada), este costo no aplica.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_sui_agua(
    input_csv: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    anio: int | None = None,
) -> dict[str, Path]:
    if not input_csv.exists():
        raise FileNotFoundError(
            f"No existe: {input_csv}\n"
            "Descarga los datos de tarifas de acueducto desde:\n"
            "  https://sui.superservicios.gov.co/  →  Acueducto → Tarifas\n"
            "y ejecuta: python -m src.extract.sui_agua --input-csv ruta/tarifas.csv"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_input_csv(input_csv, anio)

    output_path = output_dir / "sui_costo_agua_municipal.csv"
    observations_path = output_dir / "sui_agua_observaciones.txt"

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    write_observations(observations_path, len(df), str(input_csv))

    return {
        "costo_agua": output_path,
        "observaciones": observations_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Carga tarifas de acueducto SUI desde CSV descargado manualmente."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV con codigo_dane y tarifa_cop_m3. Descarga desde https://sui.superservicios.gov.co/",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anio", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_sui_agua(
            input_csv=args.input_csv.resolve(),
            output_dir=args.output_dir.resolve(),
            anio=args.anio,
        )
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {error}")
        return 1

    print("SUI costo agua cargado.")
    for label, path in outputs.items():
        print(f"  - {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
