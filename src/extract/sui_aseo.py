"""Extrae costo de servicio de aseo (SUI) como proxy de infraestructura economica.

Fuente de datos
---------------
SUI - Sistema Unico de Informacion de Servicios Publicos (Superservicios).
Archivo: Variable-calculo-tarifa-a-corte-de-marzo-2026_0.xlsx
Descarga: https://sui.superservicios.gov.co/ -> Aseo -> Variables Tarifa

Nota: Este archivo contiene tarifas de ASEO (residuos solidos), NO de agua potable.
Para tarifas de agua potable usar sui_agua.py con datos del componente acueducto.

Variables del archivo
----------------------
    DEPARTAMENTO                     : nombre del departamento
    MUNICIPIO                        : nombre del municipio (texto, no codigo)
    AÑO CARGUE                       : año de la tarifa
    PERIODO CARGUE                   : mes/periodo
    COSTO DE RECOLECCION Y TRANSPORTE: costo COP por tonelada
    COSTO DE DISPOSICION FINAL       : costo COP por tonelada
    TONELADAS DE RESIDUOS            : total mensual
    NUMERO DE SUSCRIPTORES ATENDIDOS : suscriptores del servicio

Uso en el pipeline
------------------
El costo de aseo es un proxy de la calidad y cobertura de infraestructura
de servicios publicos en el municipio. Municipios con costos mas bajos
por tonelada suelen tener mejor infraestructura o economias de escala,
lo que se correlaciona positivamente con el entorno para inversiones.

    costo_aseo_cop_ton: costo total (recoleccion + disposicion) por tonelada
    suscriptores_aseo:  numero de suscriptores (proxy de poblacion urbana servida)

Nota de join
------------
El archivo no tiene codigo DANE. El join se hace por departamento + municipio
(texto normalizado). Los municipios sin match se registran como NA.

Salida
------
    data/clean/sui_aseo/sui_aseo_municipal.csv

Columnas:
    codigo_dane         CHAR(5)
    costo_aseo_cop_ton  DOUBLE   costo total aseo (recoleccion + disposicion)
    suscriptores_aseo   INT      suscriptores en el periodo mas reciente
    anio_referencia     INT
    fuente              VARCHAR

Uso
---
    python -m src.extract.sui_aseo
    python -m src.extract.sui_aseo --input-xlsx ruta/al/archivo.xlsx
    python -m src.extract.sui_aseo --force
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_XLSX = (
    PROJECT_ROOT / "data" / "raw" / "Variable-calculo-tarifa-a-corte-de-marzo-2026_0.xlsx"
)
DEFAULT_MUNICIPALITIES_CSV = (
    PROJECT_ROOT / "data" / "clean" / "base_municipios" / "municipios_distritos_colombia.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "sui_aseo"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()


def _detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normed = {_norm(c): c for c in df.columns}
    for cand in candidates:
        if _norm(cand) in normed:
            return normed[_norm(cand)]
    return None


def _read_tarifa(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sheet = xl.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    # Limpiar columnas con caracteres extraños en encoding
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _aggregate_municipal(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega tarifas al nivel municipal: promedio del anio mas reciente disponible."""
    depto_col = _detect_col(df, ["DEPARTAMENTO", "Departamento"])
    mun_col = _detect_col(df, ["MUNICIPIO", "Municipio"])
    anio_col = _detect_col(df, ["AÑO CARGUE", "ANO CARGUE", "Año Cargue", "ANIO CARGUE"])
    rec_col = _detect_col(df, [
        "COSTO DE RECOLECCION  Y TRANSPORTE",
        "COSTO DE RECOLECCION Y TRANSPORTE",
        "Costo Recoleccion Transporte",
    ])
    disp_col = _detect_col(df, [
        "COSTO DE DISPOSICION FINAL",
        "Costo Disposicion Final",
    ])
    ton_col = _detect_col(df, ["TONELADAS DE RESIDUOS", "Toneladas"])
    sus_col = _detect_col(df, [
        "NUMERO DE SUSCRIPTORES ATENDIDOS",
        "Suscriptores Atendidos",
    ])

    if not depto_col or not mun_col:
        raise ValueError(
            f"No se encontraron columnas DEPARTAMENTO/MUNICIPIO.\n"
            f"Columnas: {list(df.columns)[:10]}"
        )

    df = df.copy()
    df["_depto_norm"] = df[depto_col].apply(_norm)
    df["_mun_norm"] = df[mun_col].apply(_norm)
    df["_anio"] = pd.to_numeric(df[anio_col], errors="coerce") if anio_col else 0

    for col in [rec_col, disp_col, ton_col, sus_col]:
        if col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Costo total = recoleccion + disposicion
    rec = df[rec_col].fillna(0) if rec_col else pd.Series(0, index=df.index)
    disp = df[disp_col].fillna(0) if disp_col else pd.Series(0, index=df.index)
    df["_costo_total"] = rec + disp

    # Usar el anio mas reciente por municipio
    latest = df.sort_values("_anio", ascending=False).groupby(
        ["_depto_norm", "_mun_norm"], as_index=False
    ).first()

    result = pd.DataFrame()
    result["_depto_norm"] = latest["_depto_norm"]
    result["_mun_norm"] = latest["_mun_norm"]
    result["costo_aseo_cop_ton"] = latest["_costo_total"]
    result["suscriptores_aseo"] = latest[sus_col].astype("Int64") if sus_col else pd.NA
    result["anio_referencia"] = latest["_anio"].astype("Int64")

    return result


def _join_dane(aggregated: pd.DataFrame, municipalities_csv: Path) -> pd.DataFrame:
    """Une con tabla de municipios por nombre normalizado."""
    munis = pd.read_csv(municipalities_csv, dtype={"codigo_dane": "string"})
    munis["codigo_dane"] = munis["codigo_dane"].str.zfill(5)
    munis["_depto_norm"] = munis["departamento"].apply(_norm)
    munis["_mun_norm"] = munis["municipio"].apply(_norm)

    merged = aggregated.merge(
        munis[["codigo_dane", "_depto_norm", "_mun_norm"]],
        on=["_depto_norm", "_mun_norm"],
        how="left",
    )

    n_match = merged["codigo_dane"].notna().sum()
    n_total = len(merged)
    print(f"  Join por nombre: {n_match}/{n_total} municipios con codigo_dane")

    result = merged[merged["codigo_dane"].notna()].copy()
    result = result[["codigo_dane", "costo_aseo_cop_ton", "suscriptores_aseo", "anio_referencia"]].copy()
    result["fuente"] = "SUI Superservicios - Variables Tarifa Aseo"
    return result


def run_sui_aseo(
    input_xlsx: Path = DEFAULT_INPUT_XLSX,
    municipalities_csv: Path = DEFAULT_MUNICIPALITIES_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Path]:
    """Lee el Excel de tarifas de aseo SUI y exporta CSV municipal."""

    output_path = output_dir / "sui_aseo_municipal.csv"
    if output_path.exists() and not force:
        print(f"  SUI Aseo: usando resultado previo {output_path}")
        return {"summary": output_path}

    if not input_xlsx.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo SUI Aseo en: {input_xlsx}\n"
            "Descarga desde: https://sui.superservicios.gov.co/ -> Aseo -> Variables Tarifa"
        )

    print(f"  Leyendo tarifa aseo: {input_xlsx}")
    df = _read_tarifa(input_xlsx)
    print(f"  Registros crudos: {len(df)}")

    aggregated = _aggregate_municipal(df)
    print(f"  Municipios unicos: {len(aggregated)}")

    result = _join_dane(aggregated, municipalities_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"  SUI Aseo completado: {len(result)} municipios -> {output_path}")
    return {"summary": output_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte tarifas SUI Aseo a CSV municipal para el pipeline."
    )
    parser.add_argument("--input-xlsx", type=Path, default=DEFAULT_INPUT_XLSX)
    parser.add_argument("--municipalities-csv", type=Path, default=DEFAULT_MUNICIPALITIES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_sui_aseo(
            input_xlsx=args.input_xlsx.resolve(),
            municipalities_csv=args.municipalities_csv.resolve(),
            output_dir=args.output_dir.resolve(),
            force=args.force,
        )
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {error}")
        return 1

    print("SUI Aseo completado.")
    for label, path in outputs.items():
        print(f"  - {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
