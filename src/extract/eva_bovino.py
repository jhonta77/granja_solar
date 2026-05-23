"""Convierte la base pecuaria EVA/ICA a inventario bovino municipal.

Entrada esperada:
    data/raw/BasePecuaria20192023 (1).xlsx

Salida:
    data/clean/upra_agropecuario/eva_inventario_bovino_municipal.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_XLSX = PROJECT_ROOT / "data" / "raw" / "BasePecuaria20192023 (1).xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "upra_agropecuario"


def run_eva_bovino(
    input_xlsx: Path = DEFAULT_INPUT_XLSX,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    year: int | None = None,
) -> dict[str, Path]:
    if not input_xlsx.exists():
        raise FileNotFoundError(f"No existe la base pecuaria EVA/ICA: {input_xlsx}")

    df = pd.read_excel(input_xlsx, sheet_name="InvBovino", header=3, dtype=str)
    code_col = next(c for c in df.columns if "municipio" in c.lower() and "dane" in c.lower())
    year_col = next(c for c in df.columns if c.lower().startswith("a"))
    total_col = next(c for c in df.columns if "total bovinos" in c.lower())
    farms_col = next((c for c in df.columns if "fincas" in c.lower()), None)
    mun_col = next(c for c in df.columns if c.lower() == "municipio")
    dep_col = next(c for c in df.columns if c.lower() == "departamento")

    df["_year"] = pd.to_numeric(df[year_col], errors="coerce")
    selected_year = int(year if year is not None else df["_year"].max())
    latest = df[df["_year"].eq(selected_year)].copy()

    result = pd.DataFrame(
        {
            "codigo_dane": latest[code_col].astype("string").str.strip().str.zfill(5),
            "departamento": latest[dep_col].astype("string").str.strip(),
            "municipio": latest[mun_col].astype("string").str.strip(),
            "inventario_bovinos": pd.to_numeric(latest[total_col], errors="coerce").astype("Int64"),
            "fincas_con_bovinos": (
                pd.to_numeric(latest[farms_col], errors="coerce").astype("Int64")
                if farms_col
                else pd.NA
            ),
            "anio_referencia_agro": selected_year,
            "fuente_agropecuaria": "EVA-MADR/ICA Base Pecuaria Inventario Bovino",
        }
    )
    result = result.dropna(subset=["codigo_dane", "inventario_bovinos"]).drop_duplicates("codigo_dane")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eva_inventario_bovino_municipal.csv"
    observations_path = output_dir / "eva_inventario_bovino_observaciones.txt"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    observations_path.write_text(
        "\n".join(
            [
                "EVA/MADR - Inventario bovino municipal",
                "=======================================",
                "",
                f"Archivo fuente: {input_xlsx}",
                f"Anio usado: {selected_year}",
                f"Municipios con inventario bovino: {len(result)}",
                "",
                "Este archivo contiene inventario bovino y fincas con bovinos.",
                "No equivale a carga_bovina_ua_ha porque falta area de pasturas municipal.",
                "Se usa en rentabilidad como proxy de costo de oportunidad agropecuario.",
            ]
        ),
        encoding="utf-8",
    )
    return {"inventario_bovino": output_path, "observaciones": observations_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convierte inventario bovino EVA/ICA a CSV municipal.")
    parser.add_argument("--input-xlsx", type=Path, default=DEFAULT_INPUT_XLSX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--year", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_eva_bovino(
        input_xlsx=args.input_xlsx.resolve(),
        output_dir=args.output_dir.resolve(),
        year=args.year,
    )
    print("Inventario bovino EVA generado.")
    for label, path in outputs.items():
        print(f"- {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
