"""Extrae el Indice Municipal de Riesgo de Calamidades (IMRC) del DNP.

Fuente de datos
---------------
DNP - Departamento Nacional de Planeacion, Colombia.
Archivo: IMRC-BASE-DE-DATOS-2024.xlsx
Descarga: https://www.dnp.gov.co/programas/desarrollo-territorial/gestion-del-riesgo

El IMRC combina amenaza + exposicion + vulnerabilidad - capacidad de gestion
para producir un indice 0-100 por municipio, donde 100 = maximo riesgo.

El archivo tiene dos componentes:
    Exceso de Lluvias: riesgo por inundaciones y exceso hidrico
    Deficit de Lluvias: riesgo por sequia extrema e incendios forestales

Columnas clave usadas
----------------------
    DIVIPOLA           → codigo_dane (CHAR 5, codigo DIVIPOLA del municipio)
    IMRC_E             → imrc_exceso      (indice ajustado exceso lluvias, 0-100)
    IMRC_D             → imrc_deficit     (indice ajustado deficit lluvias, 0-100)
    IR_E               → ir_exceso_raw    (riesgo bruto exceso antes de capacidad)
    IR_D               → ir_deficit_raw   (riesgo bruto deficit)
    %AA                → pct_area_amenaza_exceso  (% area amenazada por inundacion)
    IPM                → vulnerabilidad_social_pct (pobreza multidimensional)

Columna derivada para el pipeline
-----------------------------------
    riesgo_inundacion_idx = imrc_exceso / 100   (0-1, 1=maximo riesgo)

Esta columna alimenta _build_score_riesgo() en dimensiones_scoring.py:
    mayor riesgo inundacion → penaliza score_riesgo.

Salida principal
----------------
    data/clean/ideam_riesgo/ideam_riesgo_municipal.csv

Uso
---
    python -m src.extract.imrc_dnp
    python -m src.extract.imrc_dnp --input-xlsx ruta/IMRC-BASE-DE-DATOS-2024.xlsx
    python -m src.extract.imrc_dnp --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_XLSX = (
    PROJECT_ROOT / "data" / "raw" / "IMRC-BASE-DE-DATOS-2024.xlsx"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "ideam_riesgo"

SHEET_EXCESO = "Exceso de Lluvias"
SHEET_DEFICIT = "Deficit de Lluvias"


# ---------------------------------------------------------------------------
# Lectura y limpieza del Excel IMRC
# ---------------------------------------------------------------------------

def _read_imrc_sheet(path: Path, sheet: str) -> pd.DataFrame:
    """Lee una hoja del IMRC. La fila 0 del DataFrame es la cabecera real."""
    raw = pd.read_excel(path, sheet_name=sheet, header=0, dtype=str)
    real_cols = raw.iloc[0].tolist()
    df = raw.iloc[1:].reset_index(drop=True)
    df.columns = [str(c).strip() for c in real_cols]
    return df


def _detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Busca la primera columna disponible (insensible a mayusculas y tildes)."""
    import unicodedata

    def norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s))
        return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()

    normed = {norm(c): c for c in df.columns}
    for cand in candidates:
        key = norm(cand)
        if key in normed:
            return normed[key]
    return None


def _extract_exceso(df: pd.DataFrame) -> pd.DataFrame:
    divipola_col = _detect_col(df, ["DIVIPOLA", "DIVPOLA", "Cod_municipio"])
    imrc_e_col = _detect_col(df, [
        "Indice de riesgo de desastres ajustado por capacidades (IMRC_E)",
        "IMRC_E", "imrc_e",
    ])
    ir_e_col = _detect_col(df, [
        "Indice de Rieso de Desastres Exceso (IR_E)",
        "Indice de Riesgo de Desastres Exceso (IR_E)",
        "IR_E",
    ])
    pct_aa_col = _detect_col(df, ["%AA", "pct_aa", "%area_amenazada"])
    ipm_col = _detect_col(df, [
        "Vulnerabilidad social (% IPM)", "IPM", "Vulnerabilidad social",
    ])

    if not divipola_col or not imrc_e_col:
        raise ValueError(
            f"IMRC Exceso: no se encontraron columnas DIVIPOLA o IMRC_E.\n"
            f"Columnas disponibles: {list(df.columns)[:10]}"
        )

    result = pd.DataFrame()
    result["codigo_dane"] = df[divipola_col].astype(str).str.strip().str.zfill(5)
    result["imrc_exceso"] = pd.to_numeric(df[imrc_e_col], errors="coerce")

    if ir_e_col:
        result["ir_exceso_raw"] = pd.to_numeric(df[ir_e_col], errors="coerce")
    if pct_aa_col:
        result["pct_area_amenaza_exceso"] = pd.to_numeric(df[pct_aa_col], errors="coerce")
    if ipm_col:
        result["vulnerabilidad_social_pct"] = pd.to_numeric(df[ipm_col], errors="coerce")

    return result.dropna(subset=["codigo_dane"]).reset_index(drop=True)


def _extract_deficit(df: pd.DataFrame) -> pd.DataFrame:
    divipola_col = _detect_col(df, ["DIVIPOLA", "DIVPOLA", "Cod_municipio"])
    imrc_d_col = _detect_col(df, [
        "Indice de riesgo de desastres ajustado por capacidades (IMRC_D)",
        "IMRC_D", "imrc_d",
    ])
    ir_d_col = _detect_col(df, [
        "Indice de Rieso de Desastres -  Deficit de  lluvias",
        "Indice de Rieso de Desastres - Deficit de lluvias",
        "IR_D",
    ])
    pct_ase_col = _detect_col(df, ["% ASE", "%ASE", "pct_ase"])

    if not divipola_col or not imrc_d_col:
        raise ValueError(
            f"IMRC Deficit: no se encontraron columnas DIVIPOLA o IMRC_D.\n"
            f"Columnas disponibles: {list(df.columns)[:10]}"
        )

    result = pd.DataFrame()
    result["codigo_dane"] = df[divipola_col].astype(str).str.strip().str.zfill(5)
    result["imrc_deficit"] = pd.to_numeric(df[imrc_d_col], errors="coerce")

    if ir_d_col:
        result["ir_deficit_raw"] = pd.to_numeric(df[ir_d_col], errors="coerce")
    if pct_ase_col:
        result["pct_area_sequia_extrema"] = pd.to_numeric(df[pct_ase_col], errors="coerce")

    return result.dropna(subset=["codigo_dane"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_imrc_dnp(
    input_xlsx: Path = DEFAULT_INPUT_XLSX,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Path]:
    """Lee el IMRC DNP y exporta CSV listo para score_riesgo."""

    output_path = output_dir / "ideam_riesgo_municipal.csv"
    if output_path.exists() and not force:
        print(f"  IMRC DNP: usando resultado previo {output_path}")
        return {"summary": output_path}

    if not input_xlsx.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo IMRC en: {input_xlsx}\n"
            "Descarga desde: https://www.dnp.gov.co/programas/desarrollo-territorial/gestion-del-riesgo\n"
            "Guarda como: data/raw/IMRC-BASE-DE-DATOS-2024.xlsx"
        )

    print(f"  Leyendo IMRC: {input_xlsx}")

    df_exceso_raw = _read_imrc_sheet(input_xlsx, SHEET_EXCESO)
    df_deficit_raw = _read_imrc_sheet(input_xlsx, SHEET_DEFICIT)

    df_exceso = _extract_exceso(df_exceso_raw)
    df_deficit = _extract_deficit(df_deficit_raw)

    print(f"  Exceso de lluvias: {len(df_exceso)} municipios")
    print(f"  Deficit de lluvias: {len(df_deficit)} municipios")

    df = df_exceso.merge(df_deficit, on="codigo_dane", how="outer")

    # Columna principal que alimenta _build_score_riesgo()
    # IMRC va 0-100; dividimos por 100 para normalizar a 0-1
    df["riesgo_inundacion_idx"] = (
        pd.to_numeric(df["imrc_exceso"], errors="coerce") / 100.0
    ).clip(0.0, 1.0)

    df["riesgo_sequia_idx"] = (
        pd.to_numeric(df.get("imrc_deficit", pd.Series(dtype="float")), errors="coerce") / 100.0
    ).clip(0.0, 1.0)

    # Riesgo combinado (media simple de los dos componentes disponibles)
    exceso = df["riesgo_inundacion_idx"]
    sequia = df["riesgo_sequia_idx"]
    df["riesgo_combinado_idx"] = exceso.where(sequia.isna(), (exceso.fillna(0) + sequia.fillna(0)) / 2)

    df["fuente"] = "IMRC DNP 2024 - Indice Municipal Riesgo Calamidades"

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig", errors="replace")

    print(f"  IMRC completado: {len(df)} municipios -> {output_path}")
    return {"summary": output_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte el Excel IMRC-DNP a CSV para el pipeline de scoring."
    )
    parser.add_argument(
        "--input-xlsx", type=Path, default=DEFAULT_INPUT_XLSX,
        help="Ruta al archivo IMRC-BASE-DE-DATOS-2024.xlsx",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Regenerar aunque exista CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_imrc_dnp(
            input_xlsx=args.input_xlsx.resolve(),
            output_dir=args.output_dir.resolve(),
            force=args.force,
        )
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {error}")
        return 1

    print("IMRC DNP completado.")
    for label, path in outputs.items():
        print(f"  - {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
