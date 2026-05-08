"""Carga productividad agropecuaria municipal (UGG/ha) desde EVA pecuaria ICA.

Fuente principal
----------------
EVA Pecuaria - Inventario Bovino 2019-2023 (ICA / MADR).
Archivo: BasePecuaria20192023.xlsx  (hoja InvBovino)
Descarga: https://www.agronet.gov.co/estadistica/Paginas/home.aspx?cod=1
  Seccion: Pecuaria -> Bovinos -> por municipio

Calculo de UGG (Unidades Gran Ganado)
--------------------------------------
El archivo EVA tiene el inventario desglosado por categoria de animal.
Se convierte a UGG usando factores FEDEGAN:

    Terneras/Terneros < 1 ano     -> 0.4 UGG
    Hembras/Machos 1-2 anos       -> 0.6 UGG
    Hembras/Machos 2-3 anos       -> 0.8 UGG
    Hembras > 3 anos (vacas)      -> 1.0 UGG
    Machos > 3 anos (toros)       -> 1.2 UGG

Carga bovina proxy
------------------
El archivo EVA no incluye area de pasturas por municipio.
Se usa area_km2_igac * 100 (ha totales del municipio) como denominador.
Esto subestima la carga real donde no toda el area es pastura, pero
produce una metrica RELATIVA consistente para comparar municipios.

Para interpretar correctamente el score, el resultado se clasifica
en bandas de sistema productivo (FEDEGAN/AGROSAVIA):

    extensivo_bajo:  ugg_ha < 1      (pasturas nativas, sin tecnificacion)
    tradicional:     1 <= ugg_ha < 3 (tropico bajo convencional: 1.5-1.8)
    tecnificado:     ugg_ha >= 3     (fincas tecnificadas: 3-4 cabezas/ha)

Salida
------
    data/clean/upra_agropecuario/upra_agropecuario_municipal.csv

Columnas:
    codigo_dane             CHAR(5)
    inventario_bovinos      INT     total cabezas
    ugg_total               DOUBLE  total en unidades gran ganado
    area_municipio_ha       DOUBLE  area total municipio en hectareas
    ugg_ha_proxy            DOUBLE  UGG / ha total municipio (proxy)
    carga_bovina_ua_ha      DOUBLE  alias de ugg_ha_proxy (compatibilidad)
    sistema_productivo      VARCHAR extensivo_bajo | tradicional | tecnificado
    anio_referencia_agro    INT
    fuente_agropecuaria     VARCHAR
    fecha_carga_utc         VARCHAR

Uso
---
    python -m src.extract.upra_agropecuario
    python -m src.extract.upra_agropecuario --input-xlsx data/raw/BasePecuaria20192023.xlsx
"""

from __future__ import annotations

import argparse
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_XLSX = PROJECT_ROOT / "data" / "raw" / "BasePecuaria20192023 (1).xlsx"
DEFAULT_MUNICIPALITIES_CSV = (
    PROJECT_ROOT / "data" / "clean" / "base_municipios" / "municipios_distritos_colombia.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "upra_agropecuario"

# Factores UGG por categoria (FEDEGAN)
_UGG_FACTORS: dict[str, float] = {
    "terneras_lt1": 0.4,
    "terneros_lt1": 0.4,
    "hembras_1_2":  0.6,
    "machos_1_2":   0.6,
    "hembras_2_3":  0.8,
    "machos_2_3":   0.8,
    "hembras_gt3":  1.0,  # vacas adultas
    "machos_gt3":   1.2,  # toros
}

# Bandas de sistema productivo (FEDEGAN / AGROSAVIA)
_BANDS = [
    (0.0, 1.0,  "extensivo_bajo"),
    (1.0, 3.0,  "tradicional"),
    (3.0, 9999, "tecnificado"),
]


def _norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", "" if value is None else str(value))
    return "".join(c for c in text if not unicodedata.combining(c)).lower().strip()


def _detect_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Busca columna cuyo nombre normalizado contenga todos los keywords."""
    for col in df.columns:
        col_norm = _norm(col)
        if all(kw in col_norm for kw in keywords):
            return col
    return None


def _parse_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _classify_band(val: float) -> str:
    for low, high, label in _BANDS:
        if low <= val < high:
            return label
    return "tecnificado"


def _read_invbovino(path: Path) -> pd.DataFrame:
    """Lee hoja InvBovino saltando las filas de titulo."""
    xl = pd.ExcelFile(path)
    # Preferir hoja InvBovino; si no existe usar la primera
    sheet = "InvBovino" if "InvBovino" in xl.sheet_names else xl.sheet_names[0]
    # Las primeras 3 filas son titulo/fuente; fila 3 es el header real
    df = pd.read_excel(path, sheet_name=sheet, header=3, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _compute_ugg(df: pd.DataFrame) -> pd.Series:
    """Suma UGG ponderados a partir de las columnas de categoria."""
    ugg = pd.Series(0.0, index=df.index)

    col_map = {
        "terneras_lt1": ["terneras", "1"],
        "terneros_lt1": ["terneros", "1"],
        "hembras_1_2":  ["hembras", "1", "2"],
        "machos_1_2":   ["machos", "1", "2"],
        "hembras_2_3":  ["hembras", "2", "3"],
        "machos_2_3":   ["machos", "2", "3"],
        "hembras_gt3":  ["hembras", "3"],
        "machos_gt3":   ["machos", "3"],
    }

    # Asignar columnas a categorias
    used: set[str] = set()
    assignments: dict[str, str] = {}

    for cat, keywords in col_map.items():
        for col in df.columns:
            if col in used:
                continue
            col_norm = _norm(col)
            if all(kw in col_norm for kw in keywords):
                assignments[cat] = col
                used.add(col)
                break

    for cat, col in assignments.items():
        factor = _UGG_FACTORS[cat]
        ugg += _parse_num(df[col]).fillna(0) * factor

    n_assigned = len(assignments)
    print(f"  Categorias UGG asignadas: {n_assigned}/8 -> {list(assignments.keys())}")
    return ugg


def load_pecuaria(path: Path, municipalities_csv: Path) -> pd.DataFrame:
    df = _read_invbovino(path)

    # Columnas clave
    dane_col = _detect_col(df, ["dane", "municipio"]) or _detect_col(df, ["codigo", "mun"])
    anio_col = _detect_col(df, ["a", "o"]) or _detect_col(df, ["ano"]) or _detect_col(df, ["year"])
    total_col = _detect_col(df, ["total", "bovino"])

    # Buscar codigo dane municipio (columna con "dane" y "municipio" o similar)
    dane_mun_col = None
    for col in df.columns:
        n = _norm(col)
        if "dane" in n and "municipio" in n:
            dane_mun_col = col
            break
        if "codigo" in n and "municipio" in n:
            dane_mun_col = col
            break
    if dane_mun_col is None:
        # fallback: buscar columna que tenga codigos tipo "05001"
        for col in df.columns:
            sample = df[col].dropna().head(10)
            if sample.astype(str).str.match(r"^\d{4,5}$").sum() >= 5:
                dane_mun_col = col
                break

    if dane_mun_col is None:
        raise ValueError(
            f"No se encontro columna de codigo DANE municipio.\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Columna de ano
    anio_col = None
    for col in df.columns:
        n = _norm(col)
        if n in {"ano", "a~o", "year", "anio"} or ("a" in n and "o" in n and len(n) <= 4):
            anio_col = col
            break
    if anio_col is None:
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0 and vals.between(2015, 2030).mean() > 0.8:
                anio_col = col
                break

    print(f"  DANE col: {dane_mun_col!r}  |  Ano col: {anio_col!r}")

    # Calcular UGG
    df["_ugg"] = _compute_ugg(df)
    df["_dane"] = df[dane_mun_col].astype(str).str.strip().str.zfill(5)
    df["_anio"] = pd.to_numeric(df[anio_col], errors="coerce") if anio_col else 0

    # Total bovinos
    total_col = _detect_col(df, ["total"])
    df["_total"] = _parse_num(df[total_col]).fillna(0) if total_col else df["_ugg"]

    # Quedarse con el ano mas reciente por municipio
    df = df.sort_values("_anio", ascending=False)
    df = df.groupby("_dane", as_index=False).first()

    # Unir con municipios para obtener area_km2_igac
    munis = pd.read_csv(municipalities_csv, dtype={"codigo_dane": "string"})
    munis["codigo_dane"] = munis["codigo_dane"].str.zfill(5)

    merged = df.merge(
        munis[["codigo_dane", "area_km2_igac"]],
        left_on="_dane",
        right_on="codigo_dane",
        how="left",
    )

    n_match = merged["codigo_dane"].notna().sum()
    print(f"  Match con municipios IGAC: {n_match}/{len(merged)}")

    area_ha = pd.to_numeric(merged["area_km2_igac"], errors="coerce") * 100

    result = pd.DataFrame()
    result["codigo_dane"] = merged["_dane"]
    result["inventario_bovinos"] = merged["_total"].astype("Int64")
    result["ugg_total"] = merged["_ugg"].round(2)
    result["area_municipio_ha"] = area_ha.round(2)
    result["ugg_ha_proxy"] = (merged["_ugg"] / area_ha).where(area_ha > 0).round(4)
    result["carga_bovina_ua_ha"] = result["ugg_ha_proxy"]  # alias para scoring
    result["sistema_productivo"] = result["ugg_ha_proxy"].apply(
        lambda x: _classify_band(float(x)) if pd.notna(x) else pd.NA
    )
    result["anio_referencia_agro"] = merged["_anio"].astype("Int64")
    result["fuente_agropecuaria"] = "EVA-ICA BasePecuaria"
    result = result.dropna(subset=["codigo_dane", "ugg_ha_proxy"])
    result = result.drop_duplicates(subset=["codigo_dane"])
    result["fecha_carga_utc"] = datetime.now(timezone.utc).isoformat()

    # Resumen de bandas
    if len(result) > 0:
        dist = result["sistema_productivo"].value_counts()
        print(f"  Distribucion sistema productivo:\n{dist.to_string()}")

    return result


def write_observations(path: Path, n: int, fuente_path: str) -> None:
    lines = [
        "UPRA/EVA - Productividad agropecuaria municipal (UGG/ha)",
        "=========================================================",
        "",
        f"Registros cargados: {n}",
        f"Archivo fuente: {fuente_path}",
        "",
        "Metodologia:",
        "  - Se calculan UGG (Unidades Gran Ganado) ponderando cada categoria",
        "    de animal por su factor FEDEGAN:",
        "      Terneras/Terneros < 1 ano -> 0.4 UGG",
        "      Hembras/Machos 1-2 anos   -> 0.6 UGG",
        "      Hembras/Machos 2-3 anos   -> 0.8 UGG",
        "      Hembras > 3 anos (vacas)  -> 1.0 UGG",
        "      Machos > 3 anos (toros)   -> 1.2 UGG",
        "",
        "  - ugg_ha_proxy = UGG_total / (area_km2_igac * 100)",
        "    NOTA: el denominador es el area TOTAL del municipio, no solo pasturas.",
        "    Esto subestima la carga real; se usa como metrica relativa comparativa.",
        "",
        "Bandas de sistema productivo (FEDEGAN/AGROSAVIA):",
        "  extensivo_bajo:  ugg_ha < 1      (pasturas nativas, baja productividad)",
        "  tradicional:     1 <= ugg_ha < 3 (tropico bajo convencional: 1.5-1.8)",
        "  tecnificado:     ugg_ha >= 3     (fincas tecnificadas: 3-4 cabezas/ha)",
        "",
        "Uso en scoring multidimensional:",
        "  score_agropecuario:",
        "    carga alta (tecnificado) -> mayor costo de oportunidad -> prioridad agrovoltaica.",
        "    La variable se normaliza contra las bandas, no contra min/max municipal.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_upra_agropecuario(
    input_xlsx: Path = DEFAULT_INPUT_XLSX,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    municipalities_csv: Path = DEFAULT_MUNICIPALITIES_CSV,
) -> dict[str, Path]:
    if not input_xlsx.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {input_xlsx}\n"
            "Descarga BasePecuaria desde:\n"
            "  https://www.agronet.gov.co/estadistica/Paginas/home.aspx?cod=1\n"
            "  Seccion: Pecuaria -> Bovinos\n"
            "y copia el archivo a data/raw/"
        )
    if not municipalities_csv.exists():
        raise FileNotFoundError(
            f"No existe el archivo de municipios: {municipalities_csv}\n"
            "Ejecuta primero: python -m src.extract.igac_municipios"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_pecuaria(input_xlsx, municipalities_csv)

    output_path = output_dir / "upra_agropecuario_municipal.csv"
    observations_path = output_dir / "upra_agropecuario_observaciones.txt"

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    write_observations(observations_path, len(df), str(input_xlsx))

    return {
        "agropecuario": output_path,
        "observaciones": observations_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Carga productividad agropecuaria (UGG/ha) desde EVA pecuaria ICA."
    )
    parser.add_argument(
        "--input-xlsx",
        type=Path,
        default=DEFAULT_INPUT_XLSX,
        help="Archivo BasePecuaria Excel (InvBovino).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--municipalities-csv",
        type=Path,
        default=DEFAULT_MUNICIPALITIES_CSV,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_upra_agropecuario(
            input_xlsx=args.input_xlsx.resolve(),
            output_dir=args.output_dir.resolve(),
            municipalities_csv=args.municipalities_csv.resolve(),
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
