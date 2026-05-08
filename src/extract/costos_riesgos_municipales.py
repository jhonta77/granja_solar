from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUNICIPALITIES_PATH = (
    PROJECT_ROOT / "data" / "clean" / "base_municipios" / "municipios_distritos_colombia.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "costos_riesgos_municipales"

TIERRA_URL = "https://www.datos.gov.co/resource/rttb-pk7n.json"
AGUA_URL = "https://www.datos.gov.co/resource/un66-nbty.json"
VIENTO_URL = "https://www.datos.gov.co/resource/sgfv-3yp8.json"


def normalize_text(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().upper()


def clean_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("$", "").replace("COP", "").replace(" ", "")
    text = re.sub(r"[^0-9,.\-]", "", text)
    if not text:
        return None
    if "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def minmax_inverse(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    minimum = values.min(skipna=True)
    maximum = values.max(skipna=True)
    if pd.isna(minimum) or pd.isna(maximum) or minimum == maximum:
        return values.map(lambda value: 1.0 if pd.notna(value) else pd.NA).astype("Float64")
    return (1 - ((values - minimum) / (maximum - minimum))).clip(0, 1)


def fetch_socrata(
    url: str,
    limit: int = 5000,
    max_rows: int | None = None,
    params: dict[str, str] | None = None,
    timeout: int = 120,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    offset = 0
    session = requests.Session()
    base_params = params or {}
    while True:
        page_limit = limit
        if max_rows is not None:
            remaining = max_rows - len(rows)
            if remaining <= 0:
                break
            page_limit = min(page_limit, remaining)
        request_params = {**base_params, "$limit": page_limit, "$offset": offset}
        response = session.get(url, params=request_params, timeout=timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError:
            if rows:
                break
            raise
        page = response.json()
        if not page:
            break
        rows.extend(page)
        if len(page) < page_limit:
            break
        offset += page_limit
    return pd.DataFrame(rows)


def load_municipalities(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe base municipal: {path}")
    df = pd.read_csv(path, dtype={"codigo_dane": "string"})
    df["codigo_dane"] = df["codigo_dane"].astype("string").str.zfill(5)
    df["municipio_normalizado_join"] = df["municipio"].map(normalize_text)
    df["departamento_normalizado_join"] = df["departamento"].map(normalize_text)
    unique_municipality_names = (
        df.groupby("municipio_normalizado_join")["codigo_dane"].transform("nunique").eq(1)
    )
    df["nombre_municipio_unico_pais"] = unique_municipality_names.astype(int)
    return df


def join_to_municipalities(
    source: pd.DataFrame,
    municipalities: pd.DataFrame,
    *,
    municipality_col: str,
    department_col: str | None = None,
) -> pd.DataFrame:
    working = source.copy()
    working["municipio_normalizado_join"] = working[municipality_col].map(normalize_text)
    if department_col and department_col in working.columns:
        working["departamento_normalizado_join"] = working[department_col].map(normalize_text)
        joined = working.merge(
            municipalities,
            on=["departamento_normalizado_join", "municipio_normalizado_join"],
            how="left",
            suffixes=("", "_base"),
        )
        joined["tipo_cruce_codigo_dane"] = joined["codigo_dane"].notna().map(
            {True: "departamento_municipio", False: "sin_cruce"}
        )
        return joined

    unique_municipalities = municipalities[municipalities["nombre_municipio_unico_pais"].eq(1)]
    joined = working.merge(
        unique_municipalities,
        on="municipio_normalizado_join",
        how="left",
        suffixes=("", "_base"),
    )
    joined["tipo_cruce_codigo_dane"] = joined["codigo_dane"].notna().map(
        {True: "municipio_unico_pais", False: "sin_cruce"}
    )
    return joined


def price_from_land_range(row: pd.Series) -> float | None:
    for column in ["precio_tierra_ha_cop", "valor_ha", "precio_ha", "valor"]:
        if column in row.index:
            value = clean_number(row[column])
            if value is not None and value > 0:
                return value

    text = str(row.get("rango_prec", "")).lower()
    numbers = [float(match.replace(",", ".")) for match in re.findall(r"\d+(?:[,.]\d+)?", text)]
    if len(numbers) >= 2:
        return ((numbers[0] + numbers[1]) / 2) * 1_000_000
    if len(numbers) == 1:
        if "mayor" in text:
            return numbers[0] * 1_000_000
        return numbers[0] * 1_000_000

    code_value = clean_number(row.get("cod_precio"))
    if code_value is not None and code_value > 0:
        return code_value * 1_000_000
    return None


def build_land(raw: pd.DataFrame, municipalities: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return municipalities[["codigo_dane", "municipio", "departamento"]].assign(
            precio_tierra_ha_cop=pd.NA,
            score_tierra=pd.NA,
            flag_dato_tierra=0,
            fuente_tierra="UPRA Datos Abiertos rttb-pk7n",
            tipo_cruce_tierra="sin_datos_fuente",
        )

    raw = raw.copy()
    raw["precio_tierra_ha_cop_registro"] = raw.apply(price_from_land_range, axis=1)
    joined = join_to_municipalities(
        raw,
        municipalities,
        municipality_col="municipio",
        department_col="departamen" if "departamen" in raw.columns else None,
    )
    grouped = (
        joined.dropna(subset=["codigo_dane"])
        .groupby(["codigo_dane", "municipio_base", "departamento"], dropna=False)
        .agg(
            precio_tierra_ha_cop=("precio_tierra_ha_cop_registro", "median"),
            registros_tierra=("precio_tierra_ha_cop_registro", "count"),
            area_ha_upra=("area_ha", lambda values: pd.to_numeric(values, errors="coerce").sum()),
            tipo_cruce_tierra=("tipo_cruce_codigo_dane", "first"),
        )
        .reset_index()
        .rename(columns={"municipio_base": "municipio"})
    )
    result = municipalities[["codigo_dane", "municipio", "departamento"]].merge(
        grouped.drop(columns=["municipio", "departamento"], errors="ignore"),
        on="codigo_dane",
        how="left",
    )
    result["score_tierra"] = minmax_inverse(result["precio_tierra_ha_cop"])
    result["flag_dato_tierra"] = result["precio_tierra_ha_cop"].notna().astype(int)
    result["fuente_tierra"] = "UPRA Datos Abiertos rttb-pk7n"
    result["tipo_cruce_tierra"] = result["tipo_cruce_tierra"].fillna("sin_cruce")
    return result


def build_water(raw: pd.DataFrame, municipalities: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return municipalities[["codigo_dane", "municipio", "departamento"]].assign(
            tarifa_acueducto_m3_cop=pd.NA,
            score_agua=pd.NA,
            flag_dato_agua=0,
            fuente_agua="SUI Datos Abiertos un66-nbty",
            tipo_cruce_agua="sin_datos_fuente",
        )

    raw = raw.copy()
    tariff_candidates = [
        column
        for column in [
            "consumo_b_sico_servicio_de",
            "consumo_basico_servicio_de",
            "consumo_b_sico_servicio_de_1",
            "consumo_complementario",
            "consumo_suntuario_servicio",
        ]
        if column in raw.columns
    ]
    raw["tarifa_acueducto_m3_cop_registro"] = raw[tariff_candidates].apply(
        lambda row: next((value for value in (clean_number(item) for item in row) if value is not None), None),
        axis=1,
    )
    joined = join_to_municipalities(raw, municipalities, municipality_col="municipio")
    grouped = (
        joined.dropna(subset=["codigo_dane"])
        .groupby(["codigo_dane", "municipio_base", "departamento"], dropna=False)
        .agg(
            tarifa_acueducto_m3_cop=("tarifa_acueducto_m3_cop_registro", "median"),
            registros_agua=("tarifa_acueducto_m3_cop_registro", "count"),
            tipo_cruce_agua=("tipo_cruce_codigo_dane", "first"),
        )
        .reset_index()
        .rename(columns={"municipio_base": "municipio"})
    )
    result = municipalities[["codigo_dane", "municipio", "departamento"]].merge(
        grouped.drop(columns=["municipio", "departamento"], errors="ignore"),
        on="codigo_dane",
        how="left",
    )
    result["score_agua"] = minmax_inverse(result["tarifa_acueducto_m3_cop"])
    result["flag_dato_agua"] = result["tarifa_acueducto_m3_cop"].notna().astype(int)
    result["fuente_agua"] = "SUI Datos Abiertos un66-nbty"
    result["tipo_cruce_agua"] = result["tipo_cruce_agua"].fillna("sin_cruce")
    return result


def build_wind(raw: pd.DataFrame, municipalities: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return municipalities[["codigo_dane", "municipio", "departamento"]].assign(
            velocidad_viento_ms=pd.NA,
            velocidad_viento_max_ms=pd.NA,
            score_riesgo_viento=pd.NA,
            flag_dato_viento=0,
            fuente_viento="IDEAM Datos Abiertos sgfv-3yp8",
            tipo_cruce_viento="sin_datos_fuente",
        )

    raw = raw.copy()
    if {"velocidad_viento_ms_registro", "velocidad_viento_max_ms_registro"}.issubset(raw.columns):
        raw["velocidad_viento_ms_registro"] = raw["velocidad_viento_ms_registro"].map(clean_number)
        raw["velocidad_viento_max_ms_registro"] = raw["velocidad_viento_max_ms_registro"].map(clean_number)
    else:
        raw["velocidad_viento_ms_registro"] = raw["valorobservado"].map(clean_number)
        raw["velocidad_viento_max_ms_registro"] = raw["velocidad_viento_ms_registro"]
        if "unidadmedida" in raw.columns:
            raw = raw[raw["unidadmedida"].astype(str).str.lower().str.contains("m/s", regex=False, na=False)]
    joined = join_to_municipalities(
        raw,
        municipalities,
        municipality_col="municipio",
        department_col="departamento",
    )
    grouped = (
        joined.dropna(subset=["codigo_dane"])
        .groupby(["codigo_dane", "municipio_base", "departamento_base"], dropna=False)
        .agg(
            velocidad_viento_ms=("velocidad_viento_ms_registro", "mean"),
            velocidad_viento_max_ms=("velocidad_viento_max_ms_registro", "max"),
            registros_viento=("velocidad_viento_ms_registro", "count"),
            tipo_cruce_viento=("tipo_cruce_codigo_dane", "first"),
        )
        .reset_index()
        .rename(columns={"municipio_base": "municipio", "departamento_base": "departamento"})
    )
    result = municipalities[["codigo_dane", "municipio", "departamento"]].merge(
        grouped.drop(columns=["municipio", "departamento"], errors="ignore"),
        on="codigo_dane",
        how="left",
    )
    result["score_riesgo_viento"] = minmax_inverse(result["velocidad_viento_max_ms"])
    result["flag_dato_viento"] = result["velocidad_viento_max_ms"].notna().astype(int)
    result["fuente_viento"] = "IDEAM Datos Abiertos sgfv-3yp8"
    result["tipo_cruce_viento"] = result["tipo_cruce_viento"].fillna("sin_cruce")
    return result


def build_combined(
    municipalities: pd.DataFrame,
    land: pd.DataFrame,
    water: pd.DataFrame,
    wind: pd.DataFrame,
) -> pd.DataFrame:
    combined = municipalities[["codigo_dane", "municipio", "departamento"]].copy()
    for frame in [land, water, wind]:
        keep = [column for column in frame.columns if column not in ["municipio", "departamento"]]
        combined = combined.merge(frame[keep], on="codigo_dane", how="left")
    return combined


def write_observations(path: Path, combined: pd.DataFrame, source_counts: dict[str, int]) -> None:
    lines = [
        "Costos y riesgos municipales",
        "=============================",
        "",
        "Fuentes:",
        f"- Tierra rural: {TIERRA_URL}",
        f"- Agua/acueducto: {AGUA_URL}",
        f"- Viento IDEAM: {VIENTO_URL}",
        "",
        "Metodologia:",
        "- Se usa la base municipal IGAC como catalogo maestro.",
        "- La llave final es codigo_dane; municipio y departamento quedan para lectura y auditoria.",
        "- Tierra rural usa mediana municipal del precio por hectarea cuando existe dato cruzado.",
        "- Si la fuente de tierra solo trae rango, se aproxima con el punto medio del rango en millones COP/ha.",
        "- Agua usa mediana municipal del cargo variable de consumo basico de acueducto disponible.",
        "- Viento usa promedio y maximo municipal de observaciones IDEAM en m/s.",
        "- Los scores se normalizan como 1 - minmax porque menor costo o menor viento fuerte es mejor.",
        "- Municipios sin dato no se eliminan; quedan NaN y flag_dato_* = 0.",
        "- La imputacion para el score economico-climatico se realiza despues en viabilidad_municipal.py con mediana departamental y luego nacional.",
        "- Si Socrata corta una paginacion con error despues de traer registros, se conserva lo descargado y la cobertura queda reflejada en los conteos.",
        "",
        "Registros descargados:",
    ]
    for name, count in source_counts.items():
        lines.append(f"- {name}: {count}")
    lines.extend(
        [
            "",
            "Cobertura municipal:",
            f"- tierra con dato: {int(combined['flag_dato_tierra'].fillna(0).eq(1).sum())}",
            f"- agua con dato: {int(combined['flag_dato_agua'].fillna(0).eq(1).sum())}",
            f"- viento con dato: {int(combined['flag_dato_viento'].fillna(0).eq(1).sum())}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    municipalities_path: Path = DEFAULT_MUNICIPALITIES_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_rows: int | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    municipalities = load_municipalities(municipalities_path)

    raw_land = fetch_socrata(
        TIERRA_URL,
        limit=1000,
        max_rows=max_rows,
        params={
            "$select": "cod_depart,departamen,cod_dane_m,municipio,categoria_,rango_prec,cod_precio,area_ha,consecutiv"
        },
    )
    raw_water = fetch_socrata(AGUA_URL, limit=1000, max_rows=max_rows)
    if max_rows is None:
        try:
            raw_wind = fetch_socrata(
                VIENTO_URL,
                limit=1000,
                max_rows=max_rows,
                params={
                    "$select": (
                        "departamento,municipio,avg(valorobservado) as velocidad_viento_ms_registro,"
                        "max(valorobservado) as velocidad_viento_max_ms_registro,count(*) as registros_viento_api"
                    ),
                    "$where": "unidadmedida='m/s'",
                    "$group": "departamento,municipio",
                },
            )
        except requests.RequestException:
            raw_wind = fetch_socrata(
                VIENTO_URL,
                limit=1000,
                max_rows=100000,
                params={
                    "$select": "departamento,municipio,valorobservado,unidadmedida",
                    "$where": "unidadmedida='m/s'",
                },
            )
    else:
        raw_wind = fetch_socrata(
            VIENTO_URL,
            limit=1000,
            max_rows=max_rows,
            params={
                "$select": "departamento,municipio,valorobservado,unidadmedida",
                "$where": "unidadmedida='m/s'",
            },
        )

    land = build_land(raw_land, municipalities)
    water = build_water(raw_water, municipalities)
    wind = build_wind(raw_wind, municipalities)
    combined = build_combined(municipalities, land, water, wind)

    paths = {
        "tierra": output_dir / "tierra_municipal.csv",
        "agua": output_dir / "agua_municipal.csv",
        "viento": output_dir / "viento_municipal.csv",
        "combined": output_dir / "costos_riesgos_municipales.csv",
        "observations": output_dir / "observaciones_costos_riesgos.txt",
    }
    land.to_csv(paths["tierra"], index=False, encoding="utf-8-sig")
    water.to_csv(paths["agua"], index=False, encoding="utf-8-sig")
    wind.to_csv(paths["viento"], index=False, encoding="utf-8-sig")
    combined.to_csv(paths["combined"], index=False, encoding="utf-8-sig")
    write_observations(
        paths["observations"],
        combined,
        {
            "tierra_rttb_pk7n": len(raw_land),
            "agua_un66_nbty": len(raw_water),
            "viento_sgfv_3yp8": len(raw_wind),
        },
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga y consolida costos de tierra, agua y riesgo por viento a escala municipal."
    )
    parser.add_argument("--municipalities-path", type=Path, default=DEFAULT_MUNICIPALITIES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limita filas por fuente para pruebas. Por defecto descarga todo lo disponible por paginacion.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_pipeline(
        municipalities_path=args.municipalities_path,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
    )
    print("Costos y riesgos municipales generados.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
