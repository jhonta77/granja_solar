from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from src.db.mysql_cli import MySQLSettings, query_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "powerbi"


@dataclass(frozen=True)
class ExportSpec:
    name: str
    sql: str
    transform: Callable[[pd.DataFrame], pd.DataFrame]
    fallback_csv: Path | None = None


def clean_column_name(column: str) -> str:
    return (
        column.strip()
        .lower()
        .replace(" ", "_")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(column) for column in df.columns]
    return df


def as_text_code(df: pd.DataFrame, column: str = "codigo_dane") -> pd.DataFrame:
    if column not in df.columns:
        return df
    df = df.copy()
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )
    return df


def coerce_numeric(df: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    exclude = exclude or set()
    df = df.copy()
    for column in df.columns:
        if column in exclude:
            continue
        if df[column].dtype == object:
            converted = pd.to_numeric(df[column], errors="coerce")
            non_null_count = int(df[column].notna().sum())
            converted_count = int(converted.notna().sum())
            if non_null_count == converted_count:
                df[column] = converted
    return df


def add_score_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    score_column = "v_i_multidimensional"
    if score_column not in df.columns:
        return df

    score = pd.to_numeric(df[score_column], errors="coerce")
    df["ranking_viabilidad"] = score.rank(method="dense", ascending=False).astype("Int64")
    df["nivel_viabilidad"] = pd.cut(
        score,
        bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
        labels=["Baja", "Media baja", "Media alta", "Alta"],
    )
    return df


def add_financial_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    roi_column = next((column for column in df.columns if column.lower() == "roi_neto"), None)
    payback_column = next((column for column in df.columns if column.lower() == "payback_anios"), None)

    if roi_column:
        roi = pd.to_numeric(df[roi_column], errors="coerce")
        df["ranking_roi"] = roi.rank(method="dense", ascending=False).astype("Int64")
        df["roi_categoria"] = pd.cut(
            roi,
            bins=[-float("inf"), 0, 0.05, 0.10, float("inf")],
            labels=["Negativo", "Bajo", "Medio", "Alto"],
        )

    if payback_column:
        payback = pd.to_numeric(df[payback_column], errors="coerce")
        df["payback_categoria"] = pd.cut(
            payback,
            bins=[0, 8, 15, 25, float("inf")],
            labels=["Corto", "Medio", "Largo", "Fuera de horizonte"],
        )

    return df


def transform_default(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = as_text_code(df)
    return coerce_numeric(df, exclude={"codigo_dane"})


def transform_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = transform_default(df)
    return add_score_bands(df)


def transform_financial(df: pd.DataFrame) -> pd.DataFrame:
    df = transform_default(df)
    return add_financial_bands(df)


def get_export_specs() -> list[ExportSpec]:
    return [
        ExportSpec(
            name="dashboard_municipal",
            sql="SELECT * FROM vw_dashboard_municipal;",
            transform=transform_default,
        ),
        ExportSpec(
            name="contexto_municipal",
            sql="SELECT * FROM vw_municipal_contexto;",
            transform=transform_default,
        ),
        ExportSpec(
            name="comparacion_scores",
            sql="SELECT * FROM vw_comparacion_scores;",
            transform=transform_scores,
        ),
        ExportSpec(
            name="rentabilidad_municipal",
            sql="SELECT * FROM rentabilidad_municipal;",
            transform=transform_financial,
            fallback_csv=PROJECT_ROOT
            / "data"
            / "clean"
            / "rentabilidad_municipal"
            / "rentabilidad_municipal.csv",
        ),
        ExportSpec(
            name="clusters_municipios",
            sql="SELECT * FROM cluster_municipal;",
            transform=transform_default,
        ),
        ExportSpec(
            name="departamentos",
            sql="SELECT * FROM departamentos;",
            transform=transform_default,
        ),
        ExportSpec(
            name="municipios",
            sql="SELECT * FROM municipios;",
            transform=transform_default,
        ),
    ]


def export_powerbi_files(
    output_dir: Path,
    settings: MySQLSettings,
    *,
    database: str,
    encoding: str = "utf-8-sig",
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for spec in get_export_specs():
        source = "mysql"
        try:
            df = query_dataframe(spec.sql, settings=settings, database=database)
        except Exception as error:
            if spec.fallback_csv is None or not spec.fallback_csv.exists():
                manifest_rows.append(
                    {
                        "tabla": spec.name,
                        "archivo": "",
                        "filas": 0,
                        "columnas": 0,
                        "origen": source,
                        "estado": "error",
                        "detalle": str(error),
                    }
                )
                continue
            df = pd.read_csv(spec.fallback_csv)
            source = "csv_fallback"

        df = spec.transform(df)
        output_path = output_dir / f"{spec.name}.csv"
        df.to_csv(output_path, index=False, encoding=encoding)
        manifest_rows.append(
            {
                "tabla": spec.name,
                "archivo": str(output_path),
                "filas": len(df),
                "columnas": len(df.columns),
                "origen": source,
                "estado": "ok",
                "detalle": "",
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    build_single_model_csv(output_dir, encoding=encoding)
    manifest.to_csv(output_dir / "manifest_powerbi.csv", index=False, encoding=encoding)
    return manifest


def build_single_model_csv(output_dir: Path, *, encoding: str) -> None:
    dashboard_path = output_dir / "dashboard_municipal.csv"
    scores_path = output_dir / "comparacion_scores.csv"
    financial_path = output_dir / "rentabilidad_municipal.csv"
    if not dashboard_path.exists() or not scores_path.exists():
        return

    dashboard = pd.read_csv(dashboard_path, dtype={"codigo_dane": "string"})
    scores = pd.read_csv(scores_path, dtype={"codigo_dane": "string"})
    model = dashboard.merge(
        scores[
            [
                "codigo_dane",
                "v_i_multidimensional",
                "score_fisico",
                "score_electrico",
                "score_economico",
                "score_agropecuario",
                "score_riesgo",
                "clasificacion_multidim",
                "ranking_viabilidad",
                "nivel_viabilidad",
            ]
        ],
        on="codigo_dane",
        how="left",
        suffixes=("", "_score"),
    )

    if financial_path.exists():
        financial = pd.read_csv(financial_path, dtype={"codigo_dane": "string"})
        financial_columns = [
            column
            for column in [
                "codigo_dane",
                "generacion_kwh_ha_year",
                "ingreso_energia_cop_ha_year",
                "costo_total_estimado_cop_ha_year",
                "margen_estimado_cop_ha_year",
                "relacion_beneficio_costo_rentabilidad",
                "score_rentabilidad",
                "clasificacion_rentabilidad",
                "score_rentabilidad_ajustada",
                "clasificacion_rentabilidad_ajustada",
            ]
            if column in financial.columns
        ]
        model = model.merge(
            financial[financial_columns],
            on="codigo_dane",
            how="left",
            suffixes=("", "_rentabilidad"),
        )

    model.to_csv(output_dir / "modelo_powerbi.csv", index=False, encoding=encoding)


def parse_args() -> argparse.Namespace:
    env_settings = MySQLSettings.from_env()
    parser = argparse.ArgumentParser(
        description="Exporta tablas y vistas MySQL transformadas para consumirlas en Power BI Desktop."
    )
    parser.add_argument(
        "--database",
        default=env_settings.database,
        help="Base de datos MySQL origen. Por defecto usa MYSQL_DATABASE.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Carpeta destino para los CSV de Power BI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = MySQLSettings.from_env(database_override=args.database)
    manifest = export_powerbi_files(
        Path(args.output_dir),
        settings=settings,
        database=args.database,
    )

    ok_count = int((manifest["estado"] == "ok").sum())
    error_count = int((manifest["estado"] == "error").sum())
    print(f"Exportacion Power BI terminada: {ok_count} archivos OK, {error_count} errores.")
    print(f"Carpeta destino: {Path(args.output_dir).resolve()}")
    if error_count:
        print(manifest.loc[manifest["estado"] == "error", ["tabla", "detalle"]].to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
