from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "simem"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "simem"

SIMEM_BACKEND_DATOS = "https://www.simem.co/backend-datos/"
SIMEM_BACKEND_FILES = "https://www.simem.co/backend-files/"

# Datasets hidrologicos utiles como primera capa para el proyecto.
# No son "zonas solares"; son contexto energetico/hidrico para integrar despues.
DEFAULT_HYDRO_DATASET_IDS = ["A0CF2A", "BA1C55", "B0E933"]
DEFAULT_HYDRO_TERMS = [
    "hidraulica",
    "hidraulicas",
    "hidraulico",
    "hidraulicos",
    "hidricos",
    "hidrico",
    "embalse",
    "embalses",
    "aportes hidricos",
    "aporte hidrico",
    "reservas hidraulicas",
    "plantas hidraulicas",
]

REQUEST_HEADERS = {
    "Cache-Control": "no-cache",
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


def normalize_text(value: Any) -> str:
    """Normaliza texto para busqueda robusta sin depender de acentos."""

    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.lower().strip()


def normalize_column_name(column_name: Any) -> str:
    """Convierte nombres de columnas a snake_case reproducible."""

    normalized = normalize_text(column_name)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized or "columna_sin_nombre"


def unique_column_names(columns: list[str]) -> list[str]:
    """Evita nombres duplicados despues de normalizar columnas."""

    seen: dict[str, int] = {}
    result: list[str] = []
    for column in columns:
        count = seen.get(column, 0)
        if count == 0:
            result.append(column)
        else:
            result.append(f"{column}_{count + 1}")
        seen[column] = count + 1
    return result


def parse_date(value: Any) -> date | None:
    """Extrae una fecha de valores ISO o cadenas de SIMEM."""

    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def request_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = 90,
    retries: int = 3,
) -> dict[str, Any] | list[Any]:
    """Consulta JSON con reintentos simples para evitar fallas temporales."""

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=REQUEST_HEADERS,
                timeout=timeout,
            )
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"No fue posible consultar {url}. Ultimo error: {last_error}")


def write_json(path: Path, data: Any) -> None:
    """Guarda JSON identado para trazabilidad."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Lee un JSON local cacheado."""

    return json.loads(path.read_text(encoding="utf-8"))


def fetch_public_catalog(raw_dir: Path, force: bool = False) -> pd.DataFrame:
    """
    Descarga el catalogo publico de datasets SIMEM.

    El endpoint exige un parametro id aunque devuelve el catalogo publico completo.
    Se usa un UUID nulo para dejar claro que no depende de un dataset especifico.
    """

    raw_path = raw_dir / "catalogo_publico_simem.json"
    if raw_path.exists() and not force:
        data = read_json(raw_path)
    else:
        data = request_json(
            SIMEM_BACKEND_DATOS + "file-generation/catalog-variable-dataset",
            params={"id": "00000000-0000-0000-0000-000000000000"},
            timeout=120,
        )
        write_json(raw_path, data)

    rows = data.get("data", []) if isinstance(data, dict) else []
    catalog = pd.DataFrame(rows)
    if catalog.empty:
        raise ValueError("El catalogo publico de SIMEM llego vacio.")

    return catalog


def filter_hydro_catalog(catalog: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    """Filtra candidatos relacionados con hidroelectricidad o embalses."""

    if catalog.empty:
        return pd.DataFrame()

    normalized_terms = [normalize_text(term) for term in terms]
    result = catalog.copy()
    searchable_columns = [
        column
        for column in ["nombreConjuntoDatos", "idDataset", "tipoPublicacion"]
        if column in result.columns
    ]

    def row_matches(row: pd.Series) -> bool:
        text = " ".join(normalize_text(row.get(column)) for column in searchable_columns)
        return any(term in text for term in normalized_terms)

    result = result[result.apply(row_matches, axis=1)].copy()
    if "nombreConjuntoDatos" in result.columns:
        result = result.sort_values("nombreConjuntoDatos")
    return result


def find_catalog_row(catalog: pd.DataFrame, dataset_id: str) -> dict[str, Any] | None:
    """Busca un dataset por idDataset en el catalogo publico."""

    if catalog.empty or "idDataset" not in catalog.columns:
        return None

    mask = catalog["idDataset"].astype(str).str.upper() == dataset_id.upper()
    if not mask.any():
        return None
    return catalog[mask].iloc[0].to_dict()


def infer_date_range(
    catalog_row: dict[str, Any] | None,
    start_date: str | None,
    end_date: str | None,
    days: int,
) -> tuple[str, str, list[str]]:
    """Define rango de consulta usando finDato cuando existe."""

    warnings: list[str] = []
    if end_date:
        end = parse_date(end_date)
    else:
        end = parse_date((catalog_row or {}).get("finDato"))
        if end is None:
            end = date.today() - timedelta(days=2)
            warnings.append(
                "No se pudo inferir finDato desde catalogo; se uso hoy menos 2 dias."
            )

    if end is None:
        raise ValueError("No se pudo definir fecha final para SIMEM.")

    if start_date:
        start = parse_date(start_date)
    else:
        start = end - timedelta(days=max(days, 1) - 1)

    if start is None:
        raise ValueError("No se pudo definir fecha inicial para SIMEM.")
    if start > end:
        raise ValueError(f"Fecha inicial {start} mayor que fecha final {end}.")

    return start.isoformat(), end.isoformat(), warnings


def fetch_dataset_detail(dataset_id: str, raw_dir: Path, force: bool = False) -> dict[str, Any]:
    """Consulta metadatos de detalle de un dataset publico SIMEM."""

    raw_path = raw_dir / dataset_id.upper() / "detalle_datos_publicos.json"
    if raw_path.exists() and not force:
        return read_json(raw_path)

    data = request_json(
        SIMEM_BACKEND_FILES + "api/detalle-datos-publicos",
        params={"datasetId": dataset_id},
        timeout=90,
    )
    write_json(raw_path, data)
    return data


def fetch_public_data(
    dataset_id: str,
    start_date: str,
    end_date: str,
    raw_dir: Path,
    force: bool = False,
    column_destiny_name: str = "null",
    values: str = "null",
) -> dict[str, Any]:
    """Descarga registros del API PublicData de SIMEM."""

    safe_range = f"{start_date}_{end_date}".replace(":", "-")
    raw_path = raw_dir / dataset_id.upper() / f"public_data_{safe_range}.json"
    if raw_path.exists() and not force:
        return read_json(raw_path)

    data = request_json(
        SIMEM_BACKEND_FILES + "api/PublicData",
        params={
            "startDate": start_date,
            "endDate": end_date,
            "datasetId": dataset_id,
            "columnDestinyName": column_destiny_name,
            "values": values,
        },
        timeout=120,
    )
    write_json(raw_path, data)
    return data


def records_to_dataframe(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limpia registros JSON con transformaciones minimas para integracion."""

    df = pd.DataFrame(records)
    if df.empty:
        return df, pd.DataFrame(columns=["columna_original", "columna_normalizada"])

    original_columns = list(df.columns)
    normalized_columns = unique_column_names([normalize_column_name(col) for col in original_columns])
    mapping = pd.DataFrame(
        {
            "columna_original": original_columns,
            "columna_normalizada": normalized_columns,
        }
    )
    df.columns = normalized_columns

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].astype("string").str.strip()

        normalized = normalize_text(column)
        if "fecha" in normalized:
            df[column] = pd.to_datetime(df[column], errors="coerce")
            continue

        if any(token in normalized for token in ["codigo", "nombre", "region"]):
            continue

        converted = pd.to_numeric(df[column], errors="coerce")
        non_null = df[column].notna().sum()
        if non_null > 0 and converted.notna().sum() / non_null >= 0.9:
            df[column] = converted

    return df, mapping


def columns_to_dataframe(columns: list[dict[str, Any]]) -> pd.DataFrame:
    """Normaliza la lista de columnas documentadas por SIMEM."""

    if not columns:
        return pd.DataFrame(columns=["nameColumn", "dataType", "description"])
    return pd.DataFrame(columns)


def metadata_to_dataframe(metadata: dict[str, Any]) -> pd.DataFrame:
    """Convierte metadatos SIMEM a formato llave-valor."""

    return pd.DataFrame(
        [{"campo": key, "valor": value} for key, value in metadata.items()]
    )


def quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula resumen basico de calidad para los registros descargados."""

    if df.empty:
        return pd.DataFrame(
            [{"filas": 0, "columnas": 0, "duplicados_exactos": 0}]
        )

    rows: list[dict[str, Any]] = []
    for column in df.columns:
        rows.append(
            {
                "columna": column,
                "tipo_dato": str(df[column].dtype),
                "nulos": int(df[column].isna().sum()),
                "porcentaje_nulos": round(float(df[column].isna().mean() * 100), 4),
                "cardinalidad": int(df[column].nunique(dropna=True)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.insert(0, "filas_dataset", len(df))
    summary.insert(1, "columnas_dataset", len(df.columns))
    summary.insert(2, "duplicados_exactos_dataset", int(df.duplicated().sum()))
    return summary


def is_hydro_dataset(dataset_name: str, hydro_terms: list[str]) -> bool:
    """Evalua si el nombre parece hidro/hidrologico segun terminos configurados."""

    normalized = normalize_text(dataset_name)
    return any(normalize_text(term) in normalized for term in hydro_terms)


def export_dataset(
    dataset_id: str,
    catalog: pd.DataFrame,
    raw_dir: Path,
    output_dir: Path,
    start_date: str | None,
    end_date: str | None,
    days: int,
    hydro_terms: list[str],
    force: bool,
) -> dict[str, Any]:
    """Descarga y exporta un dataset SIMEM en carpetas ordenadas."""

    dataset_id = dataset_id.upper()
    catalog_row = find_catalog_row(catalog, dataset_id)
    inferred_start, inferred_end, warnings = infer_date_range(
        catalog_row,
        start_date=start_date,
        end_date=end_date,
        days=days,
    )

    detail = fetch_dataset_detail(dataset_id, raw_dir=raw_dir, force=force)
    public_data = fetch_public_data(
        dataset_id,
        start_date=inferred_start,
        end_date=inferred_end,
        raw_dir=raw_dir,
        force=force,
    )

    result = public_data.get("result", {}) if isinstance(public_data, dict) else {}
    detail_result = detail.get("result", {}) if isinstance(detail, dict) else {}

    dataset_name = result.get("name") or detail_result.get("name") or ""
    metadata = result.get("metadata") or detail_result.get("metadata") or {}
    columns = result.get("columns") or detail_result.get("columns") or []
    records = result.get("records") or []

    records_df, column_mapping = records_to_dataframe(records)
    columns_df = columns_to_dataframe(columns)
    metadata_df = metadata_to_dataframe(metadata)
    quality_df = quality_summary(records_df)

    dataset_dir = output_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    records_df.to_csv(dataset_dir / "registros.csv", index=False, encoding="utf-8-sig")
    columns_df.to_csv(dataset_dir / "columnas.csv", index=False, encoding="utf-8-sig")
    metadata_df.to_csv(dataset_dir / "metadatos.csv", index=False, encoding="utf-8-sig")
    column_mapping.to_csv(
        dataset_dir / "mapeo_columnas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    quality_df.to_csv(
        dataset_dir / "calidad_basica.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hydro_flag = is_hydro_dataset(dataset_name, hydro_terms)
    if not hydro_flag:
        warnings.append(
            "El nombre del dataset no coincide con terminos hidro/hidrologicos; "
            "no debe tratarse como hidroeléctrica sin justificacion adicional."
        )

    observations = build_dataset_observations(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        start_date=inferred_start,
        end_date=inferred_end,
        records_count=len(records_df),
        columns_count=len(records_df.columns),
        metadata=metadata,
        warnings=warnings,
        hydro_flag=hydro_flag,
    )
    (dataset_dir / "observaciones.txt").write_text(observations, encoding="utf-8")

    return {
        "dataset_id": dataset_id,
        "nombre_dataset": dataset_name,
        "fecha_inicio": inferred_start,
        "fecha_fin": inferred_end,
        "registros": len(records_df),
        "columnas": len(records_df.columns),
        "es_hidrologico_por_nombre": hydro_flag,
        "salida": str(dataset_dir),
        "advertencias": " | ".join(warnings),
    }


def build_dataset_observations(
    dataset_id: str,
    dataset_name: str,
    start_date: str,
    end_date: str,
    records_count: int,
    columns_count: int,
    metadata: dict[str, Any],
    warnings: list[str],
    hydro_flag: bool,
) -> str:
    """Construye observaciones trazables para informe academico."""

    lines = [
        "Observaciones SIMEM",
        "===================",
        f"dataset_id: {dataset_id}",
        f"nombre: {dataset_name}",
        f"rango_consultado: {start_date} a {end_date}",
        f"registros_exportados: {records_count}",
        f"columnas_exportadas: {columns_count}",
        f"entidad_fuente: {metadata.get('entity', 'No reportada')}",
        f"periodicidad: {metadata.get('periodicity', 'No reportada')}",
        f"granularidad: {metadata.get('granularity', 'No reportada')}",
        f"ultima_actualizacion_fuente: {metadata.get('lastUpdate', 'No reportada')}",
        f"clasificado_como_hidrologico_por_nombre: {hydro_flag}",
        "",
        "Uso metodologico:",
        "- Fuente oficial consultada: SIMEM/XM, API publica PublicData.",
        "- Transformaciones aplicadas: normalizacion de nombres de columnas, recorte de espacios, conversion evidente de fechas y numericos.",
        "- No se imputan datos, no se integran datasets y no se generan graficos en esta etapa.",
        "",
        "Advertencias:",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No se detectaron advertencias automaticas.")
    return "\n".join(lines) + "\n"


def export_catalogs(
    catalog: pd.DataFrame,
    hydro_catalog: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Exporta catalogos SIMEM completo e hidrologico."""

    output_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(
        output_dir / "catalogo_publico_simem.csv",
        index=False,
        encoding="utf-8-sig",
    )
    hydro_catalog.to_csv(
        output_dir / "catalogo_hidroelectricas_simem.csv",
        index=False,
        encoding="utf-8-sig",
    )


def build_global_observations(
    dataset_summaries: list[dict[str, Any]],
    hydro_catalog: pd.DataFrame,
    requested_ids: list[str],
) -> str:
    """Genera resumen global de la extraccion SIMEM."""

    lines = [
        "Resumen global SIMEM",
        "====================",
        "Fuente: SIMEM/XM - backend-datos y backend-files API publica.",
        f"datasets_solicitados: {', '.join(requested_ids)}",
        f"candidatos_hidrologicos_en_catalogo: {len(hydro_catalog)}",
        "",
        "Datasets descargados:",
    ]
    for item in dataset_summaries:
        lines.append(
            "- "
            f"{item['dataset_id']} | {item['nombre_dataset']} | "
            f"registros={item['registros']} | "
            f"hidrologico_por_nombre={item['es_hidrologico_por_nombre']}"
        )

    lines.extend(
        [
            "",
            "Nota metodologica:",
            "- F99E13 corresponde a 'Listado de Subestaciones del STN y el STR'. Es util para infraestructura electrica, no para hidroelectricas.",
            "- Para hidroelectricidad/hidrologia se priorizan datasets SIMEM con nombres como embalses, aportes hidricos y reservas hidraulicas.",
            "- Estos datos no definen por si solos la viabilidad solar; se deben integrar despues con PVOUT, pendiente, restricciones territoriales e infraestructura.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Define interfaz de ejecucion por consola."""

    parser = argparse.ArgumentParser(
        description=(
            "Extrae catalogo y datasets publicos de SIMEM para variables "
            "hidrologicas/hidroelectricas e infraestructura electrica."
        )
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        dest="dataset_ids",
        help=(
            "idDataset SIMEM a descargar. Puede repetirse. "
            "Si se omite, descarga datasets base de embalses/aportes/reservas."
        ),
    )
    parser.add_argument(
        "--include-f99e13",
        action="store_true",
        help="Incluye F99E13, que es subestaciones STN/STR, no hidroelectricas.",
    )
    parser.add_argument("--start-date", help="Fecha inicial YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Fecha final YYYY-MM-DD.")
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Dias a consultar si no se define start-date. Por defecto: 1.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Carpeta para JSON crudos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta para CSV limpios y observaciones.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Vuelve a consultar la API aunque existan JSON cacheados.",
    )
    return parser.parse_args()


def main() -> int:
    """Punto de entrada para ejecutar la extraccion SIMEM."""

    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_ids = args.dataset_ids or DEFAULT_HYDRO_DATASET_IDS.copy()
    if args.include_f99e13 and "F99E13" not in [item.upper() for item in dataset_ids]:
        dataset_ids.append("F99E13")
    dataset_ids = [dataset_id.upper() for dataset_id in dataset_ids]

    print("Consultando catalogo publico SIMEM...")
    catalog = fetch_public_catalog(args.raw_dir, force=args.force)
    hydro_catalog = filter_hydro_catalog(catalog, DEFAULT_HYDRO_TERMS)
    export_catalogs(catalog, hydro_catalog, args.output_dir)

    print(f"Catalogo publico: {len(catalog)} datasets.")
    print(f"Candidatos hidro/hidrologicos detectados: {len(hydro_catalog)}.")

    summaries: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        print(f"Descargando dataset SIMEM {dataset_id}...")
        try:
            summary = export_dataset(
                dataset_id=dataset_id,
                catalog=catalog,
                raw_dir=args.raw_dir,
                output_dir=args.output_dir,
                start_date=args.start_date,
                end_date=args.end_date,
                days=args.days,
                hydro_terms=DEFAULT_HYDRO_TERMS,
                force=args.force,
            )
            summaries.append(summary)
            print(
                f"  OK: {summary['nombre_dataset']} "
                f"({summary['registros']} registros)."
            )
            if summary["advertencias"]:
                print(f"  Advertencias: {summary['advertencias']}")
        except Exception as error:  # noqa: BLE001
            print(f"  ERROR en {dataset_id}: {error}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(
        args.output_dir / "resumen_datasets_descargados.csv",
        index=False,
        encoding="utf-8-sig",
    )
    observations = build_global_observations(
        summaries,
        hydro_catalog=hydro_catalog,
        requested_ids=dataset_ids,
    )
    (args.output_dir / "observaciones_simem.txt").write_text(
        observations,
        encoding="utf-8",
    )

    print("")
    print("Extraccion SIMEM finalizada.")
    print(f"Salidas: {args.output_dir}")
    print("Archivos principales:")
    print("- catalogo_hidroelectricas_simem.csv")
    print("- resumen_datasets_descargados.csv")
    print("- observaciones_simem.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
