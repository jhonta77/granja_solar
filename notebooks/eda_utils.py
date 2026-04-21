from __future__ import annotations

import csv
import hashlib
import itertools
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".xlsm", ".txt"}
NULL_TOKENS = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "sin dato",
    "sin datos",
    "s/d",
    "-",
    "--",
}
DATE_KEYWORDS = {
    "fecha",
    "date",
    "periodo",
    "period",
    "tiempo",
    "time",
    "timestamp",
    "mes",
    "month",
}
DATE_EXACT_NAMES = {
    "fecha",
    "date",
    "periodo",
    "period",
    "timestamp",
    "fecha_archivo",
}
NUMERIC_PRIORITY_KEYWORDS = {
    "demanda",
    "consumo",
    "energia",
    "energy",
    "mw",
    "mwh",
    "gwh",
    "kwh",
    "potencia",
    "carga",
    "generacion",
    "generacion_total",
}
CATEGORY_PRIORITY_KEYWORDS = {
    "region",
    "departamento",
    "depto",
    "zona",
    "mercado",
    "municipio",
    "ciudad",
    "area",
    "agente",
    "sistema",
}
IDENTIFIER_KEYWORDS = {
    "id",
    "codigo",
    "cod",
    "nit",
    "documento",
    "consecutivo",
    "llave",
    "key",
}
MEASURE_EXCLUSION_KEYWORDS = {
    "promedio",
    "media",
    "min",
    "max",
    "anio",
    "ano",
    "year",
    "mes",
    "month",
}


def strip_accents(value: str) -> str:
    """Elimina acentos para facilitar comparaciones y nombres de salida."""

    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def slugify(value: str) -> str:
    """Genera nombres seguros para carpetas y archivos."""

    text = strip_accents(str(value)).lower().strip()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text.strip("_") or "dataset"


def bounded_slug(value: str, max_length: int = 80) -> str:
    """Genera un slug reproducible con longitud segura para Windows."""

    text = slugify(value)
    if len(text) <= max_length:
        return text
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    prefix_length = max(1, max_length - len(digest) - 1)
    return f"{text[:prefix_length].rstrip('_')}_{digest}"


def normalize_column_label(column_name: Any) -> str:
    """Estandariza nombres de columnas a formato snake_case simple."""

    text = strip_accents(str(column_name)).lower().strip()
    text = re.sub(r"[^\w\s/.-]", " ", text)
    text = re.sub(r"[/.\s-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "columna"


def make_unique(names: list[str]) -> list[str]:
    """Resuelve colisiones en nombres de columnas normalizados."""

    seen: dict[str, int] = {}
    unique_names: list[str] = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        unique_names.append(name if count == 1 else f"{name}_{count}")
    return unique_names


def ensure_project_folders(project_root: Path) -> dict[str, Path]:
    """Crea la estructura minima del proyecto si no existe."""

    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    clean_dir = data_dir / "clean"
    notebooks_dir = project_root / "notebooks"

    for folder in (data_dir, raw_dir, clean_dir, notebooks_dir):
        folder.mkdir(parents=True, exist_ok=True)

    return {
        "project_root": project_root,
        "data_dir": data_dir,
        "raw_dir": raw_dir,
        "clean_dir": clean_dir,
        "notebooks_dir": notebooks_dir,
    }


def list_supported_files(raw_dir: Path) -> list[Path]:
    """Lista archivos tabulares soportados dentro de data/raw."""

    return sorted(
        [
            path
            for path in raw_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )


def inspect_excel_sheets(file_path: Path) -> list[str]:
    """Devuelve los nombres de hojas de un archivo Excel."""

    try:
        with pd.ExcelFile(file_path) as workbook:
            return workbook.sheet_names
    except ImportError as error:
        raise ImportError(
            "No fue posible abrir el archivo Excel. Instala openpyxl en el entorno."
        ) from error


def sniff_delimiter(file_path: Path) -> str:
    """Infiere delimitador basico para archivos de texto delimitado."""

    encodings = ("utf-8-sig", "utf-8", "latin-1")
    sample = ""
    for encoding in encodings:
        try:
            with file_path.open("r", encoding=encoding, errors="replace") as handler:
                sample = "".join([handler.readline() for _ in range(5)])
            break
        except OSError:
            continue

    if not sample:
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except csv.Error:
        delimiter_counts = {
            delimiter: sample.count(delimiter) for delimiter in [",", ";", "\t", "|"]
        }
        return max(delimiter_counts, key=delimiter_counts.get)


def extract_xm_source_type(file_path: Path) -> str:
    """Extrae el prefijo funcional de nombres XM como PRON_AREAS0407."""

    match = re.match(r"^(.*?)(\d{4})$", file_path.stem)
    if match:
        return match.group(1)
    return file_path.stem


def parse_xm_file_date(file_path: Path) -> pd.Timestamp | pd.NaT:
    """Reconstruye una fecha desde carpeta YYYY-MM y sufijo MMDD del archivo."""

    folder_match = re.search(r"(?P<year>\d{4})-(?P<month>\d{2})$", file_path.parent.name)
    suffix_match = re.search(r"(?P<month>\d{2})(?P<day>\d{2})$", file_path.stem)
    if not folder_match or not suffix_match:
        return pd.NaT

    year = int(folder_match.group("year"))
    month = int(suffix_match.group("month"))
    day = int(suffix_match.group("day"))

    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return pd.NaT


def _looks_numeric_token(value: Any) -> bool:
    """Determina si un token parece numerico."""

    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def build_xm_txt_column_names(file_path: Path, column_count: int) -> list[str]:
    """Asigna nombres utiles a TXT XM sin encabezado."""

    source_type = normalize_column_label(extract_xm_source_type(file_path))

    if "pron_barra" in source_type and column_count >= 5:
        metadata_columns = ["area_operativa", "barra", "nivel_tension", "variable"]
        value_columns = [
            f"hora_{index:02d}" for index in range(1, column_count - len(metadata_columns) + 1)
        ]
        return metadata_columns + value_columns

    if "pron_areas" in source_type and column_count >= 4:
        metadata_columns = ["subarea", "hora", "variable"]
        value_columns = [
            f"dia_{index:02d}" for index in range(1, column_count - len(metadata_columns) + 1)
        ]
        return metadata_columns + value_columns

    if source_type in {"pron_ucp", "pronucp"} and column_count >= 4:
        metadata_columns = ["unidad", "hora", "variable"]
        value_columns = [
            f"dia_{index:02d}" for index in range(1, column_count - len(metadata_columns) + 1)
        ]
        return metadata_columns + value_columns

    if source_type in {"pronsin"} and column_count >= 2:
        metadata_columns = ["hora"]
        value_columns = [
            f"dia_{index:02d}" for index in range(1, column_count - len(metadata_columns) + 1)
        ]
        return metadata_columns + value_columns

    if source_type in {"pron_sin"} and column_count >= 3:
        metadata_columns = ["sistema", "hora"]
        value_columns = [
            f"dia_{index:02d}" for index in range(1, column_count - len(metadata_columns) + 1)
        ]
        return metadata_columns + value_columns

    generic_columns: list[str] = []
    for index in range(column_count):
        if index < min(3, column_count):
            generic_columns.append(f"dimension_{index + 1}")
        else:
            generic_columns.append(f"valor_{index - 2:02d}")
    return generic_columns


def read_xm_txt_file(file_path: Path) -> pd.DataFrame:
    """Lee TXT delimitados de XM y agrega metadatos de contexto."""

    delimiter = sniff_delimiter(file_path)
    encodings = ("utf-8-sig", "utf-8", "latin-1")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            txt_df = pd.read_csv(
                file_path,
                sep=delimiter,
                header=None,
                encoding=encoding,
                dtype="string",
                engine="python",
            )
            break
        except Exception as error:
            last_error = error
    else:
        raise ValueError(
            f"No fue posible leer el TXT '{file_path.name}'."
        ) from last_error

    txt_df = txt_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if txt_df.empty:
        return txt_df

    txt_df.columns = build_xm_txt_column_names(file_path, txt_df.shape[1])
    txt_df["archivo_origen"] = file_path.name
    txt_df["carpeta_origen"] = file_path.parent.name
    txt_df["ruta_relativa_origen"] = str(file_path)
    txt_df["tipo_archivo"] = extract_xm_source_type(file_path)
    txt_df["fecha_archivo"] = parse_xm_file_date(file_path)
    return txt_df


def read_tabular_file(file_path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    """Carga archivos CSV o Excel con validaciones basicas."""

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Formato no soportado para '{file_path.name}'. Se esperaba CSV o Excel."
        )

    if suffix == ".csv":
        encodings = ("utf-8-sig", "utf-8", "latin-1")
        last_error: Exception | None = None
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding, sep=None, engine="python")
            except Exception as error:
                last_error = error
        raise ValueError(
            f"No fue posible leer el CSV '{file_path.name}'."
        ) from last_error

    if suffix == ".txt":
        return read_xm_txt_file(file_path)

    try:
        excel_sheet = sheet_name if sheet_name is not None else 0
        return pd.read_excel(file_path, sheet_name=excel_sheet)
    except ValueError as error:
        available = inspect_excel_sheets(file_path)
        raise ValueError(
            f"La hoja '{sheet_name}' no existe en '{file_path.name}'. "
            f"Hojas disponibles: {available}"
        ) from error
    except ImportError as error:
        raise ImportError(
            "No fue posible leer el archivo Excel. Instala openpyxl en el entorno."
        ) from error


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normaliza columnas y devuelve el mapeo original-normalizado."""

    original_columns = [str(column) for column in df.columns]
    normalized_columns = [normalize_column_label(column) for column in original_columns]
    unique_columns = make_unique(normalized_columns)

    mapping = pd.DataFrame(
        {
            "columna_original": original_columns,
            "columna_normalizada": unique_columns,
        }
    )

    normalized_df = df.copy()
    normalized_df.columns = unique_columns
    return normalized_df, mapping


def clean_string_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Limpia espacios y tokens nulos comunes en columnas de texto."""

    cleaned_df = df.copy()
    notes: list[str] = []

    for column in cleaned_df.columns:
        series = cleaned_df[column]
        if not (is_object_dtype(series) or is_string_dtype(series)):
            continue

        cleaned = series.astype("string")
        cleaned = cleaned.str.replace("\u00A0", " ", regex=False)
        cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
        cleaned = cleaned.str.strip()
        cleaned = cleaned.mask(cleaned.str.lower().isin(NULL_TOKENS), pd.NA)

        if not cleaned.equals(series.astype("string")):
            notes.append(
                f"Se limpiaron espacios o marcadores de nulos en la columna '{column}'."
            )

        cleaned_df[column] = cleaned

    return cleaned_df, notes


def _is_year_only_series(series: pd.Series) -> bool:
    """Indica si la serie parece contener solo anios, no fechas completas."""

    non_null = series.dropna()
    if non_null.empty:
        return False

    text = non_null.astype("string").str.strip()
    if not text.str.fullmatch(r"\d{4}").all():
        return False

    years = pd.to_numeric(text, errors="coerce")
    return years.between(1900, 2100, inclusive="both").all()


def should_try_datetime(series: pd.Series, column_name: str) -> bool:
    """Decide si una columna amerita intento de conversion a fecha."""

    normalized_name = normalize_column_label(column_name)
    if is_datetime64_any_dtype(series):
        return True

    if re.match(r"^(dia|hora)_\d+$", normalized_name):
        return False

    if normalized_name.endswith("_origen") and not any(
        keyword in normalized_name for keyword in DATE_KEYWORDS
    ):
        return False

    if normalized_name in {"ano", "anio", "year", "mes", "month"}:
        return False

    if is_numeric_dtype(series):
        return any(keyword in normalized_name for keyword in DATE_KEYWORDS)

    if not (is_object_dtype(series) or is_string_dtype(series)):
        return False

    non_null = series.dropna()
    if non_null.empty:
        return False

    text = non_null.astype("string").str.strip()
    if _is_year_only_series(text):
        return False

    if (
        normalized_name in DATE_EXACT_NAMES
        or normalized_name.startswith("fecha_")
        or normalized_name.endswith("_fecha")
        or normalized_name.endswith("_date")
    ):
        return True

    sample = text.head(100)
    ratio_with_separators = sample.str.contains(r"[/:-]", regex=True).mean()
    ratio_iso = sample.str.match(r"\d{4}[-/]\d{1,2}([-/]\d{1,2})?$", na=False).mean()
    ratio_month_text = sample.str.contains(
        r"ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic|jan|apr|aug|dec",
        case=False,
        regex=True,
    ).mean()

    return max(ratio_with_separators, ratio_iso, ratio_month_text) >= 0.5


def attempt_datetime_conversion(
    series: pd.Series, column_name: str
) -> tuple[pd.Series | None, str | None]:
    """Intenta convertir una columna a fecha sin forzar inferencias inseguras."""

    if is_datetime64_any_dtype(series):
        return series, "La columna ya estaba en formato fecha."

    non_null = series.dropna()
    if non_null.empty:
        return None, None

    candidates: list[tuple[float, str, pd.Series]] = []
    normalized_name = normalize_column_label(column_name)
    name_suggests_date = any(keyword in normalized_name for keyword in DATE_KEYWORDS)

    for dayfirst, label in ((True, "dayfirst=True"), (False, "dayfirst=False")):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format.*",
                category=UserWarning,
            )
            parsed = pd.to_datetime(
                series,
                errors="coerce",
                dayfirst=dayfirst,
                format="mixed",
            )
        ratio = parsed.notna().sum() / len(non_null)
        candidates.append((ratio, label, parsed))

    if is_numeric_dtype(series) and name_suggests_date:
        numeric_series = pd.to_numeric(series, errors="coerce")
        parsed_excel = pd.to_datetime(
            numeric_series, errors="coerce", unit="D", origin="1899-12-30"
        )
        ratio_excel = parsed_excel.notna().sum() / len(non_null)
        candidates.append((ratio_excel, "serial_excel", parsed_excel))

    best_ratio, best_label, best_series = max(candidates, key=lambda item: item[0])
    threshold = 0.60 if name_suggests_date else 0.80

    if best_ratio < threshold:
        return None, None

    return best_series, (
        f"Se convirtio '{column_name}' a fecha usando {best_label} "
        f"con exito sobre {best_ratio:.1%} de los valores no nulos."
    )


def _prepare_numeric_text(series: pd.Series) -> pd.Series:
    """Prepara texto numerico para evaluaciones de conversion."""

    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.str.replace("\u00A0", "", regex=False)
    cleaned = cleaned.str.replace(" ", "", regex=False)
    cleaned = cleaned.str.replace(r"^\((.+)\)$", r"-\1", regex=True)
    cleaned = cleaned.str.replace(r"[$€£%]", "", regex=True)
    cleaned = cleaned.str.replace(r"(?i)\b(cop|usd|eur)\b", "", regex=True)
    return cleaned.str.strip()


def _parse_numeric_candidate(text: pd.Series, mode: str) -> pd.Series:
    """Aplica una estrategia concreta de parseo numerico."""

    if mode == "identity":
        candidate = text
    elif mode == "comma_decimal":
        candidate = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    elif mode == "comma_thousand":
        candidate = text.str.replace(",", "", regex=False)
    elif mode == "dot_thousand":
        candidate = text.str.replace(".", "", regex=False)
    else:
        raise ValueError(f"Modo de parseo no soportado: {mode}")

    return pd.to_numeric(candidate, errors="coerce")


def attempt_numeric_conversion(
    series: pd.Series, column_name: str
) -> tuple[pd.Series | None, str | None]:
    """Intenta convertir una columna de texto a numerica si la evidencia es fuerte."""

    if is_numeric_dtype(series) or is_datetime64_any_dtype(series) or is_bool_dtype(series):
        return None, None

    normalized_name = normalize_column_label(column_name)
    if any(keyword in normalized_name for keyword in IDENTIFIER_KEYWORDS):
        return None, None

    non_null = series.dropna()
    if non_null.empty:
        return None, None

    text = _prepare_numeric_text(non_null)
    if text.empty:
        return None, None

    leading_zero_ratio = text.str.match(r"^0\d+$", na=False).mean()
    uniqueness_ratio = text.nunique(dropna=True) / max(len(text), 1)
    if leading_zero_ratio >= 0.5 and uniqueness_ratio >= 0.9:
        return None, None

    contains_comma = text.str.contains(",", regex=False).mean()
    contains_dot = text.str.contains(".", regex=False).mean()
    comma_decimal_like = text.str.contains(r",\d+$", regex=True).mean()
    comma_thousand_like = text.str.contains(r"^\d{1,3}(?:,\d{3})+$", regex=True).mean()
    dot_decimal_like = text.str.contains(r"\.\d+$", regex=True).mean()
    dot_thousand_like = text.str.contains(r"^\d{1,3}(?:\.\d{3})+$", regex=True).mean()

    candidate_modes = ["identity", "comma_decimal", "comma_thousand", "dot_thousand"]
    best_score = -1.0
    best_mode: str | None = None
    best_series: pd.Series | None = None

    for mode in candidate_modes:
        parsed_non_null = _parse_numeric_candidate(text, mode)
        ratio = parsed_non_null.notna().sum() / len(text)
        score = ratio

        if mode == "identity":
            score += 0.04 * dot_decimal_like
            score -= 0.08 * comma_thousand_like
        elif mode == "comma_decimal":
            score += 0.08 * comma_decimal_like
            score += 0.05 * (contains_comma > 0)
            score -= 0.10 * dot_decimal_like
        elif mode == "comma_thousand":
            score += 0.08 * comma_thousand_like
            score += 0.05 * (contains_comma > 0)
            score -= 0.05 * comma_decimal_like
        elif mode == "dot_thousand":
            score += 0.08 * dot_thousand_like
            score += 0.05 * (contains_dot > 0)
            score -= 0.10 * dot_decimal_like

        if score > best_score:
            best_score = score
            best_mode = mode
            best_series = parsed_non_null

    if best_mode is None or best_series is None:
        return None, None

    parse_ratio = best_series.notna().sum() / len(text)
    name_suggests_measure = any(
        keyword in normalized_name for keyword in NUMERIC_PRIORITY_KEYWORDS
    )
    threshold = 0.75 if name_suggests_measure else 0.90

    if parse_ratio < threshold:
        return None, None

    converted = pd.Series(pd.NA, index=series.index, dtype="Float64")
    converted.loc[non_null.index] = best_series.astype("Float64")

    return converted, (
        f"Se convirtio '{column_name}' a numerica usando el modo '{best_mode}' "
        f"con exito sobre {parse_ratio:.1%} de los valores no nulos."
    )


def preprocess_dataset(raw_df: pd.DataFrame) -> dict[str, Any]:
    """Aplica transformaciones minimas y deja trazabilidad completa."""

    df = raw_df.copy()
    df, column_mapping = normalize_columns(df)
    transformations = [
        "Se normalizaron los nombres de columnas a un formato estandar sin tildes."
    ]

    df, string_notes = clean_string_columns(df)
    transformations.extend(string_notes)

    converted_dates: list[str] = []
    for column in list(df.columns):
        if not should_try_datetime(df[column], column):
            continue

        converted_series, note = attempt_datetime_conversion(df[column], column)
        if converted_series is None or note is None:
            continue

        df[column] = converted_series
        converted_dates.append(column)
        transformations.append(note)

    converted_numeric: list[str] = []
    for column in list(df.columns):
        converted_series, note = attempt_numeric_conversion(df[column], column)
        if converted_series is None or note is None:
            continue

        df[column] = converted_series
        converted_numeric.append(column)
        transformations.append(note)

    derived_temporal_columns: list[str] = []
    for column in converted_dates:
        if not is_datetime64_any_dtype(df[column]):
            continue

        year_column = f"{column}_anio"
        month_column = f"{column}_mes"

        if year_column not in df.columns:
            df[year_column] = df[column].dt.year.astype("Int64")
            derived_temporal_columns.append(year_column)

        if month_column not in df.columns:
            df[month_column] = df[column].dt.month.astype("Int64")
            derived_temporal_columns.append(month_column)

    if derived_temporal_columns:
        transformations.append(
            "Se derivaron columnas temporales auxiliares: "
            + ", ".join(derived_temporal_columns)
            + "."
        )

    return {
        "raw_df": raw_df,
        "clean_df": df,
        "column_mapping": column_mapping,
        "transformations": transformations,
        "converted_dates": converted_dates,
        "converted_numeric": converted_numeric,
        "derived_temporal_columns": derived_temporal_columns,
    }


def is_derived_temporal_name(column_name: str) -> bool:
    """Identifica columnas auxiliares derivadas de fechas."""

    normalized_name = normalize_column_label(column_name)
    return (
        normalized_name in {"ano", "anio", "year", "mes", "month"}
        or normalized_name.endswith("_anio")
        or normalized_name.endswith("_ano")
        or normalized_name.endswith("_year")
        or normalized_name.endswith("_mes")
        or normalized_name.endswith("_month")
    )


def infer_column_role(series: pd.Series, total_rows: int) -> str:
    """Asigna un rol analitico simple a una columna."""

    if is_datetime64_any_dtype(series):
        return "fecha"
    if is_bool_dtype(series):
        return "booleana"
    if is_numeric_dtype(series):
        return "numerica"

    unique_non_null = series.nunique(dropna=True)
    limit = max(10, min(50, int(total_rows * 0.2))) if total_rows else 10
    if unique_non_null <= limit:
        return "categorica"
    return "texto_alta_cardinalidad"


def build_column_summary(preprocess_result: dict[str, Any]) -> pd.DataFrame:
    """Construye una tabla detallada de estructura y calidad por columna."""

    raw_df = preprocess_result["raw_df"]
    clean_df = preprocess_result["clean_df"]
    mapping = preprocess_result["column_mapping"].set_index("columna_normalizada")

    rows: list[dict[str, Any]] = []
    total_rows = len(clean_df)

    for column in clean_df.columns:
        series = clean_df[column]
        non_null = series.dropna()
        unique_non_null = int(non_null.nunique(dropna=True))
        nulls = int(series.isna().sum())
        null_pct = (nulls / total_rows * 100) if total_rows else 0.0
        cardinality_pct = (unique_non_null / total_rows * 100) if total_rows else 0.0
        is_constant = unique_non_null <= 1 and total_rows > 0
        is_candidate_key = nulls == 0 and unique_non_null == total_rows and total_rows > 0
        is_high_cardinality = unique_non_null >= max(20, int(total_rows * 0.5))
        if column in mapping.index:
            original_column = mapping.loc[column, "columna_original"]
            original_dtype = (
                str(raw_df[original_column].dtype)
                if original_column in raw_df.columns
                else "desconocido"
            )
        else:
            original_column = "<columna_derivada>"
            original_dtype = "derivada"
        sample_values = [str(value) for value in non_null.head(3).tolist()]

        rows.append(
            {
                "columna_original": original_column,
                "columna_normalizada": column,
                "tipo_original": original_dtype,
                "tipo_final": str(series.dtype),
                "rol_inferido": infer_column_role(series, total_rows),
                "nulos": nulls,
                "pct_nulos": round(null_pct, 2),
                "valores_unicos_no_nulos": unique_non_null,
                "pct_cardinalidad_sobre_filas": round(cardinality_pct, 2),
                "columna_constante": is_constant,
                "alta_cardinalidad": is_high_cardinality,
                "llave_candidata_simple": is_candidate_key,
                "convertida_a_fecha": column in preprocess_result["converted_dates"],
                "convertida_a_numerica": column in preprocess_result["converted_numeric"],
                "muestra_valores": " | ".join(sample_values),
            }
        )

    return pd.DataFrame(rows).sort_values("columna_normalizada").reset_index(drop=True)


def build_null_report(df: pd.DataFrame) -> pd.DataFrame:
    """Genera el reporte de nulos absoluto y porcentual por columna."""

    total_rows = len(df)
    rows = []
    for column in df.columns:
        nulls = int(df[column].isna().sum())
        rows.append(
            {
                "columna": column,
                "nulos": nulls,
                "pct_nulos": round((nulls / total_rows * 100) if total_rows else 0.0, 2),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["pct_nulos", "nulos", "columna"], ascending=[False, False, True]
    )


def build_cardinality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula cardinalidad y razon de unicidad por columna."""

    total_rows = len(df)
    rows = []
    for column in df.columns:
        non_null = df[column].dropna()
        unique_non_null = int(non_null.nunique(dropna=True))
        rows.append(
            {
                "columna": column,
                "valores_unicos_no_nulos": unique_non_null,
                "pct_cardinalidad_sobre_filas": round(
                    (unique_non_null / total_rows * 100) if total_rows else 0.0, 2
                ),
                "pct_cobertura_no_nulos": round(
                    (non_null.size / total_rows * 100) if total_rows else 0.0, 2
                ),
                "columna_constante": unique_non_null <= 1 and total_rows > 0,
                "alta_cardinalidad": unique_non_null >= max(20, int(total_rows * 0.5)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["pct_cardinalidad_sobre_filas", "columna"], ascending=[False, True]
    )


def build_numeric_descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    """Genera estadisticas descriptivas para variables numericas."""

    numeric_columns = [column for column in df.columns if is_numeric_dtype(df[column])]
    if not numeric_columns:
        return pd.DataFrame()

    description = (
        df[numeric_columns]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        .transpose()
        .reset_index()
        .rename(columns={"index": "columna"})
    )
    return description


def build_categorical_count_tables(
    df: pd.DataFrame, max_tables: int = 8, top_n: int = 20
) -> dict[str, pd.DataFrame]:
    """Construye conteos para variables categoricas manejables."""

    count_tables: dict[str, pd.DataFrame] = {}
    total_rows = len(df)

    candidates: list[tuple[float, str]] = []
    for column in df.columns:
        series = df[column]
        if is_datetime64_any_dtype(series) or is_numeric_dtype(series):
            continue

        unique_non_null = series.nunique(dropna=True)
        if unique_non_null <= 1:
            continue

        score = 0.0
        normalized_name = normalize_column_label(column)
        if any(keyword in normalized_name for keyword in CATEGORY_PRIORITY_KEYWORDS):
            score += 100.0
        score += max(0.0, 40.0 - unique_non_null)
        score += min(series.notna().sum() / max(total_rows, 1), 1.0) * 10.0
        candidates.append((score, column))

    for _, column in sorted(candidates, reverse=True)[:max_tables]:
        counts = (
            df[column]
            .fillna("<<NULO>>")
            .value_counts(dropna=False)
            .head(top_n)
            .rename_axis("categoria")
            .reset_index(name="conteo")
        )
        counts["pct_sobre_filas"] = round((counts["conteo"] / max(total_rows, 1)) * 100, 2)
        count_tables[column] = counts

    return count_tables


def _is_key_friendly_column(df: pd.DataFrame, column: str) -> bool:
    """Filtra columnas razonables para explorar llaves candidatas compuestas."""

    series = df[column]
    normalized_name = normalize_column_label(column)

    if series.nunique(dropna=True) <= 1:
        return False

    if any(keyword in normalized_name for keyword in NUMERIC_PRIORITY_KEYWORDS):
        return False

    if is_bool_dtype(series):
        return False

    if is_numeric_dtype(series):
        if pd.api.types.is_float_dtype(series):
            return False
        if any(keyword in normalized_name for keyword in MEASURE_EXCLUSION_KEYWORDS):
            return False

    return True


def detect_candidate_keys(df: pd.DataFrame, max_pool_size: int = 8) -> pd.DataFrame:
    """Busca llaves candidatas simples y compuestas."""

    total_rows = len(df)
    if total_rows == 0:
        return pd.DataFrame(
            columns=[
                "tipo_llave",
                "columnas",
                "numero_columnas",
                "duplicados",
                "nulos_en_columnas",
                "clasificacion",
            ]
        )

    rows: list[dict[str, Any]] = []
    pool: list[tuple[float, str]] = []

    for column in df.columns:
        series = df[column]
        nulls = int(series.isna().sum())
        unique_non_null = int(series.nunique(dropna=True))
        duplicates = total_rows - unique_non_null
        uniqueness_ratio = unique_non_null / total_rows
        normalized_name = normalize_column_label(column)
        key_friendly = _is_key_friendly_column(df, column)

        if key_friendly and nulls == 0 and unique_non_null == total_rows:
            rows.append(
                {
                    "tipo_llave": "simple",
                    "columnas": column,
                    "numero_columnas": 1,
                    "duplicados": 0,
                    "nulos_en_columnas": 0,
                    "clasificacion": "llave_candidata",
                }
            )
        elif key_friendly and nulls == 0 and uniqueness_ratio >= 0.98:
            rows.append(
                {
                    "tipo_llave": "simple",
                    "columnas": column,
                    "numero_columnas": 1,
                    "duplicados": duplicates,
                    "nulos_en_columnas": 0,
                    "clasificacion": "llave_posible",
                }
            )

        if not key_friendly:
            continue

        score = uniqueness_ratio * 100
        if any(keyword in normalized_name for keyword in IDENTIFIER_KEYWORDS):
            score += 30
        if any(keyword in normalized_name for keyword in DATE_KEYWORDS):
            score += 10
        if any(keyword in normalized_name for keyword in CATEGORY_PRIORITY_KEYWORDS):
            score += 5
        pool.append((score, column))

    composite_pool = [column for _, column in sorted(pool, reverse=True)[:max_pool_size]]

    for size in (2, 3):
        for combo in itertools.combinations(composite_pool, size):
            subset = df.loc[:, combo]
            nulls = int(subset.isna().any(axis=1).sum())
            duplicates = int(subset.duplicated().sum())
            if nulls == 0 and duplicates == 0:
                rows.append(
                    {
                        "tipo_llave": "compuesta",
                        "columnas": " + ".join(combo),
                        "numero_columnas": size,
                        "duplicados": 0,
                        "nulos_en_columnas": 0,
                        "clasificacion": "llave_candidata",
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "tipo_llave",
                "columnas",
                "numero_columnas",
                "duplicados",
                "nulos_en_columnas",
                "clasificacion",
            ]
        )

    candidate_keys = pd.DataFrame(rows).drop_duplicates()
    return candidate_keys.sort_values(
        ["clasificacion", "numero_columnas", "columnas"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def detect_relevant_numeric_columns(df: pd.DataFrame, limit: int = 6) -> list[str]:
    """Prioriza variables numericas para visualizacion."""

    candidates: list[tuple[float, str]] = []
    total_rows = len(df)
    for column in df.columns:
        series = df[column]
        if not is_numeric_dtype(series):
            continue

        if series.nunique(dropna=True) <= 1:
            continue

        normalized_name = normalize_column_label(column)
        non_null = series.dropna()
        uniqueness_ratio = (
            non_null.nunique(dropna=True) / max(len(non_null), 1) if not non_null.empty else 0.0
        )
        if normalized_name in {"hora", "hour", "mes", "month", "ano", "anio", "year"}:
            continue
        if any(keyword in normalized_name for keyword in IDENTIFIER_KEYWORDS):
            continue
        if is_derived_temporal_name(column):
            continue
        is_xm_projection_column = bool(re.match(r"^(dia|hora)_\d+$", normalized_name))
        if (
            uniqueness_ratio >= 0.95
            and not any(keyword in normalized_name for keyword in NUMERIC_PRIORITY_KEYWORDS)
            and not is_xm_projection_column
        ):
            continue

        score = 0.0
        if any(keyword in normalized_name for keyword in NUMERIC_PRIORITY_KEYWORDS):
            score += 100.0
        if is_xm_projection_column:
            score += 60.0
        score += min(series.notna().sum() / max(total_rows, 1), 1.0) * 20.0
        score += min(series.nunique(dropna=True), 100) / 5.0
        candidates.append((score, column))

    return [column for _, column in sorted(candidates, reverse=True)[:limit]]


def detect_relevant_categorical_columns(df: pd.DataFrame, limit: int = 5) -> list[str]:
    """Prioriza variables categoricas utiles para comparaciones."""

    candidates: list[tuple[float, str]] = []
    total_rows = len(df)
    for column in df.columns:
        series = df[column]
        if is_numeric_dtype(series) or is_datetime64_any_dtype(series):
            continue

        unique_non_null = series.nunique(dropna=True)
        if unique_non_null <= 1:
            continue

        normalized_name = normalize_column_label(column)
        score = 0.0
        if any(keyword in normalized_name for keyword in CATEGORY_PRIORITY_KEYWORDS):
            score += 100.0
        if unique_non_null <= 30:
            score += 40.0 - unique_non_null
        score += min(series.notna().sum() / max(total_rows, 1), 1.0) * 10.0
        candidates.append((score, column))

    return [column for _, column in sorted(candidates, reverse=True)[:limit]]


def select_primary_datetime_column(df: pd.DataFrame) -> str | None:
    """Selecciona la mejor columna temporal disponible."""

    candidates: list[tuple[float, str]] = []
    total_rows = len(df)
    for column in df.columns:
        series = df[column]
        if not is_datetime64_any_dtype(series):
            continue

        normalized_name = normalize_column_label(column)
        score = min(series.notna().sum() / max(total_rows, 1), 1.0) * 10.0
        if any(keyword in normalized_name for keyword in DATE_KEYWORDS):
            score += 50.0
        candidates.append((score, column))

    if not candidates:
        return None

    return sorted(candidates, reverse=True)[0][1]


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Exporta un DataFrame a CSV UTF-8 con separador coma."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_text_lines(lines: list[str], output_path: Path) -> None:
    """Guarda observaciones en un archivo de texto."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def create_histograms(df: pd.DataFrame, columns: list[str], output_dir: Path) -> list[str]:
    """Genera histogramas simples para variables numericas relevantes."""

    created_files: list[str] = []
    for column in columns:
        data = pd.to_numeric(df[column], errors="coerce").dropna()
        if data.empty:
            continue

        fig, axis = plt.subplots(figsize=(8, 5))
        axis.hist(data.to_numpy(), bins=30)
        axis.set_title(f"Histograma de {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Frecuencia")
        fig.tight_layout()

        output_path = output_dir / f"histograma_{bounded_slug(column)}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        created_files.append(output_path.name)

    return created_files


def create_boxplots(df: pd.DataFrame, columns: list[str], output_dir: Path) -> list[str]:
    """Genera boxplots para variables numericas relevantes."""

    created_files: list[str] = []
    for column in columns:
        data = pd.to_numeric(df[column], errors="coerce").dropna()
        if data.empty:
            continue

        fig, axis = plt.subplots(figsize=(8, 4))
        axis.boxplot(data.to_numpy(), vert=False)
        axis.set_title(f"Boxplot de {column}")
        axis.set_xlabel(column)
        fig.tight_layout()

        output_path = output_dir / f"boxplot_{bounded_slug(column)}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        created_files.append(output_path.name)

    return created_files


def create_categorical_bars(
    count_tables: dict[str, pd.DataFrame], output_dir: Path
) -> list[str]:
    """Genera graficos de barras para categorias relevantes."""

    created_files: list[str] = []
    for column, table in count_tables.items():
        if table.empty:
            continue

        fig, axis = plt.subplots(figsize=(9, 5))
        axis.bar(table["categoria"].astype(str), table["conteo"])
        axis.set_title(f"Top categorias de {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Conteo")
        axis.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        output_path = output_dir / f"categorias_{bounded_slug(column)}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        created_files.append(output_path.name)

    return created_files


def _infer_time_frequency(series: pd.Series) -> str | None:
    """Sugiere una frecuencia de agregacion razonable para series temporales."""

    unique_dates = series.dropna().sort_values().dt.normalize().nunique()
    if unique_dates <= 1:
        return None
    if unique_dates > 365:
        return "M"
    if unique_dates > 90:
        return "W"
    return "D"


def create_temporal_plots(
    df: pd.DataFrame,
    datetime_column: str | None,
    numeric_columns: list[str],
    output_dir: Path,
) -> list[str]:
    """Genera series temporales agregadas para variables numericas clave."""

    if datetime_column is None:
        return []

    created_files: list[str] = []
    time_series = df[[datetime_column] + numeric_columns].copy()
    time_series = time_series.dropna(subset=[datetime_column])
    if time_series.empty:
        return created_files

    frequency = _infer_time_frequency(time_series[datetime_column])
    for column in numeric_columns[:3]:
        data = time_series[[datetime_column, column]].dropna()
        if data.empty:
            continue

        data = data.sort_values(datetime_column)
        if frequency is not None:
            aggregated = (
                data.set_index(datetime_column)[column]
                .resample(frequency)
                .mean()
                .dropna()
                .reset_index()
            )
        else:
            aggregated = data

        if aggregated.empty:
            continue

        fig, axis = plt.subplots(figsize=(9, 5))
        axis.plot(aggregated[datetime_column], aggregated[column])
        axis.set_title(f"Serie temporal de {column}")
        axis.set_xlabel(datetime_column)
        axis.set_ylabel(column)
        fig.autofmt_xdate()
        fig.tight_layout()

        output_path = output_dir / f"serie_temporal_{bounded_slug(column)}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        created_files.append(output_path.name)

    return created_files


def create_category_comparison_plots(
    df: pd.DataFrame,
    category_columns: list[str],
    numeric_columns: list[str],
    output_dir: Path,
) -> list[str]:
    """Genera comparaciones por categoria cuando existen variables compatibles."""

    created_files: list[str] = []
    if not category_columns or not numeric_columns:
        return created_files

    for category_column in category_columns[:3]:
        for numeric_column in numeric_columns[:2]:
            data = df[[category_column, numeric_column]].dropna()
            if data.empty:
                continue

            summary = (
                data.groupby(category_column, dropna=False)[numeric_column]
                .mean()
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )
            if summary.empty:
                continue

            fig, axis = plt.subplots(figsize=(9, 5))
            axis.bar(summary[category_column].astype(str), summary[numeric_column])
            axis.set_title(f"{numeric_column} promedio por {category_column}")
            axis.set_xlabel(category_column)
            axis.set_ylabel(f"Promedio de {numeric_column}")
            axis.tick_params(axis="x", rotation=45)
            fig.tight_layout()

            output_path = output_dir / (
                f"comparacion_{bounded_slug(numeric_column, 50)}_por_{bounded_slug(category_column, 50)}.png"
            )
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            created_files.append(output_path.name)

    return created_files


def create_missing_heatmap(df: pd.DataFrame, output_dir: Path) -> str | None:
    """Genera un heatmap simple de nulos si aporta valor."""

    null_columns = [column for column in df.columns if df[column].isna().any()]
    if len(null_columns) < 2 or len(df) < 2:
        return None

    sample = df[null_columns].isna().iloc[: min(len(df), 300)].transpose().astype(int)

    fig, axis = plt.subplots(figsize=(10, max(4, len(null_columns) * 0.4)))
    image = axis.imshow(sample, aspect="auto", interpolation="nearest")
    axis.set_title("Mapa de nulos")
    axis.set_xlabel("Filas muestreadas")
    axis.set_ylabel("Columnas")
    axis.set_yticks(range(len(sample.index)))
    axis.set_yticklabels(sample.index)
    fig.colorbar(image, ax=axis, label="Nulo")
    fig.tight_layout()

    output_path = output_dir / "heatmap_nulos.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path.name


def save_categorical_tables(count_tables: dict[str, pd.DataFrame], output_dir: Path) -> list[str]:
    """Exporta tablas de frecuencia por categoria."""

    created_files: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for column, table in count_tables.items():
        output_path = output_dir / f"frecuencia_{bounded_slug(column)}.csv"
        table.to_csv(output_path, index=False, encoding="utf-8-sig")
        created_files.append(output_path.name)
    return created_files


def source_output_dir(file_path: Path, output_root: Path, sheet_name: str | None = None) -> Path:
    """Calcula la carpeta de salida para una fuente."""

    folder_name = bounded_slug(
        file_path.stem if sheet_name is None else f"{file_path.stem}_{sheet_name}",
        max_length=120,
    )
    return output_root / folder_name


def is_analysis_complete(output_dir: Path) -> bool:
    """Indica si una fuente ya tiene las salidas minimas completas."""

    required_files = [
        "resumen_columnas.csv",
        "nulos_por_columna.csv",
        "cardinalidad.csv",
        "mapeo_columnas.csv",
        "llaves_candidatas.csv",
        "dataset_limpio_minimo.csv",
        "observaciones_calidad.txt",
    ]
    return output_dir.exists() and all((output_dir / file_name).exists() for file_name in required_files)


def build_quality_observations(
    source_label: str,
    preprocess_result: dict[str, Any],
    column_summary: pd.DataFrame,
    null_report: pd.DataFrame,
    candidate_keys: pd.DataFrame,
    exact_duplicates: int,
    relevant_numeric: list[str],
    relevant_categorical: list[str],
    datetime_column: str | None,
) -> list[str]:
    """Redacta un resumen automatico de hallazgos para consola y archivo."""

    df = preprocess_result["clean_df"]
    total_rows, total_columns = df.shape
    observations: list[str] = []

    observations.append(f"Fuente analizada: {source_label}")
    observations.append(f"Dimension del dataset: {total_rows} filas x {total_columns} columnas.")
    observations.append(
        "Columnas detectadas: " + ", ".join(df.columns.tolist()) if total_columns else "Sin columnas detectadas."
    )
    observations.append("")
    observations.append("Transformaciones minimas aplicadas:")
    observations.extend(
        [f"- {note}" for note in preprocess_result["transformations"]]
        if preprocess_result["transformations"]
        else ["- No se requirieron transformaciones."]
    )

    observations.append("")
    observations.append("Hallazgos de calidad:")
    observations.append(f"- Filas duplicadas exactas: {exact_duplicates}.")

    high_null_columns = null_report[null_report["pct_nulos"] >= 20]["columna"].tolist()
    observations.append(
        "- Columnas con nulos >= 20%: "
        + (", ".join(high_null_columns) if high_null_columns else "ninguna.")
    )

    constant_columns = column_summary[column_summary["columna_constante"]][
        "columna_normalizada"
    ].tolist()
    observations.append(
        "- Columnas constantes: "
        + (", ".join(constant_columns) if constant_columns else "ninguna.")
    )

    high_cardinality_columns = column_summary[column_summary["alta_cardinalidad"]][
        "columna_normalizada"
    ].tolist()
    observations.append(
        "- Columnas de alta cardinalidad: "
        + (", ".join(high_cardinality_columns) if high_cardinality_columns else "ninguna.")
    )

    suspicious_measure_columns = [
        column
        for column in df.columns
        if any(keyword in normalize_column_label(column) for keyword in NUMERIC_PRIORITY_KEYWORDS)
        and not is_numeric_dtype(df[column])
    ]
    observations.append(
        "- Columnas de interes energetico no convertidas a numerico: "
        + (
            ", ".join(suspicious_measure_columns)
            if suspicious_measure_columns
            else "ninguna."
        )
    )

    suspicious_date_columns = [
        column
        for column in df.columns
        if any(keyword in normalize_column_label(column) for keyword in DATE_KEYWORDS)
        and not is_derived_temporal_name(column)
        and not is_datetime64_any_dtype(df[column])
    ]
    observations.append(
        "- Columnas con semantica temporal no convertidas a fecha: "
        + (", ".join(suspicious_date_columns) if suspicious_date_columns else "ninguna.")
    )

    observations.append("")
    observations.append("Variables prioritarias para el informe:")
    observations.append(
        "- Variables numericas relevantes: "
        + (", ".join(relevant_numeric) if relevant_numeric else "no se detectaron.")
    )
    observations.append(
        "- Variables categoricas relevantes: "
        + (", ".join(relevant_categorical) if relevant_categorical else "no se detectaron.")
    )
    observations.append(
        "- Columna temporal principal: "
        + (datetime_column if datetime_column else "no se detecto.")
    )

    observations.append("")
    observations.append("Posibles llaves para integracion futura:")
    if candidate_keys.empty:
        observations.append("- No se detectaron llaves candidatas exactas o casi unicas.")
    else:
        for row in candidate_keys.head(10).to_dict("records"):
            observations.append(
                f"- {row['clasificacion']} ({row['tipo_llave']}): {row['columnas']}."
            )

    observations.append("")
    observations.append("Limitaciones detectadas para el proyecto:")
    limitation_count = 0
    if datetime_column is None:
        observations.append("- No se detecto una fecha confiable para analisis temporal.")
        limitation_count += 1
    if not relevant_numeric:
        observations.append("- No se identificaron medidas numericas prioritarias para demanda o energia.")
        limitation_count += 1
    if not relevant_categorical:
        observations.append("- No se detectaron categorias geograficas o de mercado claramente utilizables.")
        limitation_count += 1
    if exact_duplicates > 0:
        observations.append("- Existen duplicados exactos que deben revisarse antes de integrar fuentes.")
        limitation_count += 1
    if limitation_count == 0:
        observations.append("- El dataset puede analizarse, pero requiere contraste posterior con otras fuentes del proyecto.")

    return observations


def analyze_source(
    file_path: Path,
    output_root: Path,
    sheet_name: str | None = None,
    make_graphs: bool = False,
) -> dict[str, Any]:
    """Ejecuta el EDA completo de una fuente individual y exporta sus salidas."""

    raw_df = read_tabular_file(file_path, sheet_name=sheet_name)
    if raw_df.empty:
        raise ValueError("El dataset cargado esta vacio.")

    source_label = file_path.name if sheet_name is None else f"{file_path.name} | hoja={sheet_name}"
    output_dir = source_output_dir(file_path, output_root, sheet_name)
    graphs_dir = output_dir / "graficos"
    categorical_tables_dir = output_dir / "tablas_categoricas"

    output_dir.mkdir(parents=True, exist_ok=True)
    if make_graphs:
        graphs_dir.mkdir(parents=True, exist_ok=True)

    preprocess_result = preprocess_dataset(raw_df)
    clean_df = preprocess_result["clean_df"]
    column_summary = build_column_summary(preprocess_result)
    null_report = build_null_report(clean_df)
    cardinality_report = build_cardinality_report(clean_df)
    numeric_description = build_numeric_descriptive_table(clean_df)
    categorical_count_tables = build_categorical_count_tables(clean_df)
    candidate_keys = detect_candidate_keys(clean_df)
    exact_duplicates = int(clean_df.duplicated().sum())
    relevant_numeric = detect_relevant_numeric_columns(clean_df)
    relevant_categorical = detect_relevant_categorical_columns(clean_df)
    datetime_column = select_primary_datetime_column(clean_df)

    observations = build_quality_observations(
        source_label=source_label,
        preprocess_result=preprocess_result,
        column_summary=column_summary,
        null_report=null_report,
        candidate_keys=candidate_keys,
        exact_duplicates=exact_duplicates,
        relevant_numeric=relevant_numeric,
        relevant_categorical=relevant_categorical,
        datetime_column=datetime_column,
    )

    save_dataframe(column_summary, output_dir / "resumen_columnas.csv")
    save_dataframe(null_report, output_dir / "nulos_por_columna.csv")
    save_dataframe(cardinality_report, output_dir / "cardinalidad.csv")
    save_dataframe(preprocess_result["column_mapping"], output_dir / "mapeo_columnas.csv")
    save_dataframe(candidate_keys, output_dir / "llaves_candidatas.csv")
    save_dataframe(clean_df, output_dir / "dataset_limpio_minimo.csv")
    save_text_lines(observations, output_dir / "observaciones_calidad.txt")

    if not numeric_description.empty:
        save_dataframe(numeric_description, output_dir / "estadisticas_numericas.csv")

    if exact_duplicates > 0:
        duplicate_rows = clean_df.loc[clean_df.duplicated(keep=False)].copy()
        save_dataframe(duplicate_rows, output_dir / "filas_duplicadas_exactas.csv")

    exported_category_tables = save_categorical_tables(
        categorical_count_tables, categorical_tables_dir
    )

    created_graphs: list[str] = []
    if make_graphs:
        created_graphs.extend(create_histograms(clean_df, relevant_numeric, graphs_dir))
        created_graphs.extend(create_boxplots(clean_df, relevant_numeric, graphs_dir))
        created_graphs.extend(create_categorical_bars(categorical_count_tables, graphs_dir))
        created_graphs.extend(
            create_temporal_plots(clean_df, datetime_column, relevant_numeric, graphs_dir)
        )
        created_graphs.extend(
            create_category_comparison_plots(
                clean_df, relevant_categorical, relevant_numeric, graphs_dir
            )
        )
        heatmap_name = create_missing_heatmap(clean_df, graphs_dir)
        if heatmap_name:
            created_graphs.append(heatmap_name)

    return {
        "source_label": source_label,
        "output_dir": output_dir,
        "rows": clean_df.shape[0],
        "columns": clean_df.shape[1],
        "column_names": clean_df.columns.tolist(),
        "exact_duplicates": exact_duplicates,
        "datetime_column": datetime_column,
        "relevant_numeric": relevant_numeric,
        "relevant_categorical": relevant_categorical,
        "candidate_keys": candidate_keys,
        "observations": observations,
        "created_graphs": created_graphs,
        "exported_category_tables": exported_category_tables,
    }
