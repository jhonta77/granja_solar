from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

import pandas as pd


def strip_accents(value: str) -> str:
    """Elimina acentos para facilitar comparaciones de texto."""

    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def slugify(value: str) -> str:
    """Genera nombres seguros para archivos, ids y etiquetas."""

    text = strip_accents(str(value)).lower().strip()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text.strip("_") or "dataset"


def normalize_column_label(column_name: object) -> str:
    """Estandariza nombres de columnas a formato snake_case simple."""

    text = strip_accents(str(column_name)).lower().strip()
    text = re.sub(r"[^\w\s/.-]", " ", text)
    text = re.sub(r"[/.\s-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "columna"


def sniff_delimiter(file_path: Path) -> str:
    """Infiere el delimitador principal de un TXT tabular."""

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
    """Lee TXT delimitados de XM y agrega metadatos base de contexto."""

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
        raise ValueError(f"No fue posible leer el TXT '{file_path.name}'.") from last_error

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
