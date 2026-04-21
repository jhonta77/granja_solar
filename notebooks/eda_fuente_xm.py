from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    if __package__ is None or __package__ == "":
        CURRENT_DIR = Path(__file__).resolve().parent
        if str(CURRENT_DIR) not in sys.path:
            sys.path.insert(0, str(CURRENT_DIR))
        from eda_utils import (  # type: ignore
            analyze_source,
            ensure_project_folders,
            is_analysis_complete,
            inspect_excel_sheets,
            list_supported_files,
            slugify,
            source_output_dir,
        )
    else:
        from .eda_utils import (
            analyze_source,
            ensure_project_folders,
            is_analysis_complete,
            inspect_excel_sheets,
            list_supported_files,
            slugify,
            source_output_dir,
        )
except ModuleNotFoundError as error:
    missing_name = getattr(error, "name", None)
    if missing_name in {"eda_utils", "notebooks"}:
        raise
    raise SystemExit(
        "Falta una dependencia de Python para ejecutar el EDA. "
        f"Modulo faltante: {missing_name}. "
        "Instala o activa un entorno con pandas, matplotlib y openpyxl."
    ) from error


def build_argument_parser() -> argparse.ArgumentParser:
    """Configura argumentos para ejecutar el EDA desde VS Code o terminal."""

    parser = argparse.ArgumentParser(
        description=(
            "EDA reproducible para fuentes individuales del proyecto de "
            "seleccion de zonas viables para granja solar en Colombia."
        )
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Ruta opcional a un archivo especifico. Si no se indica, se procesan "
            "todos los CSV y Excel dentro de data/raw."
        ),
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="Nombre de hoja a procesar cuando la fuente es Excel.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Carpeta raiz de salida. Por defecto usa data/clean.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocesa fuentes aunque ya existan salidas completas.",
    )
    parser.add_argument(
        "--make-graphs",
        action="store_true",
        help="Genera PNG del EDA. Por defecto no se crean diagramas para evitar ruido.",
    )
    return parser


def resolve_source_path(project_root: Path, raw_dir: Path, source_arg: str | None) -> list[Path]:
    """Resuelve la fuente indicada por el usuario o usa data/raw por defecto."""

    if source_arg is None:
        return list_supported_files(raw_dir)

    candidate = Path(source_arg)
    if candidate.exists():
        if candidate.is_dir():
            return list_supported_files(candidate.resolve())
        return [candidate.resolve()]

    candidate_in_raw = raw_dir / source_arg
    if candidate_in_raw.exists():
        if candidate_in_raw.is_dir():
            return list_supported_files(candidate_in_raw.resolve())
        return [candidate_in_raw.resolve()]

    recursive_matches = sorted(
        [path.resolve() for path in raw_dir.rglob(source_arg) if path.is_file()]
    )
    if recursive_matches:
        return recursive_matches

    raise FileNotFoundError(
        f"No se encontro la fuente '{source_arg}'. Verifica la ruta o copia el archivo en {raw_dir}."
    )


def print_dataset_summary(result: dict[str, object]) -> None:
    """Imprime un resumen compacto y sustentable por dataset."""

    print("\n" + "=" * 80)
    print(f"Fuente: {result['source_label']}")
    print(f"Dimension: {result['rows']} filas x {result['columns']} columnas")
    print(f"Columnas: {', '.join(result['column_names'])}")
    print(f"Duplicados exactos: {result['exact_duplicates']}")

    datetime_column = result["datetime_column"]
    print(
        "Columna temporal principal: "
        + (str(datetime_column) if datetime_column else "no detectada")
    )

    relevant_numeric = result["relevant_numeric"]
    print(
        "Variables numericas relevantes: "
        + (", ".join(relevant_numeric) if relevant_numeric else "no detectadas")
    )

    relevant_categorical = result["relevant_categorical"]
    print(
        "Variables categoricas relevantes: "
        + (", ".join(relevant_categorical) if relevant_categorical else "no detectadas")
    )

    candidate_keys = result["candidate_keys"]
    if hasattr(candidate_keys, "empty") and not candidate_keys.empty:  # type: ignore[attr-defined]
        key_strings = candidate_keys["columnas"].head(5).tolist()  # type: ignore[index]
        print("Posibles llaves de integracion futura: " + ", ".join(key_strings))
    else:
        print("Posibles llaves de integracion futura: no detectadas")

    output_dir = result["output_dir"]
    print(f"Salidas exportadas en: {output_dir}")
    print("Hallazgos principales:")
    for line in result["observations"][:20]:
        print(line)


def write_global_summary(results: list[dict[str, object]], output_root: Path) -> Path:
    """Consolida un resumen global de la corrida."""

    summary_lines: list[str] = []
    summary_lines.append("Resumen global del EDA por fuente")
    summary_lines.append("")

    if not results:
        summary_lines.append("No se procesaron fuentes.")
    else:
        for result in results:
            candidate_keys = result["candidate_keys"]
            if hasattr(candidate_keys, "empty") and not candidate_keys.empty:  # type: ignore[attr-defined]
                keys_text = ", ".join(candidate_keys["columnas"].head(3).tolist())  # type: ignore[index]
            else:
                keys_text = "no detectadas"

            summary_lines.append(f"Fuente: {result['source_label']}")
            summary_lines.append(
                f"- Dimension: {result['rows']} filas x {result['columns']} columnas."
            )
            summary_lines.append(
                f"- Duplicados exactos: {result['exact_duplicates']}."
            )
            summary_lines.append(
                "- Variables numericas relevantes: "
                + (
                    ", ".join(result["relevant_numeric"])
                    if result["relevant_numeric"]
                    else "no detectadas"
                )
            )
            summary_lines.append(
                "- Variables categoricas relevantes: "
                + (
                    ", ".join(result["relevant_categorical"])
                    if result["relevant_categorical"]
                    else "no detectadas"
                )
            )
            summary_lines.append(f"- Llaves candidatas: {keys_text}.")
            summary_lines.append(f"- Carpeta de salida: {result['output_dir']}")
            summary_lines.append("")

    summary_path = output_root / "resumen_eda_global.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return summary_path


def main() -> int:
    """Punto de entrada del EDA por fuente individual."""

    project_root = Path(__file__).resolve().parents[1]
    folders = ensure_project_folders(project_root)
    raw_dir = folders["raw_dir"]
    default_output_root = folders["clean_dir"]

    parser = build_argument_parser()
    args = parser.parse_args()

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else default_output_root.resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    failure_path = output_root / f"errores_{slugify('eda_fuentes')}.txt"
    if failure_path.exists():
        failure_path.unlink()

    try:
        source_files = resolve_source_path(project_root, raw_dir, args.source)
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        return 1

    if not source_files:
        print(
            f"No se encontraron archivos CSV, Excel o TXT en {raw_dir}. "
            "Copia las fuentes a data/raw o usa --source."
        )
        return 0

    print("EDA por fuente individual para el proyecto de granja solar")
    print(f"Directorio base del proyecto: {project_root}")
    print(f"Directorio de entrada: {raw_dir}")
    print(f"Directorio de salida: {output_root}")

    results: list[dict[str, object]] = []
    failures: list[str] = []
    skipped: list[Path] = []

    for file_path in source_files:
        suffix = file_path.suffix.lower()
        sheet_names: list[str | None]

        if suffix in {".xlsx", ".xls", ".xlsm"}:
            try:
                available_sheets = inspect_excel_sheets(file_path)
            except Exception as error:
                failures.append(f"{file_path.name}: {error}")
                print(f"\nERROR al inspeccionar hojas de {file_path.name}: {error}")
                continue

            print(f"\nArchivo Excel detectado: {file_path.name}")
            print("Hojas disponibles: " + ", ".join(available_sheets))

            if args.sheet:
                if args.sheet not in available_sheets:
                    failures.append(
                        f"{file_path.name}: la hoja '{args.sheet}' no existe."
                    )
                    print(
                        f"ADVERTENCIA: la hoja '{args.sheet}' no existe en {file_path.name}. Se omite."
                    )
                    continue
                sheet_names = [args.sheet]
            else:
                sheet_names = available_sheets
        else:
            sheet_names = [None]

        for sheet_name in sheet_names:
            output_dir = source_output_dir(file_path, output_root=output_root, sheet_name=sheet_name)
            if not args.no_resume and is_analysis_complete(output_dir):
                skipped.append(output_dir)
                label = file_path.name if sheet_name is None else f"{file_path.name} | {sheet_name}"
                print(f"\nOMITIDO por resume: {label}")
                print(f"Salida existente: {output_dir}")
                continue

            try:
                result = analyze_source(
                    file_path,
                    output_root=output_root,
                    sheet_name=sheet_name,
                    make_graphs=args.make_graphs,
                )
                results.append(result)
                print_dataset_summary(result)
            except Exception as error:
                label = file_path.name if sheet_name is None else f"{file_path.name} | {sheet_name}"
                failures.append(f"{label}: {error}")
                print(f"\nERROR al procesar {label}: {error}")

    summary_path = write_global_summary(results, output_root)
    print("\n" + "=" * 80)
    print(f"Resumen global exportado en: {summary_path}")
    if skipped:
        skipped_path = output_root / "fuentes_omitidas_por_resume.txt"
        skipped_path.write_text(
            "\n".join(str(path) for path in skipped),
            encoding="utf-8",
        )
        print(f"Fuentes omitidas por resume: {len(skipped)}")
        print(f"Listado en: {skipped_path}")
    else:
        skipped_path = output_root / "fuentes_omitidas_por_resume.txt"
        if skipped_path.exists():
            skipped_path.unlink()

    if failures:
        failure_path.write_text("\n".join(failures), encoding="utf-8")
        print("Fuentes con error:")
        for item in failures:
            print(f"- {item}")
        print(f"Detalle de errores en: {failure_path}")
        return 1 if not results else 0

    print("Ejecucion finalizada sin errores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
