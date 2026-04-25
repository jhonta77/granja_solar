from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.xm_file_utils import (  # noqa: E402
    extract_xm_source_type,
    read_xm_txt_file,
    slugify,
)


ACTIVE_VARIABLES = {"EN", "Ener.Act.", "ENER.ACT.", "Ener.Act"}


def list_xm_txt_files(raw_dir: Path) -> list[Path]:
    """Lista archivos TXT XM en carpetas anidadas."""

    return sorted(path for path in raw_dir.rglob("*.txt") if path.is_file())


def get_value_columns(df: pd.DataFrame) -> list[str]:
    """Detecta columnas horarias o diarias de valor."""

    return [
        column
        for column in df.columns
        if str(column).startswith("hora_") or str(column).startswith("dia_")
    ]


def infer_xm_family(file_path: Path) -> str:
    """Normaliza la familia funcional de un archivo XM."""

    return slugify(extract_xm_source_type(file_path))


def _column_number(column_name: str) -> int:
    """Extrae el numero de columnas tipo hora_01 o dia_07."""

    return int(str(column_name).split("_")[-1])


def read_xm_file_long(file_path: Path) -> pd.DataFrame:
    """
    Convierte un TXT XM ancho a formato largo.

    La salida queda lista para agregacion: tipo_archivo, zona, variable, fecha,
    hora y valor. Esto evita crear carpetas por archivo y permite ranking global.
    """

    wide_df = read_xm_txt_file(file_path)
    if wide_df.empty:
        return pd.DataFrame()

    family = infer_xm_family(file_path)
    value_columns = get_value_columns(wide_df)
    if not value_columns:
        return pd.DataFrame()

    metadata_columns = [column for column in wide_df.columns if column not in value_columns]
    long_df = wide_df.melt(
        id_vars=metadata_columns,
        value_vars=value_columns,
        var_name="columna_tiempo",
        value_name="valor",
    )

    long_df["valor"] = pd.to_numeric(long_df["valor"], errors="coerce")
    long_df = long_df.dropna(subset=["valor"]).copy()
    if long_df.empty:
        return pd.DataFrame()

    long_df["tipo_archivo"] = family
    long_df["archivo_origen"] = file_path.name
    long_df["fecha_archivo"] = pd.to_datetime(long_df["fecha_archivo"], errors="coerce")

    if "variable" in long_df.columns:
        long_df["variable"] = long_df["variable"].astype("string").str.strip()
    else:
        long_df["variable"] = "valor"

    if family == "pron_barra":
        long_df["zona"] = long_df.get("barra", "sin_zona").astype("string")
        long_df["zona_grupo"] = long_df.get("area_operativa", long_df["zona"]).astype("string")
        long_df["hora"] = long_df["columna_tiempo"].map(_column_number)
        long_df["fecha"] = long_df["fecha_archivo"]
    elif family == "pron_areas":
        long_df["zona"] = long_df.get("subarea", "sin_zona").astype("string")
        long_df["zona_grupo"] = long_df["zona"]
        long_df["hora"] = pd.to_numeric(long_df.get("hora"), errors="coerce")
        long_df["fecha"] = long_df["fecha_archivo"] + pd.to_timedelta(
            long_df["columna_tiempo"].map(_column_number) - 1, unit="D"
        )
    elif family in {"pron_ucp", "pronucp"}:
        long_df["zona"] = long_df.get("unidad", "sin_zona").astype("string")
        long_df["zona_grupo"] = long_df["zona"]
        long_df["hora"] = pd.to_numeric(long_df.get("hora"), errors="coerce")
        long_df["fecha"] = long_df["fecha_archivo"] + pd.to_timedelta(
            long_df["columna_tiempo"].map(_column_number) - 1, unit="D"
        )
    elif family == "pronsin":
        long_df["zona"] = "SIN"
        long_df["zona_grupo"] = "SIN"
        long_df["hora"] = pd.to_numeric(long_df.get("hora"), errors="coerce")
        long_df["fecha"] = long_df["fecha_archivo"] + pd.to_timedelta(
            long_df["columna_tiempo"].map(_column_number) - 1, unit="D"
        )
    elif family == "pron_sin":
        long_df["zona"] = long_df.get("sistema", "SIN").astype("string")
        long_df["zona_grupo"] = long_df["zona"]
        long_df["hora"] = pd.to_numeric(long_df.get("hora"), errors="coerce")
        long_df["fecha"] = long_df["fecha_archivo"] + pd.to_timedelta(
            long_df["columna_tiempo"].map(_column_number) - 1, unit="D"
        )
    else:
        zone_column = next(
            (
                column
                for column in ["subarea", "barra", "unidad", "sistema", "dimension_1"]
                if column in long_df.columns
            ),
            None,
        )
        long_df["zona"] = (
            long_df[zone_column].astype("string") if zone_column else "sin_zona"
        )
        long_df["zona_grupo"] = long_df["zona"]
        long_df["hora"] = pd.NA
        long_df["fecha"] = long_df["fecha_archivo"]

    result_columns = [
        "tipo_archivo",
        "zona_grupo",
        "zona",
        "variable",
        "fecha",
        "hora",
        "valor",
        "archivo_origen",
    ]
    return long_df[result_columns].copy()


def normalize_minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Normaliza una serie a 0-1."""

    numeric = pd.to_numeric(series, errors="coerce")
    minimum = numeric.min()
    maximum = numeric.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(1.0, index=series.index)

    normalized = (numeric - minimum) / (maximum - minimum)
    return normalized if higher_is_better else 1.0 - normalized


def build_zone_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega mediciones por familia XM y zona."""

    if long_df.empty:
        return pd.DataFrame()

    filtered_groups: list[pd.DataFrame] = []
    for _, family_group in long_df.groupby("tipo_archivo", dropna=False):
        family_data = family_group.copy()
        if "variable" in family_data.columns:
            active_mask = family_data["variable"].isin(ACTIVE_VARIABLES)
            if active_mask.any():
                family_data = family_data[active_mask].copy()
        filtered_groups.append(family_data)

    filtered = pd.concat(filtered_groups, ignore_index=True)

    summary = (
        filtered.groupby(["tipo_archivo", "zona_grupo", "zona"], dropna=False)
        .agg(
            registros=("valor", "size"),
            valor_promedio=("valor", "mean"),
            valor_maximo=("valor", "max"),
            valor_p95=("valor", lambda values: values.quantile(0.95)),
            desviacion=("valor", "std"),
            fecha_min=("fecha", "min"),
            fecha_max=("fecha", "max"),
            dias_cubiertos=("fecha", lambda values: values.dt.normalize().nunique()),
            archivos=("archivo_origen", "nunique"),
        )
        .reset_index()
    )
    summary["coeficiente_variacion"] = (
        summary["desviacion"] / summary["valor_promedio"].replace(0, pd.NA)
    )
    return summary


def add_relative_scores(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula probabilidades relativas y score compacto.

    No son probabilidades estadisticas calibradas; son indicadores 0-1
    normalizados dentro de cada familia XM para ordenar zonas comparables.
    """

    if summary.empty:
        return summary

    scored_groups: list[pd.DataFrame] = []
    for _, group in summary.groupby("tipo_archivo", dropna=False):
        scored = group.copy()
        scored["prob_demanda_promedio"] = normalize_minmax(scored["valor_promedio"])
        scored["prob_pico_demanda"] = normalize_minmax(scored["valor_p95"])
        scored["prob_cobertura_datos"] = normalize_minmax(scored["registros"])
        scored["prob_estabilidad"] = normalize_minmax(
            scored["coeficiente_variacion"].fillna(0), higher_is_better=False
        )
        scored["suma_probabilidades"] = (
            scored["prob_demanda_promedio"]
            + scored["prob_pico_demanda"]
            + scored["prob_cobertura_datos"]
            + scored["prob_estabilidad"]
        )
        scored["score_relativo"] = (
            0.40 * scored["prob_demanda_promedio"]
            + 0.40 * scored["prob_pico_demanda"]
            + 0.15 * scored["prob_cobertura_datos"]
            + 0.05 * scored["prob_estabilidad"]
        )
        scored["probabilidad_relativa"] = normalize_minmax(scored["score_relativo"])
        scored_groups.append(scored)

    return (
        pd.concat(scored_groups, ignore_index=True)
        .sort_values(["tipo_archivo", "score_relativo"], ascending=[True, False])
        .reset_index(drop=True)
    )


def select_top_n(scored: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Selecciona top N por familia XM."""

    if scored.empty:
        return scored

    return (
        scored.groupby("tipo_archivo", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def create_top_charts(top_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Grafica solo el top N por familia para evitar ruido visual."""

    charts_dir = output_dir / "graficos"
    charts_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for family, group in top_df.groupby("tipo_archivo", dropna=False):
        if group.empty:
            continue

        plot_data = group.sort_values("score_relativo", ascending=True)
        labels = plot_data["zona"].astype(str)

        fig, axis = plt.subplots(figsize=(10, max(4, len(plot_data) * 0.45)))
        axis.barh(labels, plot_data["score_relativo"])
        axis.set_title(f"Top zonas por score relativo - {family}")
        axis.set_xlabel("Score relativo 0-1")
        axis.set_ylabel("Zona")
        fig.tight_layout()

        output_path = charts_dir / f"top_zonas_{slugify(str(family))}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        created.append(output_path)

    return created


def build_observations(
    files: list[Path],
    scored: pd.DataFrame,
    top_df: pd.DataFrame,
    top_n: int,
    make_figures: bool,
) -> list[str]:
    """Construye observaciones metodologicas del ranking compacto."""

    lines: list[str] = []
    lines.append("Ranking compacto XM por zonas")
    lines.append("")
    lines.append(f"Archivos TXT procesados: {len(files)}")
    lines.append(f"Zonas/familias agregadas: {len(scored)}")
    lines.append(f"Top exportado por familia: {top_n}")
    lines.append("Diagramas generados: " + ("si" if make_figures else "no"))
    lines.append("")
    lines.append("Metodo:")
    lines.append(
        "- Se convierten los TXT XM de formato ancho a formato largo con fecha, hora, zona, variable y valor."
    )
    lines.append(
        "- Se priorizan variables activas cuando existen: EN o Ener.Act.; otras variables quedan fuera si hay energia activa disponible."
    )
    lines.append(
        "- Se agregan valor promedio, percentil 95, maximo, cobertura de datos y coeficiente de variacion por zona."
    )
    lines.append(
        "- Se calculan indicadores 0-1 por familia: prob_demanda_promedio, prob_pico_demanda, prob_cobertura_datos y prob_estabilidad."
    )
    lines.append(
        "- score_relativo = 0.40 promedio + 0.40 pico + 0.15 cobertura + 0.05 estabilidad."
    )
    lines.append("")
    lines.append("Advertencia:")
    lines.append(
        "- Estas probabilidades son relativas para ordenar zonas dentro de cada familia XM; no son probabilidades estadisticas calibradas."
    )
    lines.append(
        "- Este ranking usa demanda/proyeccion energetica disponible, no reemplaza el score final de viabilidad solar con pendiente, red, restricciones y costos."
    )
    lines.append(
        "- El objetivo es reducir ruido y graficar solo candidatos prioritarios para exploracion."
    )
    lines.append("")
    lines.append("Top detectado:")
    for row in top_df.head(30).to_dict("records"):
        lines.append(
            "- {tipo}: {zona} | score={score:.3f} | promedio={promedio:.3f} | p95={p95:.3f}".format(
                tipo=row["tipo_archivo"],
                zona=row["zona"],
                score=row["score_relativo"],
                promedio=row["valor_promedio"],
                p95=row["valor_p95"],
            )
        )
    return lines


def cache_path_for_file(file_path: Path, raw_dir: Path, cache_dir: Path) -> Path:
    """Construye una ruta de cache sensible a ruta, tamano y fecha de modificacion."""

    relative = file_path.resolve().relative_to(raw_dir.resolve())
    stat = file_path.stat()
    fingerprint = f"{relative.as_posix()}|{stat.st_size}|{stat.st_mtime_ns}"
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{slugify(str(relative.with_suffix('')))}_{digest}.csv"


def read_or_build_cached_long(
    file_path: Path,
    raw_dir: Path,
    cache_dir: Path,
    use_cache: bool,
) -> pd.DataFrame:
    """Lee cache de un TXT transformado o lo construye si no existe."""

    if not use_cache:
        return read_xm_file_long(file_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path_for_file(file_path, raw_dir, cache_dir)
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if "fecha" in cached.columns:
            cached["fecha"] = pd.to_datetime(cached["fecha"], errors="coerce")
        return cached

    long_df = read_xm_file_long(file_path)
    if not long_df.empty:
        long_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return long_df


def remove_stale_charts(output_dir: Path) -> None:
    """Elimina PNG top_zonas previos cuando el usuario no quiere diagramas."""

    charts_dir = output_dir / "graficos"
    if not charts_dir.exists():
        return

    for png_path in charts_dir.glob("top_zonas_*.png"):
        png_path.unlink()

    try:
        if not any(charts_dir.iterdir()):
            charts_dir.rmdir()
    except OSError:
        pass


def run_compact_ranking(
    raw_dir: Path,
    output_dir: Path,
    top_n: int = 10,
    use_cache: bool = True,
    make_figures: bool = False,
) -> dict[str, Any]:
    """Ejecuta ranking compacto completo."""

    files = list_xm_txt_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No se encontraron TXT XM en {raw_dir}")

    cache_dir = PROJECT_ROOT / "data" / "interim" / "xm_top10_cache"
    long_frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for file_path in files:
        try:
            long_df = read_or_build_cached_long(
                file_path=file_path,
                raw_dir=raw_dir,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
            if not long_df.empty:
                long_frames.append(long_df)
        except Exception as error:
            failures.append(f"{file_path}: {error}")

    if not long_frames:
        raise ValueError("No se pudo construir ninguna tabla larga desde los TXT XM.")

    long_all = pd.concat(long_frames, ignore_index=True)
    summary = build_zone_summary(long_all)
    scored = add_relative_scores(summary)
    top_df = select_top_n(scored, top_n)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "xm_resumen_zonas.csv"
    top_path = output_dir / "xm_top10_zonas.csv"
    observations_path = output_dir / "xm_observaciones_top10.txt"
    failures_path = output_dir / "xm_errores_lectura.txt"

    scored.to_csv(summary_path, index=False, encoding="utf-8-sig")
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")
    observations_path.write_text(
        "\n".join(build_observations(files, scored, top_df, top_n, make_figures)),
        encoding="utf-8",
    )
    if failures:
        failures_path.write_text("\n".join(failures), encoding="utf-8")
    elif failures_path.exists():
        failures_path.unlink()

    if make_figures:
        charts = create_top_charts(top_df, output_dir)
    else:
        remove_stale_charts(output_dir)
        charts = []

    return {
        "files": files,
        "summary": scored,
        "top": top_df,
        "summary_path": summary_path,
        "top_path": top_path,
        "observations_path": observations_path,
        "charts": charts,
        "cache_dir": cache_dir,
        "failures": failures,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Configura argumentos CLI."""

    parser = argparse.ArgumentParser(
        description="Ranking compacto XM: consolida TXT, calcula score relativo y grafica top N."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw"),
        help="Carpeta raw con TXT XM en subcarpetas.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "clean" / "xm_top10"),
        help="Carpeta de salida compacta.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Numero de zonas a exportar por familia XM.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="No usa cache intermedio; reprocesa todos los TXT desde cero.",
    )
    parser.add_argument(
        "--make-figures",
        action="store_true",
        help="Genera PNG del top. Por defecto no se crean diagramas.",
    )
    return parser


def main() -> int:
    """Punto de entrada CLI."""

    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        result = run_compact_ranking(
            raw_dir=Path(args.raw_dir).resolve(),
            output_dir=Path(args.output_dir).resolve(),
            top_n=args.top_n,
            use_cache=not args.no_cache,
            make_figures=args.make_figures,
        )
    except Exception as error:
        print(f"ERROR: {error}")
        return 1

    print("Ranking compacto XM finalizado.")
    print(f"Archivos TXT procesados: {len(result['files'])}")
    print(f"Zonas agregadas: {len(result['summary'])}")
    print(f"Top exportado: {result['top_path']}")
    print(f"Resumen completo: {result['summary_path']}")
    print(f"Observaciones: {result['observations_path']}")
    print(f"Cache intermedio: {result['cache_dir']}")
    print(f"Graficos generados: {len(result['charts'])}")
    if result["failures"]:
        print(f"Archivos con error: {len(result['failures'])}")
    print(
        "Nota: el score es relativo y sirve para priorizar exploracion; "
        "no reemplaza el score espacial final."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
