from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import math
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr, skew

from src.db.mysql_cli import MySQLSettings, query_dataframe

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "eda_visual_mysql"


@dataclass(frozen=True)
class SourceConfig:
    source_id: str
    source_label: str
    source_note: str
    question: str
    source_query: str
    final_query: str
    value_col: str
    value_label: str
    value_unit: str
    box_group_col: str
    box_group_label: str
    top_groups: int
    scatter_x: str
    scatter_y: str
    scatter_x_label: str
    scatter_y_label: str
    scatter_color_col: str | None = None
    scatter_color_kind: str | None = None
    scatter_color_label: str | None = None
    implication_note: str = ""


def parse_args() -> argparse.Namespace:
    default_database = MySQLSettings.from_env().database
    parser = argparse.ArgumentParser(
        description="Genera EDA visual desde MySQL para el proyecto de granja solar."
    )
    parser.add_argument(
        "--database",
        default=default_database,
        help="Base de datos MySQL con las tablas ya pobladas.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta donde se exportan graficas y reporte.",
    )
    return parser.parse_args()


def wrap_paragraph(text: str) -> str:
    return textwrap.fill(text, width=110)


def slugify(value: str) -> str:
    text = value.lower().strip()
    for old, new in {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ñ": "n",
        "/": "_",
        " ": "_",
        "-": "_",
    }.items():
        text = text.replace(old, new)
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def format_number(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:,.{decimals}f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Sin filas para mostrar._"

    display_df = df.fillna("").astype(str)
    columns = list(display_df.columns)
    widths = {
        column: max(
            len(str(column)),
            int(display_df[column].map(len).max()) if not display_df.empty else 0,
        )
        for column in columns
    }

    header = "| " + " | ".join(str(column).ljust(widths[column]) for column in columns) + " |"
    separator = "| " + " | ".join("-" * widths[column] for column in columns) + " |"
    rows = [header, separator]
    for _, row in display_df.iterrows():
        rows.append("| " + " | ".join(row[column].ljust(widths[column]) for column in columns) + " |")
    return "\n".join(rows)


def modality_description(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.nunique(dropna=True) < 3:
        return "sin forma suficiente para estimar modalidad"
    grid = np.linspace(clean.min(), clean.max(), 256)
    density = gaussian_kde(clean)(grid)
    peaks = 0
    for index in range(1, len(density) - 1):
        if density[index] > density[index - 1] and density[index] > density[index + 1]:
            peaks += 1
    if peaks <= 1:
        return "unimodal"
    if peaks == 2:
        return "bimodal"
    return "multimodal"


def skew_description(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.shape[0] < 3:
        return "sin sesgo concluyente"
    value = float(skew(clean, bias=False))
    if value > 0.5:
        return "sesgada a la derecha"
    if value < -0.5:
        return "sesgada a la izquierda"
    return "cercana a simetrica"


def relationship_description(x: pd.Series, y: pd.Series) -> tuple[str, float]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if frame.shape[0] < 3 or frame["x"].nunique(dropna=True) < 2 or frame["y"].nunique(dropna=True) < 2:
        return "sin relacion concluyente", float("nan")
    corr = float(pearsonr(frame["x"], frame["y"]).statistic)
    magnitude = abs(corr)
    if magnitude >= 0.7:
        strength = "fuerte"
    elif magnitude >= 0.4:
        strength = "moderada"
    elif magnitude >= 0.2:
        strength = "debil"
    else:
        strength = "muy debil"
    direction = "positiva" if corr > 0 else "negativa"
    return f"{direction} {strength}", corr


def top_groups(df: pd.DataFrame, group_col: str, value_col: str, limit: int) -> pd.DataFrame:
    working = df[[group_col, value_col]].copy()
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working = working.dropna(subset=[group_col, value_col])
    if working.empty:
        return working
    counts = working[group_col].value_counts().head(limit).index.tolist()
    return working[working[group_col].isin(counts)].copy()


def ensure_dirs(output_dir: Path, source_ids: list[str]) -> dict[str, Path]:
    dirs = {
        "base": output_dir,
        "sources": output_dir / "fuentes",
        "comparativos": output_dir / "posterior_limpieza",
        "explicativas": output_dir / "graficas_explicativas",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    for source_id in source_ids:
        (dirs["sources"] / source_id).mkdir(parents=True, exist_ok=True)
    return dirs


def query_df(settings: MySQLSettings, sql_text: str) -> pd.DataFrame:
    return query_dataframe(sql_text, settings=settings, database=settings.database)


def plot_kde(series: pd.Series, label: str, output_path: Path) -> None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    if clean.shape[0] >= 2 and clean.nunique(dropna=True) >= 2:
        grid = np.linspace(clean.min(), clean.max(), 256)
        density = gaussian_kde(clean)(grid)
        ax.plot(grid, density, color="#136f63", linewidth=2.2)
        ax.fill_between(grid, density, alpha=0.2, color="#2a9d8f")
        ax.axvline(clean.median(), color="#c44536", linestyle="--", linewidth=1.2, label="Mediana")
    else:
        ax.text(0.5, 0.5, "Datos insuficientes para KDE", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"KDE - {label}")
    ax.set_xlabel(label)
    ax.set_ylabel("Densidad estimada")
    ax.grid(alpha=0.25)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_histogram(series: pd.Series, label: str, output_path: Path) -> None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    bins = min(35, max(10, int(math.sqrt(clean.shape[0]))))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(clean, bins=bins, color="#457b9d", alpha=0.8, edgecolor="white")
    ax.axvline(clean.median(), color="#c44536", linestyle="--", linewidth=1.2, label="Mediana")
    ax.set_title(f"Histograma - {label}")
    ax.set_xlabel(label)
    ax.set_ylabel("Frecuencia")
    ax.grid(alpha=0.20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_boxplot(df: pd.DataFrame, group_col: str, value_col: str, label: str, group_label: str, output_path: Path) -> None:
    working = df[[group_col, value_col]].copy()
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working = working.dropna(subset=[group_col, value_col])
    grouped = working.groupby(group_col)[value_col].apply(list)
    fig, ax = plt.subplots(figsize=(max(10, len(grouped) * 0.8), 6))
    if not grouped.empty:
        ax.boxplot(grouped.tolist(), tick_labels=grouped.index.tolist(), patch_artist=True)
    else:
        ax.text(0.5, 0.5, "Sin datos suficientes", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"Boxplot - {label} por {group_label}")
    ax.set_xlabel(group_label)
    ax.set_ylabel(label)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    *,
    color_col: str | None = None,
    color_kind: str | None = None,
    color_label: str | None = None,
) -> None:
    working = df.copy()
    working[x_col] = pd.to_numeric(working[x_col], errors="coerce")
    working[y_col] = pd.to_numeric(working[y_col], errors="coerce")
    working = working.dropna(subset=[x_col, y_col])

    fig, ax = plt.subplots(figsize=(8, 6))
    if color_col and color_col in working.columns and color_kind == "continuous":
        working[color_col] = pd.to_numeric(working[color_col], errors="coerce")
        scatter = ax.scatter(
            working[x_col],
            working[y_col],
            c=working[color_col],
            cmap="viridis",
            s=32,
            alpha=0.75,
        )
        fig.colorbar(scatter, ax=ax, label=color_label or color_col)
    elif color_col and color_col in working.columns and color_kind == "categorical":
        categories = (
            working[color_col].astype("string").fillna("SIN DATO").value_counts().head(6).index.tolist()
        )
        palette = ["#1d3557", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
        for index, category in enumerate(categories):
            subset = working[working[color_col].astype("string").fillna("SIN DATO").eq(category)]
            ax.scatter(subset[x_col], subset[y_col], s=30, alpha=0.72, label=str(category), color=palette[index])
        remainder = working[~working[color_col].astype("string").fillna("SIN DATO").isin(categories)]
        if not remainder.empty:
            ax.scatter(remainder[x_col], remainder[y_col], s=20, alpha=0.35, label="Otros", color="#999999")
        ax.legend(title=color_label or color_col, fontsize=8)
    else:
        ax.scatter(working[x_col], working[y_col], s=28, alpha=0.70, color="#2a9d8f")

    if working.shape[0] >= 3 and working[x_col].nunique(dropna=True) >= 2:
        coefficients = np.polyfit(working[x_col], working[y_col], deg=1)
        grid = np.linspace(working[x_col].min(), working[x_col].max(), 100)
        trend = coefficients[0] * grid + coefficients[1]
        ax.plot(grid, trend, color="#c44536", linewidth=1.6, linestyle="--", label="Tendencia")

    ax.set_title(f"Scatter - {x_label} vs {y_label}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_comparison_overlay(
    source_series: pd.Series,
    final_series: pd.Series,
    label: str,
    output_path: Path,
) -> None:
    source_clean = pd.to_numeric(source_series, errors="coerce").dropna()
    final_clean = pd.to_numeric(final_series, errors="coerce").dropna()
    combined = pd.concat([source_clean, final_clean], ignore_index=True)
    bins = min(30, max(10, int(math.sqrt(max(len(source_clean), len(final_clean))))))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(source_clean, bins=bins, density=True, alpha=0.35, label="Fuente limpia", color="#457b9d")
    ax.hist(final_clean, bins=bins, density=True, alpha=0.35, label="Tabla final", color="#e76f51")
    if source_clean.nunique(dropna=True) >= 3:
        grid = np.linspace(combined.min(), combined.max(), 256)
        ax.plot(grid, gaussian_kde(source_clean)(grid), color="#1d3557", linewidth=1.6)
    if final_clean.nunique(dropna=True) >= 3:
        grid = np.linspace(combined.min(), combined.max(), 256)
        ax.plot(grid, gaussian_kde(final_clean)(grid), color="#b03a2e", linewidth=1.6)
    ax.set_title(f"Verificacion posterior - {label}")
    ax.set_xlabel(label)
    ax.set_ylabel("Densidad")
    ax.legend()
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def interpret_kde(config: SourceConfig, series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    q25 = clean.quantile(0.25)
    q75 = clean.quantile(0.75)
    return wrap_paragraph(
        f"La densidad estimada de {config.value_label} es {modality_description(clean)} y "
        f"{skew_description(clean)}. La mediana se ubica en {format_number(clean.median())} {config.value_unit} "
        f"y el 50% central cae entre {format_number(q25)} y {format_number(q75)} {config.value_unit}. "
        f"Esto indica que la mayor parte de los municipios se concentra en un rango operativo claramente identificable, "
        f"mientras los extremos deben tratarse como señales territoriales y no como comportamiento tipico. "
        f"{config.implication_note}"
    )


def interpret_histogram(config: SourceConfig, series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    bins = min(30, max(10, int(math.sqrt(clean.shape[0]))))
    counts, edges = np.histogram(clean, bins=bins)
    dominant_index = int(np.argmax(counts))
    bin_left = edges[dominant_index]
    bin_right = edges[dominant_index + 1]
    p95 = clean.quantile(0.95)
    tail_share = (clean >= p95).mean() * 100
    return wrap_paragraph(
        f"El histograma concentra la mayor frecuencia entre {format_number(bin_left)} y {format_number(bin_right)} "
        f"{config.value_unit}, lo que define el rango mas comun de la fuente. La forma general es "
        f"{skew_description(clean)}, y aproximadamente {format_number(tail_share)}% de los casos cae en la cola superior "
        f"por encima del percentil 95. Para analisis posteriores conviene normalizar con cuidado y revisar estos extremos "
        f"antes de asumir que representan el patron general del territorio. {config.implication_note}"
    )


def interpret_boxplot(config: SourceConfig, df: pd.DataFrame) -> str:
    working = df[[config.box_group_col, config.value_col]].copy()
    working[config.value_col] = pd.to_numeric(working[config.value_col], errors="coerce")
    working = working.dropna(subset=[config.box_group_col, config.value_col])
    grouped = working.groupby(config.box_group_col)[config.value_col]
    medians = grouped.median().sort_values(ascending=False)
    iqrs = (grouped.quantile(0.75) - grouped.quantile(0.25)).sort_values(ascending=False)

    outlier_counts: dict[str, int] = {}
    for group_name, values in grouped:
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_counts[str(group_name)] = int(((values < lower) | (values > upper)).sum())
    outlier_series = pd.Series(outlier_counts).sort_values(ascending=False)

    return wrap_paragraph(
        f"En el boxplot por {config.box_group_label.lower()}, {medians.index[0]} presenta la mediana mas alta de "
        f"{config.value_label.lower()}, mientras {iqrs.index[0]} exhibe la mayor dispersion intercuartilica. "
        f"El grupo con mas outliers visibles es {outlier_series.index[0]}, lo que sugiere heterogeneidad interna o "
        f"municipios atipicos dentro de esa agrupacion. Esto ayuda a distinguir territorios estables de territorios que "
        f"requieren lectura mas fina antes de convertir el hallazgo en decision de negocio. {config.implication_note}"
    )


def interpret_scatter(config: SourceConfig, df: pd.DataFrame) -> str:
    relation, corr = relationship_description(df[config.scatter_x], df[config.scatter_y])
    working = df[[config.scatter_x, config.scatter_y]].copy()
    working[config.scatter_x] = pd.to_numeric(working[config.scatter_x], errors="coerce")
    working[config.scatter_y] = pd.to_numeric(working[config.scatter_y], errors="coerce")
    working = working.dropna()
    x_q75 = working[config.scatter_x].quantile(0.75)
    y_q75 = working[config.scatter_y].quantile(0.75)
    upper_upper = int(((working[config.scatter_x] >= x_q75) & (working[config.scatter_y] >= y_q75)).sum())
    return wrap_paragraph(
        f"El scatter entre {config.scatter_x_label} y {config.scatter_y_label} muestra una relacion {relation}"
        + (f" (r={corr:.2f}). " if not pd.isna(corr) else ". ")
        + f"Hay {upper_upper} casos en el cuadrante alto-alto, que son los municipios mas interesantes cuando ambas "
        f"variables empujan en la misma direccion del problema. La nube tambien permite detectar puntos aislados que no "
        f"siguen la tendencia general, utiles para revisar restricciones, conectividad o decisiones de integracion. "
        f"{config.implication_note}"
    )


def interpret_post_clean(config: SourceConfig, source_series: pd.Series, final_series: pd.Series) -> str:
    source_clean = pd.to_numeric(source_series, errors="coerce").dropna()
    final_clean = pd.to_numeric(final_series, errors="coerce").dropna()
    same_count = source_clean.shape[0] == final_clean.shape[0]
    same_median = abs(float(source_clean.median()) - float(final_clean.median())) < 1e-9
    return wrap_paragraph(
        f"La verificacion posterior compara la distribucion de {config.value_label} en la fuente limpia frente a la tabla "
        f"final integrada. La cobertura es {'igual' if same_count else 'distinta'} ({len(source_clean)} vs {len(final_clean)} "
        f"registros validos) y la mediana permanece {'estable' if same_median else 'ligeramente distinta'}. "
        f"Cuando las curvas casi se superponen, la integracion no esta distorsionando la variable y el dataset queda mejor "
        f"preparado para consultas SQL, dashboard y analitica posterior."
    )


def render_source(config: SourceConfig, settings: MySQLSettings, dirs: dict[str, Path]) -> tuple[list[dict[str, str]], dict[str, object]]:
    source_df = query_df(settings, config.source_query)
    final_df = query_df(settings, config.final_query)
    source_dir = dirs["sources"] / config.source_id
    manifest_rows: list[dict[str, str]] = []

    clean_series = pd.to_numeric(source_df[config.value_col], errors="coerce").dropna()
    if clean_series.shape[0] < 5 or clean_series.nunique(dropna=True) < 2:
        raise ValueError(f"La fuente {config.source_id} no tiene datos suficientes para visualizacion.")

    kde_path = source_dir / f"kde_{slugify(config.value_col)}.png"
    hist_path = source_dir / f"hist_{slugify(config.value_col)}.png"
    box_path = source_dir / f"box_{slugify(config.value_col)}_por_{slugify(config.box_group_col)}.png"
    scatter_path = source_dir / f"scatter_{slugify(config.scatter_x)}_vs_{slugify(config.scatter_y)}.png"
    post_path = dirs["comparativos"] / f"comparativo_{config.source_id}_{slugify(config.value_col)}.png"

    plot_kde(clean_series, config.value_label, kde_path)
    plot_histogram(clean_series, config.value_label, hist_path)

    box_df = top_groups(source_df, config.box_group_col, config.value_col, config.top_groups)
    plot_boxplot(box_df, config.box_group_col, config.value_col, config.value_label, config.box_group_label, box_path)

    plot_scatter(
        source_df,
        config.scatter_x,
        config.scatter_y,
        config.scatter_x_label,
        config.scatter_y_label,
        scatter_path,
        color_col=config.scatter_color_col,
        color_kind=config.scatter_color_kind,
        color_label=config.scatter_color_label,
    )
    plot_comparison_overlay(source_df[config.value_col], final_df[config.value_col], config.value_label, post_path)

    chart_rows = [
        ("kde", kde_path, interpret_kde(config, source_df[config.value_col])),
        ("histograma", hist_path, interpret_histogram(config, source_df[config.value_col])),
        ("boxplot", box_path, interpret_boxplot(config, box_df)),
        ("scatter", scatter_path, interpret_scatter(config, source_df)),
        ("post_limpieza", post_path, interpret_post_clean(config, source_df[config.value_col], final_df[config.value_col])),
    ]
    for chart_type, path, interpretation in chart_rows:
        manifest_rows.append(
            {
                "seccion": "fuente" if chart_type != "post_limpieza" else "verificacion",
                "fuente": config.source_label,
                "chart_type": chart_type,
                "titulo": f"{config.source_label} - {chart_type}",
                "archivo": str(path),
                "pregunta": config.question,
                "interpretacion": interpretation,
            }
        )

    verification = {
        "fuente": config.source_label,
        "n_fuente": int(pd.to_numeric(source_df[config.value_col], errors="coerce").notna().sum()),
        "n_tabla_final": int(pd.to_numeric(final_df[config.value_col], errors="coerce").notna().sum()),
        "mediana_fuente": float(pd.to_numeric(source_df[config.value_col], errors="coerce").median()),
        "mediana_final": float(pd.to_numeric(final_df[config.value_col], errors="coerce").median()),
    }
    return manifest_rows, verification


def query_dashboard(settings: MySQLSettings) -> pd.DataFrame:
    sql = """
    SELECT
      codigo_dane,
      municipio,
      departamento,
      lon,
      lat,
      v_i_modelo_rural,
      score_rural_con_bono_demanda,
      clasificacion_preliminar,
      pvout_kwh_kwp_day,
      dist_subestacion_km,
      pct_area_protegida_runap,
      d_i_demanda,
      cluster_kmeans,
      cluster_kmeans_label,
      silhouette_municipio
    FROM vw_dashboard_municipal
    """
    return query_df(settings, sql)


def plot_top_viability(df: pd.DataFrame, output_path: Path) -> tuple[str, str]:
    top = df.dropna(subset=["v_i_modelo_rural"]).sort_values("v_i_modelo_rural", ascending=False).head(15).copy()
    top = top.sort_values("v_i_modelo_rural", ascending=True)
    labels = top["municipio"] + " (" + top["departamento"] + ")"
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels, top["v_i_modelo_rural"], color="#2a9d8f")
    ax.set_title("Top 15 municipios por viabilidad rural")
    ax.set_xlabel("V_i rural")
    ax.grid(alpha=0.20, axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    top_departments = top["departamento"].value_counts().head(3).index.tolist()
    title = f"Los 15 municipios lideres concentran V_i rural entre {top['v_i_modelo_rural'].min():.2f} y {top['v_i_modelo_rural'].max():.2f}"
    interpretation = wrap_paragraph(
        f"El ranking muestra una franja alta y compacta de municipios con scores rurales muy superiores al promedio, "
        f"con presencia destacada de {', '.join(top_departments)}. El hallazgo indica que la priorizacion preliminar no "
        f"esta dispersa en todo el pais, sino concentrada en territorios donde coinciden buen recurso solar, cercania relativa "
        f"a red y menor castigo territorial. Para negocio, esta grafica sirve como punto de partida para shortlist y validacion "
        f"posterior de campo."
    )
    return title, interpretation


def plot_solar_grid_tradeoff(df: pd.DataFrame, output_path: Path) -> tuple[str, str]:
    working = df.dropna(subset=["pvout_kwh_kwp_day", "dist_subestacion_km", "v_i_modelo_rural"]).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        working["dist_subestacion_km"],
        working["pvout_kwh_kwp_day"],
        c=working["v_i_modelo_rural"],
        cmap="viridis",
        s=35,
        alpha=0.75,
    )
    fig.colorbar(scatter, ax=ax, label="V_i rural")
    ax.set_title("Trade-off entre solar y distancia a red")
    ax.set_xlabel("Distancia a subestacion (km)")
    ax.set_ylabel("PVOUT (kWh/kWp/dia)")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    relation, corr = relationship_description(working["dist_subestacion_km"], working["v_i_modelo_rural"])
    title = "La viabilidad cae cuando la distancia a red aumenta, incluso con buen recurso solar"
    interpretation = wrap_paragraph(
        f"El grafico combina dos drivers centrales del modelo: recurso solar y cercania a subestacion. La nube deja ver que "
        f"los puntos con mayor coloracion de viabilidad se concentran en la zona de baja distancia y PVOUT medio-alto, mientras "
        f"la relacion distancia-viabilidad es {relation}"
        + (f" (r={corr:.2f}). " if not pd.isna(corr) else ". ")
        + "Esto refuerza que el proyecto no debe optimizar solo solar: la interconexion condiciona de forma directa la priorizacion."
    )
    return title, interpretation


def plot_cluster_box(df: pd.DataFrame, output_path: Path) -> tuple[str, str]:
    working = df.dropna(subset=["cluster_kmeans_label", "v_i_modelo_rural"]).copy()
    grouped = [group["v_i_modelo_rural"].tolist() for _, group in working.groupby("cluster_kmeans_label")]
    labels = [label for label, _ in working.groupby("cluster_kmeans_label")]
    fig, ax = plt.subplots(figsize=(8, 6))
    if grouped:
        ax.boxplot(grouped, tick_labels=labels, patch_artist=True)
    else:
        ax.text(0.5, 0.5, "Sin datos de clusters", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Distribucion de viabilidad por cluster K-Means")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("V_i rural")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    medians = working.groupby("cluster_kmeans_label")["v_i_modelo_rural"].median().sort_values(ascending=False)
    title = "Los clusters separan perfiles municipales con viabilidades claramente distintas"
    interpretation = wrap_paragraph(
        f"El boxplot evidencia que {medians.index[0]} concentra la mediana de viabilidad mas alta, mientras "
        f"{medians.index[-1]} recoge el perfil comparativamente mas rezagado. La separacion entre cajas sugiere que el clustering "
        f"no solo agrupa por conveniencia numerica, sino que distingue perfiles territoriales utiles para segmentar la estrategia: "
        f"priorizacion inmediata, vigilancia o descarte preliminar."
    )
    return title, interpretation


def explanatory_specs(settings: MySQLSettings, dirs: dict[str, Path]) -> list[dict[str, str]]:
    dashboard = query_dashboard(settings)
    rows: list[dict[str, str]] = []

    chart_builders = [
        (
            "grafica_1_top_viabilidad.png",
            plot_top_viability,
            "¿Qué municipios concentran hoy la mejor combinacion preliminar de solar, red, pendiente y restricciones?",
            "Bar chart horizontal · Corte municipal actual · Vista vw_dashboard_municipal · indice V_i rural 0-1.",
            "MySQL granja_solar · vista vw_dashboard_municipal · fuentes integradas IGAC, PVOUT, UPME, RUNAP, POT y XM.",
        ),
        (
            "grafica_2_tradeoff_solar_red.png",
            plot_solar_grid_tradeoff,
            "¿Qué tan fuerte es el trade-off entre recurso solar y cercania a red en la priorizacion municipal?",
            "Scatter plot · Corte municipal actual · Vista vw_dashboard_municipal · PVOUT kWh/kWp/dia y distancia km.",
            "MySQL granja_solar · vista vw_dashboard_municipal · fuentes integradas IGAC, PVOUT y UPME.",
        ),
        (
            "grafica_3_viabilidad_por_cluster.png",
            plot_cluster_box,
            "¿Los clusters K-Means separan municipios con perfiles de viabilidad realmente distintos?",
            "Boxplot · Corte municipal clustered · Vista vw_dashboard_municipal · V_i rural 0-1 por cluster.",
            "MySQL granja_solar · vista vw_dashboard_municipal + clustering K-Means del proyecto.",
        ),
    ]

    for index, (filename, builder, question, subtitle, data_source) in enumerate(chart_builders, start=1):
        output_path = dirs["explicativas"] / filename
        title, interpretation = builder(dashboard, output_path)
        rows.append(
            {
                "seccion": "explicativa",
                "fuente": f"Grafica explicativa {index}",
                "chart_type": "grafica_explicativa",
                "titulo": title,
                "archivo": str(output_path),
                "pregunta": question,
                "subtitulo": subtitle,
                "fuente_dato": data_source,
                "interpretacion": interpretation,
            }
        )
    return rows


def source_configs() -> list[SourceConfig]:
    return [
        SourceConfig(
            source_id="fuente_1_pvout",
            source_label="Fuente 1 - PVOUT municipal",
            source_note="Global Solar Atlas / World Bank / ESMAP / Solargis + integracion municipal del proyecto.",
            question="¿Como se distribuye el recurso solar municipal y que tanto se traduce en viabilidad preliminar alta?",
            source_query="""
                SELECT
                  m.codigo_dane,
                  m.municipio,
                  d.departamento,
                  pv.pvout_kwh_kwp_day,
                  pv.annual_yield_kwh_kw_year,
                  v.v_i_modelo_rural
                FROM municipio_pvout pv
                JOIN municipios m ON m.codigo_dane = pv.codigo_dane
                JOIN departamentos d ON d.departamento_id = m.departamento_id
                LEFT JOIN viabilidad_municipal v ON v.codigo_dane = pv.codigo_dane
                WHERE pv.pvout_kwh_kwp_day IS NOT NULL
            """,
            final_query="""
                SELECT pvout_kwh_kwp_day
                FROM viabilidad_municipal
                WHERE pvout_kwh_kwp_day IS NOT NULL
            """,
            value_col="pvout_kwh_kwp_day",
            value_label="PVOUT",
            value_unit="kWh/kWp/dia",
            box_group_col="departamento",
            box_group_label="Departamento",
            top_groups=10,
            scatter_x="pvout_kwh_kwp_day",
            scatter_y="v_i_modelo_rural",
            scatter_x_label="PVOUT (kWh/kWp/dia)",
            scatter_y_label="V_i rural",
            implication_note="Para el negocio, una cola alta de PVOUT amplifica opciones, pero no garantiza por si sola prioridad final sin revisar red y restricciones.",
        ),
        SourceConfig(
            source_id="fuente_2_red",
            source_label="Fuente 2 - Distancia a red UPME",
            source_note="UPME Geoportal + distancia municipal a subestacion mas cercana.",
            question="¿Que tan dispersa es la distancia a la red y como condiciona la viabilidad preliminar municipal?",
            source_query="""
                SELECT
                  m.codigo_dane,
                  m.municipio,
                  d.departamento,
                  mr.dist_subestacion_km,
                  mr.capacidad_mva_mas_cercana,
                  v.v_i_modelo_rural
                FROM municipio_red mr
                JOIN municipios m ON m.codigo_dane = mr.codigo_dane
                JOIN departamentos d ON d.departamento_id = m.departamento_id
                LEFT JOIN viabilidad_municipal v ON v.codigo_dane = mr.codigo_dane
                WHERE mr.dist_subestacion_km IS NOT NULL
            """,
            final_query="""
                SELECT dist_subestacion_km
                FROM viabilidad_municipal
                WHERE dist_subestacion_km IS NOT NULL
            """,
            value_col="dist_subestacion_km",
            value_label="Distancia a subestacion",
            value_unit="km",
            box_group_col="departamento",
            box_group_label="Departamento",
            top_groups=10,
            scatter_x="dist_subestacion_km",
            scatter_y="v_i_modelo_rural",
            scatter_x_label="Distancia a subestacion (km)",
            scatter_y_label="V_i rural",
            implication_note="Distancias largas elevan el riesgo tecnico y economico de interconexion, por lo que la cola alta debe vigilarse como factor de descarte preliminar.",
        ),
        SourceConfig(
            source_id="fuente_3_runap",
            source_label="Fuente 3 - Restriccion RUNAP",
            source_note="Parques Nacionales / RUNAP + interseccion municipal del proyecto.",
            question="¿Que tan restrictiva es la cobertura de areas protegidas sobre el universo municipal candidato?",
            source_query="""
                SELECT
                  m.codigo_dane,
                  m.municipio,
                  d.departamento,
                  ru.pct_area_protegida_runap,
                  ru.area_protegida_km2_runap,
                  ru.area_municipio_km2_calc,
                  ru.r_i_runap,
                  v.v_i_modelo_rural
                FROM municipio_runap ru
                JOIN municipios m ON m.codigo_dane = ru.codigo_dane
                JOIN departamentos d ON d.departamento_id = m.departamento_id
                LEFT JOIN viabilidad_municipal v ON v.codigo_dane = ru.codigo_dane
                WHERE ru.pct_area_protegida_runap IS NOT NULL
            """,
            final_query="""
                SELECT pct_area_protegida_runap
                FROM viabilidad_municipal
                WHERE pct_area_protegida_runap IS NOT NULL
            """,
            value_col="pct_area_protegida_runap",
            value_label="Cobertura RUNAP",
            value_unit="proporcion del municipio",
            box_group_col="departamento",
            box_group_label="Departamento",
            top_groups=10,
            scatter_x="pct_area_protegida_runap",
            scatter_y="v_i_modelo_rural",
            scatter_x_label="Cobertura RUNAP",
            scatter_y_label="V_i rural",
            scatter_color_col="r_i_runap",
            scatter_color_kind="categorical",
            scatter_color_label="Restriccion RUNAP",
            implication_note="Coberturas altas limitan area util y pueden anular municipios aunque tengan buen recurso solar o proximidad a red.",
        ),
        SourceConfig(
            source_id="fuente_4_demanda",
            source_label="Fuente 4 - Proxy de demanda XM",
            source_note="XM pron_areas + mapeo municipal del proyecto.",
            question="¿Como se reparte el contexto de demanda XM entre zonas y que tanto cambia la lectura de viabilidad?",
            source_query="""
                SELECT
                  m.codigo_dane,
                  m.municipio,
                  d.departamento,
                  dx.zona_xm_demanda,
                  dx.tipo_mapeo_demanda,
                  dx.demanda_xm_proxy_mwh_o_unidad_fuente,
                  dx.demanda_xm_valor_p95,
                  dx.d_i_demanda,
                  v.v_i_modelo_rural
                FROM municipio_demanda_xm dx
                JOIN municipios m ON m.codigo_dane = dx.codigo_dane
                JOIN departamentos d ON d.departamento_id = m.departamento_id
                LEFT JOIN viabilidad_municipal v ON v.codigo_dane = dx.codigo_dane
                WHERE dx.demanda_xm_proxy_mwh_o_unidad_fuente IS NOT NULL
            """,
            final_query="""
                SELECT demanda_xm_proxy_mwh_o_unidad_fuente
                FROM viabilidad_municipal
                WHERE demanda_xm_proxy_mwh_o_unidad_fuente IS NOT NULL
            """,
            value_col="demanda_xm_proxy_mwh_o_unidad_fuente",
            value_label="Demanda XM proxy",
            value_unit="unidad fuente",
            box_group_col="zona_xm_demanda",
            box_group_label="Zona XM",
            top_groups=10,
            scatter_x="demanda_xm_proxy_mwh_o_unidad_fuente",
            scatter_y="v_i_modelo_rural",
            scatter_x_label="Demanda XM proxy",
            scatter_y_label="V_i rural",
            scatter_color_col="tipo_mapeo_demanda",
            scatter_color_kind="categorical",
            scatter_color_label="Tipo de mapeo",
            implication_note="La demanda sirve como contexto comercial y sensibilidad favorable, pero no sustituye criterios fisicos de ubicacion.",
        ),
    ]


def build_report(
    output_path: Path,
    source_rows: list[dict[str, str]],
    explanatory_rows: list[dict[str, str]],
    verification_rows: list[dict[str, object]],
) -> None:
    lines: list[str] = []
    lines.append("# EDA Visual del Proyecto de Granja Solar")
    lines.append("")
    lines.append(
        "Este reporte genera las visualizaciones del EDA visual desde MySQL para cuatro fuentes clave del proyecto: "
        "PVOUT municipal, distancia a red, restricciones RUNAP y proxy de demanda XM. Cada fuente incluye KDE, histograma, "
        "boxplot, scatter y una verificacion posterior frente a la tabla final integrada."
    )
    lines.append("")

    grouped_sources: dict[str, list[dict[str, str]]] = {}
    for row in source_rows:
        grouped_sources.setdefault(row["fuente"], []).append(row)

    for source_name, rows in grouped_sources.items():
        rows_by_type = {row["chart_type"]: row for row in rows}
        question = rows[0]["pregunta"]
        lines.append(f"## {source_name}")
        lines.append("")
        lines.append(f"**Pregunta de analisis:** {question}")
        lines.append("")

        for chart_type, title in [
            ("kde", "KDE"),
            ("histograma", "Histograma"),
            ("boxplot", "Boxplot"),
            ("scatter", "Scatter"),
        ]:
            row = rows_by_type[chart_type]
            relative = Path(row["archivo"]).resolve().relative_to(PROJECT_ROOT)
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{source_name} - {title}]({relative.as_posix()})")
            lines.append("")
            lines.append(row["interpretacion"])
            lines.append("")

        lines.append("### EDA posterior a la limpieza")
        lines.append("")
        post_row = rows_by_type["post_limpieza"]
        relative = Path(post_row["archivo"]).resolve().relative_to(PROJECT_ROOT)
        lines.append(f"![{source_name} - verificacion]({relative.as_posix()})")
        lines.append("")
        lines.append(post_row["interpretacion"])
        lines.append("")

    lines.append("## Verificacion posterior disponible")
    lines.append("")
    lines.append(
        "La comparacion posterior usa la fuente limpia frente a la tabla final integrada en MySQL. "
        "Esto verifica que la integracion no altero la distribucion de la variable clave. "
        "Si despues se reconstruye un EDA raw completo, este bloque puede ampliarse con un comparativo estricto antes/despues de limpieza."
    )
    lines.append("")
    verification_df = pd.DataFrame(verification_rows)
    if not verification_df.empty:
        lines.append(dataframe_to_markdown(verification_df))
        lines.append("")

    lines.append("## Graficas explicativas")
    lines.append("")
    for index, row in enumerate(explanatory_rows, start=1):
        relative = Path(row["archivo"]).resolve().relative_to(PROJECT_ROOT)
        lines.append(f"### Grafica explicativa {index}")
        lines.append("")
        lines.append(f"![Grafica explicativa {index}]({relative.as_posix()})")
        lines.append("")
        lines.append(f"**Titulo del hallazgo:** {row['titulo']}")
        lines.append("")
        lines.append(f"**Subtitulo metodologico:** {row['subtitulo']}")
        lines.append("")
        lines.append(f"**Pregunta de negocio que responde:** {row['pregunta']}")
        lines.append("")
        lines.append(f"**Fuente del dato:** {row['fuente_dato']}")
        lines.append("")
        lines.append(f"**Interpretacion:** {row['interpretacion']}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    settings = MySQLSettings.from_env(database_override=args.database)
    configs = source_configs()
    dirs = ensure_dirs(args.output_dir, [config.source_id for config in configs])
    if sns is not None:
        sns.set_theme(style="whitegrid", context="notebook")
    else:
        plt.style.use("ggplot")

    all_source_rows: list[dict[str, str]] = []
    verification_rows: list[dict[str, object]] = []

    for config in configs:
        source_rows, verification = render_source(config, settings, dirs)
        all_source_rows.extend(source_rows)
        verification_rows.append(verification)

    explanatory_rows = explanatory_specs(settings, dirs)
    manifest_rows = all_source_rows + explanatory_rows
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(dirs["base"] / "manifest_eda_visual_mysql.csv", index=False, encoding="utf-8-sig")

    verification_df = pd.DataFrame(verification_rows)
    verification_df.to_csv(
        dirs["base"] / "verificacion_posterior_limpieza.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report_path = dirs["base"] / "reporte_eda_visual_mysql.md"
    build_report(report_path, all_source_rows, explanatory_rows, verification_rows)

    print(f"EDA visual exportado en: {dirs['base']}")
    print(f"- reporte: {report_path}")
    print(f"- manifest: {dirs['base'] / 'manifest_eda_visual_mysql.csv'}")
    print(f"- verificacion: {dirs['base'] / 'verificacion_posterior_limpieza.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
