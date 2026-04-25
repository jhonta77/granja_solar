from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIABILITY_PATH = (
    PROJECT_ROOT / "data" / "clean" / "viabilidad_municipal" / "viabilidad_municipal_preliminar.csv"
)
DEFAULT_CLUSTERS_PATH = (
    PROJECT_ROOT / "data" / "clean" / "clusters_municipios" / "municipios_clusters_kmeans.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "visualizaciones_municipios"
SCORE_COL = "v_i_modelo_rural"

CORRELATION_COLUMNS = [
    "pvout_kwh_kwp_day",
    "annual_yield_kwh_kw_year",
    "demanda_xm_proxy_mwh_o_unidad_fuente",
    "dist_subestacion_km",
    "pct_area_protegida_runap",
    "area_km2_igac",
    "altitud_m",
    "s_i_solar",
    "d_i_demanda",
    "g_i_red",
    "p_i_pendiente_proxy",
    "u_i_uso_suelo",
    "r_i_zona_urbana_pot",
    "score_preliminar_solar_red_pendiente_runap",
    SCORE_COL,
    "score_rural_con_bono_demanda",
]

HISTOGRAM_COLUMNS = [
    "pvout_kwh_kwp_day",
    "annual_yield_kwh_kw_year",
    "demanda_xm_proxy_mwh_o_unidad_fuente",
    "dist_subestacion_km",
    "pct_area_protegida_runap",
    "s_i_solar",
    "d_i_demanda",
    "g_i_red",
    "p_i_pendiente_proxy",
    "u_i_uso_suelo",
    "r_i_zona_urbana_pot",
    SCORE_COL,
    "score_rural_con_bono_demanda",
]

LABELS = {
    "pvout_kwh_kwp_day": "PVOUT kWh/kWp/dia",
    "annual_yield_kwh_kw_year": "Generacion anual kWh/kW-anio",
    "demanda_xm_proxy_mwh_o_unidad_fuente": "Demanda XM proxy",
    "dist_subestacion_km": "Distancia a subestacion km",
    "pct_area_protegida_runap": "Porcentaje area protegida RUNAP",
    "area_km2_igac": "Area municipal km2",
    "altitud_m": "Altitud m",
    "s_i_solar": "S_i solar",
    "d_i_demanda": "D_i demanda",
    "g_i_red": "G_i red",
    "p_i_pendiente_proxy": "P_i pendiente",
    "u_i_uso_suelo": "U_i uso suelo",
    "r_i_zona_urbana_pot": "R_i zona urbana POT",
    "score_preliminar_solar_red_pendiente_runap": "Score rural sin demanda",
    SCORE_COL: "V_i rural",
    "score_rural_con_bono_demanda": "Score rural con bono demanda",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera matrices, histogramas y mapas 3D del modelo municipal."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_VIABILITY_PATH,
        help="CSV municipal integrado con variables normalizadas y score.",
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=DEFAULT_CLUSTERS_PATH,
        help="CSV con clusters K-Means. Si no existe, se omite la matriz cluster vs viabilidad.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta de salida para visualizaciones.",
    )
    parser.add_argument(
        "--exclude-demand-outliers",
        action="store_true",
        help="Excluye municipios marcados como atipicos de demanda en matrices e histogramas.",
    )
    parser.add_argument("--include-demand-outliers", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "base": output_dir,
        "matrices": output_dir / "matrices",
        "histogramas": output_dir / "histogramas",
        "mapas_3d": output_dir / "mapas_3d",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_municipal_data(input_path: Path, clusters_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"No existe la tabla municipal integrada: {input_path}")

    df = pd.read_csv(input_path, dtype={"codigo_dane": "string"})
    if "codigo_dane" not in df.columns:
        raise ValueError("La tabla municipal no contiene codigo_dane.")
    df["codigo_dane"] = df["codigo_dane"].astype("string").str.zfill(5)

    if clusters_path.exists():
        clusters = pd.read_csv(clusters_path, dtype={"codigo_dane": "string"})
        clusters["codigo_dane"] = clusters["codigo_dane"].astype("string").str.zfill(5)
        keep = [
            column
            for column in ["codigo_dane", "cluster_kmeans", "cluster_kmeans_label"]
            if column in clusters.columns
        ]
        if "cluster_kmeans" in keep:
            df = df.merge(clusters[keep], on="codigo_dane", how="left")

    return df


def filter_analysis_rows(df: pd.DataFrame, exclude_demand_outliers: bool) -> pd.DataFrame:
    working = df.copy()
    if exclude_demand_outliers and "flag_atipico_eda_demanda" in working.columns:
        flag = pd.to_numeric(working["flag_atipico_eda_demanda"], errors="coerce").fillna(0)
        working = working[flag.ne(1)].copy()
    return working


def existing_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in df.columns]


def export_correlation_matrix(df: pd.DataFrame, output_dir: Path) -> Path | None:
    columns = existing_columns(df, CORRELATION_COLUMNS)
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.loc[:, numeric.nunique(dropna=True).gt(1)]
    if numeric.shape[1] < 2:
        return None

    corr = numeric.corr()
    csv_path = output_dir / "matriz_correlacion_variables_modelo.csv"
    pairs_path = output_dir / "pares_correlacion_variables_modelo.csv"
    png_path = output_dir / "matriz_correlacion_variables_modelo.png"
    corr.to_csv(csv_path, index=True, encoding="utf-8-sig")

    pair_rows: list[dict[str, object]] = []
    for index, column_a in enumerate(corr.columns):
        for column_b in corr.columns[index + 1 :]:
            value = corr.loc[column_a, column_b]
            if pd.notna(value):
                pair_rows.append(
                    {
                        "variable_1": column_a,
                        "variable_2": column_b,
                        "correlacion": float(value),
                        "correlacion_abs": float(abs(value)),
                    }
                )
    pd.DataFrame(pair_rows).sort_values("correlacion_abs", ascending=False).to_csv(
        pairs_path,
        index=False,
        encoding="utf-8-sig",
    )

    fig_size = max(8, 0.7 * len(corr.columns))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels([LABELS.get(column, column) for column in corr.columns], rotation=90)
    ax.set_yticklabels([LABELS.get(column, column) for column in corr.columns])
    ax.set_title("Matriz de correlacion de variables")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return png_path


def export_cluster_viability_matrix(df: pd.DataFrame, output_dir: Path) -> Path | None:
    required = {"cluster_kmeans", "clasificacion_preliminar"}
    if not required.issubset(df.columns):
        return None

    working = df.dropna(subset=["cluster_kmeans", "clasificacion_preliminar"]).copy()
    if working.empty:
        return None

    working["cluster_kmeans"] = working["cluster_kmeans"].astype(int).astype(str)
    matrix = pd.crosstab(working["cluster_kmeans"], working["clasificacion_preliminar"])
    normalized = pd.crosstab(
        working["cluster_kmeans"],
        working["clasificacion_preliminar"],
        normalize="index",
    )

    matrix_path = output_dir / "matriz_cruce_cluster_viabilidad.csv"
    normalized_path = output_dir / "matriz_cruce_cluster_viabilidad_normalizada.csv"
    png_path = output_dir / "matriz_cruce_cluster_viabilidad.png"
    matrix.to_csv(matrix_path, encoding="utf-8-sig")
    normalized.to_csv(normalized_path, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1] * 1.5), max(5, matrix.shape[0] * 0.8)))
    image = ax.imshow(matrix.values)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Clase de viabilidad")
    ax.set_ylabel("Cluster K-Means")
    ax.set_title("Cruce cluster vs clase de viabilidad")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix.iloc[row, col]), ha="center", va="center")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return png_path


def export_histograms(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in existing_columns(df, HISTOGRAM_COLUMNS):
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty or series.nunique(dropna=True) < 2:
            rows.append(
                {
                    "variable": column,
                    "grafico_generado": 0,
                    "motivo": "Sin datos numericos suficientes o variable constante.",
                    "n": int(series.shape[0]),
                }
            )
            continue

        bins = min(35, max(8, int(math.sqrt(series.shape[0]))))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(series, bins=bins)
        ax.set_title(f"Histograma - {LABELS.get(column, column)}")
        ax.set_xlabel(LABELS.get(column, column))
        ax.set_ylabel("Frecuencia")
        fig.tight_layout()
        png_path = output_dir / f"histograma_{column}.png"
        fig.savefig(png_path, dpi=180)
        plt.close(fig)

        rows.append(
            {
                "variable": column,
                "grafico_generado": 1,
                "archivo": str(png_path),
                "n": int(series.shape[0]),
                "min": float(series.min()),
                "media": float(series.mean()),
                "mediana": float(series.median()),
                "max": float(series.max()),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "resumen_histogramas.csv", index=False, encoding="utf-8-sig")
    return summary


def scaled_marker_size(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(series.median())
    min_value = values.min()
    max_value = values.max()
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(25, index=values.index)
    return 15 + 55 * ((values - min_value) / (max_value - min_value))


def export_3d_maps(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    required = {"lon", "lat", SCORE_COL}
    if not required.issubset(df.columns):
        return []

    paths: list[Path] = []
    working = df.dropna(subset=["lon", "lat", SCORE_COL]).copy()
    if working.empty:
        return []

    for column in ["lon", "lat", SCORE_COL, "s_i_solar"]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=["lon", "lat", SCORE_COL])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sizes = scaled_marker_size(working["s_i_solar"]) if "s_i_solar" in working.columns else 25
    scatter = ax.scatter(
        working["lon"],
        working["lat"],
        working[SCORE_COL],
        c=working[SCORE_COL],
        s=sizes,
        alpha=0.75,
    )
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_zlabel("V_i rural")
    ax.set_title("Mapa 3D municipal: altura = V_i rural, tamano = S_i solar")
    fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.1, label="V_i rural")
    fig.tight_layout()
    path = output_dir / "mapa_3d_viabilidad_todos_municipios.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    if "cluster_kmeans" in working.columns:
        clustered = working.dropna(subset=["cluster_kmeans"]).copy()
        if not clustered.empty:
            clustered["cluster_kmeans"] = pd.to_numeric(
                clustered["cluster_kmeans"], errors="coerce"
            )
            clustered = clustered.dropna(subset=["cluster_kmeans"])
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            sizes = scaled_marker_size(clustered["s_i_solar"]) if "s_i_solar" in clustered.columns else 25
            scatter = ax.scatter(
                clustered["lon"],
                clustered["lat"],
                clustered[SCORE_COL],
                c=clustered["cluster_kmeans"],
                s=sizes,
                alpha=0.8,
            )
            ax.set_xlabel("Longitud")
            ax.set_ylabel("Latitud")
            ax.set_zlabel("V_i rural")
            ax.set_title("Mapa 3D municipal: altura = V_i rural, color = cluster")
            fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.1, label="Cluster")
            fig.tight_layout()
            path = output_dir / "mapa_3d_clusters_kmeans.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)

    return paths


def write_observations(
    output_dir: Path,
    df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    corr_path: Path | None,
    cross_path: Path | None,
    map_paths: list[Path],
    histogram_summary: pd.DataFrame,
    exclude_demand_outliers: bool,
) -> Path:
    observations = [
        "Visualizaciones municipales",
        "===========================",
        "",
        f"Municipios en tabla base: {len(df)}",
        f"Municipios usados en matrices/histogramas: {len(analysis_df)}",
        f"Excluye atipicos de demanda: {exclude_demand_outliers}",
        "",
        "Advertencias metodologicas:",
        "- La matriz de correlacion mide asociacion lineal entre variables numericas; no prueba causalidad.",
        "- La matriz cluster vs viabilidad es un cruce de categorias, no una matriz de confusion supervisada estricta.",
        "- El mapa 3D es una visualizacion georreferenciada simple en lon/lat; no reemplaza un SIG ni un mapa cartografico oficial.",
        "- El tamano de puntos en el mapa 3D usa S_i solar cuando esta disponible.",
        "- La demanda se visualiza como contexto; no forma parte del V_i rural.",
        "",
        "Archivos principales:",
        f"- correlacion: {corr_path if corr_path else 'no generada'}",
        f"- cruce cluster/viabilidad: {cross_path if cross_path else 'no generado'}",
        f"- histogramas generados: {int(histogram_summary['grafico_generado'].sum()) if not histogram_summary.empty else 0}",
    ]
    observations.extend([f"- mapa 3D: {path}" for path in map_paths])

    path = output_dir / "observaciones_visualizaciones.txt"
    path.write_text("\n".join(observations), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    dirs = ensure_dirs(args.output_dir)

    df = load_municipal_data(args.input, args.clusters)
    exclude_demand_outliers = args.exclude_demand_outliers and not args.include_demand_outliers
    analysis_df = filter_analysis_rows(df, exclude_demand_outliers=exclude_demand_outliers)

    corr_path = export_correlation_matrix(analysis_df, dirs["matrices"])
    cross_path = export_cluster_viability_matrix(analysis_df, dirs["matrices"])
    histogram_summary = export_histograms(analysis_df, dirs["histogramas"])
    map_paths = export_3d_maps(df, dirs["mapas_3d"])
    observations_path = write_observations(
        dirs["base"],
        df,
        analysis_df,
        corr_path,
        cross_path,
        map_paths,
        histogram_summary,
        exclude_demand_outliers,
    )

    print("Visualizaciones municipales generadas.")
    print(f"- matrices: {dirs['matrices']}")
    print(f"- histogramas: {dirs['histogramas']}")
    print(f"- mapas_3d: {dirs['mapas_3d']}")
    print(f"- observaciones: {observations_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
