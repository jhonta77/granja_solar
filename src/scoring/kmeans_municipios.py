from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data" / "clean" / "viabilidad_municipal" / "viabilidad_municipal_preliminar.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "clusters_municipios"

DEFAULT_RANDOM_SEED = 42
DEFAULT_K = 5
DEFAULT_K_MIN = 1
DEFAULT_K_MAX = 10
SCORE_COLUMN = "v_i_modelo_rural"
FEATURE_COLUMNS = [
    "s_i_solar",
    "g_i_red",
    "p_i_pendiente_proxy",
    "u_i_uso_suelo",
]


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Carga tabla municipal integrada."""

    if not input_path.exists():
        raise FileNotFoundError(f"No existe tabla municipal integrada: {input_path}")
    df = pd.read_csv(input_path, dtype={"codigo_dane": "string"})
    if "codigo_dane" not in df.columns:
        raise ValueError("La tabla de entrada no tiene codigo_dane.")
    df["codigo_dane"] = df["codigo_dane"].astype("string").str.zfill(5)
    if SCORE_COLUMN not in df.columns:
        raise ValueError(
            f"Falta {SCORE_COLUMN}. Ejecuta primero src/scoring/viabilidad_municipal.py."
        )
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para K-Means: {missing}")
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    include_restricted: bool,
    exclude_demand_outliers: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Selecciona municipios y prepara matriz numerica 0-1 para K-Means."""

    working = df.copy()
    working["cluster_exclusion_reason"] = ""

    mask = pd.Series(True, index=working.index)
    if not include_restricted and "r_i_preliminar" in working.columns:
        restricted = pd.to_numeric(working["r_i_preliminar"], errors="coerce").fillna(0).eq(0)
        working.loc[restricted, "cluster_exclusion_reason"] += "r_i_preliminar=0;"
        mask &= ~restricted

    if exclude_demand_outliers and "flag_atipico_eda_demanda" in working.columns:
        outliers = pd.to_numeric(working["flag_atipico_eda_demanda"], errors="coerce").fillna(1).eq(1)
        working.loc[outliers, "cluster_exclusion_reason"] += "flag_atipico_eda_demanda=1;"
        mask &= ~outliers

    selected = working[mask].copy()
    excluded = working[~mask].copy()
    if selected.empty:
        raise ValueError("No quedan municipios para clustering despues de filtros.")

    features = selected[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    imputation_values = features.median(numeric_only=True)
    features = features.fillna(imputation_values)
    features = features.clip(0, 1)

    for column, value in imputation_values.items():
        selected[f"{column}_imputado_kmeans"] = selected[column].isna().astype(int)
        selected[f"{column}_valor_imputacion_kmeans"] = value

    return selected, excluded


def infer_elbow_k(evaluation: pd.DataFrame) -> int | None:
    """Estima el codo con la mayor distancia a la linea entre extremos."""

    valid = evaluation.dropna(subset=["k", "inertia"]).sort_values("k").reset_index(drop=True)
    if len(valid) < 3:
        return int(valid.loc[0, "k"]) if not valid.empty else None

    points = valid[["k", "inertia"]].astype(float).to_numpy()
    mins = points.min(axis=0)
    ranges = points.max(axis=0) - mins
    ranges[ranges == 0] = 1
    normalized = (points - mins) / ranges

    start = normalized[0]
    end = normalized[-1]
    line = end - start
    norm = np.linalg.norm(line)
    if norm == 0:
        return int(valid.loc[0, "k"])

    offsets = start - normalized
    distances = np.abs(line[0] * offsets[:, 1] - line[1] * offsets[:, 0]) / norm
    return int(valid.loc[int(np.argmax(distances)), "k"])


def add_evaluation_recommendations(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Agrega reduccion de inercia y banderas de K recomendado."""

    columns = [
        "k",
        "inertia",
        "silhouette",
        "delta_inertia",
        "pct_reduccion_inertia",
        "delta_silhouette",
        "recomendado_codo",
        "recomendado_silhouette",
        "random_seed",
    ]
    if evaluation.empty:
        return pd.DataFrame(columns=columns)

    working = evaluation.sort_values("k").reset_index(drop=True).copy()
    working["delta_inertia"] = working["inertia"].shift(1) - working["inertia"]
    working["pct_reduccion_inertia"] = working["delta_inertia"] / working["inertia"].shift(1)
    working["delta_silhouette"] = working["silhouette"] - working["silhouette"].shift(1)

    elbow_k = infer_elbow_k(working)
    silhouette_candidates = working.dropna(subset=["silhouette"]).sort_values(
        ["silhouette", "k"],
        ascending=[False, True],
    )
    silhouette_k = (
        int(silhouette_candidates.iloc[0]["k"]) if not silhouette_candidates.empty else None
    )

    working["recomendado_codo"] = working["k"].eq(elbow_k).astype(int)
    working["recomendado_silhouette"] = working["k"].eq(silhouette_k).astype(int)
    return working[columns]


def evaluate_k_values(
    features: pd.DataFrame,
    k_min: int,
    k_max: int,
    random_seed: int,
) -> pd.DataFrame:
    """Calcula inercia y silhouette para un rango de k."""

    rows: list[dict[str, Any]] = []
    n = len(features)
    upper = min(k_max, n - 1)
    for k in range(max(1, k_min), upper + 1):
        model = KMeans(n_clusters=k, random_state=random_seed, n_init=20)
        labels = model.fit_predict(features)
        silhouette = (
            float(silhouette_score(features, labels))
            if 2 <= k <= n - 1
            else float("nan")
        )
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": silhouette,
                "random_seed": random_seed,
            }
        )
    return add_evaluation_recommendations(pd.DataFrame(rows))


def fit_kmeans(
    selected: pd.DataFrame,
    k: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, KMeans]:
    """Entrena K-Means y devuelve municipios etiquetados y centroides."""

    features = selected[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median(numeric_only=True)).clip(0, 1)
    if k < 2:
        raise ValueError("k debe ser al menos 2.")
    if k >= len(features):
        raise ValueError("k debe ser menor que la cantidad de municipios seleccionados.")

    model = KMeans(n_clusters=k, random_state=random_seed, n_init=20)
    labels = model.fit_predict(features)
    distances = model.transform(features)

    clustered = selected.copy()
    clustered["cluster_kmeans"] = labels
    clustered["cluster_kmeans_label"] = clustered["cluster_kmeans"].map(lambda value: f"cluster_{value}")
    clustered["distancia_centroide_kmeans"] = distances[np.arange(len(features)), labels]
    clustered["silhouette_municipio"] = silhouette_samples(features, labels)
    clustered["silhouette_modelo_k"] = silhouette_score(features, labels)

    centroids = pd.DataFrame(model.cluster_centers_, columns=FEATURE_COLUMNS)
    centroids.insert(0, "cluster_kmeans", range(k))
    centroids["cluster_kmeans_label"] = centroids["cluster_kmeans"].map(lambda value: f"cluster_{value}")
    centroids["random_seed"] = random_seed
    return clustered, centroids, model


def build_cluster_profile(clustered: pd.DataFrame) -> pd.DataFrame:
    """Resume perfil de cada cluster."""

    aggregations = {
        "codigo_dane": "count",
        SCORE_COLUMN: "mean",
        "s_i_solar": "mean",
        "d_i_demanda": "mean",
        "g_i_red": "mean",
        "p_i_pendiente_proxy": "mean",
        "u_i_uso_suelo": "mean",
        "pvout_kwh_kwp_day": "mean",
        "dist_subestacion_km": "mean",
        "pct_area_protegida_runap": "mean",
        "flag_revision_demanda": "mean",
        "flag_atipico_eda_demanda": "mean",
        "silhouette_municipio": "mean",
        "distancia_centroide_kmeans": "mean",
    }
    available = {column: func for column, func in aggregations.items() if column in clustered.columns}
    profile = clustered.groupby("cluster_kmeans").agg(available).reset_index()
    profile = profile.rename(
        columns={
            "codigo_dane": "municipios",
            SCORE_COLUMN: "v_i_promedio",
            "s_i_solar": "s_i_solar_promedio",
            "d_i_demanda": "d_i_demanda_promedio",
            "g_i_red": "g_i_red_promedio",
            "p_i_pendiente_proxy": "p_i_pendiente_promedio",
            "u_i_uso_suelo": "u_i_uso_suelo_promedio",
            "pvout_kwh_kwp_day": "pvout_promedio",
            "dist_subestacion_km": "dist_subestacion_km_promedio",
            "pct_area_protegida_runap": "pct_area_protegida_runap_promedio",
            "flag_revision_demanda": "pct_revision_demanda",
            "flag_atipico_eda_demanda": "pct_atipico_demanda",
            "silhouette_municipio": "silhouette_promedio",
            "distancia_centroide_kmeans": "distancia_centroide_promedio",
        }
    )

    def describe_cluster(row: pd.Series) -> str:
        strengths: list[str] = []
        weaknesses: list[str] = []
        if row.get("s_i_solar_promedio", 0) >= 0.70:
            strengths.append("alto recurso solar")
        if row.get("g_i_red_promedio", 0) >= 0.95:
            strengths.append("muy cerca de red")
        if row.get("p_i_pendiente_promedio", 0) >= 0.75:
            strengths.append("pendiente favorable")
        if row.get("u_i_uso_suelo_promedio", 0) >= 0.85:
            strengths.append("baja restriccion RUNAP")
        if row.get("d_i_demanda_promedio", 0) >= 0.50:
            strengths.append("demanda proxy alta como contexto")
        if row.get("p_i_pendiente_promedio", 1) < 0.50:
            weaknesses.append("pendiente desfavorable")
        if row.get("u_i_uso_suelo_promedio", 1) < 0.70:
            weaknesses.append("mayor presencia RUNAP")
        if row.get("pct_atipico_demanda", 0) > 0:
            weaknesses.append("demanda en revision como contexto")
        if not strengths:
            strengths.append("condiciones intermedias")
        if weaknesses:
            return "; ".join(strengths) + " | revisar: " + ", ".join(weaknesses)
        return "; ".join(strengths)

    profile["interpretacion_cluster"] = profile.apply(describe_cluster, axis=1)
    return profile.sort_values("v_i_promedio", ascending=False)


def value_for_k(evaluation: pd.DataFrame, k: int, column: str) -> float | None:
    if evaluation.empty or column not in evaluation.columns:
        return None
    rows = evaluation[evaluation["k"].eq(k)]
    if rows.empty:
        return None
    value = rows.iloc[0][column]
    return None if pd.isna(value) else float(value)


def recommended_k(evaluation: pd.DataFrame, column: str) -> int | None:
    if evaluation.empty or column not in evaluation.columns:
        return None
    rows = evaluation[pd.to_numeric(evaluation[column], errors="coerce").fillna(0).eq(1)]
    if rows.empty:
        return None
    return int(rows.iloc[0]["k"])


def build_selection_summary(
    evaluation: pd.DataFrame,
    clustered: pd.DataFrame,
    excluded: pd.DataFrame,
    k: int,
    k_selection_mode: str,
    random_seed: int,
    include_restricted: bool,
    exclude_demand_outliers: bool,
) -> pd.DataFrame:
    """Construye resumen de seleccion de K para informe y dashboard."""

    elbow_k = recommended_k(evaluation, "recomendado_codo")
    silhouette_k = recommended_k(evaluation, "recomendado_silhouette")
    silhouette_used = value_for_k(evaluation, k, "silhouette")
    inertia_used = value_for_k(evaluation, k, "inertia")

    row = {
        "k_usado": k,
        "k_recomendado_codo": elbow_k,
        "k_recomendado_silhouette": silhouette_k,
        "silhouette_k_usado": silhouette_used,
        "inertia_k_usado": inertia_used,
        "municipios_agrupados": len(clustered),
        "municipios_excluidos": len(excluded),
        "modo_seleccion_k": k_selection_mode,
        "random_seed": random_seed,
        "include_restricted": include_restricted,
        "exclude_demand_outliers": exclude_demand_outliers,
        "variables_entrenamiento": ", ".join(FEATURE_COLUMNS),
        "criterio_k_usado": (
            "K elegido automaticamente desde la evaluacion del metodo del codo."
            if k_selection_mode == "auto_elbow"
            else "K elegido automaticamente desde el mayor coeficiente silhouette."
            if k_selection_mode == "auto_silhouette"
            else "Se conserva el K indicado por parametro para que el resultado sea reproducible; "
            "el codo y silhouette quedan como evidencia de seleccion."
        ),
    }
    return pd.DataFrame([row])


def export_evaluation_plots(evaluation: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Exporta graficos para sustentar codo y silhouette."""

    paths: dict[str, Path] = {}
    if evaluation.empty:
        return paths

    plot = evaluation.sort_values("k").copy()
    elbow_k = recommended_k(plot, "recomendado_codo")
    silhouette_k = recommended_k(plot, "recomendado_silhouette")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot["k"], plot["inertia"], marker="o")
    if elbow_k is not None:
        row = plot[plot["k"].eq(elbow_k)].iloc[0]
        ax.scatter([row["k"]], [row["inertia"]], s=90)
        ax.axvline(elbow_k, linestyle="--", linewidth=1)
        ax.text(elbow_k, row["inertia"], f"  codo k={elbow_k}", va="bottom")
    ax.set_title("Metodo del codo para K-Means")
    ax.set_xlabel("Numero de clusters K")
    ax.set_ylabel("Inercia")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "grafico_metodo_codo_kmeans.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["elbow_plot"] = path

    silhouette_plot = plot.dropna(subset=["silhouette"])
    if not silhouette_plot.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(silhouette_plot["k"], silhouette_plot["silhouette"], marker="o")
        if silhouette_k is not None:
            row = silhouette_plot[silhouette_plot["k"].eq(silhouette_k)].iloc[0]
            ax.scatter([row["k"]], [row["silhouette"]], s=90)
            ax.axvline(silhouette_k, linestyle="--", linewidth=1)
            ax.text(
                silhouette_k,
                row["silhouette"],
                f"  max silhouette k={silhouette_k}",
                va="bottom",
            )
        ax.set_title("Coeficiente silhouette por K")
        ax.set_xlabel("Numero de clusters K")
        ax.set_ylabel("Silhouette promedio")
        ax.set_ylim(-1, 1)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output_dir / "grafico_silhouette_kmeans.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths["silhouette_plot"] = path

    return paths


def write_observations(
    path: Path,
    clustered: pd.DataFrame,
    excluded: pd.DataFrame,
    profile: pd.DataFrame,
    evaluation: pd.DataFrame,
    k: int,
    k_selection_mode: str,
    random_seed: int,
    include_restricted: bool,
    exclude_demand_outliers: bool,
) -> None:
    """Documenta configuracion del clustering."""

    lines = [
        "K-Means municipal",
        "=================",
        "",
        f"Semilla fija: {random_seed}",
        f"k usado: {k}",
        f"Modo de seleccion de k: {k_selection_mode}",
        f"Municipios agrupados: {len(clustered)}",
        f"Municipios excluidos por filtros: {len(excluded)}",
        f"Incluye municipios restringidos: {include_restricted}",
        f"Excluye atipicos de demanda: {exclude_demand_outliers}",
        f"K recomendado por metodo del codo: {recommended_k(evaluation, 'recomendado_codo')}",
        f"K recomendado por mayor silhouette: {recommended_k(evaluation, 'recomendado_silhouette')}",
        f"Silhouette con k usado: {value_for_k(evaluation, k, 'silhouette')}",
        f"Inercia con k usado: {value_for_k(evaluation, k, 'inertia')}",
        "",
        "Variables usadas:",
    ]
    for column in FEATURE_COLUMNS:
        lines.append(f"- {column}")

    lines.extend(
        [
            "",
            "Notas metodologicas:",
            "- Todas las variables ya estan en escala 0-1.",
            "- K-Means no decide viabilidad; agrupa municipios con perfiles similares.",
            "- Los clusters deben interpretarse junto con V_i rural, R_i, POT y demanda como contexto.",
            "- La demanda no se usa como variable de entrenamiento; no debe definir ubicacion de granja.",
            "- Los valores faltantes se imputan con la mediana de la variable solo para entrenar K-Means.",
            "- El metodo del codo usa la inercia y una heuristica de maxima distancia a la linea entre extremos.",
            "- Silhouette se calcula para k >= 2; valores mas altos indican clusters mas separados y cohesionados.",
            "- Se generan CSV y PNG para EDA, informe y sustentacion.",
            "",
            "Perfil de clusters:",
        ]
    )
    for _, row in profile.iterrows():
        lines.append(
            f"- cluster_{int(row['cluster_kmeans'])}: {int(row['municipios'])} municipios; "
            f"V_i promedio={row.get('v_i_promedio', float('nan')):.3f}; "
            f"silhouette promedio={row.get('silhouette_promedio', float('nan')):.3f}; "
            f"{row['interpretacion_cluster']}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_k(
    evaluation: pd.DataFrame,
    requested_k: int,
    auto_k: bool,
    k_selection_metric: str,
) -> tuple[int, str]:
    """Define el K final desde parametro manual o recomendacion automatica."""

    if not auto_k:
        return requested_k, "manual"

    if evaluation.empty:
        raise ValueError("auto-k requiere evaluacion de K. No uses --no-evaluate-k con --auto-k.")

    metric_to_column = {
        "elbow": "recomendado_codo",
        "silhouette": "recomendado_silhouette",
    }
    column = metric_to_column[k_selection_metric]
    selected_k = recommended_k(evaluation, column)
    if selected_k is None:
        raise ValueError(f"No se pudo seleccionar K automaticamente con {k_selection_metric}.")
    return selected_k, f"auto_{k_selection_metric}"


def run_kmeans(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    k: int = DEFAULT_K,
    auto_k: bool = False,
    k_selection_metric: str = "elbow",
    random_seed: int = DEFAULT_RANDOM_SEED,
    include_restricted: bool = False,
    exclude_demand_outliers: bool = False,
    evaluate_k: bool = True,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
) -> dict[str, Path]:
    """Ejecuta clustering K-Means municipal."""

    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(input_path)
    selected, excluded = build_feature_matrix(
        df,
        include_restricted=include_restricted,
        exclude_demand_outliers=exclude_demand_outliers,
    )
    features = selected[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median(numeric_only=True)).clip(0, 1)

    evaluation = pd.DataFrame()
    if evaluate_k:
        evaluation = evaluate_k_values(features, k_min=k_min, k_max=k_max, random_seed=random_seed)

    k, k_selection_mode = resolve_k(
        evaluation=evaluation,
        requested_k=k,
        auto_k=auto_k,
        k_selection_metric=k_selection_metric,
    )
    clustered, centroids, _ = fit_kmeans(selected, k=k, random_seed=random_seed)
    profile = build_cluster_profile(clustered)
    summary = build_selection_summary(
        evaluation=evaluation,
        clustered=clustered,
        excluded=excluded,
        k=k,
        k_selection_mode=k_selection_mode,
        random_seed=random_seed,
        include_restricted=include_restricted,
        exclude_demand_outliers=exclude_demand_outliers,
    )

    clustered_path = output_dir / "municipios_clusters_kmeans.csv"
    profile_path = output_dir / "perfil_clusters_kmeans.csv"
    centroids_path = output_dir / "centroides_clusters_kmeans.csv"
    evaluation_path = output_dir / "evaluacion_kmeans_k.csv"
    summary_path = output_dir / "resumen_seleccion_kmeans.csv"
    excluded_path = output_dir / "municipios_excluidos_kmeans.csv"
    observations_path = output_dir / "observaciones_kmeans.txt"

    clustered.to_csv(clustered_path, index=False, encoding="utf-8-sig")
    profile.to_csv(profile_path, index=False, encoding="utf-8-sig")
    centroids.to_csv(centroids_path, index=False, encoding="utf-8-sig")
    evaluation.to_csv(evaluation_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    excluded.to_csv(excluded_path, index=False, encoding="utf-8-sig")
    plot_paths = export_evaluation_plots(evaluation, output_dir)
    write_observations(
        observations_path,
        clustered=clustered,
        excluded=excluded,
        profile=profile,
        evaluation=evaluation,
        k=k,
        k_selection_mode=k_selection_mode,
        random_seed=random_seed,
        include_restricted=include_restricted,
        exclude_demand_outliers=exclude_demand_outliers,
    )

    return {
        "clusters": clustered_path,
        "profile": profile_path,
        "centroids": centroids_path,
        "evaluation": evaluation_path,
        "summary": summary_path,
        "excluded": excluded_path,
        "observations": observations_path,
        **plot_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agrupa municipios con K-Means reproducible usando variables normalizadas del modelo."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--auto-k",
        action="store_true",
        help="Usa automaticamente el K recomendado por --k-selection-metric.",
    )
    parser.add_argument(
        "--k-selection-metric",
        choices=["elbow", "silhouette"],
        default="elbow",
        help="Metrica para --auto-k: elbow usa el metodo del codo; silhouette usa el mayor coeficiente.",
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--include-restricted", action="store_true")
    parser.add_argument("--exclude-demand-outliers", action="store_true")
    parser.add_argument("--include-demand-outliers", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-evaluate-k", action="store_true")
    parser.add_argument("--k-min", type=int, default=DEFAULT_K_MIN)
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_kmeans(
        input_path=args.input_path,
        output_dir=args.output_dir,
        k=args.k,
        auto_k=args.auto_k,
        k_selection_metric=args.k_selection_metric,
        random_seed=args.random_seed,
        include_restricted=args.include_restricted,
        exclude_demand_outliers=args.exclude_demand_outliers and not args.include_demand_outliers,
        evaluate_k=not args.no_evaluate_k,
        k_min=args.k_min,
        k_max=args.k_max,
    )
    print("K-Means municipal generado.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print(f"Semilla usada: {args.random_seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
