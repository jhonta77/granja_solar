from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data" / "clean" / "viabilidad_municipal" / "viabilidad_municipal_preliminar.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "clusters_municipios"

DEFAULT_RANDOM_SEED = 42
DEFAULT_K = 5
FEATURE_COLUMNS = [
    "s_i_solar",
    "d_i_demanda",
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
    for k in range(max(2, k_min), upper + 1):
        model = KMeans(n_clusters=k, random_state=random_seed, n_init=20)
        labels = model.fit_predict(features)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(features, labels)),
                "random_seed": random_seed,
            }
        )
    return pd.DataFrame(rows)


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

    clustered = selected.copy()
    clustered["cluster_kmeans"] = labels
    clustered["cluster_kmeans_label"] = clustered["cluster_kmeans"].map(lambda value: f"cluster_{value}")

    centroids = pd.DataFrame(model.cluster_centers_, columns=FEATURE_COLUMNS)
    centroids.insert(0, "cluster_kmeans", range(k))
    centroids["cluster_kmeans_label"] = centroids["cluster_kmeans"].map(lambda value: f"cluster_{value}")
    centroids["random_seed"] = random_seed
    return clustered, centroids, model


def build_cluster_profile(clustered: pd.DataFrame) -> pd.DataFrame:
    """Resume perfil de cada cluster."""

    aggregations = {
        "codigo_dane": "count",
        "v_i_modelo_proxy_xm": "mean",
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
    }
    available = {column: func for column, func in aggregations.items() if column in clustered.columns}
    profile = clustered.groupby("cluster_kmeans").agg(available).reset_index()
    profile = profile.rename(
        columns={
            "codigo_dane": "municipios",
            "v_i_modelo_proxy_xm": "v_i_promedio",
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
            strengths.append("demanda proxy alta")
        if row.get("p_i_pendiente_promedio", 1) < 0.50:
            weaknesses.append("pendiente desfavorable")
        if row.get("u_i_uso_suelo_promedio", 1) < 0.70:
            weaknesses.append("mayor presencia RUNAP")
        if row.get("pct_atipico_demanda", 0) > 0:
            weaknesses.append("demanda con revision EDA")
        if not strengths:
            strengths.append("condiciones intermedias")
        if weaknesses:
            return "; ".join(strengths) + " | revisar: " + ", ".join(weaknesses)
        return "; ".join(strengths)

    profile["interpretacion_cluster"] = profile.apply(describe_cluster, axis=1)
    return profile.sort_values("v_i_promedio", ascending=False)


def write_observations(
    path: Path,
    clustered: pd.DataFrame,
    excluded: pd.DataFrame,
    profile: pd.DataFrame,
    k: int,
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
        f"Municipios agrupados: {len(clustered)}",
        f"Municipios excluidos por filtros: {len(excluded)}",
        f"Incluye municipios restringidos: {include_restricted}",
        f"Excluye atipicos de demanda: {exclude_demand_outliers}",
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
            "- Los clusters deben interpretarse junto con V_i, R_i y flags de demanda.",
            "- Los valores faltantes se imputan con la mediana de la variable solo para entrenar K-Means.",
            "- No se generan diagramas; las salidas son CSV para EDA e informe.",
            "",
            "Perfil de clusters:",
        ]
    )
    for _, row in profile.iterrows():
        lines.append(
            f"- cluster_{int(row['cluster_kmeans'])}: {int(row['municipios'])} municipios; "
            f"V_i promedio={row.get('v_i_promedio', float('nan')):.3f}; "
            f"{row['interpretacion_cluster']}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_kmeans(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    k: int = DEFAULT_K,
    random_seed: int = DEFAULT_RANDOM_SEED,
    include_restricted: bool = False,
    exclude_demand_outliers: bool = True,
    evaluate_k: bool = True,
    k_min: int = 2,
    k_max: int = 10,
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

    clustered, centroids, _ = fit_kmeans(selected, k=k, random_seed=random_seed)
    profile = build_cluster_profile(clustered)

    clustered_path = output_dir / "municipios_clusters_kmeans.csv"
    profile_path = output_dir / "perfil_clusters_kmeans.csv"
    centroids_path = output_dir / "centroides_clusters_kmeans.csv"
    evaluation_path = output_dir / "evaluacion_kmeans_k.csv"
    excluded_path = output_dir / "municipios_excluidos_kmeans.csv"
    observations_path = output_dir / "observaciones_kmeans.txt"

    clustered.to_csv(clustered_path, index=False, encoding="utf-8-sig")
    profile.to_csv(profile_path, index=False, encoding="utf-8-sig")
    centroids.to_csv(centroids_path, index=False, encoding="utf-8-sig")
    evaluation.to_csv(evaluation_path, index=False, encoding="utf-8-sig")
    excluded.to_csv(excluded_path, index=False, encoding="utf-8-sig")
    write_observations(
        observations_path,
        clustered=clustered,
        excluded=excluded,
        profile=profile,
        k=k,
        random_seed=random_seed,
        include_restricted=include_restricted,
        exclude_demand_outliers=exclude_demand_outliers,
    )

    return {
        "clusters": clustered_path,
        "profile": profile_path,
        "centroids": centroids_path,
        "evaluation": evaluation_path,
        "excluded": excluded_path,
        "observations": observations_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agrupa municipios con K-Means reproducible usando variables normalizadas del modelo."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--include-restricted", action="store_true")
    parser.add_argument("--include-demand-outliers", action="store_true")
    parser.add_argument("--no-evaluate-k", action="store_true")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_kmeans(
        input_path=args.input_path,
        output_dir=args.output_dir,
        k=args.k,
        random_seed=args.random_seed,
        include_restricted=args.include_restricted,
        exclude_demand_outliers=not args.include_demand_outliers,
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
