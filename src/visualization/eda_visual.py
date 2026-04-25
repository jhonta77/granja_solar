from __future__ import annotations

import argparse
import math
from pathlib import Path
import matplotlib
import pandas as pd
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.dpi"] = 150

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIABILITY_PATH = PROJECT_ROOT / "data" / "clean" / "viabilidad_municipal" / "viabilidad_municipal_preliminar.csv"
OUTPUT_BASE = PROJECT_ROOT / "data" / "clean" / "visualizaciones_municipios"

# Variables para las que se generarán KDE, boxplots e histogramas.
KDE_COLUMNS = [
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
    "v_i_modelo_proxy_xm",
]

# Columnas de segmentación para los boxplots (las más relevantes para el problema)
SEGMENTATION_COLUMNS = [
    "viabilidad_pendiente",      # viable / condicional / no_viable / desconocida
    "departamento",
    "clasificacion_preliminar",   # muy_alta, alta, media, baja, excluida...
]

# Etiquetas descriptivas
LABELS = {
    "pvout_kwh_kwp_day": "PVOUT kWh/kWp/día",
    "annual_yield_kwh_kw_year": "Generación anual kWh/kW-año",
    "demanda_xm_proxy_mwh_o_unidad_fuente": "Demanda XM proxy",
    "dist_subestacion_km": "Distancia a subestación km",
    "pct_area_protegida_runap": "Porcentaje área protegida RUNAP",
    "s_i_solar": "S_i solar",
    "d_i_demanda": "D_i demanda",
    "g_i_red": "G_i red",
    "p_i_pendiente_proxy": "P_i pendiente",
    "u_i_uso_suelo": "U_i uso suelo",
    "v_i_modelo_proxy_xm": "V_i modelo proxy XM",
}


def load_data(path: Path) -> pd.DataFrame:
    """Carga la tabla municipal integrada."""
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo: {path}. Ejecute el pipeline primero.")
    df = pd.read_csv(path, dtype={"codigo_dane": "string"})
    # Convertir columnas numéricas que pudieron leerse como texto
    for col in KDE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def generate_kde(df: pd.DataFrame, output_dir: Path) -> None:
    """Genera un KDE por cada variable numérica."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for col in KDE_COLUMNS:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        if data.nunique() < 5:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        data.plot.kde(ax=ax, bw_method=0.15)
        ax.set_title(f"KDE - {LABELS.get(col, col)}")
        ax.set_xlabel(LABELS.get(col, col))
        ax.set_ylabel("Densidad")
        fig.tight_layout()
        file_name = f"kde_{col}.png"
        fig.savefig(output_dir / file_name, dpi=180)
        plt.close(fig)
        print(f"KDE guardado: {output_dir / file_name}")


def generate_histograms(df: pd.DataFrame, output_dir: Path) -> None:
    """Genera un histograma por cada variable numérica."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for col in KDE_COLUMNS:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        if data.nunique() < 2:
            continue

        # Calcular número de bins según raíz cuadrada del tamaño
        bins = min(35, max(8, int(math.sqrt(len(data)))))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(data, bins=bins, edgecolor='white')
        ax.set_title(f"Histograma - {LABELS.get(col, col)}")
        ax.set_xlabel(LABELS.get(col, col))
        ax.set_ylabel("Frecuencia")
        fig.tight_layout()
        file_name = f"histograma_{col}.png"
        fig.savefig(output_dir / file_name, dpi=180)
        plt.close(fig)
        print(f"Histograma guardado: {output_dir / file_name}")


def generate_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    """Genera boxplots por cada variable numérica segmentada por variables categóricas relevantes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for seg_col in SEGMENTATION_COLUMNS:
        if seg_col not in df.columns:
            continue
        sub = df.dropna(subset=[seg_col]).copy()
        if sub[seg_col].nunique() < 2:
            continue
        # Para categóricas con muchas clases, mantener solo las 10 más frecuentes
        if sub[seg_col].nunique() > 15:
            top_cats = sub[seg_col].value_counts().nlargest(10).index
            sub = sub[sub[seg_col].isin(top_cats)]

        for num_col in KDE_COLUMNS:
            if num_col not in sub.columns:
                continue
            plot_data = sub.dropna(subset=[num_col])
            if plot_data.empty:
                continue

            fig, ax = plt.subplots(figsize=(max(8, 0.4 * plot_data[seg_col].nunique()), 5))
            plot_data.boxplot(column=num_col, by=seg_col, ax=ax, grid=False)
            ax.set_title(f"Boxplot de {LABELS.get(num_col, num_col)} por {seg_col}")
            ax.set_xlabel(seg_col.replace("_", " ").title())
            ax.set_ylabel(LABELS.get(num_col, num_col))
            plt.suptitle("")  # eliminar título automático
            fig.tight_layout()
            file_name = f"boxplot_{num_col}_por_{seg_col}.png"
            fig.savefig(output_dir / file_name, dpi=180)
            plt.close(fig)
            print(f"Boxplot guardado: {output_dir / file_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera gráficos KDE, Boxplot e Histogramas para el EDA Visual.")
    parser.add_argument("--input", type=Path, default=VIABILITY_PATH, help="Ruta a viabilidad_municipal_preliminar.csv")
    parser.add_argument("--output-base", type=Path, default=OUTPUT_BASE, help="Carpeta base de salida")
    args = parser.parse_args()

    df = load_data(args.input)

    kde_dir = args.output_base / "kde"
    hist_dir = args.output_base / "histogramas"
    boxplots_dir = args.output_base / "boxplots"

    print("Generando gráficos KDE...")
    generate_kde(df, kde_dir)

    print("Generando Histogramas...")
    generate_histograms(df, hist_dir)

    print("Generando gráficos Boxplot...")
    generate_boxplots(df, boxplots_dir)

    print(f"\nTodos los gráficos se generaron correctamente.")
    print(f"KDE        -> {kde_dir}")
    print(f"Histogramas -> {hist_dir}")
    print(f"Boxplots   -> {boxplots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
