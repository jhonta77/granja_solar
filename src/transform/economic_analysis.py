"""
Análisis económico complementario para la viabilidad de granjas solares.
Calcula para cada municipio:
- Ingreso anual estimado por hectárea (USD y COP) basado en PVOUT local.
- Costo estimado de conexión a la subestación más cercana (USD).
- Relación beneficio/costo simple (ingreso anual / costo de conexión anualizado).
Las salidas se integran al dataset final para enriquecer el dashboard y el informe.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Constantes económicas (referencias 2024-2025, trazables a IRENA, UPME, XM)
# ---------------------------------------------------------------------------
# Tarifa estimada de venta de energía (PPA solar típico en Colombia)
TARIFA_USD_POR_MWH = 35.0          # USD/MWh (referencia UPME / XM)
# Costo de construcción de línea de transmisión (115 kV, torre metálica)
COSTO_LINEA_USD_POR_KM = 120_000.0 # USD/km
# Vida útil para anualizar el costo de conexión
VIDA_UTIL_ANIOS = 25
# Tasa de descuento simple (WACC aproximado para proyectos solares)
TASA_DESCUENTO = 0.08
# Factor de recuperación de capital (anualidad)
FRC = TASA_DESCUENTO / (1 - (1 + TASA_DESCUENTO) ** -VIDA_UTIL_ANIOS)

# Columnas que debe contener el dataset de entrada
REQUIRED_COLUMNS = [
    "codigo_dane",
    "municipio",
    "departamento",
    "annual_yield_kwh_kw_year",
    "dist_subestacion_km",
    "v_i_modelo_proxy_xm",
]

# ---------------------------------------------------------------------------
# Rutas por defecto
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIABILITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "viabilidad_municipal"
    / "viabilidad_municipal_preliminar.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "economic_analysis"
OUTPUT_DATASET = OUTPUT_DIR / "municipios_economic_analysis.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "economic_summary.csv"


def load_viability(path: Path) -> pd.DataFrame:
    """Carga la tabla de viabilidad municipal."""
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra {path}. Ejecuta primero el pipeline de viabilidad.")
    df = pd.read_csv(path, dtype={"codigo_dane": "string"})
    # Asegurar columnas numéricas
    for col in ["annual_yield_kwh_kw_year", "dist_subestacion_km", "v_i_modelo_proxy_xm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def calcular_economia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade las columnas económicas al DataFrame:
    - ingreso_usd_por_ha_anual = (annual_yield_kwh_kw_year / 1000) * TARIFA_USD_POR_MWH * capacidad_kw_por_ha
      donde capacidad_kw_por_ha se estima en 125 kW/ha (valor medio de solar_economics.py).
    - costo_conexion_usd_total = dist_subestacion_km * COSTO_LINEA_USD_POR_KM
    - costo_conexion_usd_anual = costo_conexion_usd_total * FRC
    - relacion_beneficio_costo = ingreso_usd_por_ha_anual / costo_conexion_usd_anual
    """
    # Parámetros de diseño (referencia solar_economics.py escenario base)
    CAPACIDAD_KW_POR_HA = 125.0  # kW/ha

    result = df.copy()

    # Ingreso anual por hectárea
    generacion_mwh_ha_anual = (
        pd.to_numeric(result["annual_yield_kwh_kw_year"], errors="coerce")
        * CAPACIDAD_KW_POR_HA
        / 1000.0
    )
    result["generacion_mwh_ha_anual"] = generacion_mwh_ha_anual
    result["ingreso_usd_por_ha_anual"] = generacion_mwh_ha_anual * TARIFA_USD_POR_MWH

    # Costo de conexión
    dist_km = pd.to_numeric(result["dist_subestacion_km"], errors="coerce")
    result["costo_conexion_usd_total"] = dist_km * COSTO_LINEA_USD_POR_KM
    result["costo_conexion_usd_anual"] = result["costo_conexion_usd_total"] * FRC

    # Relación beneficio/costo (solo para municipios con distancia > 0)
    mask = (result["costo_conexion_usd_anual"] > 0) & (result["ingreso_usd_por_ha_anual"] > 0)
    result["relacion_beneficio_costo"] = np.where(
        mask,
        result["ingreso_usd_por_ha_anual"] / result["costo_conexion_usd_anual"],
        np.nan,
    )

    # Agregar variables de referencia
    result["tarifa_usd_mwh"] = TARIFA_USD_POR_MWH
    result["costo_linea_usd_km"] = COSTO_LINEA_USD_POR_KM
    result["capacidad_kw_por_ha_referencia"] = CAPACIDAD_KW_POR_HA
    result["vida_util_conexion_anios"] = VIDA_UTIL_ANIOS
    result["tasa_descuento_wacc"] = TASA_DESCUENTO

    # Columnas de orden
    orden = [
        "codigo_dane",
        "municipio",
        "departamento",
        "v_i_modelo_proxy_xm",
        "annual_yield_kwh_kw_year",
        "generacion_mwh_ha_anual",
        "dist_subestacion_km",
        "ingreso_usd_por_ha_anual",
        "costo_conexion_usd_total",
        "costo_conexion_usd_anual",
        "relacion_beneficio_costo",
        "tarifa_usd_mwh",
        "costo_linea_usd_km",
        "capacidad_kw_por_ha_referencia",
        "vida_util_conexion_anios",
        "tasa_descuento_wacc",
    ]
    disponibles = [c for c in orden if c in result.columns]
    return result[disponibles + [c for c in result.columns if c not in disponibles]]


def generar_resumen(df: pd.DataFrame) -> pd.DataFrame:
    """Crea una tabla resumen con los 10 municipios de mayor relación beneficio/costo."""
    top = (
        df.dropna(subset=["relacion_beneficio_costo"])
        .nlargest(10, "relacion_beneficio_costo")
        .copy()
    )
    # Formatear para presentación
    resumen = top[
        [
            "codigo_dane",
            "municipio",
            "departamento",
            "v_i_modelo_proxy_xm",
            "ingreso_usd_por_ha_anual",
            "costo_conexion_usd_total",
            "costo_conexion_usd_anual",
            "relacion_beneficio_costo",
        ]
    ].round(2)
    return resumen


def guardar_observaciones(output_dir: Path) -> None:
    """Escribe un archivo de observaciones metodológicas."""
    texto = f"""
Análisis económico complementario para viabilidad de granja solar
==================================================================

Supuestos utilizados:
- Tarifa de venta de energía: {TARIFA_USD_POR_MWH} USD/MWh (PPA solar típico en Colombia).
- Costo de línea de transmisión: {COSTO_LINEA_USD_POR_KM:,.0f} USD/km (115 kV).
- Vida útil de la conexión: {VIDA_UTIL_ANIOS} años.
- Tasa de descuento (WACC): {TASA_DESCUENTO*100:.1f}%.
- Capacidad instalada por hectárea: 125 kW/ha (referencia NREL/IRENA).
- Generación anual por hectárea: calculada a partir del PVOUT puntual de cada municipio.

Fórmulas aplicadas:
- Ingreso anual por hectárea = generacion_mwh_ha_anual * tarifa_usd_mwh
- Costo total de conexión = dist_subestacion_km * costo_linea_usd_km
- Costo anual de conexión = costo_total_conexion * FRC  (FRC = {FRC:.4f})
- Relación beneficio/costo = ingreso_anual / costo_anual_conexion

Limitaciones:
- La distancia a la subestación es euclidiana; no considera trazado real, servidumbres ni topografía.
- El costo de línea es un promedio internacional; no incluye costos de subestación elevadora ni permisos.
- La tarifa de venta es una referencia de largo plazo; el ingreso real depende del contrato (PPA, bolsa, cargo por confiabilidad).
- Este análisis es complementario y no reemplaza un estudio financiero detallado (LCOE, TIR, VPN).

Fuentes:
- IRENA (2025). Renewable Power Generation Costs in 2024.
- UPME. GeoLCOE y referencias de costos de transmisión.
- World Bank/ESMAP/Solargis. Global Solar Atlas.
- XM. Precios de bolsa y contratos.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "economic_observations.txt").write_text(texto.strip(), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Análisis económico municipal complementario.")
    parser.add_argument("--input", type=Path, default=VIABILITY_PATH, help="Tabla de viabilidad municipal")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directorio de salida")
    args = parser.parse_args()

    df = load_viability(args.input)
    eco_df = calcular_economia(df)
    resumen_df = generar_resumen(eco_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eco_df.to_csv(args.output_dir / "municipios_economic_analysis.csv", index=False, encoding="utf-8-sig")
    resumen_df.to_csv(args.output_dir / "top10_relacion_beneficio_costo.csv", index=False, encoding="utf-8-sig")
    guardar_observaciones(args.output_dir)

    print("Análisis económico completado.")
    print(f"Dataset completo: {args.output_dir / 'municipios_economic_analysis.csv'}")
    print(f"Top 10 beneficio/costo: {args.output_dir / 'top10_relacion_beneficio_costo.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
