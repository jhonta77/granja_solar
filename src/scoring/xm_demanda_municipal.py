from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUNICIPALITIES_PATH = (
    PROJECT_ROOT / "data" / "clean" / "base_municipios" / "municipios_distritos_colombia.csv"
)
DEFAULT_XM_SUMMARY_PATH = PROJECT_ROOT / "data" / "clean" / "xm_top10" / "xm_resumen_zonas.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "xm_demanda_municipal"


DEPARTMENT_XM_MAPPING: dict[str, dict[str, Any]] = {
    "ANTIOQUIA": {
        "zona_xm": "SubAntioquia",
        "tipo_mapeo": "directo_departamento",
        "confianza": "alta",
        "revision": 0,
        "nota": "Coincidencia directa entre departamento y subarea XM.",
    },
    "ATLANTICO": {
        "zona_xm": "SubAtlantico",
        "tipo_mapeo": "directo_departamento",
        "confianza": "alta",
        "revision": 0,
        "nota": "Coincidencia directa entre departamento y subarea XM.",
    },
    "BOLIVAR": {
        "zona_xm": "SubBolivar",
        "tipo_mapeo": "directo_departamento",
        "confianza": "alta",
        "revision": 0,
        "nota": "Coincidencia directa entre departamento y subarea XM.",
    },
    "SANTANDER": {
        "zona_xm": "SubSantander",
        "tipo_mapeo": "directo_departamento",
        "confianza": "alta",
        "revision": 0,
        "nota": "Coincidencia directa entre departamento y subarea XM.",
    },
    "META": {
        "zona_xm": "SubMeta",
        "tipo_mapeo": "directo_departamento",
        "confianza": "alta",
        "revision": 0,
        "nota": "Coincidencia directa entre departamento y subarea XM.",
    },
    "ARAUCA": {
        "zona_xm": "SubArauca",
        "tipo_mapeo": "directo_departamento",
        "confianza": "alta",
        "revision": 0,
        "nota": "Coincidencia directa entre departamento y subarea XM.",
    },
    "CAQUETA": {
        "zona_xm": "SubCaqueta",
        "tipo_mapeo": "directo_departamento",
        "confianza": "alta",
        "revision": 0,
        "nota": "Coincidencia directa entre departamento y subarea XM.",
    },
    "PUTUMAYO": {
        "zona_xm": "SubPutumayo",
        "tipo_mapeo": "directo_departamento",
        "confianza": "alta",
        "revision": 0,
        "nota": "Coincidencia directa entre departamento y subarea XM.",
    },
    "NORTE DE SANTANDER": {
        "zona_xm": "SubNorteSantander",
        "tipo_mapeo": "directo_departamento",
        "confianza": "alta",
        "revision": 0,
        "nota": "Coincidencia directa entre departamento y subarea XM.",
    },
    "VALLE DEL CAUCA": {
        "zona_xm": "SubValle",
        "tipo_mapeo": "directo_departamento",
        "confianza": "alta",
        "revision": 0,
        "nota": "SubValle se usa como proxy para Valle del Cauca.",
    },
    "CORDOBA": {
        "zona_xm": "SubCordoba-Sucre",
        "tipo_mapeo": "compuesto_explicito",
        "confianza": "media",
        "revision": 0,
        "nota": "Subarea XM agrupa Cordoba y Sucre; proxy regional compuesto.",
    },
    "SUCRE": {
        "zona_xm": "SubCordoba-Sucre",
        "tipo_mapeo": "compuesto_explicito",
        "confianza": "media",
        "revision": 0,
        "nota": "Subarea XM agrupa Cordoba y Sucre; proxy regional compuesto.",
    },
    "BOYACA": {
        "zona_xm": "SubBoyaca-Casanare",
        "tipo_mapeo": "compuesto_explicito",
        "confianza": "media",
        "revision": 0,
        "nota": "Subarea XM agrupa Boyaca y Casanare; proxy regional compuesto.",
    },
    "CASANARE": {
        "zona_xm": "SubBoyaca-Casanare",
        "tipo_mapeo": "compuesto_explicito",
        "confianza": "media",
        "revision": 0,
        "nota": "Subarea XM agrupa Boyaca y Casanare; proxy regional compuesto.",
    },
    "HUILA": {
        "zona_xm": "SubHuila-Tolima",
        "tipo_mapeo": "compuesto_explicito",
        "confianza": "media",
        "revision": 0,
        "nota": "Subarea XM agrupa Huila y Tolima; proxy regional compuesto.",
    },
    "TOLIMA": {
        "zona_xm": "SubHuila-Tolima",
        "tipo_mapeo": "compuesto_explicito",
        "confianza": "media",
        "revision": 0,
        "nota": "Subarea XM agrupa Huila y Tolima; proxy regional compuesto.",
    },
    "CAUCA": {
        "zona_xm": "SubCauca-Narinno",
        "tipo_mapeo": "compuesto_explicito",
        "confianza": "media",
        "revision": 0,
        "nota": "Subarea XM agrupa Cauca y Nariño; proxy regional compuesto.",
    },
    "NARINO": {
        "zona_xm": "SubCauca-Narinno",
        "tipo_mapeo": "compuesto_explicito",
        "confianza": "media",
        "revision": 0,
        "nota": "Subarea XM agrupa Cauca y Nariño; proxy regional compuesto.",
    },
    "CALDAS": {
        "zona_xm": "SubCQR",
        "tipo_mapeo": "compuesto_acronimo_inferido",
        "confianza": "media_baja",
        "revision": 1,
        "nota": "CQR se interpreta como Caldas-Quindio-Risaralda; requiere confirmacion metodologica.",
    },
    "QUINDIO": {
        "zona_xm": "SubCQR",
        "tipo_mapeo": "compuesto_acronimo_inferido",
        "confianza": "media_baja",
        "revision": 1,
        "nota": "CQR se interpreta como Caldas-Quindio-Risaralda; requiere confirmacion metodologica.",
    },
    "RISARALDA": {
        "zona_xm": "SubCQR",
        "tipo_mapeo": "compuesto_acronimo_inferido",
        "confianza": "media_baja",
        "revision": 1,
        "nota": "CQR se interpreta como Caldas-Quindio-Risaralda; requiere confirmacion metodologica.",
    },
    "CESAR": {
        "zona_xm": "SubGCM",
        "tipo_mapeo": "compuesto_acronimo_inferido",
        "confianza": "media_baja",
        "revision": 1,
        "nota": "GCM se interpreta como Guajira-Cesar-Magdalena; requiere confirmacion metodologica.",
    },
    "LA GUAJIRA": {
        "zona_xm": "SubGCM",
        "tipo_mapeo": "compuesto_acronimo_inferido",
        "confianza": "media_baja",
        "revision": 1,
        "nota": "GCM se interpreta como Guajira-Cesar-Magdalena; requiere confirmacion metodologica.",
    },
    "MAGDALENA": {
        "zona_xm": "SubGCM",
        "tipo_mapeo": "compuesto_acronimo_inferido",
        "confianza": "media_baja",
        "revision": 1,
        "nota": "GCM se interpreta como Guajira-Cesar-Magdalena; requiere confirmacion metodologica.",
    },
}


def normalize_text(value: Any) -> str:
    """Normaliza textos para mapear departamentos."""

    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().upper()


def minmax_score(series: pd.Series) -> pd.Series:
    """Normaliza demanda a escala 0-1."""

    values = pd.to_numeric(series, errors="coerce")
    minimum = values.min(skipna=True)
    maximum = values.max(skipna=True)
    if pd.isna(minimum) or pd.isna(maximum) or minimum == maximum:
        return pd.Series([pd.NA] * len(values), index=values.index, dtype="Float64")
    return ((values - minimum) / (maximum - minimum)).clip(0, 1)


def load_inputs(municipalities_path: Path, xm_summary_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga municipios y resumen XM."""

    if not municipalities_path.exists():
        raise FileNotFoundError(f"No existe base municipal: {municipalities_path}")
    if not xm_summary_path.exists():
        raise FileNotFoundError(f"No existe resumen XM: {xm_summary_path}")

    municipalities = pd.read_csv(municipalities_path, dtype={"codigo_dane": "string"})
    municipalities["codigo_dane"] = municipalities["codigo_dane"].astype("string").str.zfill(5)
    municipalities["departamento_normalizado_modelo"] = municipalities["departamento"].map(normalize_text)
    municipalities["municipio_normalizado_modelo"] = municipalities["municipio"].map(normalize_text)

    xm = pd.read_csv(xm_summary_path)
    xm_areas = xm[xm["tipo_archivo"].eq("pron_areas")].copy()
    if xm_areas.empty:
        raise ValueError("No hay registros pron_areas en el resumen XM.")
    xm_areas["zona"] = xm_areas["zona"].astype("string").str.strip()
    return municipalities, xm_areas


def build_mapping_table(xm_areas: pd.DataFrame) -> pd.DataFrame:
    """Exporta tabla de mapeo departamento/subarea con trazabilidad."""

    rows: list[dict[str, Any]] = []
    for department, metadata in DEPARTMENT_XM_MAPPING.items():
        rows.append(
            {
                "departamento_normalizado": department,
                **metadata,
                "zona_xm_existe_en_fuente": metadata["zona_xm"] in set(xm_areas["zona"].astype(str)),
            }
        )

    # Casos especiales por municipio.
    rows.append(
        {
            "departamento_normalizado": "CUNDINAMARCA",
            "zona_xm": "SubBogota",
            "tipo_mapeo": "especial_bogota_y_cundinamarca_ambigua",
            "confianza": "baja",
            "revision": 1,
            "nota": "Bogota D.C. se mapea directo a SubBogota; otros municipios de Cundinamarca se marcan ambiguos.",
            "zona_xm_existe_en_fuente": "SubBogota" in set(xm_areas["zona"].astype(str)),
        }
    )
    return pd.DataFrame(rows)


def infer_mapping(row: pd.Series) -> dict[str, Any]:
    """Asigna subarea XM a un municipio y etiqueta incertidumbre."""

    code = str(row.get("codigo_dane", "")).zfill(5)
    department = normalize_text(row.get("departamento"))
    municipality = normalize_text(row.get("municipio"))

    if code == "11001" or municipality == "BOGOTA D C":
        return {
            "zona_xm_demanda": "SubBogota",
            "tipo_mapeo_demanda": "especial_bogota_directo",
            "confianza_mapeo_demanda": "alta",
            "flag_revision_demanda": 0,
            "flag_atipico_eda_demanda": 0,
            "nota_mapeo_demanda": "Bogota D.C. se asigna directamente a SubBogota.",
        }

    if department == "CUNDINAMARCA":
        return {
            "zona_xm_demanda": "SubBogota",
            "tipo_mapeo_demanda": "ambigua_cundinamarca_subbogota",
            "confianza_mapeo_demanda": "baja",
            "flag_revision_demanda": 1,
            "flag_atipico_eda_demanda": 1,
            "nota_mapeo_demanda": "Municipio de Cundinamarca asignado provisionalmente a SubBogota; demanda dominada por Bogota.",
        }

    metadata = DEPARTMENT_XM_MAPPING.get(department)
    if metadata:
        revision = int(metadata["revision"])
        return {
            "zona_xm_demanda": metadata["zona_xm"],
            "tipo_mapeo_demanda": metadata["tipo_mapeo"],
            "confianza_mapeo_demanda": metadata["confianza"],
            "flag_revision_demanda": revision,
            "flag_atipico_eda_demanda": revision,
            "nota_mapeo_demanda": metadata["nota"],
        }

    return {
        "zona_xm_demanda": pd.NA,
        "tipo_mapeo_demanda": "sin_mapeo_xm",
        "confianza_mapeo_demanda": "sin_mapeo",
        "flag_revision_demanda": 1,
        "flag_atipico_eda_demanda": 1,
        "nota_mapeo_demanda": "No se encontro subarea XM pron_areas equivalente; revisar si pertenece a sistema aislado o fuente faltante.",
    }


def build_demand_proxy(municipalities: pd.DataFrame, xm_areas: pd.DataFrame) -> pd.DataFrame:
    """Construye demanda municipal proxy desde subareas XM."""

    mapping_records = municipalities.apply(infer_mapping, axis=1, result_type="expand")
    result = pd.concat([municipalities.copy(), mapping_records], axis=1)

    xm_columns = [
        "zona",
        "valor_promedio",
        "valor_maximo",
        "valor_p95",
        "desviacion",
        "dias_cubiertos",
        "score_relativo",
        "probabilidad_relativa",
    ]
    xm_lookup = xm_areas[xm_columns].rename(
        columns={
            "zona": "zona_xm_demanda",
            "valor_promedio": "demanda_xm_valor_promedio",
            "valor_maximo": "demanda_xm_valor_maximo",
            "valor_p95": "demanda_xm_valor_p95",
            "desviacion": "demanda_xm_desviacion",
            "dias_cubiertos": "demanda_xm_dias_cubiertos",
            "score_relativo": "demanda_xm_score_relativo_fuente",
            "probabilidad_relativa": "demanda_xm_probabilidad_relativa_fuente",
        }
    )
    result = result.merge(xm_lookup, on="zona_xm_demanda", how="left", validate="many_to_one")
    result["demanda_xm_proxy_mwh_o_unidad_fuente"] = result["demanda_xm_valor_promedio"]
    result["d_i_demanda"] = minmax_score(result["demanda_xm_proxy_mwh_o_unidad_fuente"])
    result["demanda_xm_disponible"] = result["d_i_demanda"].notna().astype(int)
    result.loc[result["demanda_xm_disponible"].eq(0), "flag_revision_demanda"] = 1
    result.loc[result["demanda_xm_disponible"].eq(0), "flag_atipico_eda_demanda"] = 1
    result["decision_eda_demanda"] = result["tipo_mapeo_demanda"].map(
        {
            "directo_departamento": "usar_como_proxy",
            "compuesto_explicito": "usar_como_proxy_compuesto",
            "especial_bogota_directo": "usar_como_proxy",
            "compuesto_acronimo_inferido": "revisar_o_tratar_como_atipico_metodologico",
            "ambigua_cundinamarca_subbogota": "revisar_o_excluir_en_sensibilidad",
            "sin_mapeo_xm": "excluir_demanda_o_buscar_fuente_alterna",
        }
    ).fillna("revisar")
    return result.sort_values(["flag_revision_demanda", "departamento", "municipio"])


def label_xm_zones(xm_areas: pd.DataFrame, demand_proxy: pd.DataFrame) -> pd.DataFrame:
    """Etiqueta zonas XM usadas/no usadas para seguimiento EDA."""

    used = demand_proxy.groupby("zona_xm_demanda", dropna=True).agg(
        municipios_asignados=("codigo_dane", "count"),
        municipios_revision=("flag_revision_demanda", "sum"),
    )
    rows = xm_areas.merge(used, left_on="zona", right_index=True, how="left")
    rows["municipios_asignados"] = rows["municipios_asignados"].fillna(0).astype(int)
    rows["municipios_revision"] = rows["municipios_revision"].fillna(0).astype(int)
    rows["estado_uso_zona_xm"] = rows["municipios_asignados"].map(
        lambda count: "usada_en_proxy_municipal" if count > 0 else "no_asignada_a_municipios"
    )
    rows["flag_atipico_eda_zona_xm"] = 0
    rows.loc[rows["zona"].eq("SubCerromatoso"), "flag_atipico_eda_zona_xm"] = 1
    rows.loc[rows["zona"].eq("SubCerromatoso"), "estado_uso_zona_xm"] = (
        "zona_especial_industrial_no_asignada"
    )
    rows["nota_eda_zona_xm"] = ""
    rows.loc[rows["zona"].eq("SubCerromatoso"), "nota_eda_zona_xm"] = (
        "SubCerromatoso parece carga/subarea especial industrial; no se asigna automaticamente a municipios."
    )
    rows.loc[rows["municipios_revision"].gt(0), "nota_eda_zona_xm"] = (
        rows["nota_eda_zona_xm"].astype(str)
        + " Tiene municipios con mapeo de demanda marcado para revision."
    )
    return rows


def write_observations(
    output_path: Path,
    demand_proxy: pd.DataFrame,
    xm_zones: pd.DataFrame,
) -> None:
    """Escribe observaciones metodologicas."""

    mapping_counts = demand_proxy["tipo_mapeo_demanda"].value_counts(dropna=False)
    no_demand = int(demand_proxy["demanda_xm_disponible"].eq(0).sum())
    revision = int(demand_proxy["flag_revision_demanda"].eq(1).sum())
    lines = [
        "Demanda XM municipal proxy",
        "==========================",
        "",
        "Fuente usada: pron_areas desde data/clean/xm_top10/xm_resumen_zonas.csv.",
        "Unidad final: municipio/distrito IGAC.",
        "",
        "Regla metodologica:",
        "- Se asigna a cada municipio una subarea XM cuando existe correspondencia directa, compuesta o inferida.",
        "- Los casos no directos quedan etiquetados con flag_revision_demanda.",
        "- flag_atipico_eda_demanda no significa error numerico; significa atipico/incertidumbre metodologica para seguimiento EDA.",
        "- No se usan pron_barra para demanda municipal porque representan barras/subestaciones, no municipios.",
        "",
        f"Municipios totales: {len(demand_proxy)}",
        f"Municipios sin demanda asignada: {no_demand}",
        f"Municipios marcados para revision: {revision}",
        "",
        "Conteo por tipo de mapeo:",
    ]
    for label, count in mapping_counts.items():
        lines.append(f"- {label}: {count}")

    lines.extend(
        [
            "",
            "Zonas XM sin asignacion o especiales:",
        ]
    )
    special = xm_zones[
        xm_zones["estado_uso_zona_xm"].ne("usada_en_proxy_municipal")
        | xm_zones["flag_atipico_eda_zona_xm"].eq(1)
    ]
    if special.empty:
        lines.append("- Ninguna.")
    else:
        for _, row in special.iterrows():
            lines.append(f"- {row['zona']}: {row['estado_uso_zona_xm']}. {row['nota_eda_zona_xm']}")

    lines.extend(
        [
            "",
            "Uso recomendado:",
            "- Mantener los flags en el score para no ocultar incertidumbre.",
            "- En el informe, reportar resultados con y sin municipios flag_atipico_eda_demanda=1 como analisis de sensibilidad.",
            "- Si se consigue una fuente municipal/regional mas precisa, reemplazar este proxy.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_xm_demand_proxy(
    municipalities_path: Path = DEFAULT_MUNICIPALITIES_PATH,
    xm_summary_path: Path = DEFAULT_XM_SUMMARY_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Ejecuta construccion de demanda XM municipal proxy."""

    output_dir.mkdir(parents=True, exist_ok=True)
    municipalities, xm_areas = load_inputs(municipalities_path, xm_summary_path)
    mapping = build_mapping_table(xm_areas)
    demand_proxy = build_demand_proxy(municipalities, xm_areas)
    xm_zones = label_xm_zones(xm_areas, demand_proxy)

    mapping_path = output_dir / "mapeo_departamento_zona_xm.csv"
    proxy_path = output_dir / "xm_demanda_municipal_proxy.csv"
    zones_path = output_dir / "xm_zonas_eda_etiquetadas.csv"
    review_path = output_dir / "xm_municipios_demanda_revision.csv"
    observations_path = output_dir / "xm_demanda_observaciones.txt"

    mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    demand_proxy.to_csv(proxy_path, index=False, encoding="utf-8-sig")
    xm_zones.to_csv(zones_path, index=False, encoding="utf-8-sig")
    demand_proxy[demand_proxy["flag_revision_demanda"].eq(1)].to_csv(
        review_path,
        index=False,
        encoding="utf-8-sig",
    )
    write_observations(observations_path, demand_proxy, xm_zones)

    return {
        "mapping": mapping_path,
        "proxy": proxy_path,
        "zones_labeled": zones_path,
        "review_municipalities": review_path,
        "observations": observations_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye demanda municipal proxy desde subareas XM pron_areas con flags EDA."
    )
    parser.add_argument("--municipalities-path", type=Path, default=DEFAULT_MUNICIPALITIES_PATH)
    parser.add_argument("--xm-summary-path", type=Path, default=DEFAULT_XM_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_xm_demand_proxy(
        municipalities_path=args.municipalities_path,
        xm_summary_path=args.xm_summary_path,
        output_dir=args.output_dir,
    )
    print("Demanda XM municipal proxy generada.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print("Los casos no claros quedaron marcados con flag_revision_demanda y flag_atipico_eda_demanda.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
