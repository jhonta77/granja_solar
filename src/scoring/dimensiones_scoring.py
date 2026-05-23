"""Motor de scoring multidimensional para viabilidad solar municipal.

Arquitectura
------------
Reemplaza el score plano V_i por 5 dimensiones independientes que pueden
calcularse con datos parciales. Cada dimension tiene peso propio y tolera
variables faltantes redistribuyendo el peso entre las disponibles.

    score_fisico        (0.30): recurso solar, pendiente, clima
    score_electrico     (0.25): distancia red, tension, capacidad
    score_economico     (0.20): tierra, vias, agua
    score_agropecuario  (0.15): carga bovina, oportunidad agrovoltaica
    score_riesgo        (0.10): RUNAP, POT, inundacion, lluvia extrema

Formula final:
    v_i_multidimensional = R_i * sum(score_dim * w_dim) / sum(w_dim_disponibles)

Retrocompatibilidad
-------------------
v_i_modelo_rural NO se modifica. Las 5 dimensiones se agregan como columnas
nuevas en la tabla de viabilidad.

Uso standalone
--------------
    from src.scoring.dimensiones_scoring import build_multidimensional_score
    df_multi = build_multidimensional_score(base_df, nasa_df, upra_tierra_df, ...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DIMENSION_WEIGHTS: dict[str, float] = {
    "score_fisico":       0.30,
    "score_electrico":    0.25,
    "score_economico":    0.20,
    "score_agropecuario": 0.15,
    "score_riesgo":       0.10,
}

_NIVEL_TENSION_SCORES = {
    "Nivel 5": 1.0,
    "Nivel 4": 0.8,
    "Nivel 3": 0.5,
    "Nivel 2": 0.2,
    "Nivel 1": 0.0,
}


def _minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    vmin = values.min(skipna=True)
    vmax = values.max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series(pd.NA, index=series.index, dtype="Float64")
    if higher_is_better:
        return ((values - vmin) / (vmax - vmin)).clip(0.0, 1.0)
    return ((vmax - values) / (vmax - vmin)).clip(0.0, 1.0)


def _weighted_dim(terms: list[tuple[pd.Series, float]]) -> pd.Series:
    """Combina terminos redistribuyendo pesos por fila si hay nulos."""
    index = terms[0][0].index
    usable = [(s, w) for s, w in terms if pd.to_numeric(s, errors="coerce").notna().any()]
    if not usable:
        return pd.Series(pd.NA, index=index, dtype="Float64")

    result = pd.Series(0.0, index=index, dtype="float64")
    denominator = pd.Series(0.0, index=index, dtype="float64")
    for series, weight in usable:
        values = pd.to_numeric(series, errors="coerce")
        has_value = values.notna()
        result += values.fillna(0.0) * weight
        denominator += has_value.astype(float) * weight

    scored = result / denominator.where(denominator > 0)
    return scored.clip(0.0, 1.0).astype("Float64")


def _first_available(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    """Retorna la primera columna disponible con valores no-nulos."""
    for col in candidates:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            return df[col]
    return None


def _build_score_fisico(df: pd.DataFrame) -> pd.Series:
    """score_fisico: recurso solar, pendiente, temperatura, nubosidad, viento, lluvia."""
    terms = []

    if "pvout_kwh_kwp_day" in df.columns:
        terms.append((_minmax(df["pvout_kwh_kwp_day"], higher_is_better=True), 0.35))

    if "score_pendiente" in df.columns:
        terms.append((pd.to_numeric(df["score_pendiente"], errors="coerce"), 0.25))

    # Temperatura: prefiere ERA5 por resolucion espacial, luego IDEAM BART, luego NASA.
    t2m_series = _first_available(df, ["era5_t2m_media_c", "ideam_t_media_c", "t2m_media"])
    if t2m_series is not None:
        t2m = pd.to_numeric(t2m_series, errors="coerce")
        terms.append((_minmax(t2m.clip(upper=40.0), higher_is_better=False), 0.15))

    # Nubosidad: ERA5 si existe, luego NASA.
    cloud_series = _first_available(df, ["era5_cloud_cover_media", "cloud_amt_media"])
    if cloud_series is not None:
        terms.append((_minmax(cloud_series, higher_is_better=False), 0.10))

    # Viento: prefiere ERA5 por resolucion espacial, luego IDEAM BART, luego NASA.
    ws_series = _first_available(df, ["era5_ws10m_media_m_s", "ideam_viento_media_m_s", "ws10m_media"])
    if ws_series is not None:
        ws = pd.to_numeric(ws_series, errors="coerce")
        terms.append((_minmax(ws.clip(0, 15), higher_is_better=False), 0.10))

    # Precipitacion: prefiere ERA5 por resolucion espacial, luego IDEAM BART, luego NASA.
    prec_series = _first_available(
        df, ["era5_prec_suma_mm_year", "ideam_prec_suma_mm_year", "prectotcorr_suma_mm_year"]
    )
    if prec_series is not None:
        prec = pd.to_numeric(prec_series, errors="coerce")
        terms.append((_minmax(prec, higher_is_better=False), 0.05))

    if not terms:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    return _weighted_dim(terms)


def _build_score_electrico(df: pd.DataFrame) -> pd.Series:
    """score_electrico: distancia subestacion, nivel tension, capacidad MVA."""
    terms = []

    if "dist_subestacion_km" in df.columns:
        terms.append((_minmax(df["dist_subestacion_km"], higher_is_better=False), 0.50))

    if "nivel_tension_mas_cercana" in df.columns:
        tension_str = df["nivel_tension_mas_cercana"].astype("string").str.strip()
        tension_num = tension_str.map(_NIVEL_TENSION_SCORES)
        if tension_num.notna().any():
            terms.append((tension_num, 0.30))

    if "capacidad_mva_mas_cercana" in df.columns:
        terms.append((_minmax(df["capacidad_mva_mas_cercana"], higher_is_better=True), 0.20))

    if not terms:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    return _weighted_dim(terms)


def _build_score_economico(
    df: pd.DataFrame,
    upra_tierra: pd.DataFrame | None,
    invias: pd.DataFrame | None,
    sui_agua: pd.DataFrame | None,
    sui_aseo: pd.DataFrame | None = None,
) -> pd.Series:
    """score_economico: tierra, vias, agua."""
    base = df[["codigo_dane"]].copy() if "codigo_dane" in df.columns else df[[df.columns[0]]].copy()
    key = "codigo_dane" if "codigo_dane" in df.columns else df.columns[0]
    base = base.rename(columns={key: "codigo_dane"})

    terms = []

    if upra_tierra is not None and "precio_tierra_cop_ha" in upra_tierra.columns:
        merged = base.merge(
            upra_tierra[["codigo_dane", "precio_tierra_cop_ha"]].copy(),
            on="codigo_dane",
            how="left",
        )
        terms.append((_minmax(merged["precio_tierra_cop_ha"], higher_is_better=False), 0.35))

    if invias is not None and "dist_via_primaria_km" in invias.columns:
        merged = base.merge(
            invias[["codigo_dane", "dist_via_primaria_km"]].copy(),
            on="codigo_dane",
            how="left",
        )
        terms.append((_minmax(merged["dist_via_primaria_km"], higher_is_better=False), 0.30))

    prec_series = _first_available(
        df, ["era5_prec_suma_mm_year", "ideam_prec_suma_mm_year", "prectotcorr_suma_mm_year"]
    )
    if prec_series is not None:
        prec = pd.to_numeric(prec_series, errors="coerce")
        prec_clipped = prec.clip(300, 3000)
        prec_norm = _minmax(prec_clipped, higher_is_better=True)
        terms.append((prec_norm, 0.20))

    if sui_agua is not None and "costo_agua_cop_m3" in sui_agua.columns:
        merged = base.merge(
            sui_agua[["codigo_dane", "costo_agua_cop_m3"]].copy(),
            on="codigo_dane",
            how="left",
        )
        terms.append((_minmax(merged["costo_agua_cop_m3"], higher_is_better=False), 0.15))

    if sui_aseo is not None and "costo_aseo_cop_ton" in sui_aseo.columns:
        merged = base.merge(
            sui_aseo[["codigo_dane", "costo_aseo_cop_ton"]].copy(),
            on="codigo_dane",
            how="left",
        )
        terms.append((_minmax(merged["costo_aseo_cop_ton"], higher_is_better=False), 0.10))

    if not terms:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    return _weighted_dim(terms)


def _build_score_agropecuario(
    df: pd.DataFrame,
    upra_tierra: pd.DataFrame | None,
    upra_agro: pd.DataFrame | None,
) -> pd.Series:
    """score_agropecuario: carga bovina (proxy oportunidad agrovoltaica), precio tierra."""
    base = df[["codigo_dane"]].copy() if "codigo_dane" in df.columns else df[[df.columns[0]]].copy()
    key = "codigo_dane" if "codigo_dane" in df.columns else df.columns[0]
    base = base.rename(columns={key: "codigo_dane"})

    terms = []

    carga_series: pd.Series | None = None
    precio_series: pd.Series | None = None

    if upra_agro is not None and "carga_bovina_ua_ha" in upra_agro.columns:
        merged_agro = base.merge(
            upra_agro[["codigo_dane", "carga_bovina_ua_ha"]].copy(),
            on="codigo_dane",
            how="left",
        )
        carga_series = pd.to_numeric(merged_agro["carga_bovina_ua_ha"], errors="coerce")
        terms.append((_minmax(carga_series, higher_is_better=True), 0.50))

    if upra_tierra is not None and "precio_tierra_cop_ha" in upra_tierra.columns:
        merged_tierra = base.merge(
            upra_tierra[["precio_tierra_cop_ha"]].assign(codigo_dane=upra_tierra["codigo_dane"]),
            on="codigo_dane",
            how="left",
        )
        precio_series = pd.to_numeric(merged_tierra["precio_tierra_cop_ha"], errors="coerce")

    if carga_series is not None and precio_series is not None:
        costo_op = carga_series * precio_series
        terms.append((_minmax(costo_op, higher_is_better=True), 0.30))

    if "apto_doble_uso_pastoreo" in df.columns:
        agrovoltaica = pd.to_numeric(df["apto_doble_uso_pastoreo"], errors="coerce").clip(0, 1)
        terms.append((agrovoltaica, 0.20))

    # Conflicto de uso del suelo: sobreutilizacion = mayor oportunidad agrovoltaica
    if "score_conflicto" in df.columns:
        terms.append((pd.to_numeric(df["score_conflicto"], errors="coerce").clip(0, 1), 0.25))

    if not terms:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    return _weighted_dim(terms)


def _build_score_riesgo(df: pd.DataFrame, ideam: pd.DataFrame | None) -> pd.Series:
    """score_riesgo: RUNAP, POT, inundacion. Mayor score = menor riesgo."""
    terms = []

    if "u_i_no_protegido_runap" in df.columns:
        terms.append((pd.to_numeric(df["u_i_no_protegido_runap"], errors="coerce"), 0.30))

    r_pot = pd.Series(1.0, index=df.index)
    if "r_i_restriccion_pot" in df.columns:
        r_pot = pd.to_numeric(df["r_i_restriccion_pot"], errors="coerce").fillna(1.0)
    if "r_i_zona_urbana_pot" in df.columns:
        r_urbana = pd.to_numeric(df["r_i_zona_urbana_pot"], errors="coerce").fillna(1.0)
        r_pot = r_pot * r_urbana
    if r_pot.notna().any():
        terms.append((r_pot.clip(0, 1), 0.20))

    if ideam is not None and "riesgo_inundacion_idx" in ideam.columns:
        base = df[["codigo_dane"]].copy() if "codigo_dane" in df.columns else df[[df.columns[0]]].copy()
        key = "codigo_dane" if "codigo_dane" in df.columns else df.columns[0]
        base = base.rename(columns={key: "codigo_dane"})
        merged = base.merge(
            ideam[["codigo_dane", "riesgo_inundacion_idx"]].copy(),
            on="codigo_dane",
            how="left",
        )
        riesgo = pd.to_numeric(merged["riesgo_inundacion_idx"], errors="coerce")
        terms.append((_minmax(riesgo, higher_is_better=False), 0.25))

    prec_series = _first_available(
        df, ["era5_prec_suma_mm_year", "ideam_prec_suma_mm_year", "prectotcorr_suma_mm_year"]
    )
    if prec_series is not None:
        prec = pd.to_numeric(prec_series, errors="coerce")
        prec_extrema = _minmax(prec.clip(lower=2000), higher_is_better=False)
        terms.append((prec_extrema, 0.25))

    # Fraccion de area en exclusion/proteccion: mas exclusion = mayor riesgo regulatorio
    if "pct_exclusion" in df.columns:
        excl = pd.to_numeric(df["pct_exclusion"], errors="coerce") / 100.0
        terms.append((_minmax(excl, higher_is_better=False), 0.15))

    if not terms:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    return _weighted_dim(terms)


def _merge_optional(base: pd.DataFrame, other: pd.DataFrame | None, cols: list[str]) -> pd.DataFrame:
    if other is None:
        return base
    available = ["codigo_dane"] + [c for c in cols if c in other.columns]
    if "codigo_dane" not in other.columns or len(available) <= 1:
        return base
    return base.merge(other[available], on="codigo_dane", how="left")


def build_multidimensional_score(
    base: pd.DataFrame,
    nasa_municipios: pd.DataFrame | None = None,
    upra_tierra: pd.DataFrame | None = None,
    upra_agro: pd.DataFrame | None = None,
    invias: pd.DataFrame | None = None,
    ideam: pd.DataFrame | None = None,
    sui_agua: pd.DataFrame | None = None,
    ideam_bart: pd.DataFrame | None = None,
    era5: pd.DataFrame | None = None,
    upra_conflicto: pd.DataFrame | None = None,
    sui_aseo: pd.DataFrame | None = None,
    dimension_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Calcula las 5 dimensiones del score multidimensional.

    Todas las fuentes son opcionales. Si una fuente no esta disponible, su
    dimension se omite y los pesos restantes se redistribuyen.

    Parametros
    ----------
    base           : tabla viabilidad_municipal_preliminar (ya calculada)
    nasa_municipios: resumen climatico NASA POWER por municipio (codigo_dane)
    upra_tierra    : precio tierra rural UPRA
    upra_agro      : carga bovina / productividad UPRA/EVA
    invias         : distancia a vias primarias
    ideam          : riesgo inundacion IDEAM/DNP
    sui_agua       : costo agua SUI

    Retorna
    -------
    DataFrame con columnas nuevas:
        score_fisico, score_electrico, score_economico,
        score_agropecuario, score_riesgo,
        v_i_multidimensional, clasificacion_multidim,
        dims_disponibles, dims_faltantes
    """
    weights = dimension_weights or DEFAULT_DIMENSION_WEIGHTS.copy()

    result = base.copy()
    result["codigo_dane"] = result["codigo_dane"].astype("string").str.zfill(5)

    nasa_cols = [
        "t2m_media", "t2m_min", "t2m_max",
        "cloud_amt_media",
        "ws10m_media", "ws10m_max",
        "prectotcorr_media_mm_day", "prectotcorr_suma_mm_year",
        "allsky_sfc_sw_dwn_media_kwh_m2_day",
    ]
    if nasa_municipios is not None:
        nasa_clean = nasa_municipios.copy()
        nasa_clean["codigo_dane"] = nasa_clean["codigo_dane"].astype("string").str.zfill(5)
        result = _merge_optional(result, nasa_clean, nasa_cols)

    # IDEAM BART: renombrar columnas con prefijo para evitar colisiones
    if ideam_bart is not None:
        bart_clean = ideam_bart.copy()
        bart_clean["codigo_dane"] = bart_clean["codigo_dane"].astype("string").str.zfill(5)
        bart_rename = {
            "t_media_c": "ideam_t_media_c",
            "prec_suma_mm_year": "ideam_prec_suma_mm_year",
            "viento_media_m_s": "ideam_viento_media_m_s",
        }
        bart_cols_available = {k: v for k, v in bart_rename.items() if k in bart_clean.columns}
        if bart_cols_available:
            bart_sub = bart_clean[["codigo_dane"] + list(bart_cols_available.keys())].rename(
                columns=bart_cols_available
            )
            result = result.merge(bart_sub, on="codigo_dane", how="left")

    # ERA5: renombrar columnas con prefijo
    if era5 is not None:
        era5_clean = era5.copy()
        era5_clean["codigo_dane"] = era5_clean["codigo_dane"].astype("string").str.zfill(5)
        era5_rename = {
            "t2m_media_c": "era5_t2m_media_c",
            "prec_suma_mm_year": "era5_prec_suma_mm_year",
            "ws10m_media_m_s": "era5_ws10m_media_m_s",
            "cloud_cover_media": "era5_cloud_cover_media",
        }
        era5_cols_available = {k: v for k, v in era5_rename.items() if k in era5_clean.columns}
        if era5_cols_available:
            era5_sub = era5_clean[["codigo_dane"] + list(era5_cols_available.keys())].rename(
                columns=era5_cols_available
            )
            result = result.merge(era5_sub, on="codigo_dane", how="left")

    # UPRA Conflicto: score_conflicto, pct_exclusion
    if upra_conflicto is not None:
        conflicto_clean = upra_conflicto.copy()
        conflicto_clean["codigo_dane"] = conflicto_clean["codigo_dane"].astype("string").str.zfill(5)
        conflicto_cols = [c for c in ["score_conflicto", "pct_exclusion", "pct_sobreutilizacion_severa",
                                       "pct_subutilizacion", "conflicto_dominante"] if c in conflicto_clean.columns]
        if conflicto_cols:
            result = result.merge(
                conflicto_clean[["codigo_dane"] + conflicto_cols],
                on="codigo_dane",
                how="left",
            )

    result["score_fisico"] = _build_score_fisico(result)
    result["score_electrico"] = _build_score_electrico(result)
    result["score_economico"] = _build_score_economico(result, upra_tierra, invias, sui_agua, sui_aseo)
    result["score_agropecuario"] = _build_score_agropecuario(result, upra_tierra, upra_agro)
    result["score_riesgo"] = _build_score_riesgo(result, ideam)

    dim_cols = list(weights.keys())
    available_dims = [d for d in dim_cols if d in result.columns and pd.to_numeric(result[d], errors="coerce").notna().any()]
    missing_dims = [d for d in dim_cols if d not in available_dims]

    result["dims_disponibles"] = ",".join(available_dims)
    result["dims_faltantes"] = ",".join(missing_dims)

    r_i = pd.to_numeric(result.get("r_i_preliminar", pd.Series(1.0, index=result.index)), errors="coerce").fillna(1.0)

    if available_dims:
        total_w = sum(weights[d] for d in available_dims)
        weighted_sum = sum(
            pd.to_numeric(result[d], errors="coerce").fillna(0.0) * weights[d]
            for d in available_dims
        )
        result["v_i_multidimensional"] = (r_i * weighted_sum / total_w).clip(0.0, 1.0)
    else:
        result["v_i_multidimensional"] = pd.NA

    result["clasificacion_multidim"] = _classify(result["v_i_multidimensional"])

    return result


def _classify(score: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(score, errors="coerce")
    positive = numeric[numeric > 0].dropna()
    if positive.empty:
        return numeric.map(lambda v: "sin_datos" if pd.isna(v) else "excluida")

    p50 = positive.quantile(0.50)
    p75 = positive.quantile(0.75)
    p90 = positive.quantile(0.90)

    def label(v: Any) -> str:
        if pd.isna(v):
            return "sin_datos"
        if v <= 0:
            return "excluida"
        if v >= p90:
            return "muy_alta"
        if v >= p75:
            return "alta"
        if v >= p50:
            return "media"
        return "baja"

    return numeric.map(label)


def write_observations(path: Path, result: pd.DataFrame, weights: dict[str, float]) -> None:
    dim_counts = result["dims_disponibles"].value_counts(dropna=False).head(5)
    lines = [
        "Scoring multidimensional - 5 dimensiones",
        "==========================================",
        "",
        "Formula:",
        "  v_i_multidimensional = R_i * sum(score_dim * w_dim) / sum(w_dim_disponibles)",
        "",
        "Pesos de dimension:",
    ]
    for dim, w in weights.items():
        lines.append(f"  {dim}: {w:.2f}")
    lines.extend([
        "",
        f"Municipios con v_i_multidimensional calculado: {result['v_i_multidimensional'].notna().sum()}",
        "",
        "Clasificacion multidim:",
    ])
    for label, count in result["clasificacion_multidim"].value_counts(dropna=False).items():
        lines.append(f"  {label}: {count}")
    lines.extend([
        "",
        "Retrocompatibilidad: v_i_modelo_rural no se modifica.",
        "Las 5 dimensiones son adicionales al score preliminar existente.",
        "",
        "Nota: dimensiones sin datos disponibles se omiten y el peso",
        "se redistribuye entre las dimensiones con datos.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
