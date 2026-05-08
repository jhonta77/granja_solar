"""Constantes tecnicas para proyectos solar fotovoltaico utility-scale.

Fuentes
-------
NREL  : National Renewable Energy Laboratory (nrel.gov)
IRENA : International Renewable Energy Agency (irena.org)
SEIA  : Solar Energy Industries Association (seia.org)
LBNL  : Lawrence Berkeley National Laboratory

Uso
---
    from src.scoring.solar_constants import PV, JOBS, AGRO

    agua_m3_ha_year = PV.water_m3_per_mwh * energia_mwh_ha_year
    empleos_om = JOBS.om_direct_per_mw * capacidad_mw
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _PVConstants:
    # Uso de suelo
    land_ha_per_mw_nrel: float = 2.02        # NREL: 5 acres/MW utility PV
    land_ha_per_mwdc_fixed: float = 1.13     # LBNL: fixed-tilt 2.8 acres/MWdc
    land_ha_per_mwdc_tracking: float = 1.70  # LBNL: single-axis tracking 4.2 acres/MWdc

    # Agua operacional (lavado de paneles)
    water_m3_per_mwh_median: float = 0.098   # NREL: 26 gal/MWh mediano
    water_m3_per_mwh_max: float = 0.125      # NREL: 33 gal/MWh maximo

    # Pendiente maxima para prefactibilidad
    slope_pct_max: float = 5.0               # NREL: restriccion tipica <5%

    # Soiling — perdidas tipicas sin limpieza (sureste humedo EE.UU. referencia)
    soiling_loss_pct_no_cleaning: float = 12.0   # NREL: 5-12% rango tipico
    soiling_cleaning_min_per_year: int = 1        # NREL: minimo rentable

    # Solar flotante (FPV)
    fpv_performance_gain_pct: float = 7.5    # World Bank/ESMAP: 5-10% mejor PR


@dataclass(frozen=True)
class _JobConstants:
    # Despliegue (construccion + precomisionamiento) — IRENA large-scale
    construction_jobyears_per_mw: float = 2.028   # IRENA 2.6 × 78% construccion
    preconstruction_jobyears_per_mw: float = 0.156 # IRENA 2.6 × 6%

    # O&M permanente — SEIA 2025
    # 20.570 empleos O&M directos / 144.500 MW acumulados
    om_direct_per_mw: float = 0.142
    om_total_per_mw: float = 0.262   # directo + indirecto + inducido

    # Distribucion por etapa (IRENA)
    pct_construccion: float = 0.78
    pct_om: float = 0.14
    pct_disenio: float = 0.06
    pct_business_dev: float = 0.01


@dataclass(frozen=True)
class _AgroConstants:
    # Factores UGG por categoria (FEDEGAN)
    ugg_ternera_lt1: float = 0.4
    ugg_ternero_lt1: float = 0.4
    ugg_hembra_1_2: float = 0.6
    ugg_macho_1_2: float = 0.6
    ugg_hembra_2_3: float = 0.8
    ugg_macho_2_3: float = 0.8
    ugg_vaca_gt3: float = 1.0
    ugg_toro_gt3: float = 1.2

    # Bandas de sistema productivo (FEDEGAN / AGROSAVIA)
    extensivo_bajo_max: float = 1.0    # < 1 UGG/ha
    tradicional_max: float = 3.0       # 1-3 UGG/ha (tipico: 1.5-1.8)
    # tecnificado: >= 3 UGG/ha (tipico: 3-4)

    # Productividad de forraje referencia (AGROSAVIA Sabanera)
    forage_dm_kg_ha_dry: float = 1_200.0   # epoca seca
    forage_dm_kg_ha_rainy: float = 4_129.0 # epoca lluviosa

    # Productividad carne (AGROSAVIA modelos regionales)
    beef_kg_ha_year_caribe_humedo: float = 892.0      # 4 animales/ha, Caribe humedo
    beef_kg_ha_year_piedemonte_intensive: float = 980.0 # piedemonte llanero intensivo
    beef_kg_ha_year_traditional: float = 350.0        # ceba tradicional

    # Aptitud bovina nacional (UPRA 2025)
    area_apta_bovinos_mha: float = 27.1   # millones de hectareas aptas pastoreo


@dataclass(frozen=True)
class _DualUseConstants:
    # Analogos agronómicos de sombra (AGROSAVIA silvopastoril Caribe seco)
    shade_pct_bajo_arboles_min: float = 40.0
    shade_pct_bajo_arboles_max: float = 60.0
    ugg_silvopastoril_referencia: float = 2.0   # UGG/ha con suplementacion

    # Rango de agua derivado (NREL) para planta utility con CF 14-27%
    water_m3_ha_year_low: float = 60.0
    water_m3_ha_year_high: float = 147.0


PV = _PVConstants()
JOBS = _JobConstants()
AGRO = _AgroConstants()
DUAL = _DualUseConstants()
