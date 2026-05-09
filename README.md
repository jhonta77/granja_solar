# Viabilidad agrivoltaica municipal — Colombia

Pipeline reproducible en Python que evalúa, municipio por municipio, qué tan
viable es instalar una **granja solar agrivoltaica** (paneles solares + ganadería
bovina en la misma hectárea) en Colombia. Combina datos espaciales, climáticos,
eléctricos, económicos y agropecuarios en un score multidimensional y los expone
en un dashboard Streamlit interactivo con modelo P&L completo.


## 1. Modelo de scoring multidimensional

La unidad de análisis es el municipio/distrito colombiano (1,104 unidades IGAC).
El score final se calcula como promedio ponderado de cinco dimensiones, todas
normalizadas a [0, 1]:

    V_i_multidimensional =
        0.30 * score_fisico       (PVOUT, pendiente, clima)
      + 0.25 * score_electrico    (distancia subestación, tensión, capacidad)
      + 0.20 * score_economico    (precio tierra, agua, vías, aseo)
      + 0.15 * score_agropecuario (carga bovina UGG/ha)
      + 0.10 * score_riesgo       (RUNAP, viento, inundación, sequía)

Motor: [`src/scoring/dimensiones_scoring.py`](src/scoring/dimensiones_scoring.py)
Salida: [`data/clean/viabilidad_municipal/viabilidad_municipal_multidimensional.csv`](data/clean/viabilidad_municipal/viabilidad_municipal_multidimensional.csv)


## 2. Modelo financiero agrivoltaico

El modelo calcula el P&L anual completo de una granja solar agrivoltaica por
municipio. Las constantes están centralizadas en
[`src/scoring/solar_constants.py`](src/scoring/solar_constants.py).

### 2.1 Hectáreas viables

    ha_viable = area_no_RUNAP × (1 − 0.28) × 100 × score_pendiente

Sobre el área no protegida por RUNAP se descuentan promedios nacionales de uso
del suelo:

| Descuento | Factor | Fuente |
|---|---|---|
| Urbano + asentamientos | 2% | DANE/IGAC POT |
| Vías + buffer ROW 50 m | 3% | INVIAS |
| Hídrico + ronda 30 m | 8% | Decreto 2811/74 |
| Cultivos activos | 15% | UPRA frontera agrícola |

### 2.2 CAPEX desglosado

| Componente | Valor | Vida útil |
|---|---|---|
| Módulo solar (600 Wp) | 420,000 COP/panel | 25 años |
| BOS (inversor + estructura + instalación) | 2,900 M COP/MW | 25 años |
| Tierra | precio_tierra × ha | 25 años |

Densidad: 825 paneles/ha (NREL: 2.02 ha/MW, módulo de 600 Wp). Amortización
lineal sin valor residual.

### 2.3 Operación

- **Ingreso solar:** `MWh × 1000 × precio_energia` (default 200 COP/kWh)
- **Ingreso ganadero:** `ha × 2.0 UGG/ha × 250 kg × 10,000 COP`
- **Costo agua:** `MWh × 0.098 m³/MWh × tarifa_acueducto` (NREL)
- **Empleados O&M:** `0.142 empleados/MW × MW × salario × 1.52 prestaciones`
- **Degradación paneles:** 0.5%/año lineal (NREL)
- **Impuesto a la renta:** 30% sobre EBIT positivo (Ley 2277/2022)

### 2.4 Indicadores

- **EBITDA / EBIT / Utilidad neta** (M COP/año, también por ha)
- **Payback** sobre flujo de caja después de impuestos
- **ROI neto** sobre CAPEX total
- **Empleos directos / totales / construcción** (SEIA, IRENA)


## 3. Dashboard Streamlit

Aplicación principal: [`streamlit_app.py`](streamlit_app.py)

Cuatro pestañas:

1. **Ranking** — Radar interactivo + barra apilada con composición ponderada del
   score multidimensional. Tabla con los aportes por dimensión.
2. **Variables crudas por dimensión** — Histogramas de las 13 variables
   pre-normalización agrupadas por dimensión. Cada gráfica trae leyenda con
   fuente, escala, score asociado y alertas de calidad de dato.
3. **Beneficio Agrivoltaico** — Comparación de tres estrategias por municipio
   (solo solar, solo ganadería, agrivoltaico) y calculadora paso-a-paso del
   ingreso. El ranking se ordena por **ingreso/ha × score multidimensional**
   para que municipios enormes pero de baja calidad (sin red eléctrica) no
   dominen solo por área.
4. **Viabilidad Financiera** — P&L completo con waterfall por hectárea/año
   (ingresos → costos → EBITDA → amortizaciones → EBIT → impuesto → utilidad
   neta). Curva de recuperación de inversión a 25 años. Selector de municipio
   que autocarga tarifa de agua y carga ganadera reales en los sliders.

Parámetros económicos unificados como constantes compartidas
(`_DEFAULT_PRECIO_ENERGIA`, `_DEFAULT_UGG_AGRO`, etc.) para que todas las
pestañas trabajen con los mismos supuestos.


## 4. Fuentes de datos

| Fuente | Variable | API/archivo |
|---|---|---|
| IGAC | Municipios, pendientes, POT | mapas.igac.gov.co (REST) |
| Solargis / Global Solar Atlas | PVOUT | `data/PVOUT.tif` (manual) |
| NASA POWER | Radiación solar | power.larc.nasa.gov |
| Copernicus ERA5 | Clima histórico | cds.climate.copernicus.eu |
| UPME | Subestaciones | geo.upme.gov.co (REST) |
| RUNAP | Áreas protegidas | parquesnacionales.gov.co |
| INVIAS | Red vial primaria | datos.gov.co |
| IDEAM | Viento, IMRC inundación/sequía | datos.gov.co + Excel manual |
| SUI | Tarifa agua, aseo | datos.gov.co |
| UPRA | Precio tierra, conflicto suelo | datos.gov.co |
| ICA / EVA | Inventario bovino | Agronet (Excel manual) |
| XM | Demanda histórica | `data/raw/2025-04/*.txt` (manual) |


## 5. Estructura del proyecto

    data/
      raw/                 # insumos manuales (PVOUT.tif, XM, IMRC, EVA bovino)
      clean/               # salidas procesadas por módulo
    src/
      extract/             # APIs y descargas (REST, datos.gov.co)
      spatial/             # IGAC, UPME, RUNAP, intersecciones
      scoring/             # dimensiones, viabilidad, K-means, rentabilidad
      transform/           # economía solar
      visualization/       # matrices, mapas 3D, histogramas
      db/                  # mysql_loader
    streamlit_app.py       # dashboard 4 pestañas


## 6. Ejecución reproducible

Comandos en Windows PowerShell. Reemplazar `python` por
`.\venv\Scripts\python.exe` si se usa entorno virtual.

### 6.1 Entorno

    python -m venv venv
    .\venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install pandas numpy requests geopandas pyogrio shapely pyproj rasterio scikit-learn matplotlib seaborn streamlit plotly pydeck python-docx openpyxl cdsapi xarray netCDF4 tqdm mysql-connector-python python-dotenv

### 6.2 Insumos manuales (no viajan por Git)

    data/PVOUT.tif
    data/raw/2025-04/*.txt
    data/raw/solargis_pvpotential_countryranking_2020_data.xlsx
    data/raw/IMRC-BASE-DE-DATOS-2024.xlsx
    data/raw/Variable-calculo-tarifa-a-corte-de-marzo-2026_0.xlsx
    data/raw/BasePecuaria20192023 (1).xlsx

### 6.3 `.env`

    MYSQL_HOST=127.0.0.1
    MYSQL_PORT=3306
    MYSQL_USER=...
    MYSQL_PASSWORD=...
    MYSQL_DATABASE=granja_solar
    CDS_API_KEY=...                # solo si se usa Copernicus

### 6.4 Pipeline

```bash
# Base espacial
python -m src.spatial.municipios_igac --export-geojson
python -m src.spatial.pvout_raster --output-dir data\clean\pvout_municipios --points-csv data\clean\base_municipios\municipios_distritos_colombia.csv
python -m src.spatial.igac_pendientes --points-csv data\clean\base_municipios\municipios_distritos_colombia.csv
python -m src.spatial.upme_subestaciones
python -m src.spatial.runap_protected_areas
python -m src.spatial.igac_usos_pot --points-csv data\clean\base_municipios\municipios_distritos_colombia.csv

# Clima, riesgo, vías, servicios, ganadería
python -m src.extract.nasa_power_municipios
python -m src.extract.copernicus_era5 --year 2025
python -m src.extract.imrc_dnp --force
python -m src.extract.invias_vias
python -m src.extract.sui_aseo --force
python -m src.extract.upra_agropecuario

# Demanda y escenarios
python -m src.extract.simem_api --include-f99e13
python -m src.scoring.xm_demanda_municipal

# Scoring + clusters
python -m src.scoring.viabilidad_municipal
python -m src.scoring.rentabilidad_municipal
python -m src.scoring.kmeans_municipios --auto-k --random-seed 42

# Visualizaciones
python -m src.visualization.municipal_visualizations

# (Opcional) Cargar a MySQL
python -m src.db.mysql_loader --apply

# Dashboard
streamlit run streamlit_app.py
```


## 7. Salidas principales

    data/clean/base_municipios/municipios_distritos_colombia.csv
    data/clean/viabilidad_municipal/viabilidad_municipal_multidimensional.csv
    data/clean/clusters_municipios/municipios_clusters_kmeans.csv
    data/clean/rentabilidad_municipal/rentabilidad_municipal.csv

Cada módulo deja un archivo `*_observaciones.txt` con notas metodológicas.


## 8. Limitaciones del modelo actual

- **PVOUT y pendiente** son muestreo puntual municipal, no promedio por
  polígono. Para municipios grandes (Cumaribo, Mitú) esto subestima
  heterogeneidad interna.
- **POT** se muestrea puntualmente; el descuento de área urbana usa promedio
  nacional 2%, no intersección poligonal por municipio.
- **Distancia a red** usa subestaciones, no líneas de transmisión ni capacidad
  real de conexión.
- **Demanda XM** es proxy regional desde subáreas, no demanda municipal directa
  — por eso no entra al `V_i` multidimensional.
- **Hectáreas viables** asume descuentos uniformes (urbano 2%, vías 3%, etc.).
  Municipios atípicos como Vichada (sabana natural) tienen descuentos reales
  menores; municipios cafeteros tienen descuentos mayores.
- **Modelo financiero** no incluye costos financieros (interés sobre deuda),
  riesgo cambiario (paneles importados USD), licenciamiento ANLA ni consulta
  previa.
- **Ranking agrivoltaico** está ponderado por score multidimensional, pero no
  modela transmisión específica del proyecto ni precio PPA negociado.

El resultado debe presentarse como **modelo preliminar de priorización**, no
como selección definitiva de predios para construir una planta.


## 9. Bug fixes documentados

Durante el desarrollo se identificaron y corrigieron tres bugs críticos:

1. **Ingreso solar mal calculado por factor 1000** — `precio_energia` está en
   COP/kWh pero `mwh` está en MWh. Faltaba `× 1000` para convertir. Antes del
   fix ningún municipio salía viable.
2. **NaN propagation en P&L** — `float(nan or default)` no funcionaba (nan es
   truthy en Python). Reemplazado con helper `_safe()` que verifica
   `pd.notna()` antes del fallback.
3. **Mediana ha_viable = 0** — incluía 633 municipios con `score_pendiente = 0`
   (excluidos por terreno empinado). Se filtra a `ha_viable > 0` antes de
   calcular el resumen nacional.


## 10. Reproducibilidad

- K-Means con semilla fija 42.
- Pendientes y POT soportan `resume`/checkpoint.
- Cada módulo exporta observaciones metodológicas en `*_observaciones.txt`.
- Constantes técnicas y económicas centralizadas en
  [`src/scoring/solar_constants.py`](src/scoring/solar_constants.py) y
  defaults compartidos en [`streamlit_app.py`](streamlit_app.py).


---

**Curso:** Introducción a la Inteligencia Artificial — ITM
**Autor:** Jhon T.
**Datos:** NASA Power · IGAC · UPME · RUNAP · UPRA · EVA-ICA · IMRC DNP · SUI · INVIAS · Solargis
**Modelo financiero:** NREL · IRENA · SEIA · World Bank/ESMAP · FEDEGAN · AGROSAVIA
