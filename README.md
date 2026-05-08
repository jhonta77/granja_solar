# PROYECTO: VIABILIDAD MUNICIPAL PARA GRANJA SOLAR EN COLOMBIA

1. Objetivo del proyecto
------------------------

Este proyecto construye un pipeline reproducible en Python para evaluar, de
forma preliminar, que municipios de Colombia tienen mejores condiciones para
instalar una granja solar utility-scale.

La evaluacion integra variables energeticas, espaciales y territoriales:

- Recurso solar: PVOUT / radiacion solar.
- Demanda energetica: proxy desde datos historicos/proyecciones XM, usado como
  contexto favorable, no como criterio determinante de ubicacion.
- Cercania a infraestructura electrica: distancia a subestaciones.
- Pendiente del terreno: clasificacion IGAC.
- Restricciones territoriales: areas protegidas RUNAP, POT rural/urbano y proxy
  de uso del suelo.
- Produccion y costo por hectarea: escenarios tecnicos-economicos.
- Agrupamiento: K-Means para clasificar municipios con perfiles similares.

El resultado principal es una tabla municipal con un score preliminar de
viabilidad:

    V_i = R_i(0.35*S_i + 0.30*G_i + 0.25*P_i + 0.10*U_i)

Donde:

- V_i: score final de viabilidad del municipio i.
- R_i: restriccion dura preliminar.
- S_i: score solar.
- D_i: score de demanda, reportado como bono/contexto favorable separado.
- G_i: score de cercania a red electrica.
- P_i: score de pendiente.
- U_i: score de uso/restriccion territorial.

Los pesos suman 1.0 y estan definidos en:

    src/scoring/viabilidad_municipal.py


1.1. Estado del flujo EDA
-------------------------

El flujo EDA anterior fue retirado del repositorio para reescribirse desde
cero. Por ahora no hay scripts ni salidas EDA vigentes dentro del proyecto.


2. Idea metodologica
--------------------

La unidad final de analisis es el municipio/distrito colombiano.

Cada fuente se procesa primero de manera independiente. Despues se integra en
una tabla municipal unica para calcular scores normalizados en escala 0-1.

El pipeline evita mezclar unidades directamente. Por eso variables como PVOUT,
distancia, pendiente y restricciones se normalizan antes de sumarse. La demanda
se normaliza aparte porque no determina la ubicacion fisica de la granja.

Variables donde "mas es mejor":

    X_norm = (X - X_min) / (X_max - X_min)

Variables donde "menos es mejor":

    X_norm = (X_max - X) / (X_max - X_min)

Restricciones duras actuales:

- Municipio con pendiente no viable o desconocida: R_i = 0.
- Municipio a mas de 50 km de subestacion: R_i = 0.
- Municipio con 80% o mas del area cubierta por RUNAP: R_i = 0.
- Municipio/punto POT en capa urbana: R_i = 0.
- Municipio/punto POT con restriccion territorial estricta: R_i = 0.

Nota: el score es preliminar. No reemplaza estudios de ingenieria, prediales,
ambientales, topograficos ni de interconexion.


3. Estructura del proyecto
--------------------------

Estructura principal:

    data/
      raw/
        2025-04/
        nasa_power/
        igac_municipios/
        igac_pendientes/
        runap/
        simem/
        upme_subestaciones/
      clean/
        base_municipios/
        pvout/
        pvout_municipios/
        pendientes_municipios/
        subestaciones_upme/
        runap_protegidas/
        usos_suelo_pot/
        simem/
        xm_top10/
        xm_demanda_municipal/
        viabilidad_municipal/
        clusters_municipios/
        visualizaciones_municipios/
        solar_costs/
    notebooks/
    src/
      extract/
      spatial/
      scoring/
      transform/
      visualization/
    README.txt


4. Fuentes de datos usadas
--------------------------

4.1 Datos cargados manualmente

Estos archivos fueron puestos manualmente en el proyecto:

- data/raw/2025-04/*.txt
  - Archivos XM historicos/proyecciones: PRON_AREAS, PRON_BARRA, PRON_SIN,
    PRONSIN, PRON_UCP y PRONUCP.
  - Uso: construccion de proxy de demanda y resumenes XM.

- data/PVOUT.tif
  - Raster PVOUT de Global Solar Atlas / World Bank / ESMAP / Solargis.
  - Uso: extraccion puntual del potencial solar por municipio.
  - Unidad interpretada: kWh/kWp/dia.

- data/raw/solargis_pvpotential_countryranking_2020_data.xlsx
  - Excel de Global Photovoltaic Power Potential by Country.
  - Uso: referencia tecnica de potencial solar por pais.

4.2 APIs y servicios publicos consultados

IGAC - limites municipales:

    https://mapas2.igac.gov.co/server/rest/services/limites/limites/MapServer/1

Uso:

- Descargar municipios y distritos de Colombia.
- Construir la base municipal estricta.
- Generar lon/lat municipal para muestreos puntuales.

Salida principal:

    data/clean/base_municipios/municipios_distritos_colombia.csv
    data/clean/base_municipios/municipios_colombia.geojson

IGAC - pendientes de Colombia:

    https://mapas.igac.gov.co/server/rest/services/ordenamientoterritorial/pendientescolombia/MapServer

Uso:

- Consultar la clase de pendiente en el punto municipal.
- Clasificar pendiente como viable, condicional o no viable.

Criterio adoptado:

- Viable: pendiente <= 7%.
- Condicional: > 7% y <= 14%.
- No viable: > 14%.

Salida principal:

    data/clean/pendientes_municipios/pendiente_puntos_extraidos.csv

UPME - subestaciones electricas:

    https://geo.upme.gov.co/server/rest/services/SUBESTACIONES/UPME_EN_DI_SUBESTACION_consulta/MapServer

Uso:

- Descargar subestaciones con coordenadas.
- Calcular distancia minima desde cada municipio a subestaciones.
- Para G_i se priorizan subestaciones en servicio y niveles de tension 4/5.

Salida principal:

    data/clean/subestaciones_upme/subestaciones_upme.csv
    data/clean/subestaciones_upme/distancia_subestacion_municipios.csv

RUNAP - areas protegidas:

    https://mapas.parquesnacionales.gov.co/arcgis/rest/services/pnn/runap/MapServer
    https://storage.googleapis.com/pnn_geodatabase/runap/latest.zip

Uso:

- Calcular proporcion municipal cubierta por areas protegidas.
- Construir U_i como proporcion no protegida.
- Aplicar restriccion dura si RUNAP cubre 80% o mas del municipio.

Salida principal:

    data/clean/runap_protegidas/runap_restricciones_municipios.csv

IGAC - zonificacion de usos segun POT:

    https://mapas.igac.gov.co/server/rest/services/ordenamientoterritorial/zonificacionusossegunpot/MapServer

Uso:

- Clasificar leyendas de uso del suelo.
- Apoyar el criterio cualitativo de doble uso con pastoreo.
- Excluir puntos municipales que caen en capa urbana POT.

Limitacion:

- En esta version el uso POT municipal es muestreo puntual. Para descontar zona
  urbana dentro de cada municipio se necesita interseccion poligonal por area.

Salida principal:

    data/clean/usos_suelo_pot/usos_pot_clasificacion_leyenda.csv
    data/clean/usos_suelo_pot/usos_pot_puntos_extraidos.csv

SIMEM/XM - API publica:

    https://www.simem.co/backend-datos/
    https://www.simem.co/backend-files/

Datasets usados:

- A0CF2A: Listado de Embalses que sirven al SIN.
- BA1C55: Aportes Hidricos en Energia.
- B0E933: Reservas Hidraulicas en Energia.
- F99E13: Listado de Subestaciones del STN y STR.

Nota metodologica:

- F99E13 lista subestaciones, pero para distancias se usa UPME porque UPME
  entrega geometria/coordenadas.
- Los datos hidrologicos SIMEM quedan disponibles para analisis futuro, pero no
  definen por si solos la viabilidad solar.

Salida principal:

    data/clean/simem/resumen_datasets_descargados.csv
    data/clean/simem/catalogo_hidroelectricas_simem.csv

NASA POWER:

    https://power.larc.nasa.gov/api/temporal/daily/point
    https://power.larc.nasa.gov/docs/services/api/temporal/daily/

Parametros usados:

- ALLSKY_SFC_SW_DWN
- CLRSKY_SFC_SW_DWN

Uso:

- Consultar radiacion solar diaria por puntos.
- Contrastar o complementar PVOUT.

Salida principal:

    data/clean/nasa_power/nasa_power_puntos_diario.csv
    data/clean/nasa_power/nasa_power_resumen_puntos.csv


5. Scripts principales
----------------------

El flujo EDA anterior fue retirado del repositorio y se reescribira desde cero.

Base municipal:

    src/spatial/municipios_igac.py

Funciones:

- Descarga municipios desde IGAC.
- Exporta CSV estricto de municipios/distritos y GeoJSON.

PVOUT:

    src/spatial/pvout_raster.py

Funciones:

- Lee data/PVOUT.tif.
- Calcula resumen por bbox Colombia.
- Extrae PVOUT puntual para municipios si se pasa un CSV con lon/lat.

Pendiente:

    src/spatial/igac_pendientes.py

Funciones:

- Consulta el servicio IGAC de pendientes.
- Permite resume/checkpoint para continuar si se pausa.
- Exporta clasificacion de pendiente por municipio.

Subestaciones:

    src/spatial/upme_subestaciones.py

Funciones:

- Descarga subestaciones UPME.
- Calcula distancia minima a red por municipio.

Areas protegidas:

    src/spatial/runap_protected_areas.py

Funciones:

- Descarga RUNAP.
- Intersecta areas protegidas con municipios.
- Exporta restricciones municipales.

Usos del suelo POT:

    src/spatial/igac_usos_pot.py

Funciones:

- Descarga metadatos y leyenda POT.
- Clasifica etiquetas de uso del suelo.
- Permite muestreo puntual si se necesita.

SIMEM:

    src/extract/simem_api.py

Funciones:

- Descarga catalogo SIMEM.
- Descarga datasets especificos.
- Etiqueta datasets hidrologicos y subestaciones.

NASA POWER:

    src/extract/nasa_power.py

Funciones:

- Consulta radiacion diaria por puntos.
- Resume resultados por punto.

Costos y riesgos municipales:

    src/extract/costos_riesgos_municipales.py

Funciones:

- Descarga precios rurales UPRA, tarifas de acueducto SUI y viento IDEAM desde Datos Abiertos.
- Cruza registros contra la base municipal por codigo DANE cuando existe o por municipio/departamento.
- Exporta precios de tierra, tarifa de agua, velocidad de viento y scores normalizados.
- Documenta cobertura, flags de disponibilidad e imputacion posterior usada por el score.

Economia solar:

    src/transform/solar_economics.py

Funciones:

- Calcula escenarios conservador, base y optimista por hectarea.
- Usa referencias IRENA, NREL, World Bank/ESMAP/Global Solar Atlas y UPME.
- En Streamlit estima energia anual por municipio y valor anual de compra si
  existe precio real COP/kWh asociado al municipio, departamento o codigo DANE.
- Para Caribe se carga el promedio oficial Minenergia 2026: 796 COP/kWh Air-e
  y 879 COP/kWh AFINIA; es precio regional por operador, no tarifa municipal
  individual.

Demanda XM:

    src/scoring/xm_top10.py
    src/scoring/xm_demanda_municipal.py

Funciones:

- Resume fuentes XM.
- Construye un proxy municipal de demanda desde PRON_AREAS.
- Etiqueta casos inciertos o atipicos para revision.

Score municipal:

    src/scoring/viabilidad_municipal.py

Funciones:

- Integra PVOUT, pendiente, red, RUNAP, POT y demanda.
- Calcula componentes normalizados.
- Calcula V_i rural sin demanda y clasificacion preliminar.
- Calcula v_i_modelo_rural_economico_climatico con tierra, agua y viento.
- Exporta score_rural_con_bono_demanda solo como sensibilidad favorable.

K-Means:

    src/scoring/kmeans_municipios.py

Funciones:

- Agrupa municipios con perfiles similares.
- Identifica K con el metodo del codo usando inercia.
- Calcula coeficiente silhouette por K y por municipio.
- Usa semilla fija 42 por defecto.
- Exporta perfiles de cluster, centroides, evaluacion de K, graficos PNG y
  municipios excluidos.

Visualizaciones:

    src/visualization/municipal_visualizations.py

Funciones:

- Genera matriz de correlacion.
- Genera cruce cluster vs viabilidad.
- Genera histogramas.
- Genera mapas 3D en PNG.

Dashboard Streamlit:

    streamlit_app.py

Funciones:

- Muestra los 10 municipios con mayor score de viabilidad.
- Permite ocultar municipios con demanda marcada como atipica sin cambiar el
  score rural.
- Explica el ranking con aportes ponderados: solar, red, pendiente y
  uso/restriccion territorial.
- Muestra mapa, histogramas, clusters K-Means, graficos de codo/silhouette y
  escenarios economicos por hectarea.


6. Ejecucion despues de clonar el repositorio
------------------------------------------------

Estos pasos son los recomendados para una persona que clona el repositorio desde
cero y necesita reconstruir los datos procesados y crear la base MySQL.

6.1 Clonar y entrar al proyecto

    git clone <url-del-repositorio>
    cd <carpeta-del-repositorio>

6.2 Crear entorno virtual e instalar dependencias

En Windows PowerShell:

    python -m venv venv
    .\venv\Scripts\activate
    python -m pip install --upgrade pip

Instalar las dependencias usadas por el proyecto:

    pip install pandas numpy requests geopandas pyogrio shapely pyproj rasterio scikit-learn matplotlib seaborn streamlit python-docx openpyxl

Nota: si se trabaja con datos geoespaciales en Windows, `geopandas`, `rasterio`
y `pyogrio` pueden requerir wheels compatibles con la version de Python
instalada.

6.3 Restaurar insumos manuales no versionados

Por tamano y por buenas practicas, `data/raw/`, `data/clean/`, `data/interim/`
y archivos pesados como `data/PVOUT.tif` estan ignorados por Git. Despues de
clonar, se deben copiar manualmente estos insumos antes de ejecutar todo el
pipeline:

    data/PVOUT.tif
    data/raw/2025-04/*.txt
    data/raw/solargis_pvpotential_countryranking_2020_data.xlsx

Los demas datos publicos se regeneran desde los scripts del proyecto usando
servicios de IGAC, UPME, RUNAP, SIMEM y NASA POWER.

6.4 Descargar/regenerar datos limpios

Ejecutar en este orden:

    .\venv\Scripts\python.exe src\spatial\municipios_igac.py --export-geojson
    .\venv\Scripts\python.exe src\spatial\pvout_raster.py --output-dir data\clean\pvout_municipios --points-csv data\clean\base_municipios\municipios_distritos_colombia.csv
    .\venv\Scripts\python.exe src\spatial\igac_pendientes.py --output-dir data\clean\pendientes_municipios --points-csv data\clean\base_municipios\municipios_distritos_colombia.csv
    .\venv\Scripts\python.exe src\spatial\upme_subestaciones.py
    .\venv\Scripts\python.exe src\spatial\runap_protected_areas.py
    .\venv\Scripts\python.exe src\spatial\igac_usos_pot.py --points-csv data\clean\base_municipios\municipios_distritos_colombia.csv
    .\venv\Scripts\python.exe src\extract\simem_api.py --include-f99e13
    .\venv\Scripts\python.exe src\extract\nasa_power.py
    .\venv\Scripts\python.exe -m src.extract.costos_riesgos_municipales
    .\venv\Scripts\python.exe src\transform\solar_economics.py --no-figure
    .\venv\Scripts\python.exe src\scoring\xm_top10.py
    .\venv\Scripts\python.exe src\scoring\xm_demanda_municipal.py
    .\venv\Scripts\python.exe src\scoring\viabilidad_municipal.py
    .\venv\Scripts\python.exe src\scoring\kmeans_municipios.py --auto-k --k-selection-metric elbow --random-seed 42
    .\venv\Scripts\python.exe src\visualization\municipal_visualizations.py

Al finalizar, las salidas principales quedan en `data/clean/`.

6.5 Configurar MySQL

Instalar MySQL Server y verificar que el cliente `mysql.exe` este disponible en
el PATH o en:

    C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe

Crear el archivo `.env` a partir del ejemplo:

    copy .env.mysql.example .env

Editar `.env` con las credenciales locales:

    MYSQL_HOST=127.0.0.1
    MYSQL_PORT=3306
    MYSQL_USER=tu_usuario
    MYSQL_PASSWORD=tu_password
    MYSQL_DATABASE=granja_solar

El usuario MySQL debe tener permisos para crear base de datos, crear tablas,
crear vistas e insertar registros.

6.6 Crear y poblar la base de datos

Despues de generar los CSV en `data/clean/`, ejecutar:

    .\venv\Scripts\python.exe -m src.db.mysql_loader --apply

Este comando:

- Genera el SQL en `data/interim/mysql/bootstrap_granja_solar.sql`.
- Crea la base definida en `MYSQL_DATABASE`.
- Elimina y reconstruye las tablas del modelo.
- Inserta los datos limpios.
- Crea las vistas `vw_dashboard_municipal`, `vw_municipal_contexto` y
  `vw_subestaciones_municipios`.

Si solo se quiere generar el SQL sin aplicarlo:

    .\venv\Scripts\python.exe -m src.db.mysql_loader

6.7 Ejecutar dashboard

Con la base creada, iniciar Streamlit:

    .\venv\Scripts\python.exe -m streamlit run streamlit_app.py

Si MySQL no esta disponible, el dashboard intenta usar algunos CSV limpios como
respaldo cuando existen en `data/clean/`.


7. Como ejecutar el pipeline
----------------------------

Activar entorno virtual:

    venv\Scripts\activate

O ejecutar directamente con:

    venv\Scripts\python.exe <script>

Orden recomendado:

1. El flujo EDA anterior fue retirado del repositorio y ya no forma parte del
   orden de ejecucion actual.

2. Base municipal IGAC:

    venv\Scripts\python.exe src\spatial\municipios_igac.py --export-geojson

Salida:

    data/clean/base_municipios/municipios_distritos_colombia.csv

3. PVOUT municipal desde raster manual:

    venv\Scripts\python.exe src\spatial\pvout_raster.py --output-dir data\clean\pvout_municipios --points-csv data\clean\base_municipios\municipios_distritos_colombia.csv

Salida:

    data/clean/pvout_municipios/pvout_puntos_extraidos.csv

4. Pendiente municipal IGAC:

    venv\Scripts\python.exe src\spatial\igac_pendientes.py --output-dir data\clean\pendientes_municipios --points-csv data\clean\base_municipios\municipios_distritos_colombia.csv

Salida:

    data/clean/pendientes_municipios/pendiente_puntos_extraidos.csv

5. Subestaciones UPME y distancia a red:

    venv\Scripts\python.exe src\spatial\upme_subestaciones.py

Salida:

    data/clean/subestaciones_upme/distancia_subestacion_municipios.csv

6. Areas protegidas RUNAP:

    venv\Scripts\python.exe src\spatial\runap_protected_areas.py

Salida:

    data/clean/runap_protegidas/runap_restricciones_municipios.csv

7. Usos del suelo POT:

    venv\Scripts\python.exe src\spatial\igac_usos_pot.py --points-csv data\clean\base_municipios\municipios_distritos_colombia.csv

Salida:

    data/clean/usos_suelo_pot/usos_pot_clasificacion_leyenda.csv
    data/clean/usos_suelo_pot/usos_pot_puntos_extraidos.csv

8. SIMEM:

    venv\Scripts\python.exe src\extract\simem_api.py --include-f99e13

Salida:

    data/clean/simem/resumen_datasets_descargados.csv

9. NASA POWER:

    venv\Scripts\python.exe src\extract\nasa_power.py

Salida:

    data/clean/nasa_power/nasa_power_resumen_puntos.csv

10. Costos y riesgos municipales:

    venv\Scripts\python.exe -m src.extract.costos_riesgos_municipales

Salida:

    data/clean/costos_riesgos_municipales/tierra_municipal.csv
    data/clean/costos_riesgos_municipales/agua_municipal.csv
    data/clean/costos_riesgos_municipales/viento_municipal.csv
    data/clean/costos_riesgos_municipales/costos_riesgos_municipales.csv

11. Economia solar por hectarea:

    venv\Scripts\python.exe src\transform\solar_economics.py --no-figure

Salida:

    data/clean/solar_costs/solar_escenarios_por_hectarea.csv
    data/clean/energy_prices/precios_compra_energia_minenergia_caribe.csv

12. Resumen de demanda XM:

    venv\Scripts\python.exe src\scoring\xm_top10.py

Salida:

    data/clean/xm_top10/xm_resumen_zonas.csv

13. Proxy municipal de demanda:

    venv\Scripts\python.exe src\scoring\xm_demanda_municipal.py

Salida:

    data/clean/xm_demanda_municipal/xm_demanda_municipal_proxy.csv

14. Score de viabilidad municipal:

    venv\Scripts\python.exe src\scoring\viabilidad_municipal.py

Salida:

    data/clean/viabilidad_municipal/viabilidad_municipal_preliminar.csv

15. Agrupamiento K-Means:

    venv\Scripts\python.exe src\scoring\kmeans_municipios.py --auto-k --k-selection-metric elbow --random-seed 42

Salida:

    data/clean/clusters_municipios/municipios_clusters_kmeans.csv
    data/clean/clusters_municipios/perfil_clusters_kmeans.csv
    data/clean/clusters_municipios/evaluacion_kmeans_k.csv
    data/clean/clusters_municipios/resumen_seleccion_kmeans.csv
    data/clean/clusters_municipios/grafico_metodo_codo_kmeans.png
    data/clean/clusters_municipios/grafico_silhouette_kmeans.png

16. Visualizaciones:

    venv\Scripts\python.exe src\visualization\municipal_visualizations.py

Salida:

    data/clean/visualizaciones_municipios/

17. Dashboard Streamlit:

    venv\Scripts\streamlit.exe run streamlit_app.py

Si el comando anterior no abre, usar:

    venv\Scripts\python.exe -m streamlit run streamlit_app.py


8. Salidas principales actuales
-------------------------------

Base municipal:

    data/clean/base_municipios/municipios_distritos_colombia.csv

Contiene 1104 municipios/distritos usados como unidad estricta del score.

PVOUT municipal:

    data/clean/pvout_municipios/pvout_puntos_extraidos.csv

Contiene PVOUT puntual y rendimiento anual estimado por municipio.

Pendiente:

    data/clean/pendientes_municipios/pendiente_puntos_extraidos.csv

Contiene clase de pendiente y score fisico preliminar.

Infraestructura electrica:

    data/clean/subestaciones_upme/distancia_subestacion_municipios.csv

Contiene distancia minima a subestacion por municipio.

Restricciones RUNAP:

    data/clean/runap_protegidas/runap_restricciones_municipios.csv

Contiene porcentaje protegido y restriccion territorial.

Usos POT:

    data/clean/usos_suelo_pot/usos_pot_puntos_extraidos.csv

Contiene capa rural/urbana muestreada y flags de exclusion urbana.

Demanda:

    data/clean/xm_demanda_municipal/xm_demanda_municipal_proxy.csv

Contiene proxy de demanda municipal y flags de incertidumbre.

Score:

    data/clean/viabilidad_municipal/viabilidad_municipal_preliminar.csv

Contiene la tabla integrada final del modelo preliminar.

K-Means:

    data/clean/clusters_municipios/municipios_clusters_kmeans.csv
    data/clean/clusters_municipios/perfil_clusters_kmeans.csv
    data/clean/clusters_municipios/centroides_clusters_kmeans.csv
    data/clean/clusters_municipios/evaluacion_kmeans_k.csv
    data/clean/clusters_municipios/resumen_seleccion_kmeans.csv
    data/clean/clusters_municipios/grafico_metodo_codo_kmeans.png
    data/clean/clusters_municipios/grafico_silhouette_kmeans.png

Visualizaciones:

    data/clean/visualizaciones_municipios/matrices/
    data/clean/visualizaciones_municipios/histogramas/
    data/clean/visualizaciones_municipios/mapas_3d/


9. Estado actual del modelo
---------------------------

Ya esta disponible:

- Base municipal oficial IGAC.
- PVOUT municipal puntual desde raster local.
- Pendiente municipal puntual desde IGAC.
- Distancia a subestaciones UPME.
- Restricciones RUNAP por interseccion espacial.
- Muestreo POT rural/urbano como restriccion preliminar.
- Proxy de demanda XM por municipio.
- Escenarios economicos por hectarea.
- Score municipal preliminar.
- K-Means con semilla fija.
- Matriz de correlacion, histogramas y mapas 3D.

Principales limitaciones:

- PVOUT y pendiente se usan como muestreo puntual municipal, no como promedio
  por poligono.
- Demanda no es municipal directa; es un proxy desde subareas XM y no entra al
  V_i rural.
- Uso del suelo POT se usa como muestreo puntual; aun no descuenta area urbana
  por interseccion poligonal dentro de cada municipio.
- Distancia a red usa subestaciones, no lineas ni capacidad real de conexion.
- No hay aun modelacion predial, licenciamiento, servidumbres ni costos reales
  de interconexion.


10. Interpretacion de resultados
-------------------------------

El ranking de municipios sale del score V_i rural, sin demanda.

Los clusters K-Means no reemplazan el ranking. Sirven para agrupar municipios
con perfiles parecidos y explicar patrones:

- Municipios con buen PVOUT y baja restriccion.
- Municipios con buena red y POT rural/compatible.
- Municipios con alta restriccion RUNAP.
- Municipios excluidos por pendiente, distancia, area protegida o zona urbana POT.

La matriz de correlacion muestra relacion lineal entre variables. No prueba
causalidad.

La matriz cluster vs viabilidad es un cruce descriptivo, no una matriz de
confusion supervisada, porque no existe una etiqueta real de "viable/no viable"
observada contra la cual validar.


11. Reproducibilidad
--------------------

El proyecto es reproducible porque:

- Los scripts tienen rutas por defecto.
- Las salidas se guardan en data/clean.
- Pendientes y POT soportan resume para continuar si se pausa una consulta.
- K-Means usa semilla fija 42 por defecto.
- Cada fuente exporta observaciones metodologicas.

Para regenerar todo desde cero se recomienda conservar primero los insumos
manuales:

- data/PVOUT.tif
- data/raw/2025-04/*.txt
- data/raw/solargis_pvpotential_countryranking_2020_data.xlsx


12. Archivos que conviene citar en el informe
---------------------------------------------

- data/clean/base_municipios/municipios_observaciones.txt
- data/clean/pvout_municipios/pvout_observaciones.txt
- data/clean/pendientes_municipios/pendientes_observaciones.txt
- data/clean/subestaciones_upme/subestaciones_observaciones.txt
- data/clean/runap_protegidas/runap_observaciones.txt
- data/clean/xm_demanda_municipal/xm_demanda_observaciones.txt
- data/clean/viabilidad_municipal/observaciones_viabilidad_municipal.txt
- data/clean/solar_costs/solar_observaciones.txt
- data/clean/clusters_municipios/observaciones_kmeans.txt
- data/clean/clusters_municipios/evaluacion_kmeans_k.csv
- data/clean/clusters_municipios/resumen_seleccion_kmeans.csv
- data/clean/visualizaciones_municipios/observaciones_visualizaciones.txt


13. Recomendacion para la siguiente fase
----------------------------------------

La siguiente mejora metodologica debe enfocarse en reemplazar proxies puntuales
por variables de area:

- PVOUT promedio o percentiles por poligono municipal.
- Proporcion real de area con pendiente <= 7%.
- Proporcion de area rural compatible con uso agropecuario o pastoreo,
  excluyendo casco urbano y zonas pobladas mediante interseccion poligonal.
- Distancia a lineas de transmision, no solo a subestaciones.
- Demanda municipal o regional con una fuente mas directa, conservada como
  factor comercial favorable y no como determinante fisico de ubicacion.

Hasta que eso se haga, el resultado debe presentarse como modelo preliminar de
priorizacion, no como seleccion definitiva de predios para construir una planta
solar.
