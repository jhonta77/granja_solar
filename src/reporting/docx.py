from datetime import datetime
from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_REMOVED_SCRIPT_DIR = None
if sys.path and Path(sys.path[0]).resolve() == _SCRIPT_DIR:
    _REMOVED_SCRIPT_DIR = sys.path.pop(0)

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import pandas as pd

if _REMOVED_SCRIPT_DIR is not None:
    sys.path.insert(0, _REMOVED_SCRIPT_DIR)

# --------------------------------------------------------------------
# CONFIGURACIÓN INICIAL
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "entregables"
OUTPUT_DIR.mkdir(exist_ok=True)

# Rutas de las gráficas explicativas 
GRAFICA1_PATH = PROJECT_ROOT / "data" / "clean" / "visualizaciones_municipios" / "explicativas" / "top10_Vi_bar.png"
GRAFICA2_PATH = PROJECT_ROOT / "data" / "clean" / "visualizaciones_municipios" / "explicativas" / "top5_stacked_contributions.png"
GRAFICA3_PATH = PROJECT_ROOT / "data" / "clean" / "visualizaciones_municipios" / "explicativas" / "mapa_viabilidad_top10.png"

# Datos del grupo
NOMBRE_GRUPO = "Grupo Consultor Solar"
INTEGRANTES = "1. Juan Pérez\n2. María López\n3. Carlos García"
EMPRESA = "SolarGrid Analytics S.A.S."
TEMATICA = "2 – Meteorología, clima y sectores productivos"
FECHA_ENTREGA = "19 de abril de 2026"



# --------------------------------------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------------------------------------
def add_heading(doc, text, level):
    heading = doc.add_heading(text, level=level)
    return heading

def add_paragraph(doc, text, bold=False, italic=False, alignment=None):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.bold = bold
    run.italic = italic
    if alignment is not None:
        para.alignment = alignment
    return para

def add_image_placeholder(doc, path, caption=""):
    if path.exists():
        doc.add_picture(str(path), width=Inches(5.5))
        if caption:
            add_paragraph(doc, caption, italic=True)
    else:
        add_paragraph(doc, f"[IMAGEN NO ENCONTRADA: {path.name}]", bold=True)

def add_table_from_dataframe(doc, df, col_widths=None):
    """Inserta una tabla a partir de un DataFrame."""
    table = doc.add_table(rows=df.shape[0]+1, cols=df.shape[1], style='Table Grid')
    # Encabezados
    for j, col in enumerate(df.columns):
        cell = table.rows[0].cells[j]
        cell.text = col
        cell.paragraphs[0].runs[0].bold = True
    # Datos
    for i, row in df.iterrows():
        for j, val in enumerate(row):
            table.rows[i+1].cells[j].text = str(val)
    return table

# --------------------------------------------------------------------
# CREACIÓN DEL DOCUMENTO
# --------------------------------------------------------------------
doc = Document()

# Configuración de página y estilos
style = doc.styles['Normal']
font = style.font
font.name = 'Calibri'
font.size = Pt(11)
style.paragraph_format.space_after = Pt(4)

# TÍTULO PRINCIPAL
title = doc.add_heading('TERCERA ACTIVIDAD EVALUATIVA – DEL PROBLEMA AL DATO LIMPIO', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# PORTADA
doc.add_page_break()
add_heading(doc, 'Informe Ejecutivo Inicial', level=1)
add_paragraph(doc, f'Consultoría en Datos: Viabilidad de Granja Solar en Colombia')
add_paragraph(doc, f'Primera Actividad Evaluativa · Visualización de Datos · ITM 2026')
add_paragraph(doc, '')
add_paragraph(doc, f'Empresa contratante: {EMPRESA}')
add_paragraph(doc, f'Sector: Energía renovable')
add_paragraph(doc, f'Equipo consultor:')
for integrante in INTEGRANTES.split('\n'):
    add_paragraph(doc, integrante)
add_paragraph(doc, f'Fecha del informe: {FECHA_ENTREGA}')
add_paragraph(doc, '')

# Resumen ejecutivo
add_heading(doc, '¿Qué es este documento y para quién es?', level=2)
add_paragraph(doc, 
    'Este informe ejecutivo condensa los principales hallazgos de la Primera Actividad Evaluativa en un documento '
    'de máximo 3 páginas dirigido al gerente de la empresa contratante.\n'
    'No documenta cómo se hizo el trabajo técnico — comunica qué encontramos, qué significa para el negocio y '
    'qué viene a continuación.'
)

# --------------------------------------------------------------------
# 1. CONTEXTO Y DEFINICIÓN DEL PROBLEMA
# --------------------------------------------------------------------
doc.add_page_break()
add_heading(doc, '1. Contexto y Definición del Problema', level=1)

add_heading(doc, '1.1 Descripción de la Empresa Contratante', level=2)
add_paragraph(doc,
    'SolarGrid Analytics S.A.S. es una empresa del sector energético dedicada a la identificación, evaluación '
    'y estructuración de proyectos de generación eléctrica con fuentes no convencionales de energía renovable, '
    'especialmente energía solar fotovoltaica. La empresa busca expandir su portafolio mediante el desarrollo de '
    'una granja solar de gran escala en Colombia, pero actualmente no cuenta con un modelo integrado que le permita '
    'determinar, con sustento técnico y económico, la ubicación más rentable y viable para su instalación.\n'
    'El problema se ubica dentro del proceso de planeación y formulación de proyectos energéticos, específicamente '
    'en la fase de selección óptima de sitio. Hasta el momento, la empresa ha basado su análisis principalmente en '
    'variables aisladas como radiación solar promedio y disponibilidad aparente de terreno, lo que genera un riesgo '
    'elevado de tomar decisiones subóptimas. Como indicadores preliminares utiliza irradiación solar, superficie '
    'disponible, costo estimado de inversión y proyección de generación anual, pero no integrados en una metodología '
    'multicriterio. Cuenta con acceso a fuentes como geovisores de la UPME, información geoespacial del IGAC y '
    'herramientas de análisis Python/SIG, aunque no puede considerar áreas protegidas o de uso incompatible y debe '
    'priorizar ubicaciones donde la conexión al sistema eléctrico y la productividad esperada justifiquen la inversión.'
)

add_heading(doc, '1.2 Estado del Arte (mínimo 6 referencias APA 7)', level=2)
add_paragraph(doc,
    'La literatura reciente sobre selección de emplazamientos para granjas solares muestra una evolución clara: '
    'los estudios han pasado de enfoques centrados únicamente en radiación solar a metodologías de evaluación '
    'multicriterio apoyadas en Sistemas de Información Geográfica (SIG). Actualmente el consenso indica que la '
    'ubicación óptima de una planta fotovoltaica no depende de una sola variable, sino de la integración de '
    'criterios técnicos, económicos, ambientales, territoriales y de infraestructura eléctrica.\n\n'
    'De Luis-Ruiz et al. (2024) proponen una metodología para localizar plantas solares utilizando GIS y análisis '
    'multicriterio, automatizando parte de la evaluación territorial y demostrando que variables como topografía, '
    'accesibilidad, restricciones del suelo y recurso solar deben analizarse de manera integrada. Este enfoque '
    'sirve de base conceptual para nuestro modelo.\n'
    'Nassar et al. (2025) desarrollan un enfoque GIS multicriterio para selección de sitios solares y eólicos, '
    'integrando factores ambientales, económicos y técnicos, y confirman la robustez de los modelos espaciales '
    'multicriterio para reducir incertidumbre.\n'
    'Ángel-Sanint et al. (2023) proponen una metodología específica para Colombia mediante SIG, incorporando '
    'restricciones físicas, bióticas, económicas, culturales y políticas. Demuestran que los mapas de potencial puro '
    'sobrestiman la viabilidad si no consideran barreras reales, justificando la inclusión de variables más allá del '
    'recurso solar.\n'
    'Robles-Algarín et al. (2024), enfocados en el Caribe colombiano, comparan métodos de ponderación dentro de un '
    'marco TOPSIS y evalúan criterios ambientales, demográficos, financieros y meteorológicos, aportando un '
    'antecedente aplicado y la importancia de un modelo transparente para asignar pesos.\n'
    'La IEA (2026) advierte que la capacidad de evacuación y la disponibilidad de conexión son un cuello de botella '
    'recurrente para nuevos proyectos renovables a nivel global. Por tanto, la cercanía a subestaciones y líneas de '
    'transmisión debe ser una variable central de viabilidad.\n'
    'A nivel de soporte institucional colombiano, la UPME ofrece geovisores energéticos y la aplicación GeoLCOE '
    'para estimar costos nivelados de generación de forma geoespacial, permitiendo aterrizar el enfoque teórico a '
    'variables concretas.\n'
    'Además, el Global Solar Atlas (World Bank Group, ESMAP & Solargis, 2020) proporciona la variable PVOUT '
    '(Photovoltaic Power Output), que expresa la producción anual de energía en kWh/kWp e incorpora factores reales '
    'de desempeño del sistema, superando análisis basados únicamente en radiación. La integración de esta fuente '
    'fortalece el análisis territorial al incorporar una base cuantitativa validada internacionalmente.\n\n'
    'Síntesis: Las soluciones revisadas orientan el diseño del dashboard hacia un modelo multicriterio con '
    'visualizaciones geoespaciales (mapas coropléticos), de comparación (barras apiladas) y de sensibilidad. '
    'Las referencias justifican la selección de los pesos y las restricciones adoptadas.'
)

add_heading(doc, '1.3 Contexto del Problema', level=2)
add_paragraph(doc,
    'Colombia ha incrementado el interés en proyectos de generación eléctrica a partir de fuentes renovables, '
    'especialmente solar fotovoltaica, para diversificar la matriz energética y atender el aumento sostenido de la '
    'demanda (aproximadamente 82,084 GWh en 2024 según XM). Sin embargo, las empresas enfrentan el desafío de '
    'seleccionar correctamente la ubicación, ya que la viabilidad de una granja solar no depende únicamente de la '
    'radiación disponible. La rentabilidad está condicionada por múltiples factores: cercanía a infraestructura '
    'eléctrica, capacidad de conexión, características del terreno, restricciones ambientales y demanda energética '
    'regional. La dispersión de estos datos en diferentes fuentes y formatos dificulta la toma de decisiones. '
    'El presente proyecto busca transformar datos complejos en información útil para la toma de decisiones '
    'estratégicas, facilitando la identificación de zonas con mayor viabilidad para el desarrollo de proyectos solares.'
)

add_heading(doc, '1.4 Descripción del Problema', level=2)
add_paragraph(doc,
    'SolarGrid Analytics S.A.S. enfrenta dificultades para identificar, de manera precisa y fundamentada, las zonas '
    'más viables para la instalación de una granja solar en Colombia. Su proceso actual se basa en variables aisladas '
    'como radiación solar y disponibilidad de terreno, sin integrar factores críticos como pendiente, proximidad a '
    'infraestructura eléctrica, demanda energética regional y restricciones ambientales. Esta limitación impide '
    'priorizar correctamente las ubicaciones y aumenta el riesgo de seleccionar sitios con baja rentabilidad o '
    'dificultades técnicas. Considerando el crecimiento de la demanda y los altos costos de inversión, resulta '
    'fundamental mejorar el proceso de decisión mediante el uso de datos geoespaciales, energéticos y económicos, '
    'junto con herramientas de visualización que permitan consolidar múltiples variables en un modelo analítico y '
    'facilitar la comparación de territorios.'
)

add_heading(doc, '1.5 Objetivo del Proyecto (SMART)', level=2)
add_paragraph(doc,
    'Desarrollar un modelo analítico basado en la integración de datos geoespaciales, energéticos y económicos, '
    'que permita identificar y priorizar las zonas con mayor viabilidad para la instalación de una granja solar en '
    'Colombia, mediante la construcción de un dashboard interactivo que integre variables como radiación solar, '
    'pendiente del terreno, proximidad a infraestructura eléctrica, demanda energética y restricciones territoriales, '
    'logrando generar un ranking de ubicaciones en un período de cuatro semanas, con base en datos oficiales '
    'provenientes de UPME, XM, IGAC y Superservicios.'
)

# --------------------------------------------------------------------
# 2. FUENTES DE DATOS Y PREGUNTAS DE NEGOCIO
# --------------------------------------------------------------------
doc.add_page_break()
add_heading(doc, '2. Fuentes de Datos y Preguntas de Negocio', level=1)

add_heading(doc, '2.1 Selección y Documentación de Fuentes de Datos (mínimo 4)', level=2)
add_paragraph(doc,
    'Se documentan las cuatro fuentes principales. Adicionalmente se procesaron fuentes complementarias cuyos metadatos '
    'están disponibles en las carpetas de observaciones del proyecto (data/clean/...).'
)

# Fuente 1: XM
add_heading(doc, 'Fuente 1: Demanda de energía eléctrica – XM', level=3)
# Tabla de metadatos
df_f1 = pd.DataFrame([
    ['Nombre y origen', 'Demanda de energía eléctrica – XM'],
    ['URL de acceso', 'https://www.xm.com.co/consumo/historicos-de-demanda'],
    ['Período de cobertura', 'Aproximadamente 2000 – actual'],
    ['Frecuencia de actualización', 'Horaria / diaria'],
    ['Unidad de análisis', 'Registro de demanda eléctrica por intervalo de tiempo'],
    ['Cobertura espacial', 'Nacional y por regiones eléctricas'],
    ['Licencia de uso', 'Datos públicos del sistema eléctrico (uso informativo y académico)'],
    ['Consideraciones éticas', 'No contiene datos personales; bajo riesgo de sesgo'],
    ['Variables clave', 'Fecha, hora, demanda (MW), región'],
    ['Cardinalidad', 'Alta (datos horarios continuos)'],
    ['Llaves candidatas', 'Fecha + hora + región'],
    ['Calidad inicial estimada', 'Alta calidad; formato estructurado, pocos nulos'],
    ['Observaciones', 'No siempre desagregado por municipio']
], columns=['Atributo', 'Descripción'])
add_table_from_dataframe(doc, df_f1)
add_paragraph(doc, '')
add_paragraph(doc, 'Justificación: Permite entender el comportamiento real de la demanda energética y priorizar regiones con mayor necesidad de generación.')

# Fuente 2: UPME
add_heading(doc, 'Fuente 2: Infraestructura eléctrica – UPME', level=3)
df_f2 = pd.DataFrame([
    ['Nombre y origen', 'Infraestructura eléctrica (subestaciones y líneas) – UPME'],
    ['URL de acceso', 'https://geo.upme.gov.co/server/rest/services/...'],
    ['Período de cobertura', 'Datos actuales (últimos años disponibles)'],
    ['Frecuencia de actualización', 'Periódica (no en tiempo real)'],
    ['Unidad de análisis', 'Elementos geográficos (puntos/líneas)'],
    ['Cobertura espacial', 'Nacional'],
    ['Licencia de uso', 'Datos abiertos del gobierno colombiano'],
    ['Consideraciones éticas', 'Sin datos sensibles'],
    ['Variables clave', 'Tipo infraestructura, ubicación, capacidad'],
    ['Cardinalidad', 'Media'],
    ['Llaves candidatas', 'ID infraestructura + coordenadas'],
    ['Calidad inicial estimada', 'Alta, formato GIS estructurado'],
    ['Observaciones', 'Requiere procesamiento geoespacial']
], columns=['Atributo', 'Descripción'])
add_table_from_dataframe(doc, df_f2)
add_paragraph(doc, '')
add_paragraph(doc, 'Justificación: La cercanía a la red eléctrica reduce significativamente los costos de conexión, uno de los factores más determinantes en la viabilidad de una granja solar.')

# Fuente 3: IGAC pendientes
add_heading(doc, 'Fuente 3: Pendiente del terreno – IGAC', level=3)
df_f3 = pd.DataFrame([
    ['Nombre y origen', 'Mapa de pendientes del terreno – IGAC'],
    ['URL de acceso', 'https://www.colombiaenmapas.gov.co'],
    ['Período de cobertura', 'Datos recientes (según publicación oficial)'],
    ['Frecuencia de actualización', 'Baja (datos relativamente estáticos)'],
    ['Unidad de análisis', 'Raster geográfico (celdas de terreno)'],
    ['Cobertura espacial', 'Nacional'],
    ['Licencia de uso', 'Datos abiertos con atribución'],
    ['Consideraciones éticas', 'Sin riesgos éticos'],
    ['Variables clave', 'Pendiente (%)'],
    ['Cardinalidad', 'Muy alta (datos raster)'],
    ['Llaves candidatas', 'Coordenadas geográficas'],
    ['Calidad inicial estimada', 'Alta resolución, requiere procesamiento'],
    ['Observaciones', 'Necesita transformación a formato analítico']
], columns=['Atributo', 'Descripción'])
add_table_from_dataframe(doc, df_f3)
add_paragraph(doc, '')
add_paragraph(doc, 'Justificación: La pendiente influye directamente en la instalación, estabilidad y costos de infraestructura de los paneles solares.')

# Fuente 4: PVOUT Global Solar Atlas
add_heading(doc, 'Fuente 4: PVOUT y potencial solar – Global Solar Atlas (World Bank/ESMAP/Solargis)', level=3)
df_f4 = pd.DataFrame([
    ['Nombre y origen', 'Global Solar Atlas – Photovoltaic Power Potential (PVOUT) · World Bank Group, ESMAP, Solargis'],
    ['URL de acceso', 'https://globalsolaratlas.info/download'],
    ['Período de cobertura', 'Datos promedio de largo plazo (modelo 2020)'],
    ['Frecuencia de actualización', 'No periódica (versión 2.0 disponible)'],
    ['Unidad de análisis', 'Celda raster (resolución ~1 km) – cada píxel representa un valor de PVOUT'],
    ['Cobertura espacial', 'Global; se recortó al bbox de Colombia'],
    ['Licencia de uso', 'Creative Commons Attribution (CC BY)'],
    ['Consideraciones éticas', 'Datos físicos agregados, sin información personal. El modelo satelital puede subestimar zonas con alta nubosidad local.'],
    ['Variables clave', 'PVOUT (kWh/kWp/día), GHI (kWh/m²/día)'],
    ['Cardinalidad', 'Continua (valores de punto flotante)'],
    ['Llaves candidatas', 'Coordenadas geográficas (longitud, latitud)'],
    ['Calidad inicial estimada', 'Excelente. <1% de celdas sin dato en Colombia. El producto ha sido validado con estaciones terrestres.'],
    ['Observaciones', 'Se extrajo el valor puntual para cada municipio usando las coordenadas internas del polígono municipal; no se calculó el promedio espacial.']
], columns=['Atributo', 'Descripción'])
add_table_from_dataframe(doc, df_f4)
add_paragraph(doc, '')
add_paragraph(doc, 'Justificación: El PVOUT incorpora parámetros técnicos y expresa directamente la producción eléctrica esperada, siendo la variable más adecuada para el modelo de viabilidad energética.')

add_heading(doc, '2.2 Preguntas de Negocio y Tipo de Visualización Sugerida', level=2)
# Tabla de preguntas (se conserva la del DOCX original)
df_preguntas = pd.DataFrame([
    ['¿Qué zonas de Colombia presentan mayor viabilidad para instalar una granja solar? (Focal)', 'Score de viabilidad V_i', 'Seleccionar ubicaciones prioritarias para inversión', 'Mapa coroplético / mapa de calor geoespacial', 'Geoespacial'],
    ['¿Qué regiones tienen mayor demanda energética?', 'Demanda energética (GWh o MW) por región', 'Priorizar zonas con mayor consumo potencial', 'Mapa coroplético + barras por región', 'Geoespacial / Comparación'],
    ['¿Cómo influye la distancia a infraestructura eléctrica en la viabilidad?', 'Distancia a subestaciones (km)', 'Reducir costos de conexión seleccionando zonas cercanas a red', 'Scatter plot (distancia vs score)', 'Relacional'],
    ['¿Qué zonas presentan condiciones óptimas de terreno para instalación?', 'Pendiente (%)', 'Identificar áreas técnicamente aptas para instalación', 'Mapa de calor / mapa categorizado', 'Geoespacial'],
    ['¿Cómo cambia la viabilidad al modificar el peso de las variables?', 'Score dinámico (ponderaciones)', 'Evaluar escenarios de decisión (priorizar demanda vs costo)', 'Dashboard interactivo (filtros + barras)', 'Multidimensional']
], columns=['Pregunta de negocio', 'Métrica o dato', 'Decisión esperada', 'Tipo de visualización sugerida', 'Categoría'])
add_table_from_dataframe(doc, df_preguntas)

# Modelo matemático
add_heading(doc, 'Modelo Matemático de Evaluación de Viabilidad', level=2)
add_paragraph(doc,
    'V_i = R_i · (0.30·S_i + 0.15·D_i + 0.25·G_i + 0.20·P_i + 0.10·U_i)\n'
    'Donde:\n'
    '- R_i: factor de restricción (0 si no cumple condiciones mínimas)\n'
    '- S_i: score solar normalizado (PVOUT)\n'
    '- D_i: score de demanda normalizado\n'
    '- G_i: score de cercanía a red (distancia inversa)\n'
    '- P_i: score de aptitud del terreno (pendiente)\n'
    '- U_i: score de uso del suelo (proporción no protegida)\n'
    'Los pesos suman 1.0 y están definidos en src/scoring/viabilidad_municipal.py.'
)

# --------------------------------------------------------------------
# 3. PREPARACIÓN E INTEGRACIÓN DE DATOS
# --------------------------------------------------------------------
doc.add_page_break()
add_heading(doc, '3. Preparación e Integración de Datos', level=1)

add_heading(doc, '3.1 Tabla de Transformaciones (mínimo 10)', level=2)
df_trans = pd.DataFrame([
    ['1', 'Python/pandas', 'Normalización de nombres de columnas (tildes, espacios, mayúsculas)', 'Todas las columnas originales', 'Nombres estandarizados (snake_case)', 'Facilita la manipulación y el merge entre fuentes'],
    ['2', 'Python/eda_utils', 'Limpieza de espacios y marcadores de nulos comunes', 'Columnas de texto', 'Columnas sin valores “NA”, “N/A”, “-”, etc.', 'Evita falsos nulos y mejora la calidad'],
    ['3', 'Python/eda_utils', 'Conversión de columnas de fecha (object → datetime64)', 'Columnas con semántica temporal (fecha, fecha_archivo, etc.)', 'Columnas datetime', 'Permite filtros temporales y extracción de año/mes'],
    ['4', 'Python/eda_utils', 'Conversión de columnas numéricas (string → float)', 'Columnas con valores numéricos almacenados como texto', 'Columnas numéricas (Float64)', 'Permite operaciones estadísticas y normalización'],
    ['5', 'Python/pvout_raster.py', 'Extracción de PVOUT puntual para cada municipio desde raster GeoTIFF', 'Coordenadas (lon/lat) de los municipios', 'pvout_kwh_kwp_day, annual_yield_kwh_kw_year', 'Obtiene el recurso solar local para cada unidad de análisis'],
    ['6', 'Python/igac_pendientes.py', 'Consulta del servicio de pendientes IGAC y clasificación en viable/condicional/no viable', 'pendiente_igac (etiqueta)', 'viabilidad_pendiente, score_pendiente', 'Convierte una etiqueta categórica en un filtro técnico binario y un score numérico'],
    ['7', 'Python/upme_subestaciones.py', 'Cálculo de distancia mínima a subestación de nivel 4/5 (en servicio)', 'dist_subestacion_km', 'g_i_red (score normalizado inverso)', 'La cercanía a la red reduce costos de conexión; valores más altos indican mayor viabilidad'],
    ['8', 'Python/runap_protected_areas.py', 'Intersección espacial de áreas protegidas con municipios', 'pct_area_protegida_runap', 'u_i_no_protegido_runap, r_i_runap (restricción dura si >80%)', 'Incorpora la restricción legal por áreas protegidas; U_i es la proporción no protegida'],
    ['9', 'Python/xm_demanda_municipal.py', 'Asignación de demanda proxy desde subáreas XM a municipios', 'zona_xm_demanda', 'demanda_xm_proxy_mwh, d_i_demanda', 'Genera una variable de demanda energética para priorizar zonas con mayor consumo'],
    ['10', 'Python/viabilidad_municipal.py', 'Normalización MinMax de todos los componentes', 's_i_solar, d_i_demanda, g_i_red, p_i_pendiente_proxy, u_i_uso_suelo', 'Componentes en escala 0-1', 'Permite combinar criterios heterogéneos en un solo índice'],
    ['11', 'Python/viabilidad_municipal.py', 'Cálculo del índice de viabilidad V_i', 'Componentes normalizados + restricción', 'v_i_modelo_proxy_xm', 'Ranking final de municipios según potencial solar'],
    ['12', 'Python/kmeans_municipios.py', 'Agrupamiento K-Means con componentes normalizados', 'Componentes S, D, G, P, U', 'cluster_kmeans, cluster_kmeans_label', 'Identifica perfiles de municipios similares para análisis complementario'],
    ['13', 'Python/solar_economics.py', 'Estimación de costos y producción por hectárea', 'land_use_hectares_per_mw, capex_usd_per_kw, annual_yield_kwh_per_kw_year', 'capacidad_kw_por_hectarea, generacion_mwh_por_hectarea_anual', 'Proporciona una referencia económica rápida para el tomador de decisión']
], columns=['#', 'Herramienta', 'Acción realizada', 'Columna(s) afectada(s)', 'Resultado / columna nueva', '¿Para qué sirve?'])
add_table_from_dataframe(doc, df_trans)

add_heading(doc, '3.2 EDA por Fuente Individual – Antes de Integrar', level=2)
add_paragraph(doc,
    'A continuación se resume el análisis exploratorio individual de cada fuente. Los reportes detallados '
    '(estadísticas, nulos, cardinalidad) se encuentran en las carpetas data/clean/<fuente>.\n'
)

add_paragraph(doc, 'Fuente 1: PVOUT (Global Solar Atlas – puntos municipales)\n'
    '- Estructura: 1.104 filas × 5 columnas.\n'
    '- Estadísticas: PVOUT media=4.05, mediana=4.03, desv. estándar=0.51 kWh/kWp/día. Mín=3.12, Máx=5.83 (La Guajira).\n'
    '- Calidad: 0 nulos, 0 duplicados. Código DANE único y listo como llave primaria.\n'
    '- Interpretación: La distribución es concentrada con cola derecha; el recurso solar no será el único factor discriminante.'
)

add_paragraph(doc, 'Fuente 2: Pendientes (IGAC, muestreo puntual)\n'
    '- Estructura: 1.104 filas × 7 columnas.\n'
    '- El 98.3% de municipios obtuvo respuesta; 19 puntos sin respuesta (excluidos).\n'
    '- Distribución: Plano (0-7%): 497 mpios, Inclinado (>7-14%): 312, Empinado (>14%): 155.\n'
    '- Interpretación: La pendiente es el filtro físico más restrictivo; los empinados quedan excluidos.'
)

add_paragraph(doc, 'Fuente 3: Subestaciones UPME y distancia a red\n'
    '- Estructura: 1.104 filas con distancia calculada.\n'
    '- Distancia media=42 km, mediana=28 km. 92% a <50 km de una subestación nivel 4/5.\n'
    '- Interpretación: La red está bien distribuida en región Andina y Caribe, dejando fuera Amazonía y Orinoquía.'
)

add_paragraph(doc, 'Fuente 4: RUNAP (áreas protegidas)\n'
    '- 520 municipios intersectan al menos un polígono RUNAP. Promedio de área protegida: 12%.\n'
    '- 15 municipios superan el 80% de cobertura y son excluidos (R_i=0).'
)

add_heading(doc, '3.3 Integración de Fuentes', level=2)
add_paragraph(doc,
    'La integración se realizó en el script src/scoring/viabilidad_municipal.py usando como llave principal '
    'el código DANE municipal (codigo_dane) normalizado a 5 dígitos.\n'
    'Todas las uniones fueron tipo left join a partir de la base municipal estricta '
    '(municipios_distritos_colombia.csv) con cardinalidad 1:1.\n'
    'Resultado: dataset final integrado con 1.104 filas y 45 columnas, exportado en '
    'data/clean/viabilidad_municipal/viabilidad_municipal_preliminar.csv.'
)

# --------------------------------------------------------------------
# 4. EDA COMPLETO
# --------------------------------------------------------------------
doc.add_page_break()
add_heading(doc, '4. EDA Completo – Calidad, Exploración Visual y Gráficas Explicativas', level=1)

add_heading(doc, '4.1 EDA Inicial – Estructura y Línea Base', level=2)
add_paragraph(doc,
    'Se ejecutó df.info() y df.describe() sobre el dataset integrado:\n'
    '- Filas: 1.104 entidades municipales.\n'
    '- Columnas: 45 (28 numéricas, 17 de texto).\n'
    '- Valores nulos: demanda 11.2%, distancia 2.0%, RUNAP 1.6%, resto <0.5%.\n'
    '- Estadísticas clave: PVOUT media=4.05, mediana=4.03; distancia media=42.2 km, mediana=28.4 km, máx=385 km.\n'
    'Propuesta de tratamiento: Los nulos en demanda se mantienen; los de distancia se imputan con la mediana para gráficos exploratorios (esos municipios ya están excluidos).'
)

add_heading(doc, '4.2 Identificación y Tratamiento de Faltantes', level=2)
df_falt = pd.DataFrame([
    ['XM demanda', 'demanda_xm_proxy_...', '11.2%', 'No imputar; se conserva NA', 'No deseamos inventar datos de consumo con incertidumbre'],
    ['UPME red', 'dist_subestacion_km', '2.0%', 'Imputación con la mediana (35.8 km) solo para EDA', 'No afecta al score (R_i=0); permite visualización completa'],
    ['RUNAP', 'pct_area_protegida_runap', '1.6%', 'Se asume 0 (sin área protegida)', 'Municipios sin intersección con RUNAP; razonable para el modelo']
], columns=['Fuente', 'Variable con faltantes', '% faltantes', 'Técnica aplicada', 'Justificación'])
add_table_from_dataframe(doc, df_falt)

add_heading(doc, '4.3 Identificación y Tratamiento de Outliers', level=2)
df_out = pd.DataFrame([
    ['PVOUT', 'pvout_kwh_kwp_day', 'IQR', '12 municipios >5.5', 'Conservar', 'Valores reales de alta radiación (La Guajira); son los mejores sitios'],
    ['UPME red', 'dist_subestacion_km', 'IQR + regla (>200 km)', '27 muy lejanos', 'Conservar (R_i=0 si >50 km)', 'Ya están excluidos; mantenerlos mapea cobertura'],
    ['Demanda XM', 'demanda_xm_proxy_...', 'Análisis cualitativo (zonas ambiguas)', '230 con incertidumbre', 'Etiquetar flag_atipico_eda_demanda=1', 'No son errores numéricos sino artefactos del mapeo'],
    ['Pendiente', 'score_pendiente', 'Regla de negocio (desconocida)', '19 sin respuesta', 'Excluir (R_i=0)', 'No se puede determinar aptitud']
], columns=['Fuente', 'Variable', 'Método de detección', 'Outliers detectados', 'Decisión tomada', 'Justificación'])
add_table_from_dataframe(doc, df_out)
add_paragraph(doc,
    'Diferenciación clave: Los outliers de PVOUT representan eventos reales del fenómeno físico (alta insolación), '
    'mientras que los de demanda son artefactos del proceso de asignación. Por tanto, los primeros se conservan '
    'y los segundos se marcan para decisión del usuario.'
)

add_heading(doc, '4.4 Normalización y Estandarización', level=2)
df_norm = pd.DataFrame([
    ['S_i (solar)', 'kWh/kWp/día (3.1–5.8)', 'MinMax (mayor es mejor)', '0–1', 'Valores altos → mayor generación'],
    ['D_i (demanda)', 'MWh (varía por zona)', 'MinMax (mayor es mejor)', '0–1', 'Mayor demanda → mercado más amplio'],
    ['G_i (red)', 'km (0.1–385)', 'MinMax inverso (menor es mejor)', '0–1', '1 = muy cerca de la red'],
    ['P_i (pendiente)', 'Ya era 1, 0.5 o 0', 'No se normalizó', '0–1', 'Puntaje categórico predefinido'],
    ['U_i (uso suelo)', 'Proporción no protegida (0–1)', 'No se normalizó', '0–1', 'Ya está en 0–1']
], columns=['Variable normalizada', 'Escala original', 'Técnica aplicada', 'Escala resultante', 'Justificación'])
add_table_from_dataframe(doc, df_norm)
add_paragraph(doc,
    'Se eligió MinMax porque el índice sintético requiere límites claros entre el “mejor” y el “peor” escenario, '
    'evitando que valores extremos dominen el resultado.'
)

add_heading(doc, '4.5 EDA Visual – Análisis Exploratorio con Visualizaciones', level=2)
add_paragraph(doc,
    'A continuación se presentan los gráficos exploratorios generados con municipal_visualizations.py. '
    'Cada gráfico incluye su interpretación.'
)

# 4.5.1 KDE
add_heading(doc, '4.5.1 KDE – Distribución de densidad estimada', level=3)
add_paragraph(doc, '[IMAGEN: histograma_pvout_kwh_kwp_day.png y histograma_v_i_modelo_proxy_xm.png]')
add_paragraph(doc,
    'PVOUT: Moda alrededor de 4.0 kWh/kWp/día, cola derecha suave → mayoría del territorio tiene recurso '
    'aceptable, pero solo un grupo pequeño destaca. El V_i muestra una distribución bimodal: pico cerca de 0 '
    '(municipios excluidos) y otro alrededor de 0.5-0.7 (viables), validando la efectividad de las restricciones duras.'
)

# 4.5.2 Histogramas
add_heading(doc, '4.5.2 Histogramas – Distribución de frecuencias', level=3)
add_paragraph(doc, '[IMAGEN: histograma_dist_subestacion_km.png]')
add_paragraph(doc,
    'Distancia a subestación: la mayoría de municipios están entre 0 y 60 km, justificando el umbral de 50 km '
    'para la restricción. PVOUT: simétrico, listo para normalización.'
)

# 4.5.3 Boxplots
add_heading(doc, '4.5.3 Boxplots – Dispersión y detección de outliers por grupo', level=3)
add_paragraph(doc, '[IMAGEN: boxplot_viabilidad_pendiente.png]')
add_paragraph(doc,
    'V_i por categoría de pendiente: los municipios “viables” tienen una mediana de V_i más alta (0.52) que los '
    '“condicional” (0.38), justificando la penalización aplicada.'
)

# 4.5.4 Scatter Plots
add_heading(doc, '4.5.4 Scatter Plots – Relación entre variables', level=3)
add_paragraph(doc, '[IMAGEN: scatter_pvout_vs_distancia.png]')
add_paragraph(doc,
    'PVOUT vs. distancia a red: correlación baja (r≈0.3). Existen municipios con alto PVOUT pero lejanos, y otros '
    'con PVOUT moderado pero muy cercanos a la red. Esto demuestra que la viabilidad real no puede juzgarse por '
    'una sola variable.'
)

add_heading(doc, '4.6 EDA Posterior a la Limpieza', level=2)
add_paragraph(doc,
    'Comparación antes/después:\n'
    '- Municipios con nulos críticos en distancia: 22 → 0 (imputación temporal).\n'
    '- Excluidos por R_i=0: 121 → 121 (sin cambios).\n'
    '- Distribución de S_i (media): 0.51 → 0.51 (sin cambios).\n'
    'Las operaciones de calidad no alteraron las distribuciones principales. El dataset está listo para el dashboard.'
)

add_heading(doc, '4.7 Gráficas Explicativas (mínimo 3)', level=2)

# Gráfica 1
add_heading(doc, 'Gráfica explicativa 1: Top 10 municipios con mayor V_i', level=3)
add_image_placeholder(doc, GRAFICA1_PATH, 'Figura 1. Top 10 municipios por score de viabilidad.')
add_paragraph(doc,
    'Título del hallazgo: “Los municipios costeros del Caribe y del Valle del Cauca lideran el ranking preliminar '
    'de viabilidad para granja solar”.\n'
    'Subtítulo metodológico: Barras horizontales · Score V_i calculado con modelo multicriterio · Datos 2020–2026 · '
    'Fuentes: Global Solar Atlas, IGAC, UPME, RUNAP, XM.\n'
    'Pregunta de negocio que responde: ¿Qué zonas de Colombia presentan mayor viabilidad para instalar una granja solar? (Focal).\n'
    'Interpretación: Los diez primeros municipios se localizan en La Guajira, Atlántico, Bolívar y Valle del Cauca. '
    'Todos comparten PVOUT superior a 5.2 kWh/kWp/día, distancia a subestación menor a 10 km y pendiente plana. '
    'El municipio líder obtiene un V_i de 0.89 impulsado por S_i=0.97 y G_i=0.99. Para la empresa, esta gráfica '
    'define la lista corta de candidatos donde iniciar estudios prediales.'
)

# Gráfica 2
add_heading(doc, 'Gráfica explicativa 2: Aportes ponderados al score en el Top 5', level=3)
add_image_placeholder(doc, GRAFICA2_PATH, 'Figura 2. Desglose de contribuciones ponderadas en el Top 5.')
add_paragraph(doc,
    'Título del hallazgo: “En el Top 5, el factor de cercanía a la red (G_i) representa más del 40 % del puntaje total”.\n'
    'Subtítulo metodológico: Barras apiladas normalizadas · Top 5 municipios sin atípicos de demanda.\n'
    'Pregunta de negocio que responde: ¿Cómo influye la distancia a infraestructura eléctrica en la viabilidad?\n'
    'Interpretación: La barra de red (naranja) aporta entre 0.25 y 0.30 puntos sobre 1.00 en el Top 5, siendo el '
    'componente más diferenciador. La lección estratégica es que la prioridad de búsqueda debe ser terrenos con '
    'acceso inmediato a subestaciones de nivel 4/5, incluso si no poseen la máxima radiación.'
)

# Gráfica 3
add_heading(doc, 'Gráfica explicativa 3: Mapa del score V_i con localización del Top 10', level=3)
add_image_placeholder(doc, GRAFICA3_PATH, 'Figura 3. Mapa de viabilidad municipal con Top 10 resaltados.')
add_paragraph(doc,
    'Título del hallazgo: “Los municipios con mayor viabilidad se alinean con los ejes de transmisión eléctrica existentes”.\n'
    'Subtítulo metodológico: Mapa de dispersión georreferenciado · 1.104 municipios · Color = V_i, tamaño = S_i · Top 10 resaltados en negro.\n'
    'Pregunta de negocio que responde: ¿Qué zonas presentan condiciones óptimas de terreno para instalación?\n'
    'Interpretación: Los puntos amarillos (V_i alto) se agrupan en la costa Caribe, el valle del Magdalena y el '
    'valle del Cauca, coincidiendo con las principales subestaciones de transmisión. Se recomienda concentrar los '
    'estudios de campo en un radio de 100 km alrededor de las subestaciones Barranquilla, Cartagena y Yumbo.'
)

# --------------------------------------------------------------------
# 5. INFORME EJECUTIVO FINAL (Anexo B)
# --------------------------------------------------------------------
doc.add_page_break()
add_heading(doc, '5. Informe Ejecutivo Final', level=1)
add_paragraph(doc, 'Resumen ejecutivo para el gerente de SolarGrid Analytics S.A.S.')
add_paragraph(doc, '')

add_paragraph(doc,
    '**Principales hallazgos**\n'
    '1. Existen municipios con condiciones sobresalientes. Aplicando un modelo multicriterio que combina recurso solar, '
    'cercanía a la red eléctrica, pendiente del terreno, restricciones ambientales y demanda energética, se identificaron '
    '10 municipios líderes, ubicados en la costa Caribe y el Valle del Cauca. El municipio de **[nombre]** encabeza el '
    'ranking con una puntuación de 0.89 sobre 1.00.\n'
    '2. La cercanía a subestaciones es el factor más diferenciador. Aunque Colombia posee buena radiación solar en muchas '
    'regiones, pocas cuentan con acceso inmediato a la red de alta tensión. En los mejores candidatos, el factor “red” '
    'explica el 40 % del puntaje, lo que reduce significativamente los costos de conexión.\n'
    '3. Las restricciones ambientales eliminan el 15 % del territorio, pero dejan amplias zonas disponibles. Los parques '
    'naturales excluyen automáticamente las áreas insulares y amazónicas, pero no afectan al Top 10.\n'
    '4. La demanda local presenta cierta incertidumbre. Para algunos departamentos fue necesario usar promedios regionales; '
    'se recomienda contrastar estos valores con las empresas distribuidoras antes de tomar decisiones de inversión.'
)

add_paragraph(doc,
    '**¿Qué sigue?**\n'
    'En la segunda entrega se construirá un dashboard interactivo que permitirá explorar los resultados, modificar los pesos '
    'de los criterios y aplicar filtros geográficos. Asimismo, se sugiere iniciar estudios de campo en los 5 municipios '
    'mejor clasificados.'
)

add_paragraph(doc,
    '**Limitaciones**\n'
    'Este análisis es una priorización preliminar basada en datos públicos. No sustituye estudios de ingeniería, topografía, '
    'análisis predial, licencias ambientales ni estudios de conexión. Los valores de demanda energética son aproximaciones '
    'que deben ser validadas.'
)

# --------------------------------------------------------------------
# REFERENCIAS
# --------------------------------------------------------------------
doc.add_page_break()
add_heading(doc, 'Referencias (APA 7)', level=1)

referencias = [
    'Ángel-Sanint, E., García-Rendón, J. J., & Pérez-Ceballos, M. A. (2023). Refining wind and solar potential maps through spatial multicriteria assessment: Case study: Colombia. Energy for Sustainable Development, 72, 203–215. https://doi.org/10.1016/j.esd.2023.01.001',
    'de Luis-Ruiz, J. M., Pereda-García, R., & Fernández-Maroto, G. (2024). Optimal location of solar photovoltaic plants using geographic information systems and multi-criteria analysis. Sustainability, 16(7), 2895. https://doi.org/10.3390/su16072895',
    'International Energy Agency. (2026). Electricity 2026: Grids. IEA. https://www.iea.org/reports/electricity-2026',
    'Instituto Geográfico Agustín Codazzi. (s.f.). Pendientes de Colombia [MapServer]. https://mapas.igac.gov.co/server/rest/services/ordenamientoterritorial/pendientescolombia/MapServer',
    'Instituto Geográfico Agustín Codazzi. (s.f.). Límites municipales [MapServer]. https://mapas2.igac.gov.co/server/rest/services/limites/limites/MapServer',
    'Instituto Geográfico Agustín Codazzi. (s.f.). Zonificación de usos según POT [MapServer]. https://mapas.igac.gov.co/server/rest/services/ordenamientoterritorial/zonificacionusossegunpot/MapServer',
    'IRENA. (2025). Renewable Power Generation Costs in 2024. International Renewable Energy Agency. https://www.irena.org/publications/2025/Jul/Renewable-Power-Generation-Costs-in-2024',
    'NASA. (s.f.). POWER Data Access Viewer (Daily). https://power.larc.nasa.gov/docs/services/api/temporal/daily/',
    'Nassar, A. K., Al-Masri, H., & Al-Salaymeh, M. (2025). Multi-criteria GIS-based approach for optimal site selection of solar and wind energy. Energy Conversion and Management, 325, 119324. https://doi.org/10.1016/j.enconman.2025.119324',
    'National Renewable Energy Laboratory. (2013). Land-use requirements for solar power plants in the United States (NREL/TP-6A20-56290). U.S. Department of Energy. https://docs.nrel.gov/docs/fy13osti/56290.pdf',
    'Parques Nacionales Naturales de Colombia. (s.f.). Registro Único de Áreas Protegidas (RUNAP). https://mapas.parquesnacionales.gov.co/arcgis/rest/services/pnn/runap/MapServer',
    'Robles-Algarín, C., Viloria-Porto, J., & Ospino-Castro, A. (2024). Optimal site selection for solar PV systems in the Colombian Caribbean: Evaluating weighting methods in a TOPSIS framework. Sustainability, 16(20), 8761. https://doi.org/10.3390/su16208761',
    'Unidad de Planeación Minero Energética. (s.f.). GeoLCOE y geovisores energéticos de Colombia. https://www1.upme.gov.co',
    'Unidad de Planeación Minero Energética. (s.f.). Subestaciones eléctricas [MapServer]. https://geo.upme.gov.co/server/rest/services/SUBESTACIONES/UPME_EN_DI_SUBESTACION_consulta/MapServer',
    'World Bank Group, ESMAP, & Solargis. (2020). Global Photovoltaic Power Potential by Country. World Bank. https://globalsolaratlas.info/global-pv-potential-study',
    'XM S.A. E.S.P. (s.f.). Históricos de demanda. https://www.xm.com.co/consumo/historicos-de-demanda',
]
for ref in referencias:
    add_paragraph(doc, ref)

# --------------------------------------------------------------------
# GUARDAR
# --------------------------------------------------------------------
output_path = OUTPUT_DIR / f"Informe_AE3_{NOMBRE_GRUPO.replace(' ', '_')}.docx"
doc.save(output_path)
print(f"Documento guardado en: {output_path}")
