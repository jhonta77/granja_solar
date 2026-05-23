# Guia Power BI Desktop

## Flujo recomendado: conexion directa MySQL

Power BI debe conectarse a MySQL y consumir la vista:

```text
granja_solar.vw_powerbi_modelo
```

Esta es la opcion recomendada porque evita duplicar datos en CSV.

Pasos en Power BI Desktop:

1. `Obtener datos -> Base de datos MySQL`.
2. Servidor: `127.0.0.1:3306`.
3. Base de datos: `granja_solar`.
4. Elegir la vista `vw_powerbi_modelo`.
5. Cargar.

Si Power BI pide el conector de MySQL, instalar `MySQL Connector/NET` y reiniciar
Power BI Desktop.

Para reconstruir la base y las vistas:

```powershell
python -m src.db.mysql_loader --apply
```

Vistas principales disponibles:

- `vw_powerbi_modelo`: vista unica recomendada para el dashboard.
- `vw_dashboard_municipal`: vista municipal base.
- `vw_comparacion_scores`: comparacion de scores multidimensionales.
- `vw_municipal_contexto`: variables fisicas, red, RUNAP, POT y demanda.
- `vw_subestaciones_municipios`: cercania a subestaciones.

## Flujo alternativo: CSV

1. Importar o actualizar la base MySQL desde:

```powershell
mysql -u root -p < data\interim\mysql\bootstrap_granja_solar.sql
```

2. Exportar los archivos limpios para Power BI:

```powershell
python -m src.reporting.powerbi_export
```

3. En Power BI Desktop usar:

```text
Obtener datos -> Carpeta -> data\powerbi
```

Tambien se pueden cargar CSV individuales desde `data\powerbi`.

## Archivos CSV generados

- `modelo_powerbi.csv`: tabla unica recomendada para empezar el dashboard.
- `dashboard_municipal.csv`: vista principal para mapas, ranking y filtros.
- `comparacion_scores.csv`: scores por dimension y ranking de viabilidad.
- `contexto_municipal.csv`: variables fisicas, red, RUNAP, POT y demanda.
- `rentabilidad_municipal.csv`: indicadores financieros; si no existe en MySQL, se toma del CSV limpio local.
- `clusters_municipios.csv`: cluster K-Means y variables asociadas.
- `municipios.csv`: dimension municipal.
- `departamentos.csv`: dimension departamental.
- `manifest_powerbi.csv`: control de filas, columnas, origen y estado de cada exportacion.

## Relaciones sugeridas

- `municipios[codigo_dane]` 1:* `dashboard_municipal[codigo_dane]`
- `municipios[codigo_dane]` 1:* `comparacion_scores[codigo_dane]`
- `municipios[codigo_dane]` 1:* `contexto_municipal[codigo_dane]`
- `municipios[codigo_dane]` 1:* `rentabilidad_municipal[codigo_dane]`
- `municipios[codigo_dane]` 1:* `clusters_municipios[codigo_dane]`
- `departamentos[departamento_id]` 1:* `municipios[departamento_id]`

Configurar `codigo_dane` como texto en Power BI para conservar los ceros a la izquierda.

## Medidas DAX iniciales

Si usas conexion directa a MySQL, crea las medidas contra la tabla/vista
`vw_powerbi_modelo`.

```DAX
Municipios = DISTINCTCOUNT(vw_powerbi_modelo[codigo_dane])

Score promedio = AVERAGE(vw_powerbi_modelo[v_i_multidimensional])

Municipios alta viabilidad =
CALCULATE(
    DISTINCTCOUNT(vw_powerbi_modelo[codigo_dane]),
    vw_powerbi_modelo[nivel_viabilidad] = "Alta"
)

Margen promedio ha =
AVERAGE(vw_powerbi_modelo[margen_estimado_cop_ha_year])

Relacion B/C promedio =
AVERAGE(vw_powerbi_modelo[relacion_beneficio_costo_rentabilidad])
```

Si quieres empezar con CSV, carga solo `modelo_powerbi.csv` y crea las medidas
contra la tabla `modelo_powerbi`.

Crear estas medidas desde `Modelado -> Nueva medida`. Si Power BI dejo todas las
columnas dentro de una sola tabla llamada `powerbi`, reemplazar el nombre de la
tabla por `powerbi`. Si cargaste los CSV como tablas separadas, usar los nombres
mostrados abajo.

```DAX
Municipios = DISTINCTCOUNT(modelo_powerbi[codigo_dane])

Score promedio = AVERAGE(modelo_powerbi[v_i_multidimensional])

Municipios alta viabilidad =
CALCULATE(
    DISTINCTCOUNT(modelo_powerbi[codigo_dane]),
    modelo_powerbi[nivel_viabilidad] = "Alta"
)
```

Si cargaste el modelo relacional con tablas separadas:

```DAX
Municipios = DISTINCTCOUNT(municipios[codigo_dane])

Score promedio = AVERAGE(comparacion_scores[v_i_multidimensional])

Municipios alta viabilidad =
CALCULATE(
    DISTINCTCOUNT(comparacion_scores[codigo_dane]),
    comparacion_scores[nivel_viabilidad] = "Alta"
)

ROI promedio = AVERAGE(rentabilidad_municipal[roi_neto])
```

## Consultas DAX para validar

Estas consultas se pueden ejecutar desde la vista `Consultas DAX`.

```DAX
EVALUATE
TOPN(
    10,
    comparacion_scores,
    comparacion_scores[v_i_multidimensional],
    DESC
)
ORDER BY comparacion_scores[v_i_multidimensional] DESC
```

```DAX
EVALUATE
SUMMARIZECOLUMNS(
    comparacion_scores[departamento],
    "Municipios", DISTINCTCOUNT(comparacion_scores[codigo_dane]),
    "Score promedio", AVERAGE(comparacion_scores[v_i_multidimensional]),
    "Municipios alta viabilidad",
        CALCULATE(
            DISTINCTCOUNT(comparacion_scores[codigo_dane]),
            comparacion_scores[nivel_viabilidad] = "Alta"
        )
)
ORDER BY [Score promedio] DESC
```
