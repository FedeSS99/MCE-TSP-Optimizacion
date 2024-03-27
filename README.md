# Aplicación del TSP al GTFS de Ciudad de México para la planificación eficiente de rutas

# Introducción

La Ciudad de México tiene una población de más de 22 millones de personas y un sistema de transporte público complejo que incluye metro, autobuses, metrobús y taxi. En este contexto, la eficiencia de las rutas de transporte es crucial para mejora de la movilidad urbana, y por ende, de la calidad de vida de las personas usuarias.

La aplicación del Problema del Viajante (TSP) al General Transit Feed Specification (GTFS) de la Ciudad de México ofrece una oportunidad para encontrar la ruta más corta que pase por un conjunto de ubicaciones y volver al punto de partida.

## Alcance

- **Análisis de datos GTFS de la CDMX**, que se compone de archivos de texto relacionados entre sí que modelan cada aspecto de los servicios de transporte público: agencias, rutas, viajes, frecuencias, horarios, entre otros. Los archivos que se publican comprenden información de Corredores Concesionados, Metro, Metrobús, Trolebús, RTP, Tren Ligero, Ferrocarril Suburbano, Cablebús y Pumabús.
- Desarrollo de un algoritmo de optimización clásica del **Problema del Viajante (TSP)** con los datos extraídos de la base GTFS de la Ciudad de México.
- Optimización de las 10 rutas más utilizadas para verificar la eficiencia de estas y recomendar una ruta más eficiente en caso de ser necesario.
- Implementación en otras rutas y documentación de la propuesta del modelo.

---

# Objetivos

## Objetivo general

> Optimizar el tiempo de traslado en las rutas del transporte público de la Ciudad de México, bajo la perspectiva del TSP utilizando los datos abiertos GTFS.
> 

## Objetivos específicos

- Extraer información en formato GTFS del transporte público y la información geográfica asociada con una antigüedad no mayor a 6 meses.
- Desarrollar un algoritmo TSP que integre los datos GTFS de la Ciudad de México.
- Identificar las 10 rutas de transporte público con mayor demanda en Ciudad de México, en traslados de lunes a viernes.
- Verificar que las 10 rutas revisadas sean el camino óptimo, caso contrario, recomendar la búsqueda de otra ruta.



# Datos

**GTFS estático (General Transit Feed Specification): Especificación General de los Aspectos de Tránsito.**

[Link a los datos](https://datos.cdmx.gob.mx/dataset/gtfs)

**Notas sobre los datos**

- El GTFS estático es un estándar para dar a conocer la operación del transporte público y la información geográfica asociada.
- Aquí se publica la última versión del GTFS estático de la Ciudad de México (31 de octubre de 2022), el cual se mantiene en continua mejora y expansión a otros modos de transporte público.
- Actualmente, se compone de ocho archivos de texto relacionados entre sí por un identificador que modelan cada aspecto de los servicios de transporte público: agencias, rutas, viajes, frecuencias, horarios, entre otros.
- Los ocho archivos son los siguientes
    - `agency.txt`: contiene el nombre de cada una de las agencias, así como páginas de internet donde se puede consultar más información sobre estas.
    - `routes.txt`: incluye el origen y destino de cada una de las líneas, corredores o rutas que integran el Sistema, así como la agencia a la cual pertenecen y el nombre corto de la ruta correspondiente.
    - `trips.txt`: contiene las distintas ramificaciones o variaciones que puede tener una ruta, se indica el sentido de la ruta de transporte y los días de servicio.
    - `calendar.txt`: especifica los días de la semana en que operan los distintos servicios de transporte.
    - `frequencies.txt`: contiene las frecuencias promedio de arribo entre las estaciones de cada viaje que se indican en el archivo trips.txt, así como los horarios de operación.
    - `shapes.txt`: incluye los puntos que conforman el trazado de las rutas de transporte.
    - `stops.txt`: indica la ubicación geográfica de las estaciones, así como el nombre con el que son comúnmente conocidas.
    - `stop_times.txt`: cuenta con la información del tiempo estimado de arribo entre las estaciones o paradas que conforman cada viaje.
 
 
