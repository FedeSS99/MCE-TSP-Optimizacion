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
