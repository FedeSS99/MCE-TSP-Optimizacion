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
 
---
 # Descripción del problema

El problema del agente viajero – Travel Salesman Problem (TSP por sus siglas en inglés) es un problema clásico de optimización con diversas aplicaciones en el mundo real donde se busca encontrar la ruta más eficiente para que un viajero visite un conjunto de ciudades y regrese a su punto de origen, minimizando la distancia total recorrida. 

Matemáticamente, el TSP puede ser descrito con los siguientes componentes.

### Parámetros

1. **Conjunto de nodos:**  $N = \{1, 2, ...,n \}$ un conjunto de ciudades que se deben visitar.
2. **Matriz de distancias: $D = [d_{ij}]_{n \times n}$**  una matriz $n \times n$, donde  $d_{ij}$ representa la distancia entre la ciudad *i* y la ciudad *j.* 

### Variables de desición

Serán de la forma $**x_{ij}$,** variables binarias, que serán 1 si el camino entre las ciudades *i* y *j*  se incluye en la solucuón y 0 en caso contrario.

### Función objetivo

Para el TSP, se busca la minimización de la distancia total recorrida, que puede representarse como

$$
min \sum_{i = 1}^n \sum_{j = 1}^n d_{ij}x_{ij}
$$

### Factibilidad

Las restricciones del problema son las siguientes:

- Cada nodo es visitado exactamente una vez
    
    $$\sum_{j = 1, j \not = i}^n x_{ij} = 1, \forall i \in N$$
    
- El agente regresa a la ciudad de origen
    
    $$\sum_{i=1, i \not = j}^n  x_{ij} = 1, \forall j \in N$$
    
- No existen subtours, es decir, viajes que no incluyan todas las ciudades.
    
    $$u_i - u_j + nx_{ij} \leq n - 1, \forall i,j \in N, i \not = j, j > 1, u_1 = 1$$
    
    Donde $u_i$  es una variable que indica el orden en el cual se visitan las ciudades. Esta formulación garantizada que el agente recorra todas las ciudades exactamente una vez y regrese al punto de partida, sin formar subtours.


  ### Ejemplo

Supongamos que un vendedor necesita visitar 4 ciudades: A, B, C y D. Las distancias entre las ciudades son las siguientes:

|  | A | B | C | D |
| --- | --- | --- | --- | --- |
| A | 0 | 10 | 15 | 20 |
| B | 10 | 0 | 35 | 25 |
| C | 15 | 35 | 0 | 30 |
| D | 20 | 25 | 30 | 0 |

*La ruta más corta posible para el vendedor en este ejemplo es A → B → D → C → A, con una distancia total de 80 unidades.*
