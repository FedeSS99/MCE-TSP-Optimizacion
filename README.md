# Aplicación del TSP para la optimización de rutas de metro en Ciudad de México usando Recocido Simulado y Búsqueda Tabú
Por Semiramís G. de la Cruz y Federico Salinas Samaniego 

# Introducción

La Ciudad de México enfrenta desafíos significativos en movilidad urbana debido a su denso y complejo sistema de transporte público. Según la Encuesta Origen-Destino de 2017 del INEGI, en la Zona Metropolitana del Valle de México, el transporte público es el medio más utilizado para ir al trabajo, con un 45\% de los viajes \cite{unam2018}. Sin embargo, las personas se enfrentan a una gran cantidad de opciones de rutas y conexiones, lo que complica la identificación de la mejor ruta. 


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

## Descripción del método constructivo del problema

1. Iteración:
    - Mientras el número de iteraciones realizadas (**`n_iters`**) sea menor que el máximo permitido (**`MaxIters`**):
2. Generación de Ruta Aleatoria:
    - Calcular el número total de nuevas selecciones de nodos (**`TotalNuevasSelecciones`**) como la cantidad total de nodos menos uno.
    - Inicializar una ruta (**`ruta`**) con el nodo inicial (**`Ini`**) al principio y al final, y marcando el resto como "*".
    - Seleccionar índices aleatorios de nodos disponibles (**`índices_aleatorios`**) sin reemplazo.
    - Concatenar los índices de nodo seleccionados con el nodo inicial para formar una lista de índices de distancia (**`índices_distancia`**).
    - Asignar los nombres correspondientes a los nodos seleccionados en la ruta.
    - Calcular la distancia recorrida (**`distancia_recorrida`**) sumando las distancias entre nodos consecutivos en **`índices_distancia`**.
3. Evaluación de la Solución Actual:
    - Incrementar el contador de iteraciones (**`n_iters`**) en uno.
    - Si la distancia recorrida (**`distancia_recorrida`**) es menor que la distancia mínima encontrada hasta el momento (**`min_dist`**):
        - Actualizar la mejor ruta (**`mejor_ruta`**) con la ruta actual.
        - Actualizar la distancia mínima (**`min_dist`**) con la distancia recorrida.
        - Imprimir el número de iteración actual (**`n_iters`**) junto con la distancia mínima encontrada (**`min_dist`**).


## Aplicación de Metaheurísticas
El proyecto aplica técnicas de Recocido Simulado y Búsqueda Tabú para la optimización de rutas del Sistema de Transporte Colectivo Metro en la Ciudad de México. Se presentan los resultados obtenidos utilizando diferentes configuraciones de parámetros para ambas heurísticas.

### Recocido Simulado

#### Parámetros Utilizados
1. **Modelo 1:** α = 0.85, N = 500
2. **Modelo 2:** α = 0.99, N = 3500

#### Modelo 1: α = 0.85, N = 500
| Solución | Costo (Horas) | Tiempo (Horas) |
|----------|----------------|----------------|
| 1        | 5.832222       | 0.013311       |
| 2        | 6.245000       | 0.013149       |
| 3        | 5.693056       | 0.013445       |
| 4        | 5.693056       | 0.012782       |
| 5        | 5.993056       | 0.013167       |

*Cuadro 2. Resultados de 5 soluciones obtenidas mediante recocido simulado con parámetros α = 0.85 y N = 500.*

#### Modelo 2: α = 0.99, N = 3500
| Solución | Costo (Horas) | Tiempo (Horas) |
|----------|----------------|----------------|
| 1        | 5.693056       | 0.177517       |
| 2        | 5.693056       | 0.174258       |
| 3        | 5.693056       | 0.188916       |
| 4        | 5.693056       | 0.196538       |
| 5        | 5.693056       | 0.183417       |

*Cuadro 3. Resultados de 5 soluciones obtenidas mediante recocido simulado con parámetros α = 0.99 y N = 3500.*

#### Métricas Resumidas
| Modelo | Costo Promedio (Horas) | Tiempo Promedio (Horas) |
|--------|-------------------------|-------------------------|
| 1      | 5.831277                | 0.013170                |
| 2      | 5.693056                | 0.184129                |

*Cuadro 4. Costo promedio y tiempo de cómputo promedio del par de sub modelos del recocido simulado.*

### Búsqueda Tabú

#### Parámetros Utilizados
1. **Combinaciones probadas:**
   - Tabu Size: {5, 10, 15}
   - Max Iters: {5, 10, 15}

#### Tabu Size = 5, Max Iters = 5
| Solución | Costo (Horas) | Tiempo (Horas) |
|----------|----------------|----------------|
| 1        | 6.276389       | 0.017797       |
| 2        | 6.047222       | 0.018772       |
| 3        | 5.979722       | 0.018454       |
| 4        | 6.113333       | 0.019072       |
| 5        | 6.006944       | 0.019015       |

*Cuadro 5. Resultados de 5 soluciones obtenidas mediante búsqueda tabú con Tabu Size = 5 y Max Iters = 5.*

#### Tabu Size = 5, Max Iters = 15
| Solución | Costo (Horas) | Tiempo (Horas) |
|----------|----------------|----------------|
| 1        | 5.693056       | 0.045797       |
| 2        | 5.693056       | 0.046560       |
| 3        | 5.693056       | 0.046212       |
| 4        | 5.693056       | 0.048388       |
| 5        | 5.693056       | 0.051388       |

*Cuadro 6. Resultados de 5 soluciones obtenidas mediante búsqueda tabú con Tabu Size = 5 y Max Iters = 15.*

### Métricas Resumidas
| Modelo | Costo Promedio (Horas) | Tiempo Promedio (Horas) |
|--------|-------------------------|-------------------------|
| 1      | 6.084722                | 0.01825                 |
| 2      | 5.693056                | 0.04766                 |

*Cuadro 7. Costo promedio y tiempo de cómputo promedio del par de submodelos de búsqueda tabú.*

## Conclusiones
Los resultados obtenidos muestran que la configuración con α = 0.99 y N = 3500 para Recocido Simulado y Tabu Size = 5 y Max Iters = 15 para Búsqueda Tabú son las más eficientes en términos de costo y tiempo de cómputo.
