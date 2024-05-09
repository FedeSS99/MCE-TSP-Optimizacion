# Proyecto de Optimización
## CIMAT Monterrey
### Aplicación del TSP al GTFS de Ciudad de México para la Planificación Eficiente de Rutas
#- Semiramis G. de la Cruz
#- Federico Salinas Samaniego

## Lectura de datos

# Librerias
import os
import pandas as pd


# Directorio
PATH = "../"
files = os.listdir(os.path.join(PATH, 'data'))

# Importar datos
df_agency = pd.read_csv(os.path.join(PATH, 'data', files[0]))
df_calendar = pd.read_csv(os.path.join(PATH, 'data', files[1]))
df_frequencies = pd.read_csv(os.path.join(PATH, 'data', files[2]))
df_routes = pd.read_csv(os.path.join(PATH, 'data', files[3]))
df_shapes = pd.read_csv(os.path.join(PATH, 'data', files[4]))
df_stops = pd.read_csv(os.path.join(PATH, 'data', files[5]))
df_stop_times = pd.read_csv(os.path.join(PATH, 'data', files[6]))
df_trips = pd.read_csv(os.path.join(PATH, 'data', files[8]))

