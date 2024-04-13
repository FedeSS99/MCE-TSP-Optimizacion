import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CA_TSP:
    def __init__(self, Long:np.ndarray, Lat:np.ndarray, Etiquetas:list[str],
                 Ini:str, MaxIters:int) -> None:
        self.__PosX, self.__PosY = Long, Lat
        self.__Etiquetas : list[str] = Etiquetas
        self.__N : int = len(self.__Etiquetas)

        self.Nodos = {}
        for p in range(self.__N):
            self.Nodos[self.__Etiquetas[p]] = (self.__PosX[p], self.__PosY[p])

        self.__Ini : str = Ini
        self.__MaxIters : int = MaxIters

        self.MatDist = self.__ObtenerMatrizDistancias()

    def __ObtenerMatrizDistancias(self):
        PosX = np.copy(self.__PosX).reshape((1,self.__N))
        PosY = np.copy(self.__PosY).reshape((1,self.__N))
        return np.sqrt((PosX - PosX.T)**2.0 + (PosY - PosY.T)**2.0)

    def __HallarRuta(self):
        TotalSelecciones = self.__N - 1
        ruta = [self.__Ini] + TotalSelecciones*["*"] + [self.__Ini]
        dist_rec = 0.0

        indice_ini = self.__Etiquetas.index(self.__Ini)

        indices_disponibles = np.delete(np.arange(self.__N), indice_ini)
        indices_aleatorios = np.random.choice(indices_disponibles, TotalSelecciones, replace=False)

        indice_prev = indice_ini
        for k in range(1, TotalSelecciones+1):
            indice_k = indices_aleatorios[k-1]

            ruta[k] = self.__Etiquetas[indice_k]
            dist_rec += self.MatDist[indice_prev, indice_k]
            indice_prev = indice_k
        
        dist_rec += self.MatDist[indice_k, indice_ini]
        
        return ruta, dist_rec


    def EncontrarSolucion(self):
        n_iters = 0
        min_dist = np.inf
        mejor_ruta = []
        while n_iters < self.__MaxIters:
            ruta_actual, dist = self.__HallarRuta()
            n_iters += 1
            print(f"#{n_iters} - {dist:.3f}", end = "\n")

            if dist < min_dist:
                mejor_ruta = [nodo for nodo in ruta_actual]
                min_dist = dist
            
        return mejor_ruta, min_dist


if __name__ == "__main__":
    data = pd.read_csv("./data/worldcitiespop.csv",
                       usecols=["Country", "City", "Population",
                                "Latitude", "Longitude"])
    data = data.loc[data["Country"] == "mx", ["City", "Population", "Latitude", "Longitude"]]
    
    q95_pop = data["Population"].quantile(0.95)
    data = data[data["Population"] >= q95_pop]

    Longitude = data["Longitude"].to_numpy()
    Latitude = data["Latitude"].to_numpy()
    Ciudades = data["City"].to_list()

    TSP_Capitales = CA_TSP(Longitude, Latitude, Ciudades, "apodaca", 10000)

    mejor_ruta, min_dist = TSP_Capitales.EncontrarSolucion()
    print(mejor_ruta, min_dist)