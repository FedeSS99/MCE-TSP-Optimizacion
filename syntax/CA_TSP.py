import numpy as np
import pandas as pd

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "serif"

class CA_TSP:
    """
    Initialize the TSP solver.
    Args:
    - Long: numpy array of longitudes for each node.
    - Lat: numpy array of latitudes for each node.
    - Names: list of names corresponding to each node.
    - Ini: initial node for the TSP solution.
    - MaxIters: maximum number of iterations for the solution search.
    """
    def __init__(self, Long:np.ndarray, Lat:np.ndarray, Names:list[str],
                 Ini:str, MaxIters:int) -> None:
        self.__Lat, self.__Long = Lat, Long
        self.__Names : list[str] = Names
        self.__N : int = len(self.__Names)

        self.Nodes = {}
        for p in range(self.__N):
            self.Nodes[self.__Names[p]] = (self.__Long[p], self.__Lat[p])

        self.__Ini : str = Ini
        self.__init_index = self.__Names.index(self.__Ini)
        self.__availableIndex = np.delete(np.arange(self.__N), self.__init_index)
        self.__MaxIters : int = MaxIters

        self.MatDist = self.__ObtainDistanceMatrix()

        print(f"-- Random Constructive Method --\nN = {self.__N}\nInitial Node: {self.__Ini}")

    def __ObtainDistanceMatrix(self):
        """
        Calculate the distance matrix based on node coordinates.

        Returns:
        - MatDist: numpy array representing the distance matrix.
        """
        Lat = np.deg2rad(np.copy(self.__Long).reshape((1,self.__N)))
        Long = np.deg2rad(np.copy(self.__Lat).reshape((1,self.__N)))

        cos_difLat = np.cos(Lat - Lat.T)
        cos_difLong = np.cos(Long - Long.T)
        prodcosLat = np.cos(Lat)*(np.cos(Lat).T)
        arg_arcsin = np.sqrt(0.5 * (1 - cos_difLat + prodcosLat* ( 1.0 - cos_difLong)))

        return 2.0 * 6371 * np.arcsin(arg_arcsin)

    def __FindRandomRoute(self):
        """
        Find a random route for the TSP.

        Returns:
        - ruta: list representing the random route.
        - dist_rec: total distance of the random route.
        """
        TotalNewSelections = self.__N - 1
        ruta = [self.__Ini] + TotalNewSelections*["*"] + [self.__Ini]
        dist_rec = 0.0

        random_indexes = np.random.choice(self.__availableIndex, TotalNewSelections, replace=False)
        dist_indices = np.concatenate(([self.__init_index], random_indexes, [self.__init_index]))

        ruta[1:-1] = [self.__Names[k] for k in random_indexes]
        dist_rec = np.sum(self.MatDist[dist_indices[:-1], dist_indices[1:]])

        return ruta, dist_rec


    def FindSolution(self):
        """
        Find the best TSP solution using random constructive method.
        """
        print("\nSearching solutions...")

        n_iters = 0
        min_dist = np.inf
        mejor_ruta = []
        while n_iters < self.__MaxIters:
            ruta_actual, dist = self.__FindRandomRoute()
            n_iters += 1

            if dist < min_dist:
                mejor_ruta = [nodo for nodo in ruta_actual]
                min_dist = dist

                print(f"#{n_iters} - {min_dist}", end="\n")

        self.best_route = mejor_ruta
        self.best_route_dist = min_dist

    def ShowCurrentSolution(self):
        FigureMap = Basemap(projection="merc", llcrnrlat=15, urcrnrlat=35,
                            llcrnrlon=-120, urcrnrlon=-85, resolution="i")

        FigureMap.drawcoastlines()
        FigureMap.drawstates()
        FigureMap.drawcountries()
        FigureMap.drawparallels(np.arange(10, 40, 10), labels=[1,0,0,0], fontsize = 10)
        FigureMap.drawmeridians(np.arange(-70, -120, -10), labels=[0,0,0,1], fontsize = 10)

        X_ruta, Y_ruta = zip(*[self.Nodes[nodo] for nodo in self.best_route])
        FigureMap.plot(X_ruta, Y_ruta, "-r", linewidth = 2.0, latlon = True, zorder = 0)
        FigureMap.scatter(X_ruta, Y_ruta, marker = "^", color = "blue", latlon = True, zorder = 1)

        plt.title(f"TSP-Constructivo   N={self.__N} Dist={self.best_route_dist:.2f}")

        plt.show()

if __name__ == "__main__":
    data = pd.read_csv("./data/worldcitiespop.csv",
                       usecols=["Country", "City", "Population",
                                "Latitude", "Longitude"])
    data = data.loc[data["Country"] == "mx", ["City", "Population", "Latitude", "Longitude"]]

    q_pop = data["Population"].quantile(0.995)
    data = data[data["Population"] >= q_pop]

    Longitude = data["Longitude"].to_numpy()
    Latitude = data["Latitude"].to_numpy()
    Ciudades = data["City"].to_list()

    TSP_Capitales = CA_TSP(Longitude, Latitude, Ciudades, "monterrey", 25_000)

    TSP_Capitales.FindSolution()
    print(TSP_Capitales.best_route, TSP_Capitales.best_route_dist)
    TSP_Capitales.ShowCurrentSolution()
