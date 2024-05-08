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
    - PosOrDist: True if the input is the position matrix or False if it's the distance matrix
    """
    def __init__(self, Matrix:np.ndarray, Names:list[str],
                 Ini:str, MaxIters:int, PosOrDist:bool = True) -> None:
        self.__Names : list[str] = Names
        self.__N : int = len(self.__Names)

        if PosOrDist:
            self.__Long, self.__Lat = np.copy(Matrix[:,0].flatten()), np.copy(Matrix[:,1].flatten())
            self.Nodes = {}
            for p in range(self.__N):
                self.Nodes[self.__Names[p]] = (self.__Long[p], self.__Lat[p])
            
            self.MatDist = self.__ObtainDistanceMatrix()
        else:
            self.MatDist = np.copy(Matrix)

        self.__Ini : str = Ini
        self.__init_index = self.__Names.index(self.__Ini)
        self.__availableIndex = np.delete(np.arange(self.__N), self.__init_index)
        self.__MaxIters : int = MaxIters


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

        return 2.0 * 6371.0 * np.arcsin(arg_arcsin)

    def __FindRandomRoute(self):
        """
        Find a random route for the TSP.

        Returns:
        - route: list representing the random route.
        - dist_rec: total distance of the random route.
        """
        TotalNewSelections = self.__N - 1
        route = [self.__Ini] + TotalNewSelections*["*"] + [self.__Ini]
        dist_rec = 0.0

        random_indexes = np.random.choice(self.__availableIndex, TotalNewSelections, replace=False)
        dist_indices = np.concatenate(([self.__init_index], random_indexes, [self.__init_index]))

        route[1:-1] = [self.__Names[k] for k in random_indexes]
        dist_rec = np.sum(self.MatDist[dist_indices[:-1], dist_indices[1:]])

        return route, dist_rec


    def FindSolution(self):
        """
        Find the best TSP solution using random constructive method.
        """
        print("\nSearching solutions...")

        n_iters = 0
        min_dist = np.inf
        while n_iters < self.__MaxIters:
            actual_route, dist = self.__FindRandomRoute()
            n_iters += 1

            if dist < min_dist:
                best_route = [nodo for nodo in actual_route]
                min_dist = dist

                print(f"#{n_iters} - {min_dist}", end="\n")

        self.best_route = best_route
        self.best_route_dist = min_dist

    def ShowCurrentSolution(self):
        FigureMap = Basemap(projection="merc", llcrnrlat=15, urcrnrlat=35,
                            llcrnrlon=-120, urcrnrlon=-85, resolution="i")

        FigureMap.drawcoastlines()
        FigureMap.drawstates()
        FigureMap.drawcountries()
        FigureMap.drawparallels(np.arange(10, 40, 10), labels=[1,0,0,0], fontsize = 10)
        FigureMap.drawmeridians(np.arange(-70, -120, -10), labels=[0,0,0,1], fontsize = 10)

        X_route, Y_route = zip(*[self.Nodes[index] for index in self.best_route])
        FigureMap.plot(X_route, Y_route, "-r", linewidth = 2.0, latlon = True, zorder = 0)
        FigureMap.scatter(X_route, Y_route, marker = "^", color = "blue", latlon = True, zorder = 1)

        plt.title(f"TSP-Constructivo   N={self.__N} Dist={self.best_route_dist:.2f}")

        plt.show()

if __name__ == "__main__":
    data = pd.read_csv("./data/worldcitiespop.csv",
                       usecols=["Country", "City", "Population",
                                "Latitude", "Longitude"])
    data = data.loc[data["Country"] == "mx", ["City", "Population", "Latitude", "Longitude"]]

    q_pop = data["Population"].quantile(0.9945)
    data = data[data["Population"] >= q_pop]

    Positions = np.vstack((data["Longitude"].to_numpy(), data["Latitude"].to_numpy())).T
    Names = data["City"].to_list()

    df_output = pd.DataFrame({"Cities": Names, 
                              "Longitude": Positions[:,0],
                              "Latitude": Positions[:,1]})
    df_output.to_csv("./data/Top10PopCities.csv", index = False)

    """
    Dists = np.array([[0, 10, 15, 20],
                      [10, 0, 35, 25],
                      [15, 35, 0, 30],
                      [20, 25, 30, 0]], dtype=np.float32)
    Names = ["A", "B", "C", "D"]
    """

    TSP_Capitales = CA_TSP(Positions, Names, "monterrey", 50_000, PosOrDist=True)

    TSP_Capitales.FindSolution()
    print(TSP_Capitales.best_route, TSP_Capitales.best_route_dist)
    TSP_Capitales.ShowCurrentSolution()