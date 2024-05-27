import networkx as nx
import pandas as pd
import pickle
import json

import customtkinter as ctk
from tkinter import messagebox

from math import exp
from random import shuffle, sample


class TSP_SimulatedAnnealing:
    def __init__(self, graph:nx.Graph, nodes_to_visit: list[str], init_temp: float, min_temp: float, cool_rate: float, max_iters: int):
        """
        Initializes the simulated annealing algorithm for solving the Traveling Salesman Problem (TSP).

        Parameters:
        graph (nx.Graph): A Graph containing nodes, edges and weights for each edge
        distance_matrix (pd.DataFrame): A DataFrame containing the distances between nodes.
        nodes_to_visit (list[str]): A list of nodes to be visited.
        init_temp (float): Initial temperature for the annealing process.
        min_temp (float): Minimum temperature for the annealing process.
        cool_rate (float): Cooling rate for the temperature.
        max_iters (int): Maximum number of iterations to perform.
        """
        self.__graph = graph
        self.__dist_matrix = self.__get_distances_matrix()
        self.__nodes_to_visit = nodes_to_visit
        self.__temperature = init_temp
        self.__min_temperature = min_temp
        self.__cool_rate = cool_rate
        self.__max_iters = max_iters

        self.__size_nodes = len(nodes_to_visit)


    def __get_distances_matrix(self):
        """
        Computes the distances matrix for all pairs of nodes in the graph.

        This private method generates a matrix where each element represents the shortest path 
        distance between a pair of nodes in the graph. The distances are computed using Dijkstra's 
        algorithm, assuming the graph is weighted.

        Returns:
            pd.DataFrame: A DataFrame where the rows and columns correspond to the graph's nodes, 
                          and each element [i, j] contains the shortest path distance from node i to node j.

        Notes:
            - The method utilizes NetworkX's shortest_path_length function with a weight parameter 
              to account for edge weights.
            - The resulting DataFrame is symmetric if the graph is undirected, with zeros on the diagonal.
        """
        matrix_distance = pd.DataFrame(index=self.__graph.nodes(), columns=self.__graph.nodes())
        for origin_node in self.__graph.nodes():
            length = nx.single_source_dijkstra_path_length(self.__graph, origin_node, weight='weight')
            for target_node, distance in length.items():
                matrix_distance.at[origin_node, target_node] = distance
        return matrix_distance.astype(float)

    def __generate_initial_path(self):
        """
        Generates the initial path by shuffling the list of nodes to visit.

        Returns:
        list: A shuffled list representing the initial path.
        """
        path = self.__nodes_to_visit[:]
        shuffle(path)
        return path
    
    def __generate_new_path(self, path):
        """
        Generates a new path by swapping two randomly selected nodes in the current path.

        Parameters:
        path (list): The current path.

        Returns:
        list: A new path with two nodes swapped.
        """
        new_path = path[:]
        i, j = sorted(sample(range(self.__size_nodes), 2))
        new_path[i:j] = reversed(new_path[i:j])
        return new_path
    
    def __compute_path_cost(self, path):
        """
        Computes the total cost of the given path based on the distance matrix.

        Parameters:
        path (list): The path for which the cost is to be computed.

        Returns:
        float: The total cost of the path.
        """
        total_cost = 0.0
        for k in range(len(path) - 1):
            total_cost += self.__dist_matrix.loc[path[k], path[k+1]]
        total_cost += self.__dist_matrix.loc[path[-1], path[0]]
        return total_cost

    def __acceptance_condition(self, new_cost, old_cost, temp):
        """
        Determines if the new path should be accepted based on the cost difference and the current temperature.

        Parameters:
        new_cost (float): The cost of the new path.
        old_cost (float): The cost of the current path.
        temp (float): The current temperature.

        Returns:
        float: The acceptance probability.
        """
        if new_cost < old_cost:
            return 1.0
        return exp((old_cost - new_cost) / temp)

    def __get_full_tsp_path(self):
        """
        Converts the best path found into a full path including all intermediate nodes.

        This method uses the shortest path between each pair of nodes in the best path to
        generate a complete route, ensuring all nodes are visited in sequence.
        """
        complete_best_route = []

        for i in range(self.__size_nodes - 1):
            complete_best_route += nx.shortest_path(self.__graph, self.best_path[i], self.best_path[i+1], weight="weight")
        complete_best_route += nx.shortest_path(self.__graph, self.best_path[-1], self.best_path[0], weight="weight")

        self.best_path = complete_best_route
        self.best_cost = self.__compute_path_cost(self.best_path) 
    
    def find_solution(self):
        """
        Executes the simulated annealing algorithm to find the best solution for the TSP.

        Returns:
        tuple: A tuple containing the best path and the best cost.
        """
        self.__current_path = self.__generate_initial_path()
        actual_cost = self.__compute_path_cost(self.__current_path)

        self.best_path = self.__current_path[:]
        self.best_cost = actual_cost

        for iteration in range(self.__max_iters):
            if self.__temperature >= self.__min_temperature:
                new_path = self.__generate_new_path(self.__current_path)
                new_cost = self.__compute_path_cost(new_path)

                if self.__acceptance_condition(new_cost, actual_cost, self.__temperature):
                    self.__current_path = new_path[:]
                    actual_cost = new_cost

                    if new_cost < self.best_cost:
                        self.best_path = self.__current_path[:]
                        self.best_cost = actual_cost

                self.__temperature *= self.__cool_rate
            else:
                break

        self.__get_full_tsp_path()

        return self.best_path, self.best_cost
    

class TSP_GUI:
    def __init__(self, root, graph):
        self.root = root
        self.root.title("TSP Solver with Simulated Annealing")
        self.__height, self.__width = 1000, 1000
        self.__canvas = ctk.CTkCanvas(self.root, width=self.__width, height=self.__height, bg="white")
        self.__canvas.place(x = 0, y = 0)
        
        self.__graph = graph
        self.__loc_stations = self.__get_trans_stations_locations()
        self.__selected_nodes = []
        
        self.__canvas.bind("<Button-1>", self.__select_node)
        self.__solve_button = ctk.CTkButton(self.root, text="Solve TSP", command=self.__solve_tsp)
        self.__clear_button = ctk.CTkButton(self.root, text="Clear graph", command=self.__clear_graph)
        self.__output_text = ctk.CTkTextbox(self. root, width = 150, height = 100)
        self.__output_text.configure(state = "disabled")

        self.__solve_button.place(x = 1035, y = 100)
        self.__clear_button.place(x = 1035, y = 150)
        self.__output_text.place(x = 1035, y = 300)

        self.__nodes_size = 3
        self.__select_nodes_size = 5
        self.__lines_width = 2
        
        self.__draw_graph()

    def __get_trans_stations_locations(self):
        json_file = "./output_metro/travel_times_metro.json"
        with open(json_file) as input_json:
            dict_times_metro = json.load(input_json)

        location_stations = dict()
        for route_id in dict_times_metro.keys():
            stations_data = dict_times_metro[route_id]
            for station_data in stations_data:
                location_stations[station_data[0]] = tuple(station_data[2:])

        lats, longs = [list(coords_tuple) for coords_tuple in zip(*list(location_stations.values()))]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(longs), max(longs)
        norm_lats = [0.05 * self.__height + 0.85 * self.__height * (lat - min_lat)/(max_lat - min_lat) for lat in lats]
        norm_longs = [0.05 * self.__width + 0.85 * self.__width * (lon - min_lon)/(max_lon - min_lon) for lon in longs]
        for n, station in enumerate(location_stations.keys()):
            location_stations[station] = (norm_longs[n], norm_lats[n])

        return location_stations
        
    def __draw_graph(self):
        for node, (x, y) in self.__loc_stations.items():
            self.__canvas.create_oval(x - self.__nodes_size, 
                                      y - self.__nodes_size,
                                      x + self.__nodes_size,
                                      y + self.__nodes_size, fill="black", tags=f"node_{node}")
        for edge in self.__graph.edges():
            x1, y1 = self.__loc_stations[edge[0]]
            x2, y2 = self.__loc_stations[edge[1]]
            self.__canvas.create_line(x1, y1, x2, y2,
                                    fill="blue",
                                    width = self.__lines_width)
        
    def __select_node(self, event):
        x, y = event.x, event.y
        for node, (node_x, node_y) in self.__loc_stations.items():
            if (node_x - 5 < x < node_x + 5) and (node_y - 5 < y < node_y + 5):
                self.__selected_nodes.append(node)
                self.__canvas.create_oval(node_x - self.__select_nodes_size,
                                          node_y - self.__select_nodes_size,
                                          node_x + self.__select_nodes_size,
                                          node_y + self.__select_nodes_size,
                                          outline="red", 
                                          width=2,
                                          tags=f"selected_node_{node}")
                break
        
    def __solve_tsp(self):
        if len(self.__selected_nodes) < 2:
            messagebox.showwarning("Insufficient Nodes", "Please select at least two nodes to solve the TSP.")
            return
        
        tsp_sa = TSP_SimulatedAnnealing(self.__graph, self.__selected_nodes, init_temp = 1000.0, min_temp = 1e-6, cool_rate = 0.995, max_iters = 5_000)
        best_path, best_cost = tsp_sa.find_solution()
        
        self.__draw_solution(best_path, best_cost)
        
    def __draw_solution(self, path, cost):
        for i in range(len(path) - 1):
            x1, y1 = self.__loc_stations[path[i]]
            x2, y2 = self.__loc_stations[path[i+1]]
            self.__canvas.create_line(x1, y1, x2, y2,
                                      fill="red",
                                      tags="solution",
                                      width = self.__lines_width)
        x1, y1 = self.__loc_stations[path[-1]]
        x2, y2 = self.__loc_stations[path[0]]
        self.__canvas.create_line(x1, y1, x2, y2,
                                  fill="red",
                                  tags="solution",
                                  width = self.__lines_width)
        
        self.__output_text.configure(state = "normal")
        self.__output_text.insert("1.0", f"Best cost: {cost:.3f} hours")
        self.__output_text.configure(state = "disabled")

    def __clear_graph(self):
        for node in self.__selected_nodes:
            self.__canvas.delete(f"selected_node_{node}")
        self.__selected_nodes.clear()
        self.__canvas.delete("solution")
        
        self.__output_text.configure(state = "normal")
        self.__output_text.delete('1.0', 'end')
        self.__output_text.configure(state = "disabled")


if __name__ == "__main__":
    metro_graph = pickle.load(open("./output_metro/metro_graph.pickle", "rb"))

    root = ctk.CTk()
    root.geometry("1200x1000")
    root.resizable(width = False, height = False)

    App = TSP_GUI(root = root, graph = metro_graph)
    App.root.mainloop()