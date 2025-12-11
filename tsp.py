import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pprint import pprint

class TSP:
    def __init__(self, tsplib_file, optimal_tour_file=None, name=None):
        """
        Inicializa un problema TSP (asimétrico, en este caso) con un archivo TSPLIB95
        
        Args:
            tsplib_file (str): Dirección del archivo .tsp.
            optimal_tour_file (str, optional): Dirección a un archivo .opt.tour para calcular el costo óptimo.
        """
        # Cargamos la instancia del problema
        self.problem = tsplib95.load(tsplib_file)

        self.name = name
        
        # Grafo NetworkX con índices normalizados indexados desde el 0
        self.G = self.problem.get_graph(normalize=True)
        
        self.n = self.G.number_of_nodes()
        self.optimal_cost = None
        
        if optimal_tour_file:
            self._load_optimal_cost(optimal_tour_file)

    def _load_optimal_cost(self, tour_file):
        """Calcular costo optimo desde un archivo tour."""
        opt_sol = tsplib95.load(tour_file)
        
        original_nodes = list(self.problem.get_nodes())
        node_map = {node: i for i, node in enumerate(original_nodes)}
        
        min_cost = float('inf')
        for tour in opt_sol.tours:
            # Convert tour labels to normalized indices
            normalized_tour = [node_map[n] for n in tour]
            cost = self.evaluate_solution(normalized_tour)
            if cost < min_cost:
                min_cost = cost
        self.optimal_cost = min_cost

    def get_neighbors(self, i):
        """
        Para un nodo indexado por i, retorna una lista de tuplas (indice_vecino, peso)
        """
        if i not in self.G:
            return []
        
        neighbors = []
        for neighbor in self.G.neighbors(i):
            weight = self.G[i][neighbor].get('weight', float('inf'))
            neighbors.append((neighbor, weight))
        return neighbors

    def validate_solution_matrix(self, matrix: np.ndarray) -> list | None:
        """
        Dada una matriz de solución binaria, se valida su calidad como solución, y se retorna
        la secuencia de vértices correspondiente. 
        """
        # Verificación previa de branching
        if not (np.all(matrix.sum(axis=1) == 1) and np.all(matrix.sum(axis=0) == 1)):
            print("Fallo de grados.")
            return None

        node = 0
        tour = [0]
        visited = {0}

        for _ in range(self.n - 1):
            next_node = np.argmax(matrix[node, :])
            if (next_node in visited):
                print(f"Error en índice {node}, repetición")
                return None

            visited.add(next_node)
            tour.append(next_node)
            node = next_node

        # Verificamos cierre
        last_node = tour[-1]
        if matrix[last_node, 0] != 1:
            print("Error: Circuito no se cierra correctamente.")


        return tour

    def evaluate_solution(self, sequence):
        """
        Dada una secuencia solución, lista de indices de nodos, la evalua calculando el costo.
        """
        if len(sequence) != self.n:
            return float('inf'), 0

        cost = 0
        
        # Calculamos el costo del camino
        for k in range(len(sequence)):
            u = sequence[k]
            v = sequence[(k + 1) % self.n]
            
            if not self.G.has_edge(u, v):
                return float('inf'), 0
            
            w = self.G[u][v].get('weight', float('inf'))
            cost += w

        return cost

    def visualize(self, sequence=None, show_labels=True, title="Visualización"):
        """
        Visualiza el grafo del problema. Si se otorga una secuencia, resalta el circuito.
        """
        plt.figure(figsize=(10, 8))
        
        pos = {}
        if self.problem.is_depictable():
            for i in self.G.nodes:
                data = self.G.nodes[i]
                if 'coord' in data and data['coord'] is not None:
                    pos[i] = data['coord']
        
        if not pos:
            pos = nx.spring_layout(self.G)

        nx.draw_networkx_nodes(self.G, pos, node_size=300, node_color='lightblue')
        
        if show_labels:
            nx.draw_networkx_labels(self.G, pos, font_size=8)

        if sequence:
            tour_edges = []
            for k in range(len(sequence)):
                u = sequence[k]
                v = sequence[(k + 1) % self.n]
                tour_edges.append((u, v))
            
            if self.n < 100: 
                nx.draw_networkx_edges(self.G, pos, alpha=0.1, edge_color='gray')
            
            nx.draw_networkx_edges(self.G, pos, edgelist=tour_edges, edge_color='red', width=2)
        else:
            if self.n < 50:
                nx.draw_networkx_edges(self.G, pos, alpha=0.3)

        plt.title(f"{title} (Costo: {self.evaluate_solution(sequence) if sequence else 'N/A'})")
        plt.axis('off')
        plt.show()
