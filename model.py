import math

import numpy as np


class VRPTWInstance:
    """Représente une instance du problème VRPTW."""

    def __init__(self, data: dict):
        self.name = data["name"]
        self.capacity = data["capacity"]
        self.depot = data["depot"]
        self.clients = data["clients"]
        self.n = len(self.clients)

        self.nodes = [self.depot] + self.clients

        self.dist = self._compute_distance_matrix()

    def _compute_distance_matrix(self):
        """Calcule la matrice de distances euclidiennes."""
        n = len(self.nodes)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = self.nodes[i]["x"] - self.nodes[j]["x"]
                dy = self.nodes[i]["y"] - self.nodes[j]["y"]
                dist[i][j] = math.sqrt(dx * dx + dy * dy)
        return dist

    def client_ids(self):
        """Retourne la liste des IDs de clients."""
        return list(range(1, self.n + 1))


class Route:
    """Une liste ordonnée de clients (sans le dépôt en début/fin)."""

    def __init__(self, client_ids=None):
        self.clients = client_ids if client_ids is not None else []

    def copy(self):
        return Route(list(self.clients))

    def __repr__(self):
        return f"Route({self.clients})"

    def __len__(self):
        return len(self.clients)


class Solution:
    """Ensemble de routes représentant une solution complète."""

    def __init__(self, routes=None):
        self.routes = routes if routes is not None else []

    def copy(self):
        return Solution([r.copy() for r in self.routes])

    def num_routes(self):
        return len(self.routes)

    def all_clients(self):
        """Retourne tous les clients visités dans l'ordre des routes."""
        result = []
        for r in self.routes:
            result.extend(r.clients)
        return result

    def __repr__(self):
        return f"Solution({len(self.routes)} routes, clients={self.all_clients()})"
