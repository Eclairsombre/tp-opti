import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import altair as alt
    from visualisation import plot_routes

    return alt, mo, plot_routes


@app.cell(hide_code=True)
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.patches as mpatches
    import random
    import time
    import copy
    import math
    import os
    import glob
    from itertools import permutations
    from collections import defaultdict

    return glob, math, np, os, pd, plt, random


@app.cell(hide_code=True)
def _(random):
    SEED = 1
    random.seed(SEED)
    return


@app.function(hide_code=True)
def parse_vrp_file(filepath):
    """
    Lit un fichier .vrp et retourne un dictionnaire contenant :
      - 'name'        : nom du problème
      - 'capacity'    : capacité max des véhicules
      - 'depot'       : dict {x, y, ready, due}
      - 'clients'     : liste de dicts {id, x, y, ready, due, demand, service}
    """
    data = {}
    clients = []
    depot = None
    mode = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('NAME:'):
                data['name'] = line.split(':', 1)[1].strip()
            elif line.startswith('NB_CLIENTS:'):
                data['nb_clients'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('MAX_QUANTITY:'):
                data['capacity'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('DATA_DEPOTS'):
                mode = 'depot'
            elif line.startswith('DATA_CLIENTS'):
                mode = 'clients'
            elif mode == 'depot' and not line.startswith('DATA_'):
                parts = line.split()
                if len(parts) >= 5:
                    depot = {
                        'id': 0,
                        'name': parts[0],
                        'x': float(parts[1]),
                        'y': float(parts[2]),
                        'ready': float(parts[3]),
                        'due': float(parts[4]),
                        'demand': 0,
                        'service': 0
                    }
            elif mode == 'clients' and not line.startswith('DATA_'):
                parts = line.split()
                if len(parts) >= 7:
                    idx = len(clients) + 1
                    clients.append({
                        'id': idx,
                        'name': parts[0],
                        'x': float(parts[1]),
                        'y': float(parts[2]),
                        'ready': float(parts[3]),
                        'due': float(parts[4]),
                        'demand': float(parts[5]),
                        'service': float(parts[6])
                    })

    data['depot'] = depot
    data['clients'] = clients
    return data


@app.cell(hide_code=True)
def _(glob, mo, os):
    DATA_DIR = r"./data"
    vrp_files = sorted(glob.glob(os.path.join(DATA_DIR, "data*.vrp")))

    datasets = {}
    rows = []

    for f in vrp_files:
        name = os.path.splitext(os.path.basename(f))[0]
        datasets[name] = parse_vrp_file(f)
        d = datasets[name]

        rows.append({
            "Dataset": name,
            "Clients": d["nb_clients"],
            "Capacity": d["capacity"],
            "Depot X": round(d["depot"]["x"]),
            "Depot Y": round(d["depot"]["y"]),
        })


    mo.vstack([
        mo.md("## Données"),
        mo.ui.table(rows),
    ])
    return (datasets,)


@app.cell
def _(math, np):
    class VRPTWInstance:
        """Représente une instance du problème VRPTW."""

        def __init__(self, data: dict):
            self.name = data['name']
            self.capacity = data['capacity']
            self.depot = data['depot']
            self.clients = data['clients']
            self.n = len(self.clients)

            # Construire les nœuds : 0=dépôt, 1..n = clients
            self.nodes = [self.depot] + self.clients

            # Matrice de distances
            self.dist = self._compute_distance_matrix()

        def _compute_distance_matrix(self):
            """Calcule la matrice de distances euclidiennes."""
            n = len(self.nodes)
            dist = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dx = self.nodes[i]['x'] - self.nodes[j]['x']
                    dy = self.nodes[i]['y'] - self.nodes[j]['y']
                    dist[i][j] = math.sqrt(dx*dx + dy*dy)
            return dist

        def client_ids(self):
            """Retourne la liste des IDs de clients (1..n)."""
            return list(range(1, self.n + 1))


    class Route:
        """Une route = liste ordonnée de clients (sans le dépôt en début/fin)."""

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

    return Route, Solution, VRPTWInstance


@app.cell(hide_code=True)
def _(datasets, mo):
    dataset = mo.ui.dropdown(options=datasets)
    return (dataset,)


@app.cell
def _(dataset, mo):
    mo.md(f"""
    Sélectionner un jeu de données : {dataset}
    """)
    return


@app.cell(hide_code=True)
def _(VRPTWInstance, dataset, mo, pd):
    instance = VRPTWInstance(dataset.value)

    clients_df = pd.DataFrame(instance.clients)

    mo.vstack([
        mo.md("## Clients du dataset sélectionné"),
        mo.ui.table(clients_df),
    ])
    return clients_df, instance


@app.cell(hide_code=True)
def _(alt, clients_df):
    _chart = (
        alt.Chart(clients_df)
        .mark_point()
        .encode(
            x=alt.X(field='x', type='quantitative', sort='ascending'),
            y=alt.Y(field='y', type='quantitative', sort='ascending'),
            color=alt.Color(field='demand', type='quantitative'),
            tooltip=[
                alt.Tooltip(field='x', format=',.2f'),
                alt.Tooltip(field='y', format=',.2f'),
                alt.Tooltip(field='demand', format=',.2f')
            ]
        )
        .properties(
            title='Carte des clients',
            height=290,
            width='container',
            config={
                'axis': {
                    'grid': True
                }
            }
        )
    )
    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validation des contraintes

    Deux types de contraintes :
    1. **Capacité** : $\sum_{i \in \text{route}} q_i \leq C$
    2. **Fenêtres de temps** : $e_i \leq t_i \leq l_i$ (avec attente si arrivée avant $e_i$)
    """)
    return


@app.cell
def _(Route, Solution, VRPTWInstance, math):
    def min_vehicles_lower_bound(inst):
        """Calcule la borne inférieure du nombre de véhicules nécessaires."""
        total_demand = sum(c['demand'] for c in inst.clients)
        return math.ceil(total_demand / inst.capacity)

    def route_capacity_ok(route: Route, inst: VRPTWInstance) -> bool:
        """Vérifie que la capacité de la route n'est pas dépassée."""
        total = sum(inst.nodes[c]['demand'] for c in route.clients)
        return total <= inst.capacity


    def route_time_ok(route: Route, inst: VRPTWInstance) -> (bool, float):
        """
        Vérifie les fenêtres de temps d'une route.
        Retourne (valide, temps_retour_depot).
        Le véhicule peut attendre si e_i n'est pas encore atteint.
        """
        t = inst.depot['ready']
        prev = 0  # index dépôt
        for cid in route.clients:
            t += inst.dist[prev][cid]
            node = inst.nodes[cid]
            if t > node['due']:
                return False, t  # Arrivée trop tard
            t = max(t, node['ready'])  # Attendre si arrivée trop tôt
            t += node['service']
            prev = cid
        # Retour au dépôt
        t += inst.dist[prev][0]
        if t > inst.depot['due']:
            return False, t
        return True, t


    def route_time_violation(route: Route, inst: VRPTWInstance) -> float:
        """
        Calcule la violation de fenêtre de temps (somme des retards).
        Retourne 0 si tout est valide.
        """
        t = inst.depot['ready']
        prev = 0
        violation = 0.0
        for cid in route.clients:
            t += inst.dist[prev][cid]
            node = inst.nodes[cid]
            if t > node['due']:
                violation += t - node['due']
            t = max(t, node['ready'])
            t += node['service']
            prev = cid
        t += inst.dist[prev][0]
        if t > inst.depot['due']:
            violation += t - inst.depot['due']
        return violation


    def solution_is_valid(sol: Solution, inst: VRPTWInstance, check_tw: bool = True) -> bool:
        """Vérifie qu'une solution est complète et valide."""
        visited = []
        for r in sol.routes:
            if not route_capacity_ok(r, inst):
                return False
            if check_tw:
                ok, _ = route_time_ok(r, inst)
                if not ok:
                    return False
            visited.extend(r.clients)
        # Tous les clients visités une seule fois
        return sorted(visited) == list(range(1, inst.n + 1))


    def solution_total_violation(sol: Solution, inst: VRPTWInstance) -> float:
        """Retourne la violation totale des fenêtres de temps d'une solution."""
        return sum(route_time_violation(r, inst) for r in sol.routes)


    return route_time_ok, solution_is_valid, solution_total_violation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fonction objectif

    $$D_{total} = \sum_{k=1}^{K} \left( d_{0, \pi_k(1)} + \sum_{j=1}^{|\pi_k|-1} d_{\pi_k(j), \pi_k(j+1)} + d_{\pi_k(|\pi_k|), 0} \right)$$

    où $\pi_k$ est la séquence de clients de la route $k$.
    """)
    return


@app.cell
def _(Route, Solution, VRPTWInstance, solution_total_violation):
    def route_distance(route: Route, inst: VRPTWInstance) -> float:
        """Distance totale d'une route (dépôt → clients → dépôt)."""
        if not route.clients:
            return 0.0
        d = inst.dist[0][route.clients[0]]
        for i in range(len(route.clients) - 1):
            d += inst.dist[route.clients[i]][route.clients[i+1]]
        d += inst.dist[route.clients[-1]][0]
        return d


    def total_distance(sol: Solution, inst: VRPTWInstance) -> float:
        """Distance totale de toute la solution."""
        return sum(route_distance(r, inst) for r in sol.routes)


    def penalized_cost(sol: Solution, inst: VRPTWInstance, penalty: float = 1000.0) -> float:
        """
        Coût pénalisé = distance + penalty * violation_TW.
        Utilisé pour guider les métaheuristiques même en cas de violation.
        """
        dist = total_distance(sol, inst)
        violation = solution_total_violation(sol, inst)
        return dist + penalty * violation

    return route_distance, total_distance


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Générateur aléatoire de solutions

    Construction par insertion aléatoire :
    1. Mélanger aléatoirement les clients
    2. Pour chaque client, essayer de l'insérer dans une route existante (contrainte capacité)
    3. Si impossible, ouvrir une nouvelle route
    """)
    return


@app.cell
def _(Route, Solution, VRPTWInstance, random, route_distance, route_time_ok):
    def random_solution(inst: VRPTWInstance, check_tw: bool = False, rng=None) -> Solution:
        """
        Génère une solution initiale aléatoire.
        - check_tw=False : ne vérifie que la capacité (phase 1)
        - check_tw=True  : vérifie aussi les fenêtres de temps (phase 2)
        """
        if rng is None:
            rng = random.Random()

        client_ids = list(range(1, inst.n + 1))
        rng.shuffle(client_ids)

        routes = []

        for cid in client_ids:
            demand = inst.nodes[cid]['demand']
            inserted = False

            # Essayer dans une route existante (ordre aléatoire)
            route_order = list(range(len(routes)))
            rng.shuffle(route_order)

            for ri in route_order:
                r = routes[ri]
                # Vérifier la capacité
                current_load = sum(inst.nodes[c]['demand'] for c in r.clients)
                if current_load + demand > inst.capacity:
                    continue

                if check_tw:
                    # Trouver la meilleure position dans la route
                    best_pos = None
                    best_cost = float('inf')
                    for pos in range(len(r.clients) + 1):
                        new_clients = r.clients[:pos] + [cid] + r.clients[pos:]
                        tmp_route = Route(new_clients)
                        ok, _ = route_time_ok(tmp_route, inst)
                        if ok:
                            cost = route_distance(tmp_route, inst)
                            if cost < best_cost:
                                best_cost = cost
                                best_pos = pos
                    if best_pos is not None:
                        r.clients.insert(best_pos, cid)
                        inserted = True
                        break
                else:
                    r.clients.append(cid)
                    inserted = True
                    break

            if not inserted:
                routes.append(Route([cid]))

        # Supprimer les routes vides
        routes = [r for r in routes if r.clients]
        return Solution(routes)


    def greedy_solution(inst: VRPTWInstance) -> Solution:
        """
        Construit une solution par heuristique greedy (nearest neighbor).
        """
        unvisited = set(range(1, inst.n + 1))
        routes = []

        while unvisited:
            route_clients = []
            current_load = 0
            current_pos = 0  # dépôt
            current_time = inst.depot['ready']

            while True:
                best = None
                best_dist = float('inf')

                for cid in unvisited:
                    d = inst.demand if hasattr(inst, 'demand') else inst.nodes[cid]['demand']
                    demand = inst.nodes[cid]['demand']
                    if current_load + demand > inst.capacity:
                        continue

                    travel = inst.dist[current_pos][cid]
                    arr = current_time + travel
                    node = inst.nodes[cid]

                    if arr > node['due']:
                        continue

                    if travel < best_dist:
                        best_dist = travel
                        best = cid

                if best is None:
                    break

                unvisited.remove(best)
                route_clients.append(best)
                current_load += inst.nodes[best]['demand']
                travel = inst.dist[current_pos][best]
                current_time += travel
                current_time = max(current_time, inst.nodes[best]['ready'])
                current_time += inst.nodes[best]['service']
                current_pos = best

            if route_clients:
                routes.append(Route(route_clients))
            elif unvisited:
                # Forcer l'ajout du client avec violation pour ne pas boucler
                cid = next(iter(unvisited))
                unvisited.remove(cid)
                routes.append(Route([cid]))

        return Solution(routes)

    return greedy_solution, random_solution


@app.cell
def _(
    greedy_solution,
    instance,
    mo,
    plot_routes,
    plt,
    random,
    random_solution,
    solution_is_valid,
    solution_total_violation,
    total_distance,
):
    rng_test = random.Random(42)
    sol_rand = random_solution(instance, check_tw=False, rng=rng_test)
    sol_greedy = greedy_solution(instance)

    print(f"=== Solution aléatoire (sans TW) ===")
    print(f"  Nombre de routes  : {sol_rand.num_routes()}")
    print(f"  Distance totale   : {total_distance(sol_rand, instance):.2f}")
    print(f"  Violation TW      : {solution_total_violation(sol_rand, instance):.2f}")
    print(f"  Valide (cap)      : {solution_is_valid(sol_rand, instance, check_tw=False)}")

    print(f"\n=== Solution greedy ===")
    print(f"  Nombre de routes  : {sol_greedy.num_routes()}")
    print(f"  Distance totale   : {total_distance(sol_greedy, instance):.2f}")
    print(f"  Violation TW      : {solution_total_violation(sol_greedy, instance):.2f}")
    print(f"  Valide (avec TW)  : {solution_is_valid(sol_greedy, instance, check_tw=True)}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    plot_routes(
        sol_rand,
        instance,
        title="Solution aléatoire",
        ax=axes[0]
    )

    plot_routes(
        sol_greedy,
        instance,
        title="Solution greedy",
        ax=axes[1]
    )

    plt.tight_layout()

    mo.vstack([
        mo.md("## Visualisation des solutions"),
        mo.as_html(fig),
    ])
    return


if __name__ == "__main__":
    app.run()
