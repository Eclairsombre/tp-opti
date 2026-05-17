import random

from model import Route, Solution, VRPTWInstance
from utils import (
    route_distance,
    route_time_ok,
)


def random_solution(inst: VRPTWInstance, check_tw: bool = False) -> Solution:
    """
    Génère une solution initiale aléatoire.
    - check_tw=False : vérifie seulement capacité
    - check_tw=True  : vérifie aussi fenêtres de temps
    """
    seed = random.Random()

    client_ids = list(range(1, inst.n + 1))
    seed.shuffle(client_ids)

    routes = []

    for cid in client_ids:
        demand = inst.nodes[cid]["demand"]
        inserted = False

        # Essayer dans une route existante (ordre aléatoire)
        route_order = list(range(len(routes)))
        seed.shuffle(route_order)

        for ri in route_order:
            r = routes[ri]
            # Vérifier la capacité
            current_load = sum(inst.nodes[c]["demand"] for c in r.clients)
            if current_load + demand > inst.capacity:
                continue

            if check_tw:
                # Trouver la meilleure position dans la route
                best_pos = None
                best_cost = float("inf")
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


def greedy_solution(
    inst: VRPTWInstance,
    check_tw: bool = True,
) -> Solution:
    """
    Construit solution greedy (nearest neighbor).
    - check_tw=False : vérifie seulement capacité
    - check_tw=True  : vérifie aussi fenêtres de temps
    """

    seed = random.Random()

    unvisited = list(range(1, inst.n + 1))
    seed.shuffle(unvisited)

    routes = []

    while unvisited:
        route_clients = []
        current_load = 0
        current_pos = 0  # dépôt
        current_time = inst.depot["ready"]

        while True:
            candidates = []

            for cid in unvisited:
                demand = inst.nodes[cid]["demand"]

                # capacité
                if current_load + demand > inst.capacity:
                    continue

                travel = inst.dist[current_pos][cid]
                node = inst.nodes[cid]

                if check_tw:
                    arr = current_time + travel

                    if arr > node["due"]:
                        continue

                candidates.append((travel, cid))

            if not candidates:
                break

            # nearest neighbor
            best_dist = min(d for d, _ in candidates)

            # tie-break random
            best_candidates = [cid for d, cid in candidates if d == best_dist]

            best = seed.choice(best_candidates)

            unvisited.remove(best)

            route_clients.append(best)
            current_load += inst.nodes[best]["demand"]

            travel = inst.dist[current_pos][best]
            current_time += travel

            if check_tw:
                current_time = max(
                    current_time,
                    inst.nodes[best]["ready"],
                )

            current_time += inst.nodes[best]["service"]

            current_pos = best

        if route_clients:
            routes.append(Route(route_clients))

        elif unvisited:
            # évite boucle infinie
            cid = unvisited.pop(0)
            routes.append(Route([cid]))

    return Solution(routes)
