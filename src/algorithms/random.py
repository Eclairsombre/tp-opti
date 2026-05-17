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
