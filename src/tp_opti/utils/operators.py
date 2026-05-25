import random

from tp_opti.model import Route, Solution, VRPTWInstance
from tp_opti.utils.validators import penalized_cost, route_capacity_ok, total_distance


def two_opt_intra(
    sol: Solution, inst: VRPTWInstance, check_tw: bool = True
) -> Solution:
    """
    2-opt intra-route : inverse un segment [i..j] dans une route.
    Retourne la meilleure solution voisine trouvée.
    """
    best_sol = sol
    best_cost = penalized_cost(sol, inst) if check_tw else total_distance(sol, inst)

    for ri, route in enumerate(sol.routes):
        n = len(route.clients)
        if n < 4:
            continue
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_clients = (
                    route.clients[:i]
                    + list(reversed(route.clients[i : j + 1]))
                    + route.clients[j + 1 :]
                )
                new_route = Route(new_clients)

                if not route_capacity_ok(new_route, inst):
                    continue

                new_sol = sol.copy()
                new_sol.routes[ri] = new_route

                cost = (
                    penalized_cost(new_sol, inst)
                    if check_tw
                    else total_distance(new_sol, inst)
                )
                if cost < best_cost:
                    best_cost = cost
                    best_sol = new_sol

    return best_sol


def relocate(sol: Solution, inst: VRPTWInstance, check_tw: bool = True) -> Solution:
    """
    Relocate : déplace un client d'une route vers la meilleure position d'une autre route.
    """
    best_sol = sol
    best_cost = penalized_cost(sol, inst) if check_tw else total_distance(sol, inst)

    for ri in range(len(sol.routes)):
        for ci, cid in enumerate(sol.routes[ri].clients):
            demand = inst.nodes[cid]["demand"]

            for rj in range(len(sol.routes)):
                if ri == rj:
                    continue

                # Vérifier la capacité de la route destination
                load_rj = sum(inst.nodes[c]["demand"] for c in sol.routes[rj].clients)
                if load_rj + demand > inst.capacity:
                    continue

                for pos in range(len(sol.routes[rj].clients) + 1):
                    new_sol = sol.copy()
                    new_sol.routes[ri].clients.pop(ci)
                    new_sol.routes[rj].clients.insert(pos, cid)

                    # Supprimer routes vides
                    new_sol.routes = [r for r in new_sol.routes if r.clients]

                    cost = (
                        penalized_cost(new_sol, inst)
                        if check_tw
                        else total_distance(new_sol, inst)
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_sol = new_sol

    return best_sol


def swap(sol: Solution, inst: VRPTWInstance, check_tw: bool = True) -> Solution:
    """
    Swap : échange deux clients entre deux routes différentes.
    """
    best_sol = sol
    best_cost = penalized_cost(sol, inst) if check_tw else total_distance(sol, inst)

    for ri in range(len(sol.routes)):
        for ci, cid_i in enumerate(sol.routes[ri].clients):
            for rj in range(ri + 1, len(sol.routes)):
                for cj, cid_j in enumerate(sol.routes[rj].clients):
                    # Vérifier capacités après échange
                    load_ri = sum(
                        inst.nodes[c]["demand"] for c in sol.routes[ri].clients
                    )
                    load_rj = sum(
                        inst.nodes[c]["demand"] for c in sol.routes[rj].clients
                    )

                    new_load_ri = (
                        load_ri
                        - inst.nodes[cid_i]["demand"]
                        + inst.nodes[cid_j]["demand"]
                    )
                    new_load_rj = (
                        load_rj
                        - inst.nodes[cid_j]["demand"]
                        + inst.nodes[cid_i]["demand"]
                    )

                    if new_load_ri > inst.capacity or new_load_rj > inst.capacity:
                        continue

                    new_sol = sol.copy()
                    new_sol.routes[ri].clients[ci] = cid_j
                    new_sol.routes[rj].clients[cj] = cid_i

                    cost = (
                        penalized_cost(new_sol, inst)
                        if check_tw
                        else total_distance(new_sol, inst)
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_sol = new_sol

    return best_sol


def neighbor_operator(
    sol: Solution, inst: VRPTWInstance, op, check_tw: bool = True
) -> Solution:
    """
    Génère un voisin aléatoire en appliquant un opérateur parmi ["2opt", "relocate", "swap"]
    """
    rng = random.Random()
    op = rng.choice(["2opt", "relocate", "swap"])
    new_sol = sol.copy()

    if op == "2opt":
        routes_with_4 = [i for i, r in enumerate(new_sol.routes) if len(r) >= 4]
        if not routes_with_4:
            op = "relocate"
        else:
            ri = rng.choice(routes_with_4)
            n = len(new_sol.routes[ri].clients)
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            new_sol.routes[ri].clients[i : j + 1] = reversed(
                new_sol.routes[ri].clients[i : j + 1]
            )

    if op == "relocate":
        if len(new_sol.routes) >= 2:
            ri = rng.randint(0, len(new_sol.routes) - 1)
            if new_sol.routes[ri].clients:
                ci = rng.randint(0, len(new_sol.routes[ri].clients) - 1)
                cid = new_sol.routes[ri].clients[ci]
                demand = inst.nodes[cid]["demand"]

                rj_candidates = [
                    j
                    for j in range(len(new_sol.routes))
                    if j != ri
                    and sum(inst.nodes[c]["demand"] for c in new_sol.routes[j].clients)
                    + demand
                    <= inst.capacity
                ]
                if rj_candidates:
                    rj = rng.choice(rj_candidates)
                    pos = rng.randint(0, len(new_sol.routes[rj].clients))
                    new_sol.routes[ri].clients.pop(ci)
                    new_sol.routes[rj].clients.insert(pos, cid)
                    new_sol.routes = [r for r in new_sol.routes if r.clients]

    elif op == "swap":
        if len(new_sol.routes) >= 2:
            ri, rj = rng.sample(range(len(new_sol.routes)), 2)
            if new_sol.routes[ri].clients and new_sol.routes[rj].clients:
                ci = rng.randint(0, len(new_sol.routes[ri].clients) - 1)
                cj = rng.randint(0, len(new_sol.routes[rj].clients) - 1)
                cid_i = new_sol.routes[ri].clients[ci]
                cid_j = new_sol.routes[rj].clients[cj]

                load_ri = sum(
                    inst.nodes[c]["demand"] for c in new_sol.routes[ri].clients
                )
                load_rj = sum(
                    inst.nodes[c]["demand"] for c in new_sol.routes[rj].clients
                )
                new_load_ri = (
                    load_ri - inst.nodes[cid_i]["demand"] + inst.nodes[cid_j]["demand"]
                )
                new_load_rj = (
                    load_rj - inst.nodes[cid_j]["demand"] + inst.nodes[cid_i]["demand"]
                )

                if new_load_ri <= inst.capacity and new_load_rj <= inst.capacity:
                    new_sol.routes[ri].clients[ci] = cid_j
                    new_sol.routes[rj].clients[cj] = cid_i

    return new_sol
