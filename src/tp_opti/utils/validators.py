import math

from tp_opti.model import Route, Solution, VRPTWInstance


def min_vehicles_lower_bound(inst):
    """Calcule la borne inférieure du nombre de véhicules nécessaires."""
    total_demand = sum(c["demand"] for c in inst.clients)
    return math.ceil(total_demand / inst.capacity)


def route_capacity_ok(route: Route, inst: VRPTWInstance) -> bool:
    """Vérifie que la capacité de la route n'est pas dépassée."""
    total = sum(inst.nodes[c]["demand"] for c in route.clients)
    return total <= inst.capacity


def route_time_ok(route: Route, inst: VRPTWInstance) -> bool | float:
    """
    Vérifie les fenêtres de temps d'une route.
    Retourne (valide, temps_retour_depot).
    Le véhicule peut attendre si e_i n'est pas encore atteint.
    """
    t = inst.depot["ready"]
    prev = 0  # index dépôt
    for cid in route.clients:
        t += inst.dist[prev][cid]
        node = inst.nodes[cid]
        if t > node["due"]:
            return False, t  # Arrivée trop tard
        t = max(t, node["ready"])  # Attendre si arrivée trop tôt
        t += node["service"]
        prev = cid
    # Retour au dépôt
    t += inst.dist[prev][0]
    if t > inst.depot["due"]:
        return False, t
    return True, t


def route_time_violation(route: Route, inst: VRPTWInstance) -> float:
    """
    Calcule la violation de fenêtre de temps (somme des retards).
    Retourne 0 si tout est valide.
    """
    t = inst.depot["ready"]
    prev = 0
    violation = 0.0
    for cid in route.clients:
        t += inst.dist[prev][cid]
        node = inst.nodes[cid]
        if t > node["due"]:
            violation += t - node["due"]
        t = max(t, node["ready"])
        t += node["service"]
        prev = cid
    t += inst.dist[prev][0]
    if t > inst.depot["due"]:
        violation += t - inst.depot["due"]
    return violation


def solution_is_valid(
    sol: Solution, inst: VRPTWInstance, check_tw: bool = True
) -> bool:
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


def route_distance(route: Route, inst: VRPTWInstance) -> float:
    """Distance totale d'une route (dépôt → clients → dépôt)."""
    if not route.clients:
        return 0.0
    d = inst.dist[0][route.clients[0]]
    for i in range(len(route.clients) - 1):
        d += inst.dist[route.clients[i]][route.clients[i + 1]]
    d += inst.dist[route.clients[-1]][0]
    return d


def total_distance(sol: Solution, inst: VRPTWInstance) -> float:
    """Distance totale de toute la solution."""
    return sum(route_distance(r, inst) for r in sol.routes)


def penalized_cost(
    sol: Solution, inst: VRPTWInstance, penalty: float = 1000.0
) -> float:
    """
    Coût pénalisé = distance + penalty * violation_TW.
    Utilisé pour guider les métaheuristiques même en cas de violation.
    """
    dist = total_distance(sol, inst)
    violation = solution_total_violation(sol, inst)
    return dist + penalty * violation
