from tp_opti.model import Route, Solution, VRPTWInstance


def two_opt_intra(
    sol: Solution, inst: VRPTWInstance, rng, check_tw: bool = True
) -> Solution:
    """
    2-opt intra-route : inverse un segment [i..j] aléatoire dans une route.
    """
    new_sol = sol.copy()
    routes_with_4 = [i for i, r in enumerate(new_sol.routes) if len(r.clients) >= 4]
    if not routes_with_4:
        return new_sol

    ri = rng.choice(routes_with_4)
    n = len(new_sol.routes[ri].clients)
    i = rng.randint(1, n - 2)
    j = rng.randint(i + 1, n - 1)
    new_sol.routes[ri].clients[i : j + 1] = list(
        reversed(new_sol.routes[ri].clients[i : j + 1])
    )
    return new_sol


def relocate(
    sol: Solution, inst: VRPTWInstance, rng, check_tw: bool = True
) -> Solution:
    """
    Relocate : déplace un client aléatoire d'une route vers une position aléatoire d'une autre route.
    """
    new_sol = sol.copy()
    if len(new_sol.routes) < 2:
        return new_sol

    ri = rng.randint(0, len(new_sol.routes) - 1)
    if not new_sol.routes[ri].clients:
        return new_sol

    ci = rng.randint(0, len(new_sol.routes[ri].clients) - 1)
    cid = new_sol.routes[ri].clients[ci]
    demand = inst.nodes[cid]["demand"]

    rj_candidates = [
        j
        for j in range(len(new_sol.routes))
        if j != ri
        and sum(inst.nodes[c]["demand"] for c in new_sol.routes[j].clients) + demand
        <= inst.capacity
    ]
    if not rj_candidates:
        return new_sol

    rj = rng.choice(rj_candidates)
    pos = rng.randint(0, len(new_sol.routes[rj].clients))
    new_sol.routes[ri].clients.pop(ci)
    new_sol.routes[rj].clients.insert(pos, cid)
    new_sol.routes = [r for r in new_sol.routes if r.clients]
    return new_sol


def swap(sol: Solution, inst: VRPTWInstance, rng, check_tw: bool = True) -> Solution:
    """
    Swap : échange deux clients aléatoires entre deux routes différentes.
    """
    new_sol = sol.copy()
    if len(new_sol.routes) < 2:
        return new_sol

    ri, rj = rng.sample(range(len(new_sol.routes)), 2)
    if not new_sol.routes[ri].clients or not new_sol.routes[rj].clients:
        return new_sol

    ci = rng.randint(0, len(new_sol.routes[ri].clients) - 1)
    cj = rng.randint(0, len(new_sol.routes[rj].clients) - 1)
    cid_i = new_sol.routes[ri].clients[ci]
    cid_j = new_sol.routes[rj].clients[cj]

    load_ri = sum(inst.nodes[c]["demand"] for c in new_sol.routes[ri].clients)
    load_rj = sum(inst.nodes[c]["demand"] for c in new_sol.routes[rj].clients)
    new_load_ri = load_ri - inst.nodes[cid_i]["demand"] + inst.nodes[cid_j]["demand"]
    new_load_rj = load_rj - inst.nodes[cid_j]["demand"] + inst.nodes[cid_i]["demand"]

    if new_load_ri <= inst.capacity and new_load_rj <= inst.capacity:
        new_sol.routes[ri].clients[ci] = cid_j
        new_sol.routes[rj].clients[cj] = cid_i

    return new_sol


def neighbor_operator(
    sol: Solution, inst: VRPTWInstance, rng, op=None, check_tw: bool = True
) -> Solution:
    """
    Génère un voisin aléatoire en déléguant à l'opérateur choisi.
    """
    if op is None:
        op = rng.choice(["2opt", "relocate", "swap"])

    if op == "2opt":
        return two_opt_intra(sol, inst, rng, check_tw)
    elif op == "relocate":
        return relocate(sol, inst, rng, check_tw)
    elif op == "swap":
        return swap(sol, inst, rng, check_tw)

    return sol.copy()


def ox_crossover(
    parent1: Solution, parent2: Solution, inst: VRPTWInstance, rng
) -> Solution:
    """
    OX Crossover (Order Crossover) adapté au VRP.
    Travaille sur la séquence de clients (sans les séparateurs).
    """
    clients_p1 = parent1.all_clients()
    clients_p2 = parent2.all_clients()
    n = len(clients_p1)

    if n < 2:
        return parent1.copy()

    # Sélectionner un segment de p1
    i, j = sorted(rng.sample(range(n), 2))

    child_clients = [None] * n
    child_clients[i : j + 1] = clients_p1[i : j + 1]
    segment_set = set(clients_p1[i : j + 1])

    # Remplir avec p2 dans l'ordre
    remaining = [c for c in clients_p2 if c not in segment_set]
    pos = 0
    for k in range(n):
        if child_clients[k] is None:
            child_clients[k] = remaining[pos]
            pos += 1

    # Reconstruire les routes avec contrainte de capacité
    routes = []
    current_clients = []
    current_load = 0

    for cid in child_clients:
        demand = inst.nodes[cid]["demand"]
        if current_load + demand > inst.capacity:
            if current_clients:
                routes.append(Route(current_clients))
            current_clients = [cid]
            current_load = demand
        else:
            current_clients.append(cid)
            current_load += demand

    if current_clients:
        routes.append(Route(current_clients))

    return Solution(routes)


def cx_crossover(
    parent1: Solution, parent2: Solution, inst: VRPTWInstance, rng
) -> Solution:
    """
    CX Crossover (Cycle Crossover) adapté au VRP.
    Identifie les cycles de positions entre p1 et p2 :
    - Cycle 1 -> positions héritées de p1
    - Cycle 2 -> positions héritées de p2
    - Cycle 3 -> p1, etc. (alternance)
    Préserve les positions absolues, sans doublon par construction.
    """
    clients_p1 = parent1.all_clients()
    clients_p2 = parent2.all_clients()
    n = len(clients_p1)
    if n < 2:
        return parent1.copy()

    # Index de position : client -> index dans p2
    pos_in_p2 = {c: idx for idx, c in enumerate(clients_p2)}

    child_clients = [None] * n
    visited = [False] * n
    use_p1 = True  # Alterner d'un cycle à l'autre

    for start in range(n):
        if visited[start]:
            continue

        # Identifier le cycle complet à partir de 'start'
        cycle = []
        k = start
        while not visited[k]:
            visited[k] = True
            cycle.append(k)
            k = pos_in_p2[clients_p1[k]]  # Suivre le cycle

        # Remplir les positions du cycle depuis p1 ou p2
        for pos in cycle:
            child_clients[pos] = clients_p1[pos] if use_p1 else clients_p2[pos]
        use_p1 = not use_p1  # Alterner pour le prochain cycle

    # Reconstruire les routes avec contrainte de capacité (identique à OX)
    routes = []
    current_clients = []
    current_load = 0
    for cid in child_clients:
        demand = inst.nodes[cid]["demand"]
        if current_load + demand > inst.capacity:
            if current_clients:
                routes.append(Route(current_clients))
            current_clients = [cid]
            current_load = demand
        else:
            current_clients.append(cid)
            current_load += demand
    if current_clients:
        routes.append(Route(current_clients))
    return Solution(routes)


def repair_time_windows(sol: Solution, inst: VRPTWInstance, rng) -> Solution:
    """
    Tente de réparer les violations de time windows en réordonnant
    les clients dans chaque route par earliest due time.
    """
    new_sol = sol.copy()
    for route in new_sol.routes:
        route.clients.sort(key=lambda cid: inst.nodes[cid]["ready_time"])
    return new_sol


def crossover_operator(
    parent1: Solution,
    parent2: Solution,
    inst: VRPTWInstance,
    rng,
    op: str | None = None,
    check_tw: bool = False,
) -> Solution:
    if op is None:
        op = rng.choice(["ox", "cx"])

    if op == "ox":
        child = ox_crossover(parent1, parent2, inst, rng)
    elif op == "cx":
        child = cx_crossover(parent1, parent2, inst, rng)
    else:
        raise ValueError(f"Opérateur de croisement inconnu : {op}")

    return repair_time_windows(child, inst, rng) if check_tw else child
