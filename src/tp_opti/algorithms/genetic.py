import random
import time

from tp_opti.algorithms.greedy import greedy_solution
from tp_opti.algorithms.random import random_solution
from tp_opti.model import Route, Solution, VRPTWInstance
from tp_opti.utils.operators import (
    penalized_cost,
    random_neighbor,
    total_distance,
)
from tp_opti.utils.validators import solution_total_violation


def sol_to_chromosome(sol: Solution) -> list:
    """Convertit une solution en chromosome (séquence linéaire de clients avec séparateurs -1)."""
    chrom = []
    for i, r in enumerate(sol.routes):
        chrom.extend(r.clients)
        if i < len(sol.routes) - 1:
            chrom.append(-1)  # Séparateur de routes
    return chrom


def chromosome_to_sol(chrom: list, inst: VRPTWInstance) -> Solution:
    """
    Convertit un chromosome en solution.
    Coupe les routes dès que la capacité est dépassée (réparation).
    """
    routes = []
    current_clients = []
    current_load = 0

    for gene in chrom:
        if gene == -1:
            if current_clients:
                routes.append(Route(current_clients))
                current_clients = []
                current_load = 0
        else:
            demand = inst.nodes[gene]["demand"]
            if current_load + demand > inst.capacity and current_clients:
                routes.append(Route(current_clients))
                current_clients = [gene]
                current_load = demand
            else:
                current_clients.append(gene)
                current_load += demand

    if current_clients:
        routes.append(Route(current_clients))

    return Solution([r for r in routes if r.clients])


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


def mutate(
    sol: Solution, inst: VRPTWInstance, rng, mutation_rate: float = 0.1
) -> Solution:
    """Mutation : relocate aléatoire d'un client."""
    if rng.random() > mutation_rate:
        return sol
    return random_neighbor(sol, inst, rng, check_tw=False)


def tournament_select(population: list, fitnesses: list, k: int, rng) -> Solution:
    """Sélection par tournoi de taille k."""
    candidates = rng.sample(range(len(population)), min(k, len(population)))
    best_idx = min(candidates, key=lambda i: fitnesses[i])
    return population[best_idx].copy()


def genetic_algorithm(
    inst: VRPTWInstance,
    check_tw: bool = True,
    pop_size: int = 30,
    generations: int = 200,
    tournament_k: int = 5,
    mutation_rate: float = 0.2,
    elite_size: int = 2,
    seed: int = 42,
) -> dict:
    """
    Algorithme Génétique pour le VRPTW.

    Paramètres :
      - pop_size    : taille de la population
      - generations : nombre de générations
      - tournament_k: taille du tournoi de sélection
      - mutation_rate: probabilité de mutation
      - elite_size  : nombre d'élites conservées par génération
      - check_tw    : prendre en compte les fenêtres de temps
    """
    rng = random.Random(seed)

    def fitness(sol):
        return penalized_cost(sol, inst) if check_tw else total_distance(sol, inst)

    # Générer la population initiale
    population = []
    population.append(greedy_solution(inst))
    for _ in range(pop_size - 1):
        population.append(random_solution(inst, check_tw=False))

    fitnesses = [fitness(s) for s in population]
    best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
    best_sol = population[best_idx].copy()
    best_fit = fitnesses[best_idx]

    history = []
    history_best = []
    nb_generated = pop_size

    start = time.time()

    for gen in range(generations):
        # Tri pour élitisme
        sorted_idx = sorted(range(len(population)), key=lambda i: fitnesses[i])
        new_pop = [population[i].copy() for i in sorted_idx[:elite_size]]
        new_fits = [fitnesses[i] for i in sorted_idx[:elite_size]]

        while len(new_pop) < pop_size:
            p1 = tournament_select(population, fitnesses, tournament_k, rng)
            p2 = tournament_select(population, fitnesses, tournament_k, rng)
            child = ox_crossover(p1, p2, inst, rng)
            child = mutate(child, inst, rng, mutation_rate)
            f = fitness(child)
            new_pop.append(child)
            new_fits.append(f)
            nb_generated += 1

        population = new_pop
        fitnesses = new_fits

        gen_best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
        gen_best_fit = fitnesses[gen_best_idx]

        if gen_best_fit < best_fit:
            best_fit = gen_best_fit
            best_sol = population[gen_best_idx].copy()

        history.append(fitnesses[gen_best_idx])
        history_best.append(best_fit)

    elapsed = time.time() - start

    return {
        "algorithm": "Algorithme Génétique",
        "solution": best_sol,
        "best_cost": best_fit,
        "distance": total_distance(best_sol, inst),
        "violation": solution_total_violation(best_sol, inst),
        "num_routes": best_sol.num_routes(),
        "nb_generated": nb_generated,
        "time": elapsed,
        "history": history,
        "history_best": history_best,
        "params": {
            "pop_size": pop_size,
            "generations": generations,
            "tournament_k": tournament_k,
            "mutation_rate": mutation_rate,
            "check_tw": check_tw,
        },
    }
