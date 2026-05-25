import math
import random
import time

from tp_opti.algorithms.greedy import greedy_solution
from tp_opti.model import Solution, VRPTWInstance
from tp_opti.utils.operators import (
    neighbor_operator,
    penalized_cost,
    total_distance,
)
from tp_opti.utils.validators import solution_total_violation


def simulated_annealing(
    inst: VRPTWInstance,
    check_tw: bool = True,
    T0: float = 500.0,
    alpha: float = 0.995,
    max_iter: int = 10000,
    seed: int = 25565,
    op: str = "2opt",
) -> dict:
    """
    Recuit Simulé pour le VRPTW.

    Paramètres :
      - inst : instance VRPTW
      - check_tw : prendre en compte les fenêtres de temps
      - T0 : température initiale
      - alpha : facteur de refroidissement (0 < alpha < 1)
      - max_iter : nombre max d'itérations
      - seed : graine aléatoire
      - op : opérateur de voisinage parmi ["2opt", "relocate", "swap"]

    Retourne un dict avec la meilleure solution, historique de convergence, etc.
    """
    rng = random.Random(seed)

    # Solution initiale (greedy)
    current: Solution = greedy_solution(inst, check_tw=check_tw)
    current_cost = (
        penalized_cost(current, inst) if check_tw else total_distance(current, inst)
    )

    best: Solution = current.copy()
    best_cost = current_cost

    T = T0
    history = []
    history_best = []
    nb_accepted = 0
    nb_generated = 0

    start = time.time()

    for it in range(max_iter):
        neighbor: Solution = neighbor_operator(current, inst, op, check_tw=check_tw)
        neighbor_cost = (
            penalized_cost(neighbor, inst)
            if check_tw
            else total_distance(neighbor, inst)
        )
        nb_generated += 1

        delta = neighbor_cost - current_cost

        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            current = neighbor
            current_cost = neighbor_cost
            nb_accepted += 1

            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost

        T *= alpha

        if it % 100 == 0:
            history.append(current_cost)
            history_best.append(best_cost)

    elapsed = time.time() - start

    return {
        "algorithm": "Recuit Simulé",
        "solution": best,
        "best_cost": best_cost,
        "distance": total_distance(best, inst),
        "violation": solution_total_violation(best, inst),
        "num_routes": best.num_routes(),
        "nb_generated": nb_generated,
        "nb_accepted": nb_accepted,
        "time": elapsed,
        "history": history,
        "history_best": history_best,
        "params": {
            "T0": T0,
            "alpha": alpha,
            "max_iter": max_iter,
            "check_tw": check_tw,
        },
    }
