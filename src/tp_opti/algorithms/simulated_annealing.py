import math
import random
import time

from tp_opti.algorithms.greedy import greedy_solution
from tp_opti.model import Solution, VRPTWInstance
from tp_opti.utils.operators import (
    neighbor_operator,
)
from tp_opti.utils.validators import (
    penalized_cost,
    solution_total_violation,
    total_distance,
)


def simulated_annealing(
    inst: VRPTWInstance,
    check_tw: bool = True,
    T0: float = 500.0,
    alpha: float = 0.995,
    max_iter: int = 10000,
    n2: int = 10,
    seed: int = 25565,
    op: str = "2opt",
) -> dict:
    """
    Recuit Simulé pour le VRPTW.

    Paramètres :
      - inst      : instance VRPTW
      - check_tw  : prendre en compte les fenêtres de temps
      - T0        : température initiale
      - alpha     : facteur de refroidissement géométrique (0 < alpha < 1)
      - max_iter  : nombre maximum d'itérations
      - n2        : nombre de mouvements par palier
      - seed      : graine aléatoire
      - op        : opérateur de voisinage parmi ["2opt", "relocate", "swap"]

    Retourne un dict avec la meilleure solution, historique de convergence, etc.
    """
    rng = random.Random(seed)

    # Solution initiale (greedy)
    current: Solution = greedy_solution(inst, check_tw=check_tw)

    def cost(sol: Solution) -> float:
        return penalized_cost(sol, inst) if check_tw else total_distance(sol, inst)

    current_cost = cost(current)

    best: Solution = current.copy()
    best_cost = current_cost

    T = T0
    history: list[float] = []
    history_best: list[float] = []
    nb_accepted = 0
    nb_generated = 0

    start = time.time()

    for k in range(max_iter):
        for _ in range(n2):
            neighbor: Solution = neighbor_operator(
                current, inst, rng, op, check_tw=check_tw
            )
            neighbor_cost = cost(neighbor)
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

        if k % 100 == 0:
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
            "n2": n2,
            "check_tw": check_tw,
        },
    }
