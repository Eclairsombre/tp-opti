import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import glob
    import os

    import altair as alt
    import marimo as mo
    import pandas as pd

    from algorithms import greedy_solution, random_solution
    from model import VRPTWInstance
    from parsing import parse_vrp_file
    from utils import solution_total_violation, total_distance
    from visualisation import plot_routes_interactive


@app.cell(hide_code=True)
def _():
    DATA_DIR = r"./data"
    vrp_files = sorted(glob.glob(os.path.join(DATA_DIR, "data*.vrp")))

    datasets = {}
    rows = []

    for f in vrp_files:
        name = os.path.splitext(os.path.basename(f))[0]
        datasets[name] = parse_vrp_file(f)
        d = datasets[name]

        rows.append(
            {
                "Dataset": name,
                "Clients": d["nb_clients"],
                "Capacity": d["capacity"],
                "Depot X": round(d["depot"]["x"]),
                "Depot Y": round(d["depot"]["y"]),
            }
        )

    mo.vstack(
        [
            mo.md("## Données"),
            mo.ui.table(rows),
        ]
    )
    return (datasets,)


@app.cell(hide_code=True)
def _(datasets):
    dataset = mo.ui.dropdown(options=datasets)
    time_windows = mo.ui.dropdown(options=[True, False])
    return dataset, time_windows


@app.cell
def _(dataset, time_windows):
    mo.md(f"""
    Sélectionner un jeu de données : {dataset}
    Activer les contraintes de temps : {time_windows}
    """)
    return


@app.cell(hide_code=True)
def _(dataset):
    instance = VRPTWInstance(dataset.value)

    clients_df = pd.DataFrame(instance.clients)

    mo.vstack(
        [
            mo.md("## Clients du dataset sélectionné"),
            mo.ui.table(clients_df),
        ]
    )
    return (instance,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Validation des contraintes

    Deux types de contraintes :
    1. **Capacité** : $\sum_{i \in \text{route}} q_i \leq C$
    2. **Fenêtres de temps** : $e_i \leq t_i \leq l_i$ (avec attente si arrivée avant $e_i$)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Générateur aléatoire de solutions

    Construction par insertion aléatoire :
    1. Mélanger aléatoirement les clients
    2. Pour chaque client, essayer de l'insérer dans une route existante (contrainte capacité)
    3. Si impossible, ouvrir une nouvelle route
    """)
    return


@app.cell
def _(instance, time_windows):
    sol_rand = random_solution(instance, check_tw=time_windows.value)

    print("Solution aléatoire")
    print(f"Distance totale : {total_distance(sol_rand, instance):.2f}")
    print(
        f"Violation Time Windows : {solution_total_violation(sol_rand, instance):.2f}"
    )

    fig_rand = plot_routes_interactive(sol_rand, instance, "Solution aléatoire")

    mo.vstack(
        [
            mo.md("## Visualisation solution aléatoire"),
            mo.ui.plotly(fig_rand),
        ]
    )
    return


@app.cell
def _(instance, time_windows):
    sol_greedy = greedy_solution(instance, check_tw=time_windows.value)

    print("\nSolution greedy")
    print(f"Distance totale : {total_distance(sol_greedy, instance):.2f}")
    print(
        f"Violation Time Windows : {solution_total_violation(sol_greedy, instance):.2f}"
    )

    fig_greedy = plot_routes_interactive(sol_greedy, instance, "Solution greedy")

    mo.vstack([mo.md("## Visualisation solution Greedy"), mo.ui.plotly(fig_greedy)])
    return


if __name__ == "__main__":
    app.run()
