import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns", app_title="VRPTW - Main")

with app.setup:
    import glob
    import os

    import altair as alt
    import marimo as mo
    import pandas as pd

    from tp_opti.algorithms import greedy_solution, random_solution, simulated_annealing, genetic_algorithm
    from tp_opti.model import VRPTWInstance
    from tp_opti.parsing import parse_vrp_file
    from tp_opti.utils.validators import (
        min_vehicles_lower_bound,
        solution_total_violation,
        total_distance,
        solution_is_valid
    )
    from tp_opti.visualisation import plot_routes_interactive


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Jeu de données
    """)
    return


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


@app.cell
def _():
    mo.md(r"""
    ## Nombre minimum de véhicules à utiliser

    La borne inférieure est :

    $$V_{min} = \left\lceil \frac{\sum_{i=1}^{n} q_i}{C} \right\rceil$$

    avec $n$ le nombre de clients, $q_i$ la demande du client $i$ et $C$ la capacité du véhicule.
    """)
    return


@app.cell(hide_code=True)
def _(datasets):
    def _():
        rows = []

        for name, data in datasets.items():
            inst = VRPTWInstance(data)
            total_q = sum(c["demand"] for c in inst.clients)

            rows.append(
                {
                    "Données": name,
                    "Clients": inst.n,
                    "Capacité": inst.capacity,
                    "Demande": total_q,
                    "Nombre minimum de véhicules": min_vehicles_lower_bound(inst),
                }
            )

        df = pd.DataFrame(rows).sort_values("Nombre minimum de véhicules")

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Données:N", sort=None),
                y="Nombre minimum de véhicules:Q",
                tooltip=list(df.columns),
            )
            .properties(
                title="Nombre minimum de véhicules par jeux de données",
                width=700,
                height=400,
            )
        )

        labels = (
            alt.Chart(df)
            .mark_text(
                dy=-8,
                fontSize=11,
                fontWeight="bold",
            )
            .encode(
                x="Données:N",
                y="Nombre minimum de véhicules:Q",
                text="Nombre minimum de véhicules:Q",
            )
        )
        return mo.vstack(
            [
                mo.md("## Nombre minimum de véhicules"),
                mo.ui.altair_chart(chart + labels),
            ]
        )

    _()
    return


@app.cell(hide_code=True)
def _(datasets):
    dataset = mo.ui.dropdown(options=datasets)
    time_windows = mo.ui.dropdown(options=[True, False])
    return dataset, time_windows


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Analyse sur un jeu de donnée
    """)
    return


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
            mo.md("## Clients du jeu de donnée sélectionné"),
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(instance, time_windows):
    sol_greedy = greedy_solution(instance, check_tw=time_windows.value)

    print("\nSolution greedy")
    print(f"Distance totale : {total_distance(sol_greedy, instance):.2f}")
    print(
        f"Violation Time Windows : {solution_total_violation(sol_greedy, instance):.2f}"
    )

    fig_greedy = plot_routes_interactive(sol_greedy, instance, "Solution gloutonne")

    mo.vstack([mo.md("## Visualisation solution gloutonne"), mo.ui.plotly(fig_greedy)])
    return


@app.cell(hide_code=True)
def _(instance, time_windows):
    result_sa = simulated_annealing(
        instance,
        check_tw=time_windows.value,
        T0=500,
        alpha=0.995,
        max_iter=5000,
        seed=42,
        op="swap"
    )

    print(f"  Distance: {result_sa['distance']:.2f}")
    print(f"  Violation Time Windows : {result_sa['violation']:.4f}")
    print(f"  Routes : {result_sa['num_routes']}")
    print(f"  Temps : {result_sa['time']:.2f}s")
    print(f"  Solutions générées : {result_sa['nb_generated']}")
    print(f"  Solutions acceptées: {result_sa['nb_accepted']}")
    print(
        f"  Valide : {solution_is_valid(result_sa['solution'], instance, check_tw=time_windows.value)}"
    )

    fig_sa = plot_routes_interactive(result_sa["solution"], instance, "Solution recuit-simulé")

    mo.vstack(
        [
            mo.md("## Visualisation solution recuit-simulé"),
            mo.ui.plotly(fig_sa),
        ]
    )
    return


@app.cell
def _(instance, time_windows):
    result_genetic = genetic_algorithm(
        instance,
        check_tw=time_windows.value,
        pop_size=30,
        generations=100,
        seed=42,
        op_mutate="swap",
        op_cross="ox"
    )

    print(f"  Distance: {result_genetic['distance']:.2f}")
    print(f"  Violation Time Windows : {result_genetic['violation']:.4f}")
    print(f"  Routes : {result_genetic['num_routes']}")
    print(f"  Temps : {result_genetic['time']:.2f}s")
    print(f"  Solutions générées : {result_genetic['nb_generated']}")
    print(
        f"  Valide : {solution_is_valid(result_genetic['solution'], instance, check_tw=time_windows.value)}"
    )

    fig_genetic = plot_routes_interactive(result_genetic["solution"], instance, "Solution algorithme génétique")

    mo.vstack(
        [
            mo.md("## Visualisation solution algorithme génétique"),
            mo.ui.plotly(fig_genetic),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
