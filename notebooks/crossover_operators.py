# neighbor_operators_demo.py

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium", app_title="VRPTW - Crossover operators")

with app.setup:
    import glob
    import os
    import random

    import marimo as mo

    from tp_opti.algorithms.random import random_solution
    from tp_opti.model import VRPTWInstance
    from tp_opti.parsing import parse_vrp_file
    from tp_opti.utils.operators import (
        crossover_operator,
    )
    from tp_opti.visualisation import plot_routes_interactive


@app.cell(hide_code=True)
def _():
    DATA_DIR = r"./data/tests"
    vrp_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.vrp")))

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


@app.cell(hide_code=True)
def _(dataset, time_windows):
    mo.md(f"""
    Sélectionner un jeu de données : {dataset}
    Activer les contraintes de temps : {time_windows}
    """)
    return


@app.cell(hide_code=True)
def _(dataset):
    instance = VRPTWInstance(dataset.value)
    return (instance,)


@app.cell(hide_code=True)
def _(instance, time_windows):
    initial_solution_1 = random_solution(
        instance,
        check_tw=time_windows.value,
    )
    initial_solution_2 = random_solution(
        instance,
        check_tw=time_windows.value,
    )

    fig_before_1 = plot_routes_interactive(
        initial_solution_1,
        instance,
        "Routes avant opérateur (parent 1)",
    )

    fig_before_2 = plot_routes_interactive(
        initial_solution_2,
        instance,
        "Routes avant opérateur (parent 2)",
    )

    mo.vstack(
        [
            mo.md("## Routes avant opérateur (parent 2)"),
            mo.hstack(
                [
                    mo.ui.plotly(fig_before_1),
                    mo.ui.plotly(fig_before_2),
                ]
            ),
        ]
    )
    return initial_solution_1, initial_solution_2


@app.cell(hide_code=True)
def _():
    operator_dropdown = mo.ui.dropdown(
        options={
            "ox": "ox",
            "cx": "cx",
        },
        value="ox",
    )
    return (operator_dropdown,)


@app.cell(hide_code=True)
def _(operator_dropdown):
    mo.md(f"""
    Sélectionner un opérateur : {operator_dropdown}
    """)
    return


@app.cell(hide_code=True)
def _(
    initial_solution_1,
    initial_solution_2,
    instance,
    operator_dropdown,
    time_windows,
):
    operator_name = operator_dropdown.value
    rng = random.Random()

    new_solution = crossover_operator(
        initial_solution_1,
        initial_solution_2,
        instance,
        rng,
        op=operator_name,
        check_tw=time_windows.value,
    )
    return (new_solution,)


@app.cell(hide_code=True)
def _(instance, new_solution):
    fig_after = plot_routes_interactive(
        new_solution,
        instance,
        "Après opérateur",
    )

    mo.vstack(
        [
            mo.md("## Routes après opérateur"),
            mo.ui.plotly(fig_after),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
