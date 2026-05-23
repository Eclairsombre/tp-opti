# neighbor_operators_demo.py

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns", app_title="VRPTW - Operators")

with app.setup:
    import glob
    import os
    import random

    import marimo as mo

    from tp_opti.algorithms.random import random_solution
    from tp_opti.model import VRPTWInstance
    from tp_opti.parsing import parse_vrp_file
    from tp_opti.utils.operators import (
        random_neighbor,
        relocate,
        swap,
        two_opt_intra,
    )
    from tp_opti.utils.validators import (
        solution_total_violation,
        total_distance,
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
    initial_solution = random_solution(
        instance,
        check_tw=time_windows.value,
    )
    return (initial_solution,)


@app.cell(hide_code=True)
def _(initial_solution, instance):
    fig_before = plot_routes_interactive(
        initial_solution,
        instance,
        "Routes avant opérateur",
    )

    mo.vstack(
        [
            mo.md("## Routes avant opérateur"),
            mo.ui.plotly(fig_before),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    operator_dropdown = mo.ui.dropdown(
            options={
                "two_opt_intra": "two_opt_intra",
                "relocate": "relocate",
                "swap": "swap",
                "random_neighbor": "random_neighbor",
            },
            value="two_opt_intra"
        )
    return (operator_dropdown,)


@app.cell(hide_code=True)
def _(operator_dropdown):
    mo.md(f"""
    Sélectionner un opérateur : {operator_dropdown}
    """)
    return


@app.cell(hide_code=True)
def _(initial_solution, instance, operator_dropdown, time_windows):
    operator_name = operator_dropdown.value

    if operator_name == "two_opt_intra":
        new_solution = two_opt_intra(
            initial_solution,
            instance,
            check_tw=time_windows.value,
        )

    elif operator_name == "relocate":
        new_solution = relocate(
            initial_solution,
            instance,
            check_tw=time_windows.value,
        )

    elif operator_name == "swap":
        new_solution = swap(
            initial_solution,
            instance,
            check_tw=time_windows.value,
        )

    elif operator_name == "random_neighbor":
        new_solution = random_neighbor(
            initial_solution,
            instance,
            rng=random.Random(25565),
            check_tw=time_windows.value,
        )

    else:
        new_solution = initial_solution
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
