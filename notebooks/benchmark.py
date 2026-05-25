import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns", app_title="VRPTW - Benchmark")

with app.setup:
    import glob
    import os

    import marimo as mo
    import pandas as pd
    import plotly.express as px

    from tp_opti.algorithms import genetic_algorithm, simulated_annealing
    from tp_opti.model import VRPTWInstance
    from tp_opti.parsing import parse_vrp_file


@app.cell
def _():
    data_dir = "./data"
    files = sorted(glob.glob(os.path.join(data_dir, "data*.vrp")))

    datasets_bm = {
        os.path.splitext(os.path.basename(f))[0]: parse_vrp_file(f) for f in files
    }
    return (datasets_bm,)


@app.cell
def _(datasets_bm):
    dataset_sel_bm = mo.ui.multiselect(
        options=list(datasets_bm.keys()),
        value=list(datasets_bm.keys())[:1],
    )
    return (dataset_sel_bm,)


@app.cell
def _():
    algo_sel_bm = mo.ui.dropdown(options=["sa", "genetic"], value="sa")
    tw_sel_bm = mo.ui.dropdown(options=[True, False], value=True)
    repeats_bm = mo.ui.slider(start=1, stop=30, step=1, value=5)

    sa_op_bm = mo.ui.dropdown(options=["2opt", "relocate", "swap"], value="swap")

    ga_mut_bm = mo.ui.dropdown(options=["2opt", "relocate", "swap"], value="swap")
    ga_cross_bm = mo.ui.dropdown(options=["ox", "cx"], value="ox")
    return algo_sel_bm, ga_cross_bm, ga_mut_bm, repeats_bm, sa_op_bm, tw_sel_bm


@app.cell
def _(
    algo_sel_bm,
    dataset_sel_bm,
    ga_cross_bm,
    ga_mut_bm,
    repeats_bm,
    sa_op_bm,
    tw_sel_bm,
):
    mo.vstack(
        [
            mo.md("## config"),
            dataset_sel_bm,
            algo_sel_bm,
            tw_sel_bm,
            repeats_bm,
            mo.md("SA op"),
            sa_op_bm,
            mo.md("GA ops"),
            ga_mut_bm,
            ga_cross_bm,
        ]
    )
    return


@app.cell
def _(
    algo_sel_bm,
    dataset_sel_bm,
    datasets_bm,
    ga_cross_bm,
    ga_mut_bm,
    repeats_bm,
    sa_op_bm,
    tw_sel_bm,
):
    rows_bm = []

    for ds_name_bm in dataset_sel_bm.value:
        inst_bm = VRPTWInstance(datasets_bm[ds_name_bm])

        for i_bm in range(repeats_bm.value):
            if algo_sel_bm.value == "sa":
                sol_sa_bm = simulated_annealing(
                    inst_bm,
                    check_tw=tw_sel_bm.value,
                    T0=500,
                    alpha=0.995,
                    max_iter=5000,
                    seed=42,
                    op=sa_op_bm.value,
                )

                rows_bm.append(
                    {
                        "dataset": ds_name_bm,
                        "algo": "sa",
                        "distance": sol_sa_bm["distance"],
                        "time": sol_sa_bm["time"],
                    }
                )

            else:
                sol_ga_bm = genetic_algorithm(
                    inst_bm,
                    check_tw=tw_sel_bm.value,
                    pop_size=30,
                    generations=100,
                    seed=42,
                    op_mutate=ga_mut_bm.value,
                    op_cross=ga_cross_bm.value,
                )

                rows_bm.append(
                    {
                        "dataset": ds_name_bm,
                        "algo": "genetic",
                        "distance": sol_ga_bm["distance"],
                        "time": sol_ga_bm["time"],
                    }
                )

    df_bm = pd.DataFrame(rows_bm)
    return (df_bm,)


@app.cell
def _(df_bm):
    agg_bm = (
        df_bm.groupby(["dataset", "algo"])
        .agg(
            {
                "distance": ["mean", "std"],
                "time": ["mean", "std"],
            }
        )
        .reset_index()
    )

    agg_bm.columns = [
        "dataset",
        "algo",
        "dist_mean",
        "dist_std",
        "time_mean",
        "time_std",
    ]
    return (agg_bm,)


@app.cell
def _(agg_bm):
    fig_dist_bm = px.bar(
        agg_bm,
        x="dataset",
        y="dist_mean",
        color="algo",
        barmode="group",
        error_y="dist_std",
        title="Distance moyenne",
    )
    mo.ui.plotly(fig_dist_bm)
    return


@app.cell
def _(agg_bm):
    fig_time_bm = px.bar(
        agg_bm,
        x="dataset",
        y="time_mean",
        color="algo",
        barmode="group",
        error_y="time_std",
        title="Temps calcul moyen",
    )
    mo.ui.plotly(fig_time_bm)
    return


@app.cell
def _():
    T0_bm = mo.ui.slider(start=100, stop=1000, step=50, value=500)
    alpha_bm = mo.ui.slider(start=0.90, stop=0.999, step=0.001, value=0.995)
    max_iter_bm = mo.ui.slider(start=1000, stop=10000, step=500, value=5000)
    return alpha_bm, max_iter_bm


@app.cell
def _(alpha_bm, dataset_sel_bm, datasets_bm, max_iter_bm, tw_sel_bm):
    sweep_rows_bm = []

    ds0_bm = dataset_sel_bm.value[0]
    inst_sweep_bm = VRPTWInstance(datasets_bm[ds0_bm])

    for T0_val_bm in [300, 500, 700, 900]:
        sol_sweep_bm = simulated_annealing(
            inst_sweep_bm,
            check_tw=tw_sel_bm.value,
            T0=T0_val_bm,
            alpha=alpha_bm.value,
            max_iter=max_iter_bm.value,
            seed=42,
            op="swap",
        )

        sweep_rows_bm.append(
            {
                "T0": T0_val_bm,
                "distance": sol_sweep_bm["distance"],
                "time": sol_sweep_bm["time"],
            }
        )

    df_sweep_bm = pd.DataFrame(sweep_rows_bm)
    return (df_sweep_bm,)


@app.cell
def _(df_sweep_bm):
    fig_sweep_dist_bm = px.line(df_sweep_bm, x="T0", y="distance")
    mo.ui.plotly(fig_sweep_dist_bm)
    return


@app.cell
def _(df_sweep_bm):
    fig_sweep_time_bm = px.line(df_sweep_bm, x="T0", y="time")
    mo.ui.plotly(fig_sweep_time_bm)
    return


if __name__ == "__main__":
    app.run()
