import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns", app_title="VRPTW - Comparaison SA vs GA")

with app.setup:
    import glob
    import os

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from tp_opti.algorithms import (
        genetic_algorithm,
        simulated_annealing,
    )
    from tp_opti.model import VRPTWInstance
    from tp_opti.parsing import parse_vrp_file
    from tp_opti.visualisation import plot_routes_interactive


@app.cell(hide_code=True)
def _():
    mo.md("""
    # VRPTW - Comparaison Recuit Simulé vs Algorithme Génétique
    """)
    return


@app.cell(hide_code=True)
def _():
    DATA_DIR = "./data"
    _vrp_files = sorted(glob.glob(os.path.join(DATA_DIR, "data*.vrp")))

    all_datasets = {}
    for _f in _vrp_files:
        _name = os.path.splitext(os.path.basename(_f))[0]
        all_datasets[_name] = VRPTWInstance(parse_vrp_file(_f))

    mo.md(
        f"**{len(all_datasets)} datasets chargés :** {', '.join(all_datasets.keys())}"
    )
    return (all_datasets,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Configuration commune
    """)
    return


@app.cell(hide_code=True)
def _(all_datasets):
    ui_datasets = mo.ui.multiselect(
        options=list(all_datasets.keys()),
        value=list(all_datasets.keys())[:2],
        label="Datasets à tester",
    )
    ui_tw = mo.ui.switch(value=True, label="Fenêtres de temps (TW)")
    ui_repeats = mo.ui.slider(1, 10, value=3, step=1, label="Répétitions par dataset")

    mo.hstack([ui_datasets, ui_tw, ui_repeats], justify="start", gap=2)
    return ui_datasets, ui_repeats, ui_tw


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Recuit Simulé - Paramètres
    """)
    return


@app.cell(hide_code=True)
def _():
    ui_sa_T0 = mo.ui.slider(
        10, 2000, value=500, step=10, label="T0 (température initiale)"
    )
    ui_sa_alpha = mo.ui.slider(
        0.900, 0.9999, value=0.995, step=0.0005, label="alpha (refroidissement)"
    )
    ui_sa_max_iter = mo.ui.slider(500, 1000000, value=10000, step=500, label="max_iter")
    ui_sa_n2 = mo.ui.slider(1, 100, value=10, step=10, label="n2")
    ui_sa_op = mo.ui.dropdown(
        options=["2opt", "relocate", "swap"],
        value="2opt",
        label="Opérateur de voisinage",
    )

    mo.vstack(
        [
            mo.hstack([ui_sa_T0, ui_sa_alpha, ui_sa_n2], justify="start", gap=2),
            mo.hstack([ui_sa_max_iter, ui_sa_op], justify="start", gap=2),
        ]
    )
    return ui_sa_T0, ui_sa_alpha, ui_sa_max_iter, ui_sa_n2, ui_sa_op


@app.cell(hide_code=True)
def _():
    run_btn_sa = mo.ui.run_button(label="▶ Lancer le Recuit Simulé")
    run_btn_sa
    return (run_btn_sa,)


@app.cell(hide_code=True)
def _(
    all_datasets,
    run_btn_sa,
    ui_datasets,
    ui_repeats,
    ui_sa_T0,
    ui_sa_alpha,
    ui_sa_max_iter,
    ui_sa_n2,
    ui_sa_op,
    ui_tw,
):
    mo.stop(
        not run_btn_sa.value,
        mo.md("*Appuyez sur Lancer pour démarrer le Recuit Simulé.*"),
    )

    _selected = ui_datasets.value or list(all_datasets.keys())[:1]
    _n_rep = ui_repeats.value
    _check_tw = ui_tw.value

    _raw = {}
    for _ds_name in _selected:
        _inst = all_datasets[_ds_name]
        _runs = []
        for _rep in range(_n_rep):
            _r = simulated_annealing(
                _inst,
                check_tw=_check_tw,
                T0=ui_sa_T0.value,
                alpha=ui_sa_alpha.value,
                max_iter=ui_sa_max_iter.value,
                n2=ui_sa_n2.value,
                seed=25565 + _rep,
                op=ui_sa_op.value,
            )
            _runs.append(_r)
        _raw[_ds_name] = _runs

    sa_results = _raw
    mo.md(f"✅ **SA terminé** - {len(_selected)} dataset(s) × {_n_rep} répétition(s)")
    return (sa_results,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Algorithme Génétique - Paramètres
    """)
    return


@app.cell(hide_code=True)
def _():
    ui_ga_pop = mo.ui.slider(5, 500, value=75, step=5, label="pop_size")
    ui_ga_gen = mo.ui.slider(10, 2000, value=200, step=10, label="generations")
    ui_ga_tourn = mo.ui.slider(1, 20, value=5, step=1, label="tournament_k")
    ui_ga_mut_rate = mo.ui.slider(
        0.01, 0.8, value=0.2, step=0.01, label="mutation_rate"
    )
    ui_ga_elite = mo.ui.slider(1, 10, value=2, step=1, label="elite_size")
    ui_ga_op_mutate = mo.ui.dropdown(
        options=["2opt", "relocate", "swap"],
        value="2opt",
        label="op_mutate",
    )
    ui_ga_op_cross = mo.ui.dropdown(
        options=["ox", "cx"],
        value="ox",
        label="op_cross",
    )

    mo.vstack(
        [
            mo.hstack([ui_ga_pop, ui_ga_gen, ui_ga_tourn], justify="start", gap=2),
            mo.hstack([ui_ga_mut_rate, ui_ga_elite], justify="start", gap=2),
            mo.hstack([ui_ga_op_mutate, ui_ga_op_cross], justify="start", gap=2),
        ]
    )
    return (
        ui_ga_elite,
        ui_ga_gen,
        ui_ga_mut_rate,
        ui_ga_op_cross,
        ui_ga_op_mutate,
        ui_ga_pop,
        ui_ga_tourn,
    )


@app.cell(hide_code=True)
def _():
    run_btn_ga = mo.ui.run_button(label="▶ Lancer l'Algorithme Génétique")
    run_btn_ga
    return (run_btn_ga,)


@app.cell(hide_code=True)
def _(
    all_datasets,
    run_btn_ga,
    ui_datasets,
    ui_ga_elite,
    ui_ga_gen,
    ui_ga_mut_rate,
    ui_ga_op_cross,
    ui_ga_op_mutate,
    ui_ga_pop,
    ui_ga_tourn,
    ui_repeats,
    ui_tw,
):
    mo.stop(
        not run_btn_ga.value,
        mo.md("*Appuyez sur Lancer pour démarrer l'Algorithme Génétique.*"),
    )

    _selected = ui_datasets.value or list(all_datasets.keys())[:1]
    _n_rep = ui_repeats.value
    _check_tw = ui_tw.value

    _raw = {}
    for _ds_name in _selected:
        _inst = all_datasets[_ds_name]
        _runs = []
        for _rep in range(_n_rep):
            _r = genetic_algorithm(
                _inst,
                check_tw=_check_tw,
                pop_size=ui_ga_pop.value,
                generations=ui_ga_gen.value,
                tournament_k=ui_ga_tourn.value,
                mutation_rate=ui_ga_mut_rate.value,
                elite_size=ui_ga_elite.value,
                seed=25565 + _rep,
                op_mutate=ui_ga_op_mutate.value,
                op_cross=ui_ga_op_cross.value,
            )
            _runs.append(_r)
        _raw[_ds_name] = _runs

    ga_results = _raw
    mo.md(f"✅ **GA terminé** - {len(_selected)} dataset(s) × {_n_rep} répétition(s)")
    return (ga_results,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Résultats - Recuit Simulé
    """)
    return


@app.cell(hide_code=True)
def _(sa_results):
    mo.stop(not sa_results)

    _rows = []
    for _ds, _runs in sa_results.items():
        _dists = [r["distance"] for r in _runs]
        _times = [r["time"] for r in _runs]
        _viols = [r["violation"] for r in _runs]
        _nb_gen = [r["nb_generated"] for r in _runs]
        _best_run = min(_runs, key=lambda r: r["distance"])
        _rows.append(
            {
                "Dataset": _ds,
                "Dist. moy.": f"{np.mean(_dists):.2f}",
                "Dist. min": f"{min(_dists):.2f}",
                "Dist. max": f"{max(_dists):.2f}",
                "Temps moy. (s)": f"{np.mean(_times):.2f}",
                "Violation moy.": f"{np.mean(_viols):.4f}",
                "Solutions générées (moy.)": f"{np.mean(_nb_gen):.0f}",
                "Routes (best)": _best_run["num_routes"],
                "Valide (best)": "✅" if _best_run["violation"] < 1e-6 else "❌",
            }
        )

    mo.ui.table(_rows)
    return


@app.cell(hide_code=True)
def _(
    all_datasets,
    sa_results,
    ui_repeats,
    ui_sa_T0,
    ui_sa_alpha,
    ui_sa_max_iter,
    ui_sa_n2,
    ui_sa_op,
    ui_tw,
):
    mo.stop(not sa_results)

    _params_txt = (
        f"T0={ui_sa_T0.value} | alpha={ui_sa_alpha.value} | "
        f"max_iter={ui_sa_max_iter.value} | n2={ui_sa_n2.value} | "
        f"op={ui_sa_op.value} | repeats={ui_repeats.value} | TW={ui_tw.value}"
    )

    _tabs = {}
    for _ds, _runs in sa_results.items():
        _best = min(_runs, key=lambda r: r["distance"])
        _inst = all_datasets[_ds]
        _fig = plot_routes_interactive(
            _best["solution"],
            _inst,
            f"SA - {_ds} (distance={_best['distance']:.2f})",
            _params_txt,
        )
        _tabs[_ds] = mo.ui.plotly(_fig)

    mo.vstack(
        [
            mo.md("### Meilleures solutions SA par dataset"),
            mo.ui.tabs(_tabs),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    sa_results,
    ui_repeats,
    ui_sa_T0,
    ui_sa_alpha,
    ui_sa_max_iter,
    ui_sa_n2,
    ui_sa_op,
    ui_tw,
):
    mo.stop(not sa_results)

    _params_txt = (
        f"T0={ui_sa_T0.value} | alpha={ui_sa_alpha.value} | "
        f"max_iter={ui_sa_max_iter.value} | n2={ui_sa_n2.value} | "
        f"op={ui_sa_op.value} | repeats={ui_repeats.value} | TW={ui_tw.value}"
    )

    _palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

    _fig_sa_conv = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Coût courant (SA)", "Meilleur coût (SA)"),
        shared_yaxes=False,
    )

    for _i, (_ds, _runs) in enumerate(sa_results.items()):
        _color = _palette[_i % len(_palette)]

        _min_h = min(len(r["history"]) for r in _runs)
        _min_hb = min(len(r["history_best"]) for r in _runs)

        _hist_arr = np.array([r["history"][:_min_h] for r in _runs])
        _histb_arr = np.array([r["history_best"][:_min_hb] for r in _runs])

        _h_mean = _hist_arr.mean(axis=0)
        _hb_mean = _histb_arr.mean(axis=0)
        _hb_std = _histb_arr.std(axis=0)

        _xs = list(range(len(_h_mean)))
        _xsb = list(range(len(_hb_mean)))

        _fig_sa_conv.add_trace(
            go.Scatter(
                x=_xs,
                y=_h_mean.tolist(),
                mode="lines",
                name=f"{_ds} (courant)",
                line=dict(color=_color, width=1, dash="dot"),
                opacity=0.6,
                legendgroup=_ds,
            ),
            row=1,
            col=1,
        )

        _fig_sa_conv.add_trace(
            go.Scatter(
                x=_xsb + _xsb[::-1],
                y=(_hb_mean + _hb_std).tolist() + (_hb_mean - _hb_std).tolist()[::-1],
                fill="toself",
                fillcolor=_color,
                line=dict(color="rgba(255,255,255,0)"),
                opacity=0.15,
                showlegend=False,
                legendgroup=_ds,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )

        _fig_sa_conv.add_trace(
            go.Scatter(
                x=_xsb,
                y=_hb_mean.tolist(),
                mode="lines",
                name=f"{_ds} (meilleur)",
                line=dict(color=_color, width=2),
                legendgroup=_ds,
            ),
            row=1,
            col=2,
        )

    _fig_sa_conv.update_xaxes(title_text="Itération (centaine)")
    _fig_sa_conv.update_yaxes(title_text="Coût", col=1)
    _fig_sa_conv.update_layout(
        title=dict(
            text=f"Convergence SA<br><sup>{_params_txt}</sup>",
            x=0.5,
        ),
        height=500,
        hovermode="x unified",
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=-0.35, xanchor="center", x=0.5),
        margin=dict(b=120),
    )

    mo.vstack(
        [
            mo.md("### Courbe de convergence - Recuit Simulé"),
            mo.ui.plotly(_fig_sa_conv),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Résultats - Algorithme Génétique
    """)
    return


@app.cell(hide_code=True)
def _(ga_results):
    mo.stop(not ga_results)

    _rows = []
    for _ds, _runs in ga_results.items():
        _dists = [r["distance"] for r in _runs]
        _times = [r["time"] for r in _runs]
        _viols = [r["violation"] for r in _runs]
        _nb_gen = [r["nb_generated"] for r in _runs]
        _best_run = min(_runs, key=lambda r: r["distance"])
        _rows.append(
            {
                "Dataset": _ds,
                "Dist. moy.": f"{np.mean(_dists):.2f}",
                "Dist. min": f"{min(_dists):.2f}",
                "Dist. max": f"{max(_dists):.2f}",
                "Temps moy. (s)": f"{np.mean(_times):.2f}",
                "Violation moy.": f"{np.mean(_viols):.4f}",
                "Solutions générées (moy.)": f"{np.mean(_nb_gen):.0f}",
                "Routes (best)": _best_run["num_routes"],
                "Valide (best)": "✅" if _best_run["violation"] < 1e-6 else "❌",
            }
        )

    mo.ui.table(_rows)
    return


@app.cell(hide_code=True)
def _(
    all_datasets,
    ga_results,
    ui_ga_elite,
    ui_ga_gen,
    ui_ga_mut_rate,
    ui_ga_op_cross,
    ui_ga_op_mutate,
    ui_ga_pop,
    ui_ga_tourn,
    ui_repeats,
    ui_tw,
):
    mo.stop(not ga_results)

    _params_txt = (
        f"pop={ui_ga_pop.value} | gen={ui_ga_gen.value} | "
        f"tournament={ui_ga_tourn.value} | mutation={ui_ga_mut_rate.value} | "
        f"elite={ui_ga_elite.value} | mut_op={ui_ga_op_mutate.value} | "
        f"cross_op={ui_ga_op_cross.value} | repeats={ui_repeats.value} | TW={ui_tw.value}"
    )

    _tabs = {}
    for _ds, _runs in ga_results.items():
        _best = min(_runs, key=lambda r: r["distance"])
        _inst = all_datasets[_ds]
        _fig = plot_routes_interactive(
            _best["solution"],
            _inst,
            f"GA - {_ds} (distance={_best['distance']:.2f})",
            _params_txt,
        )
        _tabs[_ds] = mo.ui.plotly(_fig)

    mo.vstack(
        [
            mo.md("### Meilleures solutions GA par dataset"),
            mo.ui.tabs(_tabs),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    ga_results,
    ui_ga_elite,
    ui_ga_gen,
    ui_ga_mut_rate,
    ui_ga_op_cross,
    ui_ga_op_mutate,
    ui_ga_pop,
    ui_ga_tourn,
    ui_repeats,
    ui_tw,
):
    mo.stop(not ga_results)

    _params_txt = (
        f"pop={ui_ga_pop.value} | gen={ui_ga_gen.value} | "
        f"tournament={ui_ga_tourn.value} | mutation={ui_ga_mut_rate.value} | "
        f"elite={ui_ga_elite.value} | mut_op={ui_ga_op_mutate.value} | "
        f"cross_op={ui_ga_op_cross.value} | repeats={ui_repeats.value} | TW={ui_tw.value}"
    )

    _palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

    _fig_ga_conv = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Coût courant (GA)", "Meilleur coût (GA)"),
        shared_yaxes=False,
    )

    for _i, (_ds, _runs) in enumerate(ga_results.items()):
        _color = _palette[_i % len(_palette)]

        _min_h = min(len(r["history"]) for r in _runs)
        _min_hb = min(len(r["history_best"]) for r in _runs)

        _hist_arr = np.array([r["history"][:_min_h] for r in _runs])
        _histb_arr = np.array([r["history_best"][:_min_hb] for r in _runs])

        _h_mean = _hist_arr.mean(axis=0)
        _hb_mean = _histb_arr.mean(axis=0)
        _hb_std = _histb_arr.std(axis=0)

        _xs = list(range(len(_h_mean)))
        _xsb = list(range(len(_hb_mean)))

        _fig_ga_conv.add_trace(
            go.Scatter(
                x=_xs,
                y=_h_mean.tolist(),
                mode="lines",
                name=f"{_ds} (courant)",
                line=dict(color=_color, width=1, dash="dot"),
                opacity=0.6,
                legendgroup=_ds,
            ),
            row=1,
            col=1,
        )

        _fig_ga_conv.add_trace(
            go.Scatter(
                x=_xsb + _xsb[::-1],
                y=(_hb_mean + _hb_std).tolist() + (_hb_mean - _hb_std).tolist()[::-1],
                fill="toself",
                fillcolor=_color,
                line=dict(color="rgba(255,255,255,0)"),
                opacity=0.15,
                showlegend=False,
                legendgroup=_ds,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )

        _fig_ga_conv.add_trace(
            go.Scatter(
                x=_xsb,
                y=_hb_mean.tolist(),
                mode="lines",
                name=f"{_ds} (meilleur)",
                line=dict(color=_color, width=2),
                legendgroup=_ds,
            ),
            row=1,
            col=2,
        )

    _fig_ga_conv.update_xaxes(title_text="Génération")
    _fig_ga_conv.update_yaxes(title_text="Coût", col=1)
    _fig_ga_conv.update_layout(
        title=dict(
            text=f"Convergence GA<br><sup>{_params_txt}</sup>",
            x=0.5,
        ),
        height=500,
        hovermode="x unified",
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=-0.35, xanchor="center", x=0.5),
        margin=dict(b=120),
    )

    mo.vstack(
        [
            mo.md("### Courbe de convergence - Algorithme Génétique"),
            mo.ui.plotly(_fig_ga_conv),
        ]
    )
    return


@app.cell(hide_code=True)
def _(ga_results, sa_results):
    mo.stop(
        not sa_results or not ga_results,
        mo.md("*Lancez les deux algorithmes pour débloquer la comparaison.*"),
    )
    mo.md("## Comparaison SA vs GA")
    return


@app.cell(hide_code=True)
def _(ga_results, sa_results):
    mo.stop(not sa_results or not ga_results)

    _common_ds = [ds for ds in sa_results if ds in ga_results]
    _rows = []
    for _ds in _common_ds:
        _sa_runs = sa_results[_ds]
        _ga_runs = ga_results[_ds]

        _sa_dists = [r["distance"] for r in _sa_runs]
        _ga_dists = [r["distance"] for r in _ga_runs]
        _sa_times = [r["time"] for r in _sa_runs]
        _ga_times = [r["time"] for r in _ga_runs]
        _sa_viols = [r["violation"] for r in _sa_runs]
        _ga_viols = [r["violation"] for r in _ga_runs]
        _sa_best = min(_sa_runs, key=lambda r: r["distance"])
        _ga_best = min(_ga_runs, key=lambda r: r["distance"])

        _rows.append(
            {
                "Dataset": _ds,
                "SA - Dist. min": f"{min(_sa_dists):.2f}",
                "GA - Dist. min": f"{min(_ga_dists):.2f}",
                "SA - Dist. moy.": f"{np.mean(_sa_dists):.2f}",
                "GA - Dist. moy.": f"{np.mean(_ga_dists):.2f}",
                "SA - Temps moy. (s)": f"{np.mean(_sa_times):.2f}",
                "GA - Temps moy. (s)": f"{np.mean(_ga_times):.2f}",
                "SA - Violation moy.": f"{np.mean(_sa_viols):.4f}",
                "GA - Violation moy.": f"{np.mean(_ga_viols):.4f}",
                "SA - Routes (best)": _sa_best["num_routes"],
                "GA - Routes (best)": _ga_best["num_routes"],
                "SA - Valide": "✅" if _sa_best["violation"] < 1e-6 else "❌",
                "GA - Valide": "✅" if _ga_best["violation"] < 1e-6 else "❌",
            }
        )

    mo.vstack(
        [
            mo.md("### Tableau comparatif"),
            mo.ui.table(_rows),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    ga_results,
    sa_results,
    ui_ga_elite,
    ui_ga_gen,
    ui_ga_mut_rate,
    ui_ga_op_cross,
    ui_ga_op_mutate,
    ui_ga_pop,
    ui_ga_tourn,
    ui_repeats,
    ui_sa_T0,
    ui_sa_alpha,
    ui_sa_max_iter,
    ui_sa_n2,
    ui_sa_op,
    ui_tw,
):
    mo.stop(not sa_results or not ga_results)

    _common_ds = [ds for ds in sa_results if ds in ga_results]

    _palette_sa = ["#636EFA", "#00CC96", "#19D3F3", "#B6E880", "#AB63FA"]
    _palette_ga = ["#EF553B", "#FFA15A", "#FF6692", "#FECB52", "#FF97FF"]

    _sa_txt = (
        f"SA: T0={ui_sa_T0.value} | alpha={ui_sa_alpha.value} | "
        f"max_iter={ui_sa_max_iter.value} | n2={ui_sa_n2.value} | op={ui_sa_op.value}"
    )
    _ga_txt = (
        f"GA: pop={ui_ga_pop.value} | gen={ui_ga_gen.value} | "
        f"tourn={ui_ga_tourn.value} | mut={ui_ga_mut_rate.value} | "
        f"elite={ui_ga_elite.value} | mut_op={ui_ga_op_mutate.value} | cross={ui_ga_op_cross.value}"
    )
    _common_txt = f"repeats={ui_repeats.value} | TW={ui_tw.value}"

    _fig_conv = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"secondary_y": False}]],
    )

    for _i, _ds in enumerate(_common_ds):
        _color_sa = _palette_sa[_i % len(_palette_sa)]
        _color_ga = _palette_ga[_i % len(_palette_ga)]

        # --- SA ---
        _sa_runs = sa_results[_ds]
        _min_hb_sa = min(len(r["history_best"]) for r in _sa_runs)
        _sa_hb_arr = np.array([r["history_best"][:_min_hb_sa] for r in _sa_runs])
        _sa_hb_mean = _sa_hb_arr.mean(axis=0)
        _sa_hb_std = _sa_hb_arr.std(axis=0)
        _sa_pct = [100 * k / (len(_sa_hb_mean) - 1) for k in range(len(_sa_hb_mean))]

        _fig_conv.add_trace(
            go.Scatter(
                x=_sa_pct + _sa_pct[::-1],
                y=(_sa_hb_mean + _sa_hb_std).tolist()
                + (_sa_hb_mean - _sa_hb_std).tolist()[::-1],
                fill="toself",
                fillcolor=_color_sa,
                line=dict(color="rgba(255,255,255,0)"),
                opacity=0.15,
                showlegend=False,
                legendgroup=f"{_ds}_sa",
                hoverinfo="skip",
            )
        )
        _fig_conv.add_trace(
            go.Scatter(
                x=_sa_pct,
                y=_sa_hb_mean.tolist(),
                mode="lines",
                name=f"SA - {_ds}",
                line=dict(color=_color_sa, width=2),
                legendgroup=f"{_ds}_sa",
            )
        )

        # --- GA ---
        _ga_runs = ga_results[_ds]
        _min_hb_ga = min(len(r["history_best"]) for r in _ga_runs)
        _ga_hb_arr = np.array([r["history_best"][:_min_hb_ga] for r in _ga_runs])
        _ga_hb_mean = _ga_hb_arr.mean(axis=0)
        _ga_hb_std = _ga_hb_arr.std(axis=0)
        _ga_pct = [100 * k / (len(_ga_hb_mean) - 1) for k in range(len(_ga_hb_mean))]

        _fig_conv.add_trace(
            go.Scatter(
                x=_ga_pct + _ga_pct[::-1],
                y=(_ga_hb_mean + _ga_hb_std).tolist()
                + (_ga_hb_mean - _ga_hb_std).tolist()[::-1],
                fill="toself",
                fillcolor=_color_ga,
                line=dict(color="rgba(255,255,255,0)"),
                opacity=0.15,
                showlegend=False,
                legendgroup=f"{_ds}_ga",
                hoverinfo="skip",
            )
        )
        _fig_conv.add_trace(
            go.Scatter(
                x=_ga_pct,
                y=_ga_hb_mean.tolist(),
                mode="lines",
                name=f"GA - {_ds}",
                line=dict(color=_color_ga, width=2, dash="dash"),
                legendgroup=f"{_ds}_ga",
            )
        )

    _fig_conv.update_xaxes(title_text="Progression (%)", range=[0, 100])
    _fig_conv.update_yaxes(title_text="Meilleur coût trouvé (distance)")
    _fig_conv.update_layout(
        title=dict(
            text=f"Convergence fusionnée SA vs GA<br>"
            f"<sup>{_sa_txt}</sup><br>"
            f"<sup>{_ga_txt} | {_common_txt}</sup>",
            x=0.5,
            xanchor="center",
        ),
        height=520,
        hovermode="x unified",
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=140, b=120),
    )

    _fig_bar = make_subplots(specs=[[{"secondary_y": True}]])

    _ds_labels = _common_ds
    _sa_dist_min = [min(r["distance"] for r in sa_results[ds]) for ds in _ds_labels]
    _ga_dist_min = [min(r["distance"] for r in ga_results[ds]) for ds in _ds_labels]
    _sa_dist_mean = [
        float(np.mean([r["distance"] for r in sa_results[ds]])) for ds in _ds_labels
    ]
    _ga_dist_mean = [
        float(np.mean([r["distance"] for r in ga_results[ds]])) for ds in _ds_labels
    ]
    _sa_time_mean = [
        float(np.mean([r["time"] for r in sa_results[ds]])) for ds in _ds_labels
    ]
    _ga_time_mean = [
        float(np.mean([r["time"] for r in ga_results[ds]])) for ds in _ds_labels
    ]

    _fig_bar.add_trace(
        go.Bar(
            name="SA - Dist. min",
            x=_ds_labels,
            y=_sa_dist_min,
            marker_color="#636EFA",
            opacity=0.9,
            offsetgroup="SA",
        ),
        secondary_y=False,
    )

    _fig_bar.add_trace(
        go.Bar(
            name="GA - Dist. min",
            x=_ds_labels,
            y=_ga_dist_min,
            marker_color="#EF553B",
            opacity=0.9,
            offsetgroup="GA",
        ),
        secondary_y=False,
    )

    _fig_bar.add_trace(
        go.Bar(
            name="SA - Dist. moy.",
            x=_ds_labels,
            y=_sa_dist_mean,
            marker_color="#636EFA",
            opacity=0.45,
            offsetgroup="SA",
            base=0,
        ),
        secondary_y=False,
    )

    _fig_bar.add_trace(
        go.Bar(
            name="GA - Dist. moy.",
            x=_ds_labels,
            y=_ga_dist_mean,
            marker_color="#EF553B",
            opacity=0.45,
            offsetgroup="GA",
            base=0,
        ),
        secondary_y=False,
    )

    _fig_bar.add_trace(
        go.Scatter(
            name="SA - Temps moy. (s)",
            x=_ds_labels,
            y=_sa_time_mean,
            mode="lines+markers",
            line=dict(color="#636EFA", width=2, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
        ),
        secondary_y=True,
    )

    _fig_bar.add_trace(
        go.Scatter(
            name="GA - Temps moy. (s)",
            x=_ds_labels,
            y=_ga_time_mean,
            mode="lines+markers",
            line=dict(color="#EF553B", width=2, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
        ),
        secondary_y=True,
    )

    _fig_bar.update_yaxes(title_text="Distance", secondary_y=False)
    _fig_bar.update_yaxes(title_text="Temps moyen (s)", secondary_y=True)
    _fig_bar.update_layout(
        title=dict(
            text="Qualité et Temps - SA (bleu) vs GA (rouge)<br>"
            "<sup>Barres pleines = min, barres pâles = moy. | Tirets = temps</sup>",
            x=0.5,
            xanchor="center",
        ),
        barmode="group",
        height=480,
        hovermode="x unified",
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(t=100, b=140),
    )

    mo.vstack(
        [
            mo.md("### Convergence fusionnée (axe X normalisé en % de progression)"),
            mo.ui.plotly(_fig_conv),
            mo.md("### Qualité & Temps par dataset"),
            mo.ui.plotly(_fig_bar),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
