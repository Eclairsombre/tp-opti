import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns", app_title="VRPTW - Benchmark")

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
    # VRPTW - Benchmark
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
    ## Configuration globale
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
    ui_algo = mo.ui.dropdown(
        options=["SA (Recuit Simulé)", "GA (Algorithme Génétique)"],
        value="SA (Recuit Simulé)",
        label="Algorithme",
    )

    mo.vstack(
        [
            mo.hstack([ui_datasets, ui_tw], justify="start", gap=2),
            mo.hstack([ui_repeats, ui_algo], justify="start", gap=2),
        ]
    )
    return ui_algo, ui_datasets, ui_repeats, ui_tw


@app.cell(hide_code=True)
def _(ui_algo):
    _is_sa = "SA" in ui_algo.value
    mo.md("## Paramètres - Recuit Simulé") if _is_sa else mo.md("")
    return


@app.cell(hide_code=True)
def _(ui_algo):
    _is_sa = "SA" in ui_algo.value

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
    ) if _is_sa else mo.md("*(paramètres SA non actifs)*")
    return ui_sa_T0, ui_sa_alpha, ui_sa_max_iter, ui_sa_n2, ui_sa_op


@app.cell(hide_code=True)
def _(ui_algo):
    _is_ga = "GA (Algorithme Génétique)" in ui_algo.value
    mo.md("## Paramètres - Algorithme Génétique") if _is_ga else mo.md("")
    return


@app.cell(hide_code=True)
def _(ui_algo):
    _is_ga = "GA (Algorithme Génétique)" in ui_algo.value

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
    ) if _is_ga else mo.md("*(paramètres Genetic non actifs)*")
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
    mo.md("""
    ## Lancement
    """)
    return


@app.cell(hide_code=True)
def _():
    run_btn = mo.ui.run_button(label="▶ Lancer le benchmark")
    run_btn
    return (run_btn,)


@app.cell(hide_code=True)
def _(
    all_datasets,
    run_btn,
    ui_algo,
    ui_datasets,
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
    mo.stop(not run_btn.value, mo.md("*Appuyez sur Lancer pour démarrer.*"))

    _is_sa = "SA" in ui_algo.value
    _selected = ui_datasets.value or list(all_datasets.keys())[:1]
    _n_rep = ui_repeats.value
    _check_tw = ui_tw.value

    _all_results = {}  # {dataset_name: [result_dict, ...]}

    for _ds_name in _selected:
        _inst = all_datasets[_ds_name]
        _runs = []
        for _rep in range(_n_rep):
            _seed = 25565 + _rep
            if _is_sa:
                _r = simulated_annealing(
                    _inst,
                    check_tw=_check_tw,
                    T0=ui_sa_T0.value,
                    alpha=ui_sa_alpha.value,
                    max_iter=ui_sa_max_iter.value,
                    n2=ui_sa_n2.value,
                    seed=_seed,
                    op=ui_sa_op.value,
                )
            else:
                _r = genetic_algorithm(
                    _inst,
                    check_tw=_check_tw,
                    pop_size=ui_ga_pop.value,
                    generations=ui_ga_gen.value,
                    tournament_k=ui_ga_tourn.value,
                    mutation_rate=ui_ga_mut_rate.value,
                    elite_size=ui_ga_elite.value,
                    seed=_seed,
                    op_mutate=ui_ga_op_mutate.value,
                    op_cross=ui_ga_op_cross.value,
                )
            _runs.append(_r)
        _all_results[_ds_name] = _runs

    all_results = _all_results
    mo.md(
        f"✅ **Benchmark terminé** — {len(_selected)} dataset(s) × {_n_rep} répétition(s)"
    )
    return (all_results,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Résultats - Tableau récapitulatif
    """)
    return


@app.cell(hide_code=True)
def _(all_results):
    mo.stop(not all_results)

    _rows = []
    for _ds, _runs in all_results.items():
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
def _():
    mo.md("""
    ## Meilleure solution par dataset
    """)
    return


@app.cell(hide_code=True)
def _(
    all_datasets,
    all_results,
    ui_algo,
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
    ui_sa_op,
    ui_tw,
):
    mo.stop(not all_results)

    _is_sa = "SA" in ui_algo.value

    # Texte paramètres affiché sous titre
    if _is_sa:
            _params_txt = (
                f"T0={ui_sa_T0.value} | "
                f"alpha={ui_sa_alpha.value} | "
                f"max_iter={ui_sa_max_iter.value} | "
                f"op={ui_sa_op.value} | "
                f"repeats={ui_repeats.value} | "
                f"TW={ui_tw.value}"
            )
    else:
        _params_txt = (
            f"pop={ui_ga_pop.value} | "
            f"gen={ui_ga_gen.value} | "
            f"tournament={ui_ga_tourn.value} | "
            f"mutation={ui_ga_mut_rate.value} | "
            f"elite={ui_ga_elite.value} | "
            f"mut_op={ui_ga_op_mutate.value} | "
            f"cross_op={ui_ga_op_cross.value} | "
            f"repeats={ui_repeats.value} | "
            f"TW={ui_tw.value}"
        )

    _tabs = {}
    for _ds, _runs in all_results.items():
        _best = min(_runs, key=lambda r: r["distance"])
        _inst = all_datasets[_ds]
        _fig = plot_routes_interactive(
            _best["solution"],
            _inst,
            f"{_best['algorithm']} - {_ds} (distance={_best['distance']:.2f})",
            _params_txt
        )
        _tabs[_ds] = mo.ui.plotly(_fig)

    mo.ui.tabs(_tabs)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Courbes de convergence
    """)
    return


@app.cell(hide_code=True)
def _(
    all_results,
    ui_algo,
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
    ui_sa_op,
    ui_tw,
):
    mo.stop(not all_results)

    _is_sa = "SA" in ui_algo.value
    _x_label = "Itération (Centaine)" if _is_sa else "Génération"
    _title = "Coût en fonction des itérations" if _is_sa else "Coût en fonction des générations"

    _fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Coût courant", "Meilleur coût trouvé"),
        shared_yaxes=False,
    )

    _palette = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
    ]

    # Texte paramètres affiché sous titre
    if _is_sa:
            _params_txt = (
                f"T0={ui_sa_T0.value} | "
                f"alpha={ui_sa_alpha.value} | "
                f"max_iter={ui_sa_max_iter.value} | "
                f"op={ui_sa_op.value} | "
                f"repeats={ui_repeats.value} | "
                f"TW={ui_tw.value}"
            )
    else:
        _params_txt = (
            f"pop={ui_ga_pop.value} | "
            f"gen={ui_ga_gen.value} | "
            f"tournament={ui_ga_tourn.value} | "
            f"mutation={ui_ga_mut_rate.value} | "
            f"elite={ui_ga_elite.value} | "
            f"mut_op={ui_ga_op_mutate.value} | "
            f"cross_op={ui_ga_op_cross.value} | "
            f"repeats={ui_repeats.value} | "
            f"TW={ui_tw.value}"
        )

    for _i, (_ds, _runs) in enumerate(all_results.items()):
        _color = _palette[_i % len(_palette)]

        # Moyenne des historiques sur les runs (longueurs potentiellement identiques)
        _min_len_h = min(len(r["history"]) for r in _runs)
        _min_len_hb = min(len(r["history_best"]) for r in _runs)

        _hist_arr = np.array([r["history"][:_min_len_h] for r in _runs])
        _histb_arr = np.array([r["history_best"][:_min_len_hb] for r in _runs])

        _h_mean = _hist_arr.mean(axis=0)
        _hb_mean = _histb_arr.mean(axis=0)
        _hb_std = _histb_arr.std(axis=0)

        _xs = list(range(len(_h_mean)))
        _xsb = list(range(len(_hb_mean)))

        # Coût courant (trait fin)
        _fig.add_trace(
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

        # Meilleur coût (trait épais) + zone d'écart-type
        _fig.add_trace(
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

        _fig.add_trace(
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

    _fig.update_xaxes(title_text=_x_label)
    _fig.update_yaxes(title_text="Coût", col=1)
    _fig.update_layout(
        title=dict(
            text=f"{_title}<br><sup>{_params_txt or ''}</sup>",
            x=0.5,
        ),
        height=500,
        hovermode="x unified",
        # fond
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        # légende plus basse
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.35,
            xanchor="center",
            x=0.5,
        ),
        # marge bas pour laisser place légende
        margin=dict(b=120),
    )

    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Analyse de sensibilité - 1 paramètre

    Les autres paramètres sont **fixés aux valeurs des sliders ci-dessus**.
    """)
    return


@app.cell(hide_code=True)
def _(ui_algo):
    _is_sa = "SA" in ui_algo.value

    _sa_params = ["T0", "alpha", "max_iter","n2"]
    _ga_params = [
        "pop_size",
        "generations",
        "tournament_k",
        "mutation_rate",
        "elite_size",
    ]

    _param_options = _sa_params if _is_sa else _ga_params

    ui_sweep_param = mo.ui.dropdown(
        options=_param_options,
        value=_param_options[0],
        label="Paramètre à balayer",
    )
    ui_sweep_n = mo.ui.slider(3, 15, value=6, step=1, label="Nombre de valeurs testées")

    mo.hstack([ui_sweep_param, ui_sweep_n], justify="start", gap=2)
    return ui_sweep_n, ui_sweep_param


@app.cell(hide_code=True)
def _(ui_algo, ui_sweep_param):
    _is_sa = "SA" in ui_algo.value
    _p = ui_sweep_param.value

    # Plages par défaut pour chaque paramètre
    _ranges = {
        "T0": (10, 4000
               , False),
        "alpha": (0.90, 0.9999, False),
        "max_iter": (500, 1000000, False),
        "n2": (10,100, False),
        "pop_size": (5, 200, True),
        "generations": (10, 2000, True),
        "tournament_k": (1, 20, True),
        "mutation_rate": (0.01, 0.80, False),
        "elite_size": (1, 10, True),
    }

    _lo, _hi, _is_int = _ranges[_p]

    ui_sweep_min = mo.ui.number(_lo, _hi, value=_lo, label=f"{_p} min")
    ui_sweep_max = mo.ui.number(_lo, _hi, value=_hi, label=f"{_p} max")

    mo.hstack([ui_sweep_min, ui_sweep_max], justify="start", gap=2)
    return ui_sweep_max, ui_sweep_min


@app.cell(hide_code=True)
def _():
    sweep_btn = mo.ui.run_button(label="▶ Lancer le balayage")
    sweep_btn
    return (sweep_btn,)


@app.cell(hide_code=True)
def _(
    all_datasets,
    sweep_btn,
    ui_algo,
    ui_datasets,
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
    ui_sweep_max,
    ui_sweep_min,
    ui_sweep_n,
    ui_sweep_param,
    ui_tw,
):
    mo.stop(
        not sweep_btn.value, mo.md("*Appuyez sur 'Lancer le balayage' pour démarrer.*")
    )

    _is_sa = "SA" in ui_algo.value
    _param = ui_sweep_param.value
    _n_vals = ui_sweep_n.value
    _check_tw = ui_tw.value
    _n_rep = ui_repeats.value
    _selected = ui_datasets.value or list(all_datasets.keys())[:1]

    # Génération des valeurs à tester
    _is_int_param = _param in [
        "max_iter",
        "n2",
        "pop_size",
        "generations",
        "tournament_k",
        "elite_size",
    ]
    if _is_int_param:
        _sweep_vals = [
            int(v) for v in np.linspace(ui_sweep_min.value, ui_sweep_max.value, _n_vals)
        ]
    else:
        _sweep_vals = list(np.linspace(ui_sweep_min.value, ui_sweep_max.value, _n_vals))

    # Paramètres de base (fixes)
    _base_sa = dict(
        T0=ui_sa_T0.value,
        alpha=ui_sa_alpha.value,
        max_iter=ui_sa_max_iter.value,
        n2=ui_sa_n2.value,
        op=ui_sa_op.value,
    )
    _base_ga = dict(
        pop_size=ui_ga_pop.value,
        generations=ui_ga_gen.value,
        tournament_k=ui_ga_tourn.value,
        mutation_rate=ui_ga_mut_rate.value,
        elite_size=ui_ga_elite.value,
        op_mutate=ui_ga_op_mutate.value,
        op_cross=ui_ga_op_cross.value,
    )

    # Sweep
    _sweep_results = []  # [{param_val, dataset, dist_mean, time_mean, dist_std}, ...]

    for _val in _sweep_vals:
        for _ds in _selected:
            _inst = all_datasets[_ds]
            _dists, _times = [], []
            for _rep in range(_n_rep):
                _seed = 42 + _rep
                if _is_sa:
                    _kw = {**_base_sa, _param: _val}
                    _r = simulated_annealing(
                        _inst, check_tw=_check_tw, seed=_seed, **_kw
                    )
                else:
                    _kw = {**_base_ga, _param: _val}
                    _r = genetic_algorithm(_inst, check_tw=_check_tw, seed=_seed, **_kw)
                _dists.append(_r["distance"])
                _times.append(_r["time"])
            _sweep_results.append(
                {
                    "param_val": _val,
                    "dataset": _ds,
                    "dist_mean": float(np.mean(_dists)),
                    "dist_std": float(np.std(_dists)),
                    "time_mean": float(np.mean(_times)),
                    "time_std": float(np.std(_times)),
                }
            )

    sweep_results = _sweep_results
    mo.md(
        f"✅ Sweep terminé - {len(_sweep_vals)} valeurs × {len(_selected)} dataset(s) × {_n_rep} répétition(s)"
    )
    return (sweep_results,)


@app.cell(hide_code=True)
def _(
    sweep_results,
    ui_algo,
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
    ui_sweep_param,
    ui_tw,
):
    mo.stop(not sweep_results)

    _param = ui_sweep_param.value
    _is_sa = "SA" in ui_algo.value

    _palette = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
    ]

    # Texte paramètres affiché sous titre
    if _is_sa:
        _params = {
            "T0": ui_sa_T0.value,
            "alpha": ui_sa_alpha.value,
            "max_iter": ui_sa_max_iter.value,
            "n2":ui_sa_n2.value,
            "op": ui_sa_op.value,
            "repeats": ui_repeats.value,
            "TW": ui_tw.value,
        }
    else:
        _params = {
            "pop_size": ui_ga_pop.value,
            "generations": ui_ga_gen.value,
            "tournament_k": ui_ga_tourn.value,
            "mutation_rate": ui_ga_mut_rate.value,
            "elite_size": ui_ga_elite.value,
            "op_mutate": ui_ga_op_mutate.value,
            "op_cross": ui_ga_op_cross.value,
            "repeats": ui_repeats.value,
            "TW": ui_tw.value,
        }

    # Supprimer paramètre sweep du texte
    _params.pop(_param, None)

    _params_txt = " | ".join(
        f"{k}={v}" for k, v in _params.items()
    )

    # Regrouper par dataset
    _datasets_seen = list(dict.fromkeys(r["dataset"] for r in sweep_results))

    _fig = make_subplots(
        specs=[[{"secondary_y": True}]],
    )

    for _i, _ds in enumerate(_datasets_seen):
        _color = _palette[_i % len(_palette)]

        _rows = [r for r in sweep_results if r["dataset"] == _ds]

        _xs = [r["param_val"] for r in _rows]
        _yd = [r["dist_mean"] for r in _rows]
        _yd_std = [r["dist_std"] for r in _rows]
        _yt = [r["time_mean"] for r in _rows]

        # Bande distance ±1σ
        _fig.add_trace(
            go.Scatter(
                x=_xs + _xs[::-1],
                y=[d + s for d, s in zip(_yd, _yd_std)]
                + [d - s for d, s in zip(_yd, _yd_std)][::-1],
                fill="toself",
                fillcolor=_color,
                line=dict(color="rgba(255,255,255,0)"),
                opacity=0.15,
                showlegend=False,
                hoverinfo="skip",
            ),
            secondary_y=False,
        )

        # Distance
        _fig.add_trace(
            go.Scatter(
                x=_xs,
                y=_yd,
                mode="lines+markers",
                name=f"{_ds} – distance",
                line=dict(color=_color, width=2),
                marker=dict(size=7),
            ),
            secondary_y=False,
        )

        # Temps
        _fig.add_trace(
            go.Scatter(
                x=_xs,
                y=_yt,
                mode="lines+markers",
                name=f"{_ds} – temps (s)",
                line=dict(color=_color, width=2, dash="dash"),
                marker=dict(size=5, symbol="diamond"),
            ),
            secondary_y=True,
        )

    _fig.update_xaxes(title_text=_param)

    _fig.update_yaxes(
        title_text="Distance moyenne",
        secondary_y=False,
    )

    _fig.update_yaxes(
        title_text="Temps moyen (s)",
        secondary_y=True,
    )

    _fig.update_layout(
        title=dict(
            text=f"Qualité et Temps en fonction de {_param}<br><sup>{_params_txt or ''}</sup>",
            x=0.5,
            xanchor="center",
        ),
        height=500,
        hovermode="x unified",
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.35,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            t=120,
            b=120,
        ),
    )

    mo.ui.plotly(_fig)
    return


if __name__ == "__main__":
    app.run()
