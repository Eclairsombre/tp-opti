import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_routes(solution, instance, title="Routes", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    depot = instance.depot

    ax.scatter(depot["x"], depot["y"], s=250, marker="s", label="Depot")

    xs = [c["x"] for c in instance.clients]
    ys = [c["y"] for c in instance.clients]

    ax.scatter(xs, ys, s=40)

    for i, client in enumerate(instance.clients, start=1):
        ax.text(client["x"], client["y"], str(i), fontsize=8)

    for route_idx, route in enumerate(solution.routes):
        route_x = [depot["x"]]
        route_y = [depot["y"]]

        for client_id in route.clients:
            client = instance.nodes[client_id]

            route_x.append(client["x"])
            route_y.append(client["y"])

        route_x.append(depot["x"])
        route_y.append(depot["y"])

        ax.plot(route_x, route_y, linewidth=2, label=f"Route {route_idx + 1}")

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    return fig


def plot_routes_interactive(solution, instance, title="Routes", subtitle=None):
    fig = go.Figure()

    colors = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#a65628",
        "#f781bf",
        "#999999",
    ]

    depot = instance.depot

    # --- Capacité et distance totale par route ---
    route_loads = []
    route_dists = []
    for route in solution.routes:
        load = sum(instance.nodes[c].get("demand", 0) for c in route.clients)
        route_loads.append(load)
        nodes = [depot] + [instance.nodes[c] for c in route.clients] + [depot]
        dist = sum(
            (
                (nodes[k + 1]["x"] - nodes[k]["x"]) ** 2
                + (nodes[k + 1]["y"] - nodes[k]["y"]) ** 2
            )
            ** 0.5
            for k in range(len(nodes) - 1)
        )
        route_dists.append(dist)

    # --- Dégradés gris ---
    clients = instance.clients

    def make_gray_scale(values):
        mn, mx = min(values), max(values)
        rng = mx - mn if mx != mn else 1
        result = []
        for v in values:
            t = (v - mn) / rng
            val = int(245 - t * (245 - 51))
            result.append(f"rgb({val},{val},{val})")
        return result

    demands = [c.get("demand", 0) for c in clients]
    dues = [c.get("due", 0) for c in clients]

    demand_colors = make_gray_scale(demands)
    due_colors = make_gray_scale(dues)

    # --- Traces des routes ---
    n_route_traces = len(solution.routes)
    for route_idx, route in enumerate(solution.routes):
        color = colors[route_idx % len(colors)]
        load = route_loads[route_idx]
        dist = route_dists[route_idx]
        nodes = [depot] + [instance.nodes[c] for c in route.clients] + [depot]
        xs = [n["x"] for n in nodes]
        ys = [n["y"] for n in nodes]

        hover = []
        cum_dist = 0.0
        labels = ["Depot"] + [str(c) for c in route.clients] + ["Depot"]
        for k in range(len(nodes)):
            if k > 0:
                dx = nodes[k]["x"] - nodes[k - 1]["x"]
                dy = nodes[k]["y"] - nodes[k - 1]["y"]
                cum_dist += (dx**2 + dy**2) ** 0.5
            hover.append(
                f"<b>{labels[k]}</b><br>Stop #{k}<br>Cumul dist: {cum_dist:.1f}"
            )

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"Route {route_idx + 1} — {load}/{instance.capacity} — {dist:.1f}",
                line=dict(color=color, width=2),
                hovertext=hover,
                hoverinfo="text",
            )
        )

        for k in range(len(nodes) - 1):
            mx_ = (nodes[k]["x"] + nodes[k + 1]["x"]) / 2
            my_ = (nodes[k]["y"] + nodes[k + 1]["y"]) / 2
            dx = nodes[k + 1]["x"] - nodes[k]["x"]
            dy = nodes[k + 1]["y"] - nodes[k]["y"]
            fig.add_annotation(
                x=mx_,
                y=my_,
                ax=mx_ - dx * 0.01,
                ay=my_ - dy * 0.01,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowcolor=color,
                arrowwidth=1.5,
            )

    # --- Trace clients ---
    cx = [c["x"] for c in clients]
    cy = [c["y"] for c in clients]

    def client_hover_demand():
        return [
            f"<b>Client {i + 1}</b><br>Demande : {c.get('demand', '?')}"
            for i, c in enumerate(clients)
        ]

    def client_hover_due():
        return [
            f"<b>Client {i + 1}</b><br>Fenêtre de temps : {c.get('due', '?')}"
            for i, c in enumerate(clients)
        ]

    fig.add_trace(
        go.Scatter(
            x=cx,
            y=cy,
            mode="markers+text",
            name="Clients",
            marker=dict(
                size=12,
                color=demand_colors,
                symbol="circle",
                line=dict(color="#666666", width=1.2),
            ),
            text=[str(i + 1) for i in range(len(clients))],
            textposition="top center",
            textfont=dict(size=9, color="#333333"),
            hovertext=client_hover_demand(),
            hoverinfo="text",
        )
    )

    client_trace_idx = n_route_traces

    # --- Dépôt ---
    fig.add_trace(
        go.Scatter(
            x=[depot["x"]],
            y=[depot["y"]],
            mode="markers+text",
            name="Dépôt",
            marker=dict(
                size=18,
                color="#cc0000",
                symbol="square",
                line=dict(color="white", width=1.5),
            ),
            text=["D"],
            textposition="top center",
            textfont=dict(size=10, color="#cc0000"),
            hovertext=[f"<b>Dépôt</b><br>x={depot['x']}, y={depot['y']}"],
            hoverinfo="text",
        )
    )

    # --- Boutons switch coloration clients ---
    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            x=0.0,
            xanchor="left",
            y=1.06,
            yanchor="top",
            showactive=True,
            bgcolor="white",
            bordercolor="#aaaaaa",
            borderwidth=1,
            font=dict(size=12, color="#333333"),
            buttons=[
                dict(
                    label="Demande",
                    method="update",
                    args=[
                        {
                            "marker.color": [demand_colors],
                            "hovertext": [client_hover_demand()],
                        },
                        {},
                        [client_trace_idx],
                    ],
                ),
                dict(
                    label="Fenêtre de temps",
                    method="update",
                    args=[
                        {
                            "marker.color": [due_colors],
                            "hovertext": [client_hover_due()],
                        },
                        {},
                        [client_trace_idx],
                    ],
                ),
            ],
        )
    ]

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>{subtitle or ''}</sup>",
            x=0.5,
            xanchor="center",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridcolor="#e8e8e8",
            zeroline=False,
        ),
        yaxis=dict(showgrid=True, gridcolor="#e8e8e8", zeroline=False),
        legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
        updatemenus=updatemenus,
        hovermode="closest",
        height=680,
        margin=dict(t=100),
    )

    return fig
