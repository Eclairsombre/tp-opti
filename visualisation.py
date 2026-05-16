import marimo as mo
import matplotlib.pyplot as plt


def plot_routes(solution, instance, title="Routes", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    depot = instance.depot

    # dépôt
    ax.scatter(
        depot["x"],
        depot["y"],
        s=250,
        marker="s",
        label="Depot"
    )

    # clients
    xs = [c["x"] for c in instance.clients]
    ys = [c["y"] for c in instance.clients]

    ax.scatter(xs, ys, s=40)

    # labels clients
    for i, client in enumerate(instance.clients, start=1):
        ax.text(
            client["x"],
            client["y"],
            str(i),
            fontsize=8
        )

    # routes
    for route_idx, route in enumerate(solution.routes):

        route_x = [depot["x"]]
        route_y = [depot["y"]]

        for client_id in route.clients:
            client = instance.nodes[client_id]

            route_x.append(client["x"])
            route_y.append(client["y"])

        # retour dépôt
        route_x.append(depot["x"])
        route_y.append(depot["y"])

        ax.plot(
            route_x,
            route_y,
            linewidth=2,
            label=f"Route {route_idx + 1}"
        )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    return fig