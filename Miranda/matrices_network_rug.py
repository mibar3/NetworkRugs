import matplotlib.pyplot as plt
from ba_utils.network_rugs import draw_networkrug


def draw_networkrug_matrix_with_frederike_wrapper(
    graphs,
    orders,
    color_encodings,
    pixel_size=6,
    figsize_per_cell=(4, 2),
    start_nodes=None
):
    n_rows = len(color_encodings)
    n_cols = len(orders)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_cell[0] * n_cols,
                 figsize_per_cell[1] * n_rows),
        squeeze=False
    )

    for i, color in enumerate(color_encodings):
        for j, (order_name, order_type) in enumerate(orders.items()):

            ax = axes[i, j]

            draw_networkrug(
                graphs=graphs,
                order=order_type,
                start_nodes=start_nodes,
                color_encoding=color,
                pixel_size=pixel_size,
                ax=ax
            )

            if i == 0:
                ax.set_title(order_name)

        fig.text(
            0.01,
            1 - (i + 0.5) / n_rows,
            color,
            rotation=90,
            va="center",
            ha="left"
        )

    plt.tight_layout(rect=(0.06, 0, 1, 1))
    return fig