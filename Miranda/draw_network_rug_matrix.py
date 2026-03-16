import matplotlib.pyplot as plt
from ba_utils.visualization import draw_rug_from_graphs

def draw_network_rug_matrix(
    graphs,
    orderings,
    color_encodings,
    pixel_size=6,
    figsize_per_cell=(4, 2),
    start_nodes=None
):
    n_rows = len(color_encodings)
    n_cols = len(orderings)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False
    )
    for i, color in enumerate(color_encodings):
        for j, (ordering_name, ordering_fn) in enumerate(orderings.items()):
            ax = axes[i, j]
            # Pass start_node if provided for this ordering
            start_node = start_nodes.get(ordering_name) if start_nodes else None
            ordering = ordering_fn(graphs, start_nodes=start_node) if start_node is not None else ordering_fn(graphs)
            draw_rug_from_graphs(
                graphs_data=graphs,
                ordering=ordering,
                color_encoding=color,
                pixel_size=pixel_size,
                ax=ax
            )
            if i == 0:
                ax.set_title(ordering_name)
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