import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import e3nn_jax as e3nn


def plot_history(history):
    """Agnostic history plotter for quickly plotting a dictionary of logging info."""
    figure, axs = plt.subplots(len(history), 1, figsize=(7, 3*len(history.keys())))
    if len(history.keys()) == 1:
        axs = [axs]  # make iterable
    elif len(history.keys()) == 0:
        return
    for i, key in enumerate(history):
        data = pd.Series(history[key])
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        if sum(data.isna()) > 0:
            data = data.dropna()
            print(f"NaN encountered in {key} history")
        axs[i].plot(data)
        axs[i].set_title(key)
    plt.tight_layout()


def plot_points_and_vectors(positions, max_radius = 1000.):
    """Plots positions, with vectors between them."""
    senders, receivers = e3nn.radius_graph(positions, r_max=max_radius)
    vectors = positions[senders] - positions[receivers]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if len(positions.shape) == 2:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=20)
        ax.quiver(positions[receivers, 0], positions[receivers, 1], positions[receivers, 2],
                  vectors[:, 0], vectors[:, 1], vectors[:, 2], alpha=0.4)
    else:
        # positions.shape: [n_nodes, n_vectors, 3], plot each n_vectors in different colours
        assert len(positions.shape) == 3
        c = plt.get_cmap('Set1')(np.linspace(0.2, 0.7, positions.shape[1]))
        for i in range(positions.shape[1]):
            ax.scatter(positions[:, i, 0], positions[:, i, 1], positions[:, i, 2], s=20, color=c[i])
            ax.quiver(positions[receivers, i, 0], positions[receivers, i, 1], positions[receivers, i, 2],
                      vectors[:, i, 0], vectors[:, i, 1], vectors[:, i, 2], alpha=0.4, color=c[i])
    return fig, ax
