from typing import Optional, List, Union

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import chex
import jax
import jax.numpy as jnp

from molboil.utils.graph import get_senders_and_receivers_fully_connected
from functools import partial
import matplotlib


def plot_and_maybe_save(
    plotter,
    state: chex.ArrayTree,
    key: chex.PRNGKey,
    iteration_n: int,
    save: bool,
    plots_dir: str,
):
    figures = plotter(state, key)
    for j, figure in enumerate(figures):
        if save:
            figure.savefig(
                os.path.join(plots_dir, "plot_%03i_iter_%08i.png" % (j, iteration_n))
            )
        else:
            plt.show()
        plt.close(figure)


def plot_history(history):
    """Agnostic history plotter for quickly plotting a dictionary of logging info."""
    figure, axs = plt.subplots(len(history), 1, figsize=(7, 3 * len(history.keys())))
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


def get_pairwise_distances_for_plotting(
    samples: chex.Array, n_vertices: Optional[int] = None, max_distance: float = 7.99
):
    """Get flattened array of pairwise interatomic distances for a batch. Only the first `n_vertices` atoms
    will be included if `n_vertices` is not None. Setting the max_distance is useful as it ensures that samples greater
    than the `max_distance` can be binned into the largest bin when this function is used with `get_counts`.
    """
    chex.assert_rank(samples, 3)  # [ batch_size, n_nodes, dim]
    n_vertices = samples.shape[1] if n_vertices is None else n_vertices
    n_vertices = min(samples.shape[1], n_vertices)
    senders, receivers = get_senders_and_receivers_fully_connected(n_nodes=n_vertices)
    norms = jnp.linalg.norm(samples[:, senders] - samples[:, receivers], axis=-1)
    d = norms.flatten()
    d = d.clip(max=max_distance)  # Clip keep plot reasonable.
    return d


def get_counts(
    distances: chex.Array,
    bins: chex.Array = jnp.linspace(0.0, 8.0, num=50),
    normalize: bool = True,
):
    """Get counts of distances within each of the bins.
    Can then be passed to matplotlib using `plt.stairs(counts, bins, alpha=0.4, fill=True)`.
    """
    chex.assert_rank(distances, 1)
    count_fn = lambda lower, upper: jnp.sum((distances >= lower) & (distances < upper))
    counts = jax.vmap(count_fn)(bins[:-1], bins[1:])
    if normalize:
        counts = counts / distances.shape[0]  # normalize.
    return counts


@partial(jax.jit, static_argnums=(1, 2, 3))
def bin_samples_by_dist(
    samples_list: List[chex.Array],
    max_distance: float = 100.0,
    max_bin_fallback: int = 10.0,
    num_bins: int = 100,
):
    """Get bins and counts for list of sample arrays."""
    distance_list = []
    dist_max_list = []
    for samples in samples_list:
        distance = get_pairwise_distances_for_plotting(
            samples, max_distance=max_distance
        )
        distance = jnp.where(jnp.isfinite(distance), distance, -1)
        distance_list.append(distance)
        dist_max_list.append(jnp.nanmax(distance_list[-1]))

    max_dist = jnp.nanmax(jnp.array(dist_max_list))
    max_dist = jnp.where(jnp.isfinite(max_dist), max_dist, max_bin_fallback)
    bins = jnp.linspace(0, max_dist + 0.05, num_bins)

    count_list = []
    for distance in distance_list:
        count_list.append(get_counts(distance, bins))

    return bins, count_list


def fast_particle_distance_hist(
    ax: matplotlib.axes.Axes,
    samples_list: List[chex.Array],
    label_list: List[str] = None,
    color_list: List[Union[str, int]] = None,
    max_distance: int = 100,
    max_bin_fallback: int = 10.0,
    num_bins: int = 100,
):
    """Plot histograms of the pairwise interatomic distances
    for lists of samples.

    Args:
        ax (matplotlib.axes.Axes): Axes to plot on.
        samples_list (List[chex.Array]): List of sample arrays to plot.
            Each array will generate a different histogram.
        label_list (List[str], optional): List of labels for each sample array. Defaults to None.
        color_list (List[Union[str, int]], optional): List of colors foreach sample array. Defaults to None.
        max_distance (int, optional): Maximum pairwise interactomic distnace. Defaults to 100.
        max_bin_fallback (int, optional): Maximum pairwise distance in case all entries of sample arrays are nan. Defaults to 10.0.
        num_bins (int, optional): Number of bins in histograms. Defaults to 100.

    Returns:
        _type_: _description_
    """

    # TODO: Fix bug by which the histogram max distance is set to 8.

    c = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    bins, count_list = bin_samples_by_dist(
        samples_list,
        max_distance=max_distance,
        max_bin_fallback=max_bin_fallback,
        num_bins=num_bins,
    )
    artists = []
    for idx, counts in enumerate(count_list):
        color = color_list[idx] if color_list is not None else c[idx]
        label = label_list[idx] if label_list is not None else None
        artists.append(
            ax.stairs(counts, bins, alpha=0.4, fill=True, label=label, color=color)
        )

    return artists
