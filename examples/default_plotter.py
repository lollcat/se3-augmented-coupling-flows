from typing import Callable, Optional, List

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
from flow.aug_flow_dist import FullGraphSample, AugmentedFlow, AugmentedFlowParams
from utils.graph import get_senders_and_receivers_fully_connected


mpl.rcParams['figure.dpi'] = 150
PlottingBatchSize = int
TrainData = FullGraphSample
TestData = FullGraphSample
FLowPlotter = Callable[[AugmentedFlowParams, AugmentedFlow, chex.PRNGKey, PlottingBatchSize,
                        TestData, TrainData], List[plt.Figure]]

def plot_sample_hist(samples,
                     ax: plt.Axes,
                     n_vertices: Optional[int] = None,
                     max_distance = 10, *args, **kwargs):
    """n_vertices argument allows us to look at pairwise distances for subset of vertices,
    to prevent plotting taking too long"""
    n_vertices = samples.shape[1] if n_vertices is None else n_vertices
    n_vertices = min(samples.shape[1], n_vertices)
    senders, receivers = get_senders_and_receivers_fully_connected(n_nodes=n_vertices)
    norms = jnp.linalg.norm(samples[:, senders] - samples[:, receivers], axis=-1)
    d = norms.flatten()
    d = d[jnp.isfinite(d)]
    d = d.clip(max=max_distance)  # Clip keep plot reasonable.
    ax.hist(d, bins=50, density=True, alpha=0.4, *args, **kwargs)


def default_plotter(params: AugmentedFlowParams,
                    flow: AugmentedFlow,
                    key: chex.PRNGKey,
                    n_samples: int,
                    train_data: FullGraphSample,
                    test_data: FullGraphSample,
                    plotting_n_nodes: Optional[int] = None):

    # Plot interatomic distance histograms.
    key1, key2 = jax.random.split(key)
    joint_samples_flow = jax.jit(flow.sample_apply, static_argnums=3)(params, train_data.features[0], key1,
                                                                      (n_samples,))
    features, positions_x, positions_a = jax.jit(flow.joint_to_separate_samples)(joint_samples_flow)
    positions_x_target = test_data.positions[:n_samples]
    positions_a_target = jax.jit(flow.aux_target_sample_n_apply)(params.aux_target, test_data[:n_samples], key2)

    # Plot original coords
    fig1, axs = plt.subplots(1, 2, figsize=(10, 5))
    plot_sample_hist(positions_x, axs[0], label="flow samples", n_vertices=plotting_n_nodes)
    plot_sample_hist(positions_x, axs[1], label="flow samples", n_vertices=plotting_n_nodes)
    plot_sample_hist(train_data.positions[:n_samples], axs[0],  label="train samples", n_vertices=plotting_n_nodes)
    plot_sample_hist(test_data.positions[:n_samples], axs[1],  label="test samples", n_vertices=plotting_n_nodes)

    axs[0].set_title(f"norms between original coordinates train")
    axs[1].set_title(f"norms between original coordinates test")
    axs[0].legend()
    axs[1].legend()
    fig1.tight_layout()

    # Augmented info.
    fig2, axs2 = plt.subplots(1, flow.n_augmented, figsize=(5*flow.n_augmented, 5))
    axs2 = [axs2] if isinstance(axs2, plt.Subplot) else axs2
    for i in range(flow.n_augmented):
        positions_a_single = positions_a[:, :, i]  # get single group of augmented coordinates
        positions_a_target_single = positions_a_target[:, :, i]  # Get first set of aux variables.
        chex.assert_equal_shape((positions_x, positions_a_single, positions_a_target_single))
        plot_sample_hist(positions_a_single, axs2[i], label="flow samples", n_vertices=plotting_n_nodes)
        plot_sample_hist(positions_a_target_single, axs2[i], label="test samples", n_vertices=plotting_n_nodes)
        axs2[i].set_title(f"norms between augmented coordinates (aug group {i})")
    axs2[0].legend()
    fig2.tight_layout()

    # Plot histogram (x - a)

    fig3, axs3 = plt.subplots(1, flow.n_augmented, figsize=(5*flow.n_augmented, 5))
    axs3 = [axs3] if isinstance(axs3, plt.Subplot) else axs3
    for i in range(flow.n_augmented):
        positions_a_single = positions_a[:, :, i]  # get single group of augmented coordinates
        positions_a_target_single = positions_a_target[:, :, i]  # Get first set of aux variables.
        plot_sample_hist(positions_x - positions_a_single, axs3[i], label="flow samples", n_vertices=plotting_n_nodes)
        plot_sample_hist(positions_x_target - positions_a_target_single, axs3[i], label="test samples", n_vertices=plotting_n_nodes)
        axs3[i].set_title(f"norms between graph of x - a (aug group {i}). ")
    axs3[0].legend()
    fig3.tight_layout()

    return [fig1, fig2, fig3]
