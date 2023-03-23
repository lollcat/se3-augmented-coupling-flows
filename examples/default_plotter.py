from typing import Callable, Optional, List

from functools import partial
import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl

from molboil.utils.plotting import get_counts, get_pairwise_distances_for_plotting

from examples.configs import TrainingState
from flow.aug_flow_dist import FullGraphSample, AugmentedFlow, AugmentedFlowParams


mpl.rcParams['figure.dpi'] = 150
PlottingBatchSize = int
TrainData = FullGraphSample
TestData = FullGraphSample
FLowPlotter = Callable[[AugmentedFlowParams, AugmentedFlow, chex.PRNGKey, PlottingBatchSize,
                        TestData, TrainData], List[plt.Figure]]

def plot_histogram(counts, bins, ax: plt.Axes, *args, **kwargs):
    ax.stairs(counts, bins, alpha=0.4, fill=True, *args, **kwargs)


def make_default_plotter(
        train_data: FullGraphSample,
        test_data: FullGraphSample,
        flow: AugmentedFlow,
        n_samples_from_flow: int,
        max_n_samples: int = 10000,
        plotting_n_nodes: Optional[int] = None,
):
    bins = jnp.linspace(0., 6., 50)
    pairwise_distances_train_x = get_pairwise_distances_for_plotting(train_data.positions[:max_n_samples])
    pairwise_distances_test_x = get_pairwise_distances_for_plotting(test_data.positions[:max_n_samples])
    counts_target_train_x = get_counts(pairwise_distances_train_x, bins)
    counts_target_test_x = get_counts(pairwise_distances_test_x, bins)

    n_samples = n_samples_from_flow

    @partial(jax.jit)
    def get_data_for_plotting(state: TrainingState, key: chex.PRNGKey, train_data=train_data, test_data=test_data):
        params = state.params
        key1, key2 = jax.random.split(key)
        joint_samples_flow = jax.jit(flow.sample_apply, static_argnums=3)(params, train_data.features[0], key1,
                                                                          (n_samples,))
        features, positions_x, positions_a = flow.joint_to_separate_samples(joint_samples_flow)
        positions_a_target = flow.aux_target_sample_n_apply(params.aux_target, test_data[:n_samples], key2)

        pairwise_distances_flow_x = get_pairwise_distances_for_plotting(positions_x, plotting_n_nodes)
        pairwise_distances_flow_a = jax.vmap(get_pairwise_distances_for_plotting, in_axes=(-2, None),
                                                    out_axes=0)(
            positions_a, plotting_n_nodes)
        pairwise_distance_flow_a_minus_x = jax.vmap(get_pairwise_distances_for_plotting, in_axes=(-2, None),
                                                    out_axes=0
                                                    )(positions_a - positions_x[:, :, None], plotting_n_nodes)

        pairwise_distances_target_a = jax.vmap(get_pairwise_distances_for_plotting, in_axes=(-2, None),
                                                    out_axes=0)(
            positions_a_target, plotting_n_nodes)
        pairwise_distance_target_a_minus_x = jax.vmap(get_pairwise_distances_for_plotting, in_axes=(-2, None),
                                                    out_axes=0)(
            positions_a_target - test_data.positions[:n_samples, :, None], plotting_n_nodes)


        counts_flow_x = get_counts(pairwise_distances_flow_x, bins)
        counts_flow_a = jax.vmap(get_counts, in_axes=(0, None))(pairwise_distances_flow_a, bins)
        counts_flow_a_minus_x = jax.vmap(get_counts, in_axes=(0, None))(pairwise_distance_flow_a_minus_x, bins)
        counts_targ_a = jax.vmap(get_counts, in_axes=(0, None))(pairwise_distances_target_a, bins)
        counts_targ_a_minus_x = jax.vmap(get_counts, in_axes=(0, None))(pairwise_distance_target_a_minus_x, bins)
        return counts_flow_x, counts_flow_a, counts_flow_a_minus_x, counts_targ_a, counts_targ_a_minus_x


    def default_plotter(state: TrainingState, key: chex.PRNGKey):
        # Plot interatomic distance histograms.
        counts_flow_x, counts_flow_a, counts_flow_a_minus_x, counts_targ_a, counts_targ_a_minus_x = \
            get_data_for_plotting(state, key)

        # Plot original coords
        fig1, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot_histogram(counts_flow_x, bins, axs[0], label="flow samples")
        plot_histogram(counts_flow_x, bins, axs[1], label="flow samples")
        plot_histogram(counts_target_train_x, bins, axs[0],  label="train samples")
        plot_histogram(counts_target_test_x, bins, axs[1],  label="test samples")

        axs[0].set_title(f"norms between original coordinates train")
        axs[1].set_title(f"norms between original coordinates test")
        axs[0].legend()
        axs[1].legend()
        fig1.tight_layout()

        # Augmented info.
        fig2, axs2 = plt.subplots(1, flow.n_augmented, figsize=(5*flow.n_augmented, 5))
        axs2 = [axs2] if isinstance(axs2, plt.Subplot) else axs2
        for i in range(flow.n_augmented):
            d_a_flow_single = counts_flow_a[i]  # get single group of augmented coordinates
            d_a_target = counts_targ_a[i]  # Get first set of aux variables.
            chex.assert_equal_shape((counts_flow_x, d_a_flow_single, d_a_target))
            plot_histogram(d_a_flow_single, bins, axs2[i], label="flow samples")
            plot_histogram(d_a_target, bins, axs2[i], label="test samples")
            axs2[i].set_title(f"norms between augmented coordinates (aug group {i})")
        axs2[0].legend()
        fig2.tight_layout()

        # Plot histogram (x - a)

        fig3, axs3 = plt.subplots(1, flow.n_augmented, figsize=(5*flow.n_augmented, 5))
        axs3 = [axs3] if isinstance(axs3, plt.Subplot) else axs3
        for i in range(flow.n_augmented):
            d_a_minus_x_flow_single = counts_flow_a_minus_x[i]  # get single group of augmented coordinates
            d_a_minus_x_target_single = counts_targ_a_minus_x[i]  # Get first set of aux variables.
            chex.assert_equal_shape((counts_flow_x, d_a_minus_x_flow_single, d_a_minus_x_target_single))
            plot_histogram(d_a_minus_x_flow_single, bins, axs3[i], label="flow samples")
            plot_histogram(d_a_minus_x_target_single, bins, axs3[i], label="test samples")
            axs3[i].set_title(f"norms between graph of x - a (aug group {i}). ")
        axs3[0].legend()
        fig3.tight_layout()

        return [fig1, fig2, fig3]

    return default_plotter
