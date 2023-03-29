from typing import Callable, Optional, List

from functools import partial
import chex
import jax
import matplotlib.pyplot as plt
import matplotlib as mpl

from molboil.utils.plotting import bin_samples_by_dist, get_counts, get_pairwise_distances_for_plotting

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
        max_distance: Optional[float] = 9.0,
):
    bins_x, count_list = bin_samples_by_dist([train_data.positions[:max_n_samples],
                                              test_data.positions[:max_n_samples]], max_distance=max_distance)
    n_samples = n_samples_from_flow

    @partial(jax.jit)
    def get_data_for_plotting(state: TrainingState, key: chex.PRNGKey, train_data=train_data, test_data=test_data):
        params = state.params
        key1, key2 = jax.random.split(key)
        joint_samples_flow = jax.jit(flow.sample_apply, static_argnums=3)(params, train_data.features[0], key1,
                                                                          (n_samples,))
        features, positions_x, positions_a_flow = flow.joint_to_separate_samples(joint_samples_flow)
        a_minus_x_flow = positions_a_flow - positions_x[:, :, None]
        positions_a_target = flow.aux_target_sample_n_apply(params.aux_target, test_data[:n_samples], key2)
        a_minus_x_target = positions_a_target - test_data.positions[:n_samples, :, None]

        # Get counts flow x.
        pairwise_distances_flow_x = get_pairwise_distances_for_plotting(positions_x, plotting_n_nodes, max_distance)
        counts_flow_x = get_counts(pairwise_distances_flow_x, bins_x)

        bins_a, count_list_a = jax.vmap(bin_samples_by_dist, in_axes=(-2, None), out_axes=0
                                        )([positions_a_flow, positions_a_target], max_distance)
        bins_a_minus_x, count_list_a_minus_x = jax.vmap(bin_samples_by_dist, in_axes=(-2, None), out_axes=0)(
            [a_minus_x_flow, a_minus_x_target], max_distance)
        return counts_flow_x, bins_a, count_list_a, bins_a_minus_x, count_list_a_minus_x


    def default_plotter(state: TrainingState, key: chex.PRNGKey) -> dict:
        # Plot interatomic distance histograms.
        key = jax.random.PRNGKey(0)
        counts_flow_x, bins_a, count_list_a, bins_a_minus_x, count_list_a_minus_x = get_data_for_plotting(state, key)

        # Plot original coords
        fig1, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot_histogram(counts_flow_x, bins_x, axs[0], label="flow samples")
        plot_histogram(counts_flow_x, bins_x, axs[1], label="flow samples")
        plot_histogram(count_list[0], bins_x, axs[0],  label="train samples")
        plot_histogram(count_list[1], bins_x, axs[1],  label="test samples")

        axs[0].set_title(f"norms between original coordinates train")
        axs[1].set_title(f"norms between original coordinates test")
        axs[0].legend()
        axs[1].legend()
        fig1.tight_layout()

        # Augmented info.
        fig2, axs2 = plt.subplots(1, flow.n_augmented, figsize=(5*flow.n_augmented, 5))
        axs2 = [axs2] if isinstance(axs2, plt.Subplot) else axs2
        for i in range(flow.n_augmented):
            d_a_flow_single = count_list_a[0][i]  # get single group of augmented coordinates
            d_a_target = count_list_a[1][i]  # Get first set of aux variables.
            chex.assert_equal_shape((counts_flow_x, d_a_flow_single, d_a_target))
            plot_histogram(d_a_flow_single, bins_a[i], axs2[i], label="flow samples")
            plot_histogram(d_a_target, bins_a[i], axs2[i], label="test samples")
            axs2[i].set_title(f"norms between augmented coordinates (aug group {i})")
        axs2[0].legend()
        fig2.tight_layout()

        # Plot histogram (x - a)

        fig3, axs3 = plt.subplots(1, flow.n_augmented, figsize=(5*flow.n_augmented, 5))
        axs3 = [axs3] if isinstance(axs3, plt.Subplot) else axs3
        for i in range(flow.n_augmented):
            d_a_minus_x_flow_single = count_list_a_minus_x[0][i]  # get single group of augmented coordinates
            d_a_minus_x_target_single = count_list_a_minus_x[1][i]  # Get first set of aux variables.
            chex.assert_equal_shape((counts_flow_x, d_a_minus_x_flow_single, d_a_minus_x_target_single))
            plot_histogram(d_a_minus_x_flow_single, bins_a_minus_x[i], axs3[i], label="flow samples")
            plot_histogram(d_a_minus_x_target_single, bins_a_minus_x[i], axs3[i], label="test samples")
            axs3[i].set_title(f"norms between graph of x - a (aug group {i}). ")
        axs3[0].legend()
        fig3.tight_layout()

        return [fig1, fig2, fig3]

    return default_plotter
