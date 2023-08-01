import jax.random
from hydra import compose, initialize
import hydra

from examples.load_flow_and_checkpoint import load_flow
from examples.default_plotter import *
from molboil.targets.data import load_dw4
from examples.analyse_results.get_wandb_runs import download_checkpoint


# mpl.rcParams['figure.dpi'] = 300
# # rc('font', **{'family': 'serif', 'serif': ['Times']})
# # rc('text', usetex=False)
# rc('axes', titlesize=24, labelsize=24)  # fontsize of the axes title and labels
# rc('legend', fontsize=24)
# rc('xtick', labelsize=20)
# rc('ytick', labelsize=20)
# rc("lines", linewidth=4)

def make_get_data_for_plotting(
        train_data: FullGraphSample,
        test_data: FullGraphSample,
        flow: AugmentedFlow,
        n_samples_from_flow: int,
        max_n_samples: int = 10000,
        plotting_n_nodes: Optional[int] = None,
        max_distance: Optional[float] = 20.,
):  # Override default.yaml plotter
    bins_x, count_list = bin_samples_by_dist([train_data.positions[:max_n_samples],
                                              test_data.positions[:max_n_samples]], max_distance=max_distance)
    n_samples = n_samples_from_flow


    def get_data_for_plotting(state: TrainingState, key: chex.PRNGKey, train_data=train_data, test_data=test_data):
        params = state.params
        key1, key2 = jax.random.split(key)
        joint_samples_flow = flow.sample_apply(params, train_data.features[0], key1, (n_samples,))
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

    return get_data_for_plotting, count_list, bins_x


_BASE_DIR = '../../..'


def plot_dw4(ax: Optional = None):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=f"{_BASE_DIR}/examples/config/")
    cfg = compose(config_name="dw4.yaml")

    download_checkpoint(flow_type='spherical', tags=["dw4", "ml", "florence"], seed=0, max_iter=100,
                        base_path='./examples/analyse_results/dw4_results/models')

    checkpoint_path = "examples/analyse_results/dw4_results/models/spherical_seed0.pkl"

    n_samples_from_flow_plotting = 1000
    key = jax.random.PRNGKey(0)

    flow, state = load_flow(cfg, checkpoint_path)

    train_data, valid_data, test_data = load_dw4(train_set_size=1000, test_set_size=1000, val_set_size=1000)


    get_data_for_plotting, count_list, bins_x = make_get_data_for_plotting(train_data=train_data, test_data=test_data,
                                                                           flow=flow,
                                   n_samples_from_flow=n_samples_from_flow_plotting)

    key = jax.random.PRNGKey(0)
    counts_flow_x, bins_a, count_list_a, bins_a_minus_x, count_list_a_minus_x = get_data_for_plotting(state, key)

    # Plot original coords
    if ax is None:
        fig1, axs = plt.subplots(1, 1, figsize=(5, 5))
    else:
        axs = ax
    axs = [axs]
    plot_histogram(counts_flow_x, bins_x, axs[0], label="flow")
    plot_histogram(count_list[0], bins_x, axs[0],  label="data")
    axs[0].legend(loc="upper right")
    axs[0].set_ylabel("normalized count")
    axs[0].set_xlabel("interatomic distance")
    axs[0].set_xlim(1.7, 6.3)
    axs[0].set_title("DW4")
    if ax is None:
        fig1.tight_layout()
        fig1.savefig("examples/plots/dw4.png")
        plt.show()


if __name__ == '__main__':
    # Should be run from repo base directory to work.
    plot_dw4()
