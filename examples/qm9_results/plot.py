import jax.random
from omegaconf import DictConfig
import yaml
from matplotlib import rc

from examples.load_flow_and_checkpoint import load_flow
from examples.default_plotter import *
from molboil.targets.data import load_qm9


mpl.rcParams['figure.dpi'] = 300
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
rc('axes', titlesize=24, labelsize=24)  # fontsize of the axes title and labels
rc('legend', fontsize=24)
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)
rc("lines", linewidth=4)

def make_get_data_for_plotting(
        train_data: FullGraphSample,
        test_data: FullGraphSample,
        flow: AugmentedFlow,
        n_samples_from_flow: int,
        max_n_samples: int = 10000,
        plotting_n_nodes: Optional[int] = None,
        max_distance: Optional[float] = 20.,
):  # Override default plotter
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



if __name__ == '__main__':
    flow_type = 'spherical'
    checkpoint_path = f"examples/qm9_results/models/{flow_type}_seed0.pkl"
    cfg = DictConfig(yaml.safe_load(open(f"examples/config/qm9.yaml")))
    cfg.flow.type = flow_type
    key = jax.random.PRNGKey(0)

    flow, state = load_flow(cfg, checkpoint_path)

    train_data, valid_data, test_data = load_qm9()
    n_samples_from_flow_plotting = train_data.positions.shape[0]


    get_data_for_plotting, count_list, bins_x = make_get_data_for_plotting(train_data=train_data, test_data=test_data,
                                                                           flow=flow,
                                   n_samples_from_flow=n_samples_from_flow_plotting)

    key = jax.random.PRNGKey(0)
    counts_flow_x, bins_a, count_list_a, bins_a_minus_x, count_list_a_minus_x = get_data_for_plotting(state, key)

    # Plot original coords
    fig1, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs = [axs]
    plot_histogram(counts_flow_x, bins_x, axs[0], label="flow")
    plot_histogram(count_list[0], bins_x, axs[0],  label="data")
    # axs[0].legend(loc="upper left")
    axs[0].set_ylabel("normalized count")
    axs[0].set_xlabel("interatomic distance")
    plt.title("QM9")
    fig1.tight_layout()
    fig1.savefig("examples/plots/qm9.png")
    # plt.show()
