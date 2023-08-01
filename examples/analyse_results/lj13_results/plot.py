import jax.random
from hydra import compose, initialize
import hydra

from examples.load_flow_and_checkpoint import load_flow
from examples.default_plotter import *
from molboil.targets.data import load_lj13
from examples.analyse_results.dw4_results.plot import make_get_data_for_plotting
from examples.analyse_results.get_wandb_runs import download_checkpoint
#
#
# mpl.rcParams['figure.dpi'] = 300
# # rc('font', **{'family': 'serif', 'serif': ['Times']})
# # rc('text', usetex=False)
# rc('axes', titlesize=24, labelsize=24)  # fontsize of the axes title and labels
# rc('legend', fontsize=24)
# rc('xtick', labelsize=20)
# rc('ytick', labelsize=20)
# rc("lines", linewidth=4)

_BASE_DIR = '../../..'

def plot_lj13(ax: Optional = None):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=f"{_BASE_DIR}/examples/config/")
    cfg = compose(config_name="lj13.yaml")

    download_checkpoint(flow_type='spherical', tags=["lj13", "ml", "florence"], seed=0, max_iter=400,
                        base_path='./examples/analyse_results/lj13_results/models')

    flow_type = 'spherical'
    checkpoint_path = f"examples/analyse_results/lj13_results/models/{flow_type}_seed0.pkl"

    cfg.flow.type = flow_type
    n_samples_from_flow_plotting = 1000
    key = jax.random.PRNGKey(0)

    flow, state = load_flow(cfg, checkpoint_path)

    train_data, valid_data, test_data = load_lj13(train_set_size=1000)


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
    # axs[0].legend(loc="upper left")
    axs[0].set_title("LJ13")
    axs[0].set_xlim(0.7, 3.4)
    axs[0].set_xlabel("interatomic distance")
    if ax is None:
        axs[0].set_ylabel("normalized count")
        fig1.tight_layout()
        fig1.savefig("examples/plots/lj13.png")
        plt.show()



if __name__ == '__main__':
    plot_lj13()
