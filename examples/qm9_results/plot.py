import jax.random
from omegaconf import DictConfig
import yaml
from matplotlib import rc

from examples.load_flow_and_checkpoint import load_flow
from examples.default_plotter import *
from molboil.targets.data import load_qm9
from examples.get_wandb_runs import download_checkpoint
from examples.dw4_results.plot import make_get_data_for_plotting


mpl.rcParams['figure.dpi'] = 300
# rc('font', **{'family': 'serif', 'serif': ['Times']})
# rc('text', usetex=False)
rc('axes', titlesize=24, labelsize=24)  # fontsize of the axes title and labels
rc('legend', fontsize=24)
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)
rc("lines", linewidth=4)

def plot_qm9(ax: Optional = None):
    download_checkpoint(flow_type='spherical', tags=["lj13", "post_kigali_6"], seed=0, max_iter=256,
                        base_path='./examples/lj13_results/models')

    flow_type = 'spherical'
    checkpoint_path = f"examples/qm9_results/models/{flow_type}_seed0.pkl"
    cfg = DictConfig(yaml.safe_load(open(f"examples/config/qm9.yaml")))
    cfg.flow.type = flow_type
    key = jax.random.PRNGKey(0)

    flow, state = load_flow(cfg, checkpoint_path)

    train_data, valid_data, test_data = load_qm9(train_set_size=2000)  # None)
    n_samples_from_flow_plotting = train_data.positions.shape[0]
    print(f"plotting qm9 with {n_samples_from_flow_plotting} samples")


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
    axs[0].set_title("QM9 Positional")
    if ax is None:
        axs[0].set_ylabel("normalized count")
        axs[0].set_xlabel("interatomic distance")
        axs[0].set_xlim(0.5, 9.)
        fig1.tight_layout()
        fig1.savefig("examples/plots/qm9.png")
        # plt.show()
        print("done")



if __name__ == '__main__':
    plot_qm9()
