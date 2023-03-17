from typing import Tuple, Optional, List

import hydra
from omegaconf import DictConfig
from examples.train import train, create_train_config, plot_original_aug_norms_sample_hist, plot_sample_hist, plot_orig_aug_centre_mass_diff_hist
import jax.numpy as jnp
import chex
import mdtraj
import matplotlib.pyplot as plt
import jax

from target.alanine_dipeptide import get_atom_encoding
from flow.aug_flow_dist import FullGraphSample, AugmentedFlow, AugmentedFlowParams
from examples.lj13 import to_local_config


def load_dataset(batch_size, train_data_n_points = None, test_data_n_points = None) -> \
        Tuple[FullGraphSample, FullGraphSample]:
    train_traj = mdtraj.load('target/data/aldp_500K_train_mini.h5')
    test_traj = mdtraj.load('target/data/aldp_500K_test_mini.h5')
    features = get_atom_encoding(train_traj)

    positions_train = train_traj.xyz
    positions_test = test_traj.xyz
    if train_data_n_points is not None:
        positions_train = positions_train[:train_data_n_points]
    positions_train = positions_train[:positions_train.shape[0] - (positions_train.shape[0] % batch_size)]
    if test_data_n_points is not None:
        positions_test = positions_test[:test_data_n_points]

    train_data = FullGraphSample(positions=positions_train,
                           features=jnp.repeat(features[None, :], positions_train.shape[0], axis=0))
    test_data = FullGraphSample(positions=positions_test,
                          features=jnp.repeat(features[None, :], positions_test.shape[0], axis=0))
    return train_data, test_data


def custom_aldp_plotter(params: AugmentedFlowParams,
                    flow: AugmentedFlow,
                    key: chex.PRNGKey,
                    n_samples: int,
                    train_data: FullGraphSample,
                    test_data: FullGraphSample,
                    plotting_n_nodes: Optional[int] = None) -> List[plt.Subplot]:

    # Plot interatomic distance histograms.
    key1, key2 = jax.random.split(key)
    joint_samples_flow = jax.jit(flow.sample_apply, static_argnums=3)(params, train_data.features[0], key1,
                                                                      (n_samples,))
    features, positions_x, positions_a = jax.jit(flow.joint_to_separate_samples)(joint_samples_flow)
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
    axs[0].legend()
    fig2.tight_layout()

    # Plot histogram for centre of mean
    fig3, axs3 = plt.subplots(1, 2, figsize=(10, 5))
    positions_a_single = positions_a[:, :, 0]  # get single group of augmented coordinates
    positions_a_target_single = positions_a_target[:, :, 0]  # Get first set of aux variables.

    plot_orig_aug_centre_mass_diff_hist(positions_x, positions_a_single, ax=axs3[0], label='flow samples')
    plot_orig_aug_centre_mass_diff_hist(test_data[:n_samples].positions, positions_a_target_single, ax=axs3[0],
                                        label='test samples')
    plot_original_aug_norms_sample_hist(positions_x, positions_a_single, axs3[1], label='flow samples')
    plot_original_aug_norms_sample_hist(test_data[:n_samples].positions, positions_a_target_single, axs3[1],
                                        label='test samples')
    axs3[0].legend()
    axs3[0].set_title("norms orig - aug centre of mass (aug group 1) ")
    axs3[1].set_title("norms orig - augparticles (aug group 1)")
    fig3.tight_layout()

    return [fig1, fig2, fig3]



@hydra.main(config_path="./config", config_name="aldp.yaml")
def run(cfg: DictConfig):
    local_config = False
    if local_config:
        cfg = to_local_config(cfg)

    cfg.training.plotter = custom_aldp_plotter

    experiment_config = create_train_config(cfg, dim=3, n_nodes=22,
                                            load_dataset=load_dataset)
    #experiment_config.plotter = custom_aldp_plotter
    train(experiment_config)


if __name__ == '__main__':
    run()
