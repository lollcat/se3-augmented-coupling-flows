from typing import Tuple, Optional, List

from functools import partial
import hydra
from omegaconf import DictConfig
from examples.train import train, create_train_config, plot_original_aug_norms_sample_hist, plot_sample_hist, plot_orig_aug_centre_mass_diff_hist
import jax.numpy as jnp
import numpy as np
import chex
import mdtraj
import matplotlib as mpl
import matplotlib.pyplot as plt
import jax
from openmmtools.testsystems import AlanineDipeptideVacuum

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
                    n_batches: int = 1,
                    plotting_n_nodes: Optional[int] = None) -> List[plt.Subplot]:

    # Generate samples
    sample_fn = jax.jit(flow.sample_apply, static_argnums=3)
    separate_samples_fn = jax.jit(flow.joint_to_separate_samples)
    aux_target_sample_n_apply_fn = jax.jit(flow.aux_target_sample_n_apply)
    positions_x = []
    positions_a = []
    positions_a_target = []
    for i in range(n_batches):
        key, key_ = jax.random.split(key)
        joint_samples_flow = sample_fn(params, train_data.features[0], key_,
                                                                          (n_samples,))
        _, positions_x_, positions_a_ = separate_samples_fn(joint_samples_flow)
        positions_x.append(positions_x_)
        positions_a.append(positions_a_)
        if len(test_data) > i * n_samples:
            key, key_ = jax.random.split(key)
            end = min((i + 1) * n_samples, len(test_data))
            positions_a_target_ = aux_target_sample_n_apply_fn(params.aux_target,
                                                               test_data[(i * n_samples):end], key_)
            positions_a_target.append(positions_a_target_)
    positions_x = jnp.concatenate(positions_x, axis=0)
    positions_a = jnp.concatenate(positions_a, axis=0)
    print(positions_a_target)
    positions_a_target = jnp.concatenate(positions_a_target, axis=0)
    print(positions_a_target.shape)

    # Plot original coords
    fig1, axs = plt.subplots(1, 2, figsize=(10, 5))
    plot_sample_hist(positions_x, axs[0], label="flow samples", n_vertices=plotting_n_nodes)
    plot_sample_hist(positions_x, axs[1], label="flow samples", n_vertices=plotting_n_nodes)
    plot_sample_hist(train_data.positions[:min(n_samples * n_batches, len(train_data))],
                     axs[0], label="train samples", n_vertices=plotting_n_nodes)
    plot_sample_hist(test_data.positions[:min(n_samples * n_batches, len(train_data))],
                     axs[1], label="test samples", n_vertices=plotting_n_nodes)

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
        #chex.assert_equal_shape((positions_x, positions_a_single, positions_a_target_single))
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

    # Compute Ramachandran plot angles
    aldp = AlanineDipeptideVacuum(constraints=None)
    topology = mdtraj.Topology.from_openmm(aldp.topology)
    test_traj = mdtraj.Trajectory(np.array(test_data.positions).reshape(-1, 22, 3), topology)
    sampled_traj = mdtraj.Trajectory(np.array(positions_x).reshape(-1, 22, 3), topology)
    psi_d = mdtraj.compute_psi(test_traj)[1].reshape(-1)
    phi_d = mdtraj.compute_phi(test_traj)[1].reshape(-1)
    psi = mdtraj.compute_psi(sampled_traj)[1].reshape(-1)
    phi = mdtraj.compute_phi(sampled_traj)[1].reshape(-1)

    # Compute histograms
    nbins = 200
    htest_phi, _ = np.histogram(phi_d, nbins, range=[-np.pi, np.pi], density=True);
    hgen_phi, _ = np.histogram(phi, nbins, range=[-np.pi, np.pi], density=True);
    htest_psi, _ = np.histogram(psi_d, nbins, range=[-np.pi, np.pi], density=True);
    hgen_psi, _ = np.histogram(psi, nbins, range=[-np.pi, np.pi], density=True);

    # Plot phi and psi
    fig4, ax = plt.subplots(1, 2, figsize=(20, 10))
    x = np.linspace(-np.pi, np.pi, nbins)
    ax[0].plot(x, htest_phi, linewidth=3)
    ax[0].plot(x, hgen_phi, linewidth=3)
    ax[0].tick_params(axis='both', labelsize=20)
    ax[0].set_xlabel('$\phi$', fontsize=24)
    ax[1].plot(x, htest_psi, linewidth=3)
    ax[1].plot(x, hgen_psi, linewidth=3)
    ax[1].tick_params(axis='both', labelsize=20)
    ax[1].set_xlabel('$\psi$', fontsize=24)

    # Ramachandran plot
    fig5, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist2d(phi, psi, bins=64, norm=mpl.colors.LogNorm(),
              range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('$\phi$', fontsize=24)
    ax.set_ylabel('$\psi$', fontsize=24)

    return [fig1, fig2, fig3, fig4, fig5]



@hydra.main(config_path="./config", config_name="aldp.yaml")
def run(cfg: DictConfig):
    local_config = False
    if local_config:
        cfg = to_local_config(cfg)

    custom_aldp_plotter_ = partial(custom_aldp_plotter, n_batches=cfg.eval.plot_n_batches)
    experiment_config = create_train_config(cfg, dim=3, n_nodes=22,
                                            load_dataset=load_dataset,
                                            plotter=custom_aldp_plotter_)
    #experiment_config.plotter = custom_aldp_plotter
    train(experiment_config)


if __name__ == '__main__':
    run()
