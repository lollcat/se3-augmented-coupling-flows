from typing import List

import os
from functools import partial
import hydra
from omegaconf import DictConfig
import jax.numpy as jnp
import numpy as np
import chex
import mdtraj
import matplotlib as mpl
import matplotlib.pyplot as plt
import jax
from openmmtools.testsystems import AlanineDipeptideVacuum
from boltzgen.flows import CoordinateTransform
import torch

from molboil.targets.data import load_aldp
from molboil.train.train import train
from molboil.train.base import eval_fn

from flow.build_flow import build_flow
from flow.aug_flow_dist import FullGraphSample, AugmentedFlow
from examples.create_train_config import create_train_config, create_flow_config
from examples.configs import TrainingState
from utils.aug_flow_train_and_eval import get_eval_on_test_batch

def aldp_eval_and_plot_fn(state: TrainingState,
                          key: chex.PRNGKey,
                          iteration: int,
                          save: bool,
                          plots_dir: str,
                          flow: AugmentedFlow,
                          train_data: FullGraphSample,
                          test_data: FullGraphSample,
                          n_samples: int,
                          n_batches: int = 1,
                          K_marginal_log_lik: int = 1,
                          test_invariances: bool = False) -> List[plt.Subplot]:

    # Set up coordinate transform
    ndim = 66
    transform_data = torch.tensor(np.array(train_data.positions).reshape(-1, ndim),
                                  dtype=torch.float64)
    z_matrix = [
        (0, [1, 4, 6]),
        (1, [4, 6, 8]),
        (2, [1, 4, 0]),
        (3, [1, 4, 0]),
        (4, [6, 8, 14]),
        (5, [4, 6, 8]),
        (7, [6, 8, 4]),
        (9, [8, 6, 4]),
        (10, [8, 6, 4]),
        (11, [10, 8, 6]),
        (12, [10, 8, 11]),
        (13, [10, 8, 11]),
        (15, [14, 8, 16]),
        (16, [14, 8, 6]),
        (17, [16, 14, 15]),
        (18, [16, 14, 8]),
        (19, [18, 16, 14]),
        (20, [18, 16, 19]),
        (21, [18, 16, 19])
    ]
    cart_indices = [8, 6, 14]
    ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]
    transform = CoordinateTransform(transform_data, ndim, z_matrix, cart_indices,
                                    mode="internal", ind_circ_dih=ind_circ_dih)

    # Generate samples
    params = state.params
    sample_fn = jax.jit(flow.sample_apply, static_argnums=3)
    separate_samples_fn = jax.jit(flow.joint_to_separate_samples)
    positions_x = []
    internal_gen = []
    internal_test = []
    for i in range(n_batches):
        key, key_ = jax.random.split(key)
        joint_samples_flow = sample_fn(params, train_data.features[0], key_,
                                                                          (n_samples,))
        _, positions_x_, _ = separate_samples_fn(joint_samples_flow)
        positions_x_torch = torch.tensor(np.array(positions_x_).reshape(-1, ndim),
                                         dtype=torch.float64)
        internal_gen_ = transform.inverse(positions_x_torch)[0].detach().numpy()
        positions_x.append(positions_x_)
        internal_gen.append(internal_gen_)
        if len(test_data.positions) > i * n_samples:
            key, key_ = jax.random.split(key)
            end = min((i + 1) * n_samples, len(test_data.positions))
            positions_x_torch = torch.tensor(np.array(test_data[(i * n_samples):end].positions).reshape(-1, ndim),
                                             dtype=torch.float64)
            internal_test_ = transform.inverse(positions_x_torch)[0].detach().numpy()
            internal_test.append(internal_test_)
    positions_x = jnp.concatenate(positions_x, axis=0)
    internal_gen = np.concatenate(internal_gen, axis=0)
    internal_test = np.concatenate(internal_test, axis=0)

    # Compute Ramachandran plot angles
    aldp = AlanineDipeptideVacuum(constraints=None)
    topology = mdtraj.Topology.from_openmm(aldp.topology)
    train_traj = mdtraj.Trajectory(np.array(train_data.positions).reshape(-1, 22, 3), topology)
    test_traj = mdtraj.Trajectory(np.array(test_data.positions).reshape(-1, 22, 3), topology)
    sampled_traj = mdtraj.Trajectory(np.array(positions_x).reshape(-1, 22, 3), topology)
    psi_train = mdtraj.compute_psi(train_traj)[1].reshape(-1)
    phi_train = mdtraj.compute_phi(train_traj)[1].reshape(-1)
    psi_test = mdtraj.compute_psi(test_traj)[1].reshape(-1)
    phi_test = mdtraj.compute_phi(test_traj)[1].reshape(-1)
    psi = mdtraj.compute_psi(sampled_traj)[1].reshape(-1)
    phi = mdtraj.compute_phi(sampled_traj)[1].reshape(-1)

    # Prepare output
    info = {}

    # Compute histograms
    nbins = 200
    htrain_phi, _ = np.histogram(phi_train, nbins, range=[-np.pi, np.pi], density=True);
    htest_phi, _ = np.histogram(phi_test, nbins, range=[-np.pi, np.pi], density=True);
    hgen_phi, _ = np.histogram(phi, nbins, range=[-np.pi, np.pi], density=True);
    htrain_psi, _ = np.histogram(psi_train, nbins, range=[-np.pi, np.pi], density=True);
    htest_psi, _ = np.histogram(psi_test, nbins, range=[-np.pi, np.pi], density=True);
    hgen_psi, _ = np.histogram(psi, nbins, range=[-np.pi, np.pi], density=True);

    # Compute KLDs for phi and psi
    eps = 1e-10
    kld_phi_train = np.sum(htrain_phi * np.log((htrain_phi + eps) / (hgen_phi + eps))) \
                    * 2 * np.pi / nbins
    kld_phi_test = np.sum(htest_phi * np.log((htest_phi + eps) / (hgen_phi + eps))) \
                   * 2 * np.pi / nbins
    info.update(kld_phi_train=kld_phi_train, kld_phi_test=kld_phi_test)
    kld_psi_train = np.sum(htrain_psi * np.log((htrain_psi + eps) / (hgen_psi + eps))) \
                    * 2 * np.pi / nbins
    kld_psi_test = np.sum(htest_psi * np.log((htest_psi + eps) / (hgen_psi + eps))) \
                   * 2 * np.pi / nbins
    info.update(kld_psi_train=kld_psi_train, kld_psi_test=kld_psi_test)

    # Plot phi and psi
    fig1, ax = plt.subplots(1, 2, figsize=(20, 10))
    x = np.linspace(-np.pi, np.pi, nbins)
    ax[0].plot(x, htrain_phi, linewidth=3)
    ax[0].plot(x, htest_phi, linewidth=3)
    ax[0].plot(x, hgen_phi, linewidth=3)
    ax[0].tick_params(axis='both', labelsize=20)
    ax[0].set_xlabel('$\phi$', fontsize=24)
    ax[1].plot(x, htrain_psi, linewidth=3)
    ax[1].plot(x, htest_psi, linewidth=3)
    ax[1].plot(x, hgen_psi, linewidth=3)
    ax[1].legend(['Train', 'Test', 'Model'], fontsize=20)
    ax[1].tick_params(axis='both', labelsize=20)
    ax[1].set_xlabel('$\psi$', fontsize=24)

    # Ramachandran plot
    # Compute KLDs
    nbins_ram = 64
    eps_ram = 1e-10
    hist_ram_train = np.histogram2d(phi_train, psi_train, nbins_ram,
                                    range=[[-np.pi, np.pi], [-np.pi, np.pi]],
                                    density=True)[0]
    hist_ram_test = np.histogram2d(phi_test, psi_test, nbins_ram,
                                   range=[[-np.pi, np.pi], [-np.pi, np.pi]],
                                   density=True)[0]
    hist_ram_gen = np.histogram2d(phi, psi, nbins_ram,
                                  range=[[-np.pi, np.pi], [-np.pi, np.pi]],
                                  density=True)[0]
    kld_ram_train = np.sum(hist_ram_train * np.log((hist_ram_train + eps_ram)
                                                   / (hist_ram_gen + eps_ram))) \
                    * (2 * np.pi / nbins_ram) ** 2
    kld_ram_test = np.sum(hist_ram_test * np.log((hist_ram_test + eps_ram)
                                                 / (hist_ram_gen + eps_ram))) \
                   * (2 * np.pi / nbins_ram) ** 2
    info.update(kld_ram_train=kld_ram_train, kld_ram_test=kld_ram_test)
    fig2, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist2d(phi, psi, bins=nbins_ram, norm=mpl.colors.LogNorm(),
              range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('$\phi$', fontsize=24)
    ax.set_ylabel('$\psi$', fontsize=24)

    # Internal coordinates
    ndim = internal_gen.shape[1]
    hist_range = [-5, 5]
    hists_test = np.zeros((nbins, ndim))
    hists_gen = np.zeros((nbins, ndim))
    for i in range(ndim):
        htest, _ = np.histogram(internal_test[:, i], nbins, range=hist_range, density=True);
        hgen, _ = np.histogram(internal_gen[:, i], nbins, range=hist_range, density=True);
        hists_test[:, i] = htest
        hists_gen[:, i] = hgen

    # Histograms of the groups
    ncarts = transform.transform.len_cart_inds
    permute_inv = transform.transform.permute_inv.cpu().data.numpy()
    bond_ind = transform.transform.ic_transform.bond_indices.cpu().data.numpy()
    angle_ind = transform.transform.ic_transform.angle_indices.cpu().data.numpy()
    dih_ind = transform.transform.ic_transform.dih_indices.cpu().data.numpy()
    hists_test_cart = hists_test[:, :(3 * ncarts - 6)]
    hists_test_ = np.concatenate([hists_test[:, :(3 * ncarts - 6)],
                                  np.zeros((nbins, 6)),
                                  hists_test[:, (3 * ncarts - 6):]], axis=1)
    hists_test_ = hists_test_[:, permute_inv]
    hists_test_bond = hists_test_[:, bond_ind]
    hists_test_angle = hists_test_[:, angle_ind]
    hists_test_dih = hists_test_[:, dih_ind]

    hists_gen_cart = hists_gen[:, :(3 * ncarts - 6)]
    hists_gen_ = np.concatenate([hists_gen[:, :(3 * ncarts - 6)],
                                 np.zeros((nbins, 6)),
                                 hists_gen[:, (3 * ncarts - 6):]], axis=1)
    hists_gen_ = hists_gen_[:, permute_inv]
    hists_gen_bond = hists_gen_[:, bond_ind]
    hists_gen_angle = hists_gen_[:, angle_ind]
    hists_gen_dih = hists_gen_[:, dih_ind]
    hists_test_bond = np.concatenate((hists_test_cart[:, :2],
                                      hists_test_bond), 1)
    hists_gen_bond = np.concatenate((hists_gen_cart[:, :2],
                                     hists_gen_bond), 1)
    hists_test_angle = np.concatenate((hists_test_cart[:, 2:],
                                       hists_test_angle), 1)
    hists_gen_angle = np.concatenate((hists_gen_cart[:, 2:],
                                      hists_gen_angle), 1)

    label = ['bond', 'angle', 'dih']
    hists_test_list = [hists_test_bond, hists_test_angle,
                       hists_test_dih]
    hists_gen_list = [hists_gen_bond, hists_gen_angle,
                      hists_gen_dih]
    x = np.linspace(*hist_range, nbins)
    figs_internal = []
    for i in range(len(label)):
        ncol = 4
        if i == 0:
            fig, ax = plt.subplots(6, 4, figsize=(15, 24))
            for j in range(1, 4):
                ax[5, j].set_axis_off()
        elif i == 2:
            fig, ax = plt.subplots(5, 4, figsize=(15, 20))
            ax[4, 3].set_axis_off()
        else:
            fig, ax = plt.subplots(5, 4, figsize=(15, 20))
        for j in range(hists_test_list[i].shape[1]):
            ax[j // ncol, j % ncol].plot(x, hists_test_list[i][:, j])
            ax[j // ncol, j % ncol].plot(x, hists_gen_list[i][:, j])
        figs_internal.append(fig)

    # Plot or save figures
    figures = [fig1, fig2, *figs_internal]
    fig_names = ['phi_psi', 'ramachandran', 'bond', 'angle', 'dih']
    for i, figure in enumerate(figures):
        if save:
            figure.savefig(
                os.path.join(plots_dir, "%s_iter_%08i.png" % (fig_names[i], iteration))
            )
        else:
            plt.show()
        plt.close(figure)

    # Get eval like log-likelihood on test set
    eval_on_test_batch_fn = partial(get_eval_on_test_batch,
                                    flow=flow, K=K_marginal_log_lik,
                                    test_invariances=test_invariances)
    eval_info = eval_fn(test_data, key, state.params,
                        eval_on_test_batch_fn=eval_on_test_batch_fn,
                        eval_batch_free_fn=None,
                        batch_size=n_samples)
    info.update(eval_info)

    return info



@hydra.main(config_path="./config", config_name="aldp.yaml")
def run(cfg: DictConfig):
    def load_dataset(train_set_size: int, val_set_size: int):
        return load_aldp(train_path=cfg.target.data.train, val_path=cfg.target.data.val,
                         train_n_points=train_set_size, val_n_points=val_set_size)[:2]

    train_set, val_set = load_dataset(cfg.training.train_set_size, cfg.training.test_set_size)
    flow_config = create_flow_config(cfg)
    flow = build_flow(flow_config)
    eval_and_plot_fn = partial(aldp_eval_and_plot_fn, flow=flow, train_data=train_set, test_data=val_set,
                               n_samples=cfg.training.plot_batch_size, n_batches=cfg.eval.plot_n_batches,
                               K_marginal_log_lik=cfg.training.K_marginal_log_lik)

    experiment_config = create_train_config(cfg, dim=3, n_nodes=22,
                                            load_dataset=load_dataset,
                                            eval_and_plot_fn=eval_and_plot_fn,
                                            date_folder=False)

    train(experiment_config)


if __name__ == '__main__':
    run()
