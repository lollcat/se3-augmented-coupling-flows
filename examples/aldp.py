from typing import Tuple

import hydra
from omegaconf import DictConfig
from examples.train import train, create_train_config
import jax.numpy as jnp
import mdtraj

from target.alanine_dipeptide import get_atom_encoding
from flow.aug_flow_dist import FullGraphSample

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


def to_local_config(cfg: DictConfig) -> DictConfig:
    """Change config to make it fast to run locally. Also remove saving."""
    # Training
    cfg.training.optimizer.init_lr = 2e-4
    cfg.training.batch_size = 16
    cfg.training.n_epoch = 100
    cfg.training.save = False
    cfg.training.n_plots = 3
    cfg.training.n_eval = 10
    cfg.training.plot_batch_size = 8
    cfg.training.K_marginal_log_lik = 2
    cfg.logger = DictConfig({"list_logger": None})

    # Flow
    # cfg.flow.type = ['realnvp_non_eq']
    cfg.flow.n_layers = 2
    cfg.flow.act_norm = False

    # proj flow settings
    cfg.flow.kwargs.proj.global_frame = False
    cfg.flow.kwargs.proj.process_flow_params_jointly = False
    cfg.flow.kwargs.proj.condition_on_x_proj = True

    # Configure NNs
    cfg.flow.nets.transformer.mlp_units = (16,)
    cfg.flow.nets.transformer.n_layers = 2
    cfg.flow.nets.mlp_head_config.mlp_units = (16,)
    cfg.flow.nets.egnn.mlp_units = (8,)

    debug = False
    if debug:
        cfg_train = dict(cfg['training'])
        cfg_train['scan_run'] = False
        cfg.training = DictConfig(cfg_train)
    return cfg


@hydra.main(config_path="./config", config_name="aldp.yaml")
def run(cfg: DictConfig):
    local_config = False
    if local_config:
        cfg = to_local_config(cfg)

    experiment_config = create_train_config(cfg, dim=3, n_nodes=22,
                                            load_dataset=load_dataset)
    train(experiment_config)


if __name__ == '__main__':
    run()
