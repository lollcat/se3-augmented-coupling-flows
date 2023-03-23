from typing import Tuple

import hydra
from omegaconf import DictConfig
import jax.numpy as jnp
import mdtraj
from functools import partial
import chex


from molboil.train.train import train
from target.alanine_dipeptide import get_atom_encoding
from flow.aug_flow_dist import FullGraphSample
from examples.lj13 import to_local_config
from examples.create_train_config import create_train_config, \
    build_flow, create_flow_config, get_eval_on_test_batch, make_default_plotter, TrainingState, eval_fn


def load_dataset(train_data_n_points = None, test_data_n_points = None) -> \
        Tuple[FullGraphSample, FullGraphSample]:
    train_traj = mdtraj.load('target/data/aldp_500K_train_mini.h5')
    test_traj = mdtraj.load('target/data/aldp_500K_test_mini.h5')
    features = get_atom_encoding(train_traj)

    positions_train = train_traj.xyz
    positions_test = test_traj.xyz
    if train_data_n_points is not None:
        positions_train = positions_train[:train_data_n_points]
    if test_data_n_points is not None:
        positions_test = positions_test[:test_data_n_points]

    train_data = FullGraphSample(positions=positions_train,
                           features=jnp.repeat(features[None, :], positions_train.shape[0], axis=0))
    test_data = FullGraphSample(positions=positions_test,
                          features=jnp.repeat(features[None, :], positions_test.shape[0], axis=0))
    return train_data, test_data


def make_custom_plotter_and_eval(cfg: DictConfig, load_dataset, batch_size: int = 128):
    """Define custom plotter and evaluation function. These can be anything that follows the Callable specification
    defined in `molboil.train.train.train`. """
    train_data, test_data = load_dataset(128, 128)
    flow_config = create_flow_config(cfg)
    flow = build_flow(flow_config)

    # Define plotter.
    default_plotter = make_default_plotter(train_data,
                                       test_data,
                                       flow=flow,
                                       n_samples_from_flow=batch_size,
                                       max_n_samples=1000,
                                       plotting_n_nodes=None
                                       )
    def plotter_fn(state: TrainingState, key: chex.PRNGKey) -> dict:
        # Wrap default plotter to make function typing spec clear.
        return default_plotter(state, key)

    # Define evaluation function.
    eval_on_test_batch_fn = partial(get_eval_on_test_batch,
                                    flow=flow, K=cfg.training.K_marginal_log_lik, test_invariances=True)
    eval_batch_free_fn = None

    def evaluation_fn(state: TrainingState, key: chex.PRNGKey) -> dict:
        eval_info = eval_fn(test_data, key, state.params,
                eval_on_test_batch_fn=eval_on_test_batch_fn,
                eval_batch_free_fn=eval_batch_free_fn,
                batch_size=cfg.training.batch_size)
        return eval_info

    return plotter_fn, evaluation_fn



@hydra.main(config_path="./config", config_name="aldp.yaml")
def run(cfg: DictConfig):
    local_config = True
    if local_config:
        cfg = to_local_config(cfg)
    plotter, evaluation_fn = make_custom_plotter_and_eval(cfg, load_dataset)
    experiment_config = create_train_config(cfg, dim=3, n_nodes=22, load_dataset=load_dataset,
                                            evaluation_fn=evaluation_fn, plotter=plotter)
    train(experiment_config)


if __name__ == '__main__':
    run()
