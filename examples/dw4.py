import hydra
from omegaconf import DictConfig
from functools import partial

import jax.numpy as jnp
import numpy as np

from examples.train import train, create_train_config
from target.double_well import make_dataset
from utils.data import positional_dataset_only_to_full_graph


def load_dataset_standard(batch_size, train_set_size: int = 1000, test_set_size:int = 1000):
    # dataset from https://github.com/vgsatorras/en_flows
    # Loading following https://github.com/vgsatorras/en_flows/blob/main/dw4_experiment/dataset.py.

    data_path = 'target/data/dw4-dataidx.npy'  # 'target/data/dw_data_vertices4_dim2.npy'
    dataset = jnp.asarray(np.load(data_path, allow_pickle=True)[0])
    dataset = jnp.reshape(dataset, (-1, 4, 2))

    train_set = dataset[:train_set_size]
    train_set = train_set[:train_set_size - (train_set.shape[0] % batch_size)]

    test_set = dataset[-test_set_size:]
    return positional_dataset_only_to_full_graph(train_set), positional_dataset_only_to_full_graph(test_set)

def load_dataset_custom(batch_size, train_set_size: int = 1000, test_set_size:int = 1000, seed: int = 0,
                        temperature: float = 1.0):
    dataset = make_dataset(seed=seed, n_vertices=4, dim=2, n_samples=test_set_size+train_set_size,
                           temperature=temperature)

    train_set = dataset[:train_set_size]
    train_set = train_set[:train_set_size - (train_set.shape[0] % batch_size)]

    test_set = dataset[-test_set_size:]
    return positional_dataset_only_to_full_graph(train_set), positional_dataset_only_to_full_graph(test_set)


def to_local_config(cfg: DictConfig) -> DictConfig:
    """Change config to make it fast to run locally. Also remove saving."""
    # Training
    cfg.training.optimizer.init_lr = 2e-4
    cfg.training.batch_size = 16
    cfg.training.n_epoch = 200
    cfg.training.save = False
    cfg.training.n_plots = 3
    cfg.training.n_eval = 10
    cfg.training.plot_batch_size = 32
    cfg.training.K_marginal_log_lik = 2
    cfg.logger = DictConfig({"list_logger": None})
    # cfg.logger = DictConfig({"pandas_logger": {'save_period': 50}})

    # Flow
    cfg.flow.type = 'proj'
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


@hydra.main(config_path="./config", config_name="dw4.yaml")
def run(cfg: DictConfig):
    # assert cfg.flow.nets.type == 'egnn'  # 2D doesn't work with e3nn library.
    local_config = True
    if local_config:
        print("running locally")
        cfg = to_local_config(cfg)

    if cfg.target.custom_samples:
        print(f"loading custom dataset for temperature of {cfg.target.temperature}")
        load_dataset = partial(load_dataset_custom, temperature=cfg.target.temperature)
    else:
        load_dataset = load_dataset_standard
    experiment_config = create_train_config(cfg, dim=2, n_nodes=4,
                                            load_dataset=load_dataset)
    train(experiment_config)


#
# if __name__ == '__main__':
#     run()
