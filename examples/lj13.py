import hydra
from omegaconf import DictConfig
import jax.numpy as jnp
import numpy as np

from examples.train import train, create_train_config



def load_dataset(batch_size, train_set_size: int = 1000, val_set_size:int = 1000, seed: int = 0):
    # dataset from https://github.com/vgsatorras/en_flows
    # Loading following https://github.com/vgsatorras/en_flows/blob/main/dw4_experiment/dataset.py.

    # Train data
    data = np.load("target/data/holdout_data_LJ13.npy")
    idx = np.load("target/data/idx_LJ13.npy")
    train_set = data[idx[:train_set_size]]
    train_set = jnp.reshape(train_set, (-1, 13, 3))
    train_set = train_set[:train_set_size - (train_set.shape[0] % batch_size)]

    # Test set
    test_data_path = 'target/data/all_data_LJ13.npy'  # target/data/lj_data_vertices13_dim3.npy
    dataset = np.load(test_data_path)
    dataset = jnp.reshape(dataset, (-1, 13, 3))
    test_set = dataset[:val_set_size]
    return train_set, test_set


def to_local_config(cfg: DictConfig) -> DictConfig:
    """Change config to make it fast to run locally. Also remove saving."""
    cfg.flow.nets.egnn.mlp_units = (4,)
    cfg.flow.nets.egnn.h_embedding_dim = 3
    cfg.flow.nets.transformer.mlp_units = (4,)
    cfg.flow.n_layers = 2
    cfg.training.batch_size = 4

    cfg.training.n_epoch = 32
    cfg.training.save = False
    cfg.flow.type = ['proj']
    cfg.training.plot_batch_size = 4
    cfg.logger = DictConfig({"list_logger": None})
    return cfg


@hydra.main(config_path="./config", config_name="lj13.yaml")
def run(cfg: DictConfig):
    local_config = True
    if local_config:
        cfg = to_local_config(cfg)

    experiment_config = create_train_config(cfg,
                                            dim=3,
                                            n_nodes=13,
                                            load_dataset=load_dataset)
    train(experiment_config)



if __name__ == '__main__':
    run()
