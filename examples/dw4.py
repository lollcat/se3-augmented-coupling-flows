import hydra
from omegaconf import DictConfig
import jax
import jax.numpy as jnp
import numpy as np

from examples.train import train, create_train_config
from utils.train_and_eval import original_dataset_to_joint_dataset



def load_dataset(batch_size, train_set_size: int = 1000, test_set_size:int = 1000, seed: int = 0):
    # dataset from https://github.com/vgsatorras/en_flows
    # Loading following https://github.com/vgsatorras/en_flows/blob/main/dw4_experiment/dataset.py.

    data_path = 'target/data/dw4-dataidx.npy'  # 'target/data/dw_data_vertices4_dim2.npy'
    dataset = np.asarray(np.load(data_path, allow_pickle=True)[0])
    dataset = jnp.reshape(dataset, (-1, 4, 2))
    dataset = original_dataset_to_joint_dataset(dataset, jax.random.PRNGKey(seed))

    train_set = dataset[:train_set_size]
    train_set = train_set[:train_set_size - (train_set.shape[0] % batch_size)]

    test_set = dataset[-test_set_size:]
    return train_set, test_set


def to_local_config(cfg: DictConfig) -> DictConfig:
    """Change config to make it fast to run locally. Also remove saving."""
    cfg.training.lr = 1e-3
    cfg.flow.egnn.tanh = False
    cfg.flow.type = 'proj'  # 'proj_v2'
    cfg.flow.egnn.mlp_units = (4,)
    cfg.flow.transformer.mlp_units = (4,)
    cfg.flow.transformer.n_layers = 2
    cfg.flow.n_layers = 3
    cfg.training.batch_size = 32
    cfg.training.n_epoch = 50
    cfg.training.save = False
    cfg.training.plot_batch_size = 64
    cfg.training.K_marginal_log_lik = 5
    cfg.logger = DictConfig({"list_logger": None})

    debug = False
    if debug:
        cfg.flow.fast_compile = False
        cfg_train = dict(cfg['training'])
        cfg_train['scan_run'] = False
        cfg.training = DictConfig(cfg_train)
    return cfg

@hydra.main(config_path="./config", config_name="dw4.yaml")
def run(cfg: DictConfig):
    local_config = True
    if local_config:
        cfg = to_local_config(cfg)
    experiment_config = create_train_config(cfg, dim=2, n_nodes=4,
                                            load_dataset=load_dataset)
    train(experiment_config)



if __name__ == '__main__':
    run()
