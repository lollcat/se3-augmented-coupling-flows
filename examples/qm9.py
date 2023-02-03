import hydra
from omegaconf import DictConfig
import jax
import numpy as np

from examples.train import train, create_train_config
from utils.train_and_eval import original_dataset_to_joint_dataset



def load_dataset(batch_size, train_data_n_points = None, test_data_n_points = None, seed=0):
    # First need to run `qm9.download_data`
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))

    data_dir = "target/data/qm9_"
    train_data = np.load(data_dir + "train.npy")
    test_data = np.load(data_dir + "test.npy")
    valid_data = np.load(data_dir + "valid.npy")

    if train_data_n_points is not None:
        train_data = train_data[:train_data_n_points]
    if test_data_n_points is not None:
        test_data = test_data[:test_data_n_points]

    train_data = train_data[:train_data.shape[0] - (train_data.shape[0] % batch_size)]

    train_data = original_dataset_to_joint_dataset(train_data, key1)
    test_data = original_dataset_to_joint_dataset(test_data, key2)

    return train_data, test_data

def to_local_config(cfg: DictConfig) -> DictConfig:
    """Change config to make it fast to run locally. Also remove saving."""
    cfg.flow.egnn.mlp_units = (4,)
    cfg.flow.transformer.mlp_units = (4,)
    cfg.flow.n_layers = 1
    cfg.training.batch_size = 2
    cfg.training.n_epoch = 32
    cfg.training.save = False
    cfg.training.plot_batch_size = 2
    cfg.training.train_set_size = 100
    cfg.training.test_set_size = 100
    cfg.logger = DictConfig({"list_logger": None})
    return cfg



@hydra.main(config_path="./config", config_name="qm9.yaml")
def run(cfg: DictConfig):
    local_config = False
    if local_config:
        cfg = to_local_config(cfg)

    experiment_config = create_train_config(cfg, dim=3, n_nodes=29,
                                            load_dataset=load_dataset)
    train(experiment_config)



if __name__ == '__main__':
    run()
