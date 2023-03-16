import hydra
from omegaconf import DictConfig
import jax
import numpy as np

from examples.train import train, create_train_config
from utils.data import positional_dataset_only_to_full_graph



def load_dataset(batch_size, train_data_n_points = None, test_data_n_points = None, seed=0):
    # First need to run `qm9.download_data`
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))

    try:
        data_dir = "target/data/qm9_"
        train_data = np.load(data_dir + "train.npy")
        test_data = np.load(data_dir + "test.npy")
        valid_data = np.load(data_dir + "valid.npy")
    except:
        print("Data directory not found. Try running `dataset.py` in the `qm9` dir, otherwise speak to Laurence :)")
        raise Exception

    if train_data_n_points is not None:
        train_data = train_data[:train_data_n_points]
    if test_data_n_points is not None:
        test_data = test_data[:test_data_n_points]

    train_data = train_data[:train_data.shape[0] - (train_data.shape[0] % batch_size)]

    return positional_dataset_only_to_full_graph(train_data), positional_dataset_only_to_full_graph(test_data)

def to_local_config(cfg: DictConfig) -> DictConfig:
    """Change config to make it fast to run locally. Also remove saving."""
    cfg.flow.nets.egnn.mlp_units = (4,)
    cfg.flow.nets.transformer.mlp_units = (4,)
    cfg.flow.nets.transformer.num_heads = 2
    cfg.flow.nets.transformer.key_size = 2
    cfg.flow.nets.transformer.n_layers = 2
    cfg.flow.nets.egnn.n_layers = 2
    cfg.flow.n_layers = 2
    cfg.training.batch_size = 2
    cfg.training.n_epoch = 32
    cfg.training.save = False
    cfg.training.plot_batch_size = 2
    cfg.training.train_set_size = 1000
    cfg.training.test_set_size = 1000

    # Make MACE small for local run.
    cfg.flow.nets.mace.n_invariant_feat_residual = 16
    cfg.flow.nets.mace.residual_mlp_width = 4
    cfg.flow.nets.mace.interaction_mlp_width = 4
    cfg.flow.nets.mace.interaction_mlp_depth = 1
    cfg.flow.nets.mace.n_invariant_hidden_readout_block = 4
    cfg.flow.nets.mace.max_ell = 1
    cfg.flow.nets.mace.hidden_irreps = '4x0e+3x1o+1x2e'
    cfg.flow.nets.mace.correlation = 2

    cfg.logger = DictConfig({"list_logger": None})
    return cfg



@hydra.main(config_path="./config", config_name="qm9.yaml")
def run(cfg: DictConfig):
    local_config = False
    if local_config:
        cfg = to_local_config(cfg)

    experiment_config = create_train_config(cfg, dim=3, n_nodes=19,
                                            load_dataset=load_dataset)
    train(experiment_config)



if __name__ == '__main__':
    run()
