import hydra
from omegaconf import DictConfig

from molboil.targets.data import load_qm9
from molboil.train.train import train
from examples.create_train_config import create_train_config

def load_dataset(train_set_size, valid_set_size):
    train_data, valid_data, test_data = load_qm9(train_set_size=train_set_size)
    return train_data, valid_data[:valid_set_size]


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
