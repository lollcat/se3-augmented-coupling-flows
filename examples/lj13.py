import hydra
from omegaconf import DictConfig

from molboil.train.train import train
from molboil.targets.data import load_lj13
from examples.create_train_config import create_train_config


def load_dataset(train_set_size: int, valid_set_size: int):
    train, valid, test = load_lj13(train_set_size)
    return train, valid[:valid_set_size]

def to_local_config(cfg: DictConfig) -> DictConfig:
    """Change config to make it fast to run locally. Also remove saving."""
    # cfg.flow.nets.type = "e3gnn"
    cfg.flow.nets.egnn.mlp_units = cfg.flow.nets.e3gnn.mlp_units = (4,)
    cfg.flow.nets.egnn.h_embedding_dim = 3
    cfg.flow.nets.transformer.mlp_units = (4,)
    cfg.flow.n_layers = 2
    cfg.training.batch_size = 4

    # Make MACE small for local run.
    cfg.flow.nets.mace.n_invariant_feat_residual = 16
    cfg.flow.nets.mace.residual_mlp_width = 16
    cfg.flow.nets.mace.interaction_mlp_width = 16
    cfg.flow.nets.mace.interaction_mlp_depth = 1
    cfg.flow.nets.mace.n_invariant_hidden_readout_block = 16
    cfg.flow.nets.mace.max_ell = 2
    cfg.flow.nets.mace.hidden_irreps = '4x0e+3x1o+1x2e'
    cfg.flow.nets.mace.correlation = 2


    cfg.flow.type = 'proj_rnvp'


    cfg.training.n_epoch = 32
    cfg.training.save = False
    # cfg.flow.type = 'proj'
    cfg.training.plot_batch_size = 4
    cfg.logger = DictConfig({"list_logger": None})

    debug = False
    if debug:
        cfg_train = dict(cfg['training'])
        cfg_train['scan_run'] = False
        cfg.training = DictConfig(cfg_train)

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
