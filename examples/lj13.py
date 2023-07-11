import hydra
from omegaconf import DictConfig
import jax

from molboil.train.train import train
from molboil.targets.data import load_lj13
from target.leonard_jones import log_prob_fn
from examples.create_train_config import create_train_config


def load_dataset(train_set_size: int, valid_set_size: int, final_run=True):
    train, valid, test = load_lj13(train_set_size)
    if not final_run:
        return train, valid[:valid_set_size]
    else:
        return train, test[:valid_set_size]

def to_local_config(cfg: DictConfig) -> DictConfig:
    """Change config to make it fast to run locally. Also remove saving."""
    cfg.training.train_set_size = 4
    cfg.training.test_set_size = 4
    cfg.flow.nets.type = "egnn"
    cfg.flow.nets.egnn.mlp_units = (2, 2)
    cfg.flow.n_layers = 1
    cfg.flow.nets.egnn.n_blocks = 2
    cfg.training.batch_size = 2
    cfg.flow.type = 'proj'  # 'spherical'
    cfg.flow.kwargs.spherical.spline_num_bins = 3
    cfg.flow.n_aug = 1

    cfg.training.n_epoch = 32
    cfg.training.save = False
    cfg.flow.scaling_layer = False
    cfg.training.plot_batch_size = 4
    cfg.logger = DictConfig({"list_logger": None})

    cfg.flow.nets.mlp_head_config.mlp_units = (4,)
    cfg.flow.nets.egnn.mlp_units = (4,)
    cfg.flow.nets.egnn.n_blocks = 2
    cfg.flow.nets.non_equivariant_transformer_config.output_dim = 3
    cfg.flow.nets.non_equivariant_transformer_config.mlp_units = (4,)
    cfg.flow.nets.non_equivariant_transformer_config.n_layers = 2
    cfg.flow.nets.non_equivariant_transformer_config.num_heads = 1

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

    if cfg.training.use_64_bit:
        jax.config.update("jax_enable_x64", True)

    experiment_config = create_train_config(cfg,
                                            dim=3,
                                            n_nodes=13,
                                            load_dataset=load_dataset,
                                            target_log_prob_fn=log_prob_fn)
    train(experiment_config)


if __name__ == '__main__':
    run()
