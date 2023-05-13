import hydra
from omegaconf import DictConfig

from molboil.train.train import train
from molboil.targets.data import load_lj13
from examples.create_fab_train_config import create_train_config

from target.leonard_jones import log_prob_fn


def load_dataset(train_set_size: int, valid_set_size: int, final_run: bool = True):
    train, valid, test = load_lj13(train_set_size)
    if not final_run:
        return train, valid[:valid_set_size]
    else:
        return train, test[:valid_set_size]

def to_local_config(cfg: DictConfig) -> DictConfig:
    """Change config to make it fast to run locally. Also remove saving."""
    cfg.flow.nets.type = "egnn"
    cfg.flow.nets.egnn.mlp_units = cfg.flow.nets.e3gnn.mlp_units = (4,)
    cfg.flow.n_layers = 1
    cfg.flow.nets.egnn.n_blocks = cfg.flow.nets.e3gnn.n_blocks = 2
    cfg.training.batch_size = 2
    cfg.flow.type = 'nice'
    cfg.flow.n_aug = 1
    cfg.fab.eval_inner_batch_size = 2
    cfg.fab.eval_total_batch_size = 4
    cfg.fab.n_updates_per_smc_forward_pass = 2
    cfg.fab.n_intermediate_distributions = 4
    cfg.fab.buffer_min_length_batches = 4
    cfg.fab.buffer_max_length_batches = 10

    cfg.training.n_epoch = 32
    cfg.training.save = False
    cfg.training.plot_batch_size = 4
    cfg.logger = DictConfig({"list_logger": None})

    debug = False
    if debug:
        cfg_train = dict(cfg['training'])
        cfg_train['scan_run'] = False
        cfg.training = DictConfig(cfg_train)

    return cfg

@hydra.main(config_path="./config", config_name="lj13_fab.yaml")
def run(cfg: DictConfig):
    local_config = False
    if local_config:
        cfg = to_local_config(cfg)

    experiment_config = create_train_config(cfg,
                                            target_log_p_x_fn=log_prob_fn,
                                            dim=3,
                                            n_nodes=13,
                                            load_dataset=load_dataset)
    train(experiment_config)


if __name__ == '__main__':
    run()
