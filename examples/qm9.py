import hydra
from omegaconf import DictConfig
import jax

from molboil.targets.data import load_qm9
from molboil.train.train import train
from examples.create_train_config import create_train_config
from examples.lj13 import to_local_config


def load_dataset(train_set_size, valid_set_size, final_run: bool = True):
    train_data, valid_data, test_data = load_qm9(train_set_size=train_set_size)
    if not final_run:
        return train_data, valid_data[:valid_set_size]
    else:
        return train_data, test_data[:valid_set_size]


@hydra.main(config_path="./config", config_name="qm9.yaml")
def run(cfg: DictConfig):
    local_config = False
    if local_config:
        cfg = to_local_config(cfg)

    if cfg.training.use_64_bit:
        jax.config.update("jax_enable_x64", True)

    experiment_config = create_train_config(cfg, dim=3, n_nodes=19,
                                            load_dataset=load_dataset)
    experiment_config = experiment_config._replace(save_state_all_devices=True)
    train(experiment_config)



if __name__ == '__main__':
    run()
