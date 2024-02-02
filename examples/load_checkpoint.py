import os
import re
import yaml
import ast

import wandb
import pickle
import hydra

api = wandb.Api()

BASE_PATH = "."


def unpack_dict(wandb_dict):
    # Convert string values to dictionaries using ast.literal_eval
    for key, value in wandb_dict.items():
        wandb_dict[key] = ast.literal_eval(value)
    return wandb_dict


def get_wandb_run_by_name(run_id: str):
    run = api.run(path=f'flow-ais-bootstrap/fab/{run_id}')
    return run


def download_checkpoint_cfg(run_id: str, base_path):
    assert os.path.exists(base_path)
    # run = get_wandb_run_via_filters(tags, seed)
    run = get_wandb_run_by_name(run_id)
    cfg = run.config

    cfg = unpack_dict(cfg)
    yaml.dump(cfg, open(f"{base_path}/checkpoint_cfg.yaml", "w"))

    training_iter = 16000 - 1  # cfg['training']['n_training_iter']-1

    for file in run.files():
        if re.search(fr".*state_0*{training_iter}.pkl", str(file)):
            file.download(exist_ok=True)
            path = re.search(r"([^\s]*model_checkpoints[^\s]*)", str(file)).group()
            new_path = f"{base_path}/checkpoint.pkl"
            os.replace(path, new_path)
            print("saved" + path)

    state = pickle.load(open(new_path, "rb"))
    return state, cfg


def download_wandb_checkpoint(run_id: str):
    # Returns params from checkpoint at the end of training.
    state, cfg = download_checkpoint_cfg(run_id, base_path=BASE_PATH)
    return state, cfg


def load_checkpoint(
        run_id: str,
        requires_download: bool = True,
        use_current_cfg_as_base_config: bool = False):
    if not requires_download:
        new_path = f"{BASE_PATH}checkpoint.pkl"
        state = pickle.load(open(new_path, "rb"))
    else:
        state, _ = download_wandb_checkpoint(run_id=run_id)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path="../", job_name="colab_app"):
        cfg_model = hydra.compose(config_name=f"checkpoint_cfg")
    if use_current_cfg_as_base_config:
        with hydra.initialize(version_base=None, config_path="../examples/config", job_name="colab_app"):
            cfg = hydra.compose(config_name=f"lj13_fab")
        cfg.flow = cfg_model.flow
        cfg.target = cfg_model.target
        cfg.conditioning = cfg_model.conditioning
    else:
        cfg = cfg_model

    return state, cfg


if __name__ == '__main__':
    # run = get_wandb_run_by_name()
    state, cfg = load_checkpoint('2dk755h3')
