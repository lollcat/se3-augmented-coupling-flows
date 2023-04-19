from typing import Any

from functools import partial
import hydra
from omegaconf import DictConfig
import jax.numpy as jnp
import jax

from molboil.targets.data import load_aldp
from molboil.train.train import train
from molboil.train.base import eval_fn
from molboil.eval.aldp import eval_and_plot_fn

from flow.build_flow import build_flow
from examples.create_train_config import create_train_config, create_flow_config
from utils.aug_flow_train_and_eval import get_eval_on_test_batch


@hydra.main(config_path="./config", config_name="aldp.yaml")
def run(cfg: DictConfig):
    def load_dataset(train_set_size: int, val_set_size: int):
        return load_aldp(train_path=cfg.target.data.train, val_path=cfg.target.data.val,
                         train_n_points=train_set_size, val_n_points=val_set_size)[:2]

    train_set, val_set = load_dataset(cfg.training.train_set_size, cfg.training.test_set_size)
    flow_config = create_flow_config(cfg)
    # Add edges of aldp for harmonic potential
    if flow_config.base.x_dist.type == 'harmonic_potential':
        edges = [
            [0, 1], [1, 2], [1, 3], [1, 4], [4, 5], [4, 6], [6, 7],
            [6, 8], [8, 9], [8, 10], [10, 11], [10, 12], [10, 13], [8, 14],
            [14, 15], [14, 16], [16, 17], [16, 18], [18, 19], [18, 20], [18, 21]
        ]
        flow_config.base.x_dist.edges = edges
    flow = build_flow(flow_config)

    # Sample function for eval
    flow_sample_fn = jax.jit(flow.sample_apply, static_argnums=3)
    separate_samples_fn = jax.jit(flow.joint_to_separate_samples)
    def sample_fn(params: Any, features: jnp.array, key: jnp.array, n_samples: int):
        joint_samples_flow = flow_sample_fn(params, features, key,
                                            (n_samples,))
        _, positions_x, _ = separate_samples_fn(joint_samples_flow)
        return positions_x

    # Create eval function
    eval_on_test_batch_fn = partial(get_eval_on_test_batch,
                                    flow=flow, K=cfg.training.K_marginal_log_lik,
                                    test_invariances=True)
    eval_fn_ = partial(eval_fn, eval_on_test_batch_fn=eval_on_test_batch_fn,
                       eval_batch_free_fn=None, batch_size=cfg.training.plot_batch_size)
    eval_and_plot_fn_ = partial(eval_and_plot_fn, sample_fn=sample_fn, train_data=train_set, test_data=val_set,
                                n_samples=cfg.training.plot_batch_size, n_batches=cfg.eval.plot_n_batches,
                                eval_fn=eval_fn_)

    experiment_config = create_train_config(cfg, dim=3, n_nodes=22,
                                            load_dataset=load_dataset,
                                            eval_and_plot_fn=eval_and_plot_fn_,
                                            date_folder=False)

    train(experiment_config)


if __name__ == '__main__':
    run()
