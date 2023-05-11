import pickle

from functools import partial
import pathlib
import os
import hydra
import numpy as np
from omegaconf import DictConfig, open_dict
import jax.numpy as jnp
import jax
import haiku as hk

from molboil.targets.data import load_aldp
from molboil.utils.checkpoints import get_latest_checkpoint
from molboil.train.base import eval_fn

from flow.build_flow import build_flow
from examples.create_train_config import create_flow_config
from train.max_lik_train_and_eval import get_eval_on_test_batch


@hydra.main(config_path="./config", config_name="aldp.yaml")
def run(cfg: DictConfig):
    # Get parameters
    test_path = cfg.target.data.val
    n_points = 100000
    batch_size = 1000
    K = 10
    seed = 0
    if 'FLOW_TEST_PATH' in os.environ:
        test_path = str(os.environ['FLOW_TEST_PATH'])
    if 'FLOW_N_POINTS' in os.environ:
        n_points = int(os.environ['FLOW_N_POINTS'])
    if 'FLOW_BATCH_SIZE' in os.environ:
        batch_size = int(os.environ['FLOW_BATCH_SIZE'])
    if 'FLOW_K' in os.environ:
        K = int(os.environ['FLOW_K'])
    if 'FLOW_SEED' in os.environ:
        seed = int(os.environ['FLOW_SEED'])

    if cfg.training.use_64_bit:
        jax.config.update("jax_enable_x64", True)

    # Add edges of aldp for harmonic potential
    try:
        if cfg['flow']['base']['x_dist']['type'] == 'harmonic_potential':
            edges = [
                [0, 1], [1, 2], [1, 3], [1, 4], [4, 5], [4, 6], [6, 7],
                [6, 8], [8, 9], [8, 10], [10, 11], [10, 12], [10, 13], [8, 14],
                [14, 15], [14, 16], [16, 17], [16, 18], [18, 19], [18, 20], [18, 21]
            ]
            with open_dict(cfg):
                cfg.flow.base.x_dist.edges = edges
    except:
        pass
    flow_config = create_flow_config(cfg)
    flow = build_flow(flow_config)

    # Eval function
    eval_on_test_batch_fn = partial(get_eval_on_test_batch,
                                    flow=flow, K=K,
                                    test_invariances=False)
    eval_fn_ = partial(eval_fn, eval_on_test_batch_fn=eval_on_test_batch_fn,
                       eval_batch_free_fn=None, batch_size=batch_size)

    # Load checkpoint
    checkpoints_dir = os.path.join(cfg.training.save_dir, f"model_checkpoints")
    latest_cp = get_latest_checkpoint(checkpoints_dir, key="state_")
    with open(latest_cp, "rb") as f:
        state = pickle.load(f)

    # Get test set
    test_data = load_aldp(test_path=test_path, test_n_points=n_points)[2]

    # Run eval fn
    key = jax.random.PRNGKey(seed)
    eval_info = eval_fn(test_data, key, state.params)

    # Save results
    sample_dir = os.path.join(cfg.training.save_dir, f"log_prob")
    pathlib.Path(sample_dir).mkdir(exist_ok=True)
    np.savez(os.path.join(sample_dir, f"log_prob_%04i.npz" % seed),
             **eval_info)



if __name__ == '__main__':
    run()
