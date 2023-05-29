import pickle
from typing import Any

import pathlib
import os
import hydra
import numpy as np
from omegaconf import DictConfig, open_dict
import jax.numpy as jnp
import jax
import haiku as hk

from molboil.utils.checkpoints import get_latest_checkpoint

from flow.build_flow import build_flow
from examples.create_train_config import create_flow_config


@hydra.main(config_path="./config", config_name="aldp.yaml")
def run(cfg: DictConfig):
    # Get parameters
    seed = 0
    n_batches = 1000
    n_samples = 10000
    if 'FLOW_SEED' in os.environ:
        seed = int(os.environ['FLOW_SEED'])
    if 'FLOW_N_BATCHES' in os.environ:
        n_batches = int(os.environ['FLOW_N_BATCHES'])
    if 'FLOW_N_SAMPLES' in os.environ:
        n_samples = int(os.environ['FLOW_N_SAMPLES'])

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

    # Sample function for eval
    flow_sample_fn = jax.jit(flow.sample_and_log_prob_apply, static_argnums=3)
    separate_samples_fn = jax.jit(flow.joint_to_separate_samples)
    def sample_fn(params: Any, features: jnp.array, key: jnp.array, n_samples: int):
        joint_samples_flow, log_q = flow_sample_fn(params, features, key,
                                                   (n_samples,))
        _, positions_x, positions_a = separate_samples_fn(joint_samples_flow)
        return positions_x, positions_a, log_q

    # Load checkpoint
    checkpoints_dir = os.path.join(cfg.training.save_dir, f"model_checkpoints")
    latest_cp = get_latest_checkpoint(checkpoints_dir, key="state_")
    with open(latest_cp, "rb") as f:
        state = pickle.load(f)

    # Sample
    prng_seq = hk.PRNGSequence(seed)
    features = jnp.arange(22, dtype=int)[:, None]
    positions_x = []
    positions_a = []
    log_q = []
    for _ in range(n_batches):
        positions_x_, positions_a_, log_q_ = sample_fn(state.params, features,
                                                       next(prng_seq), n_samples)
        positions_x.append(np.array(positions_x_))
        positions_a.append(np.array(positions_a_))
        log_q.append(np.array(log_q_))
    positions_x = np.concatenate(positions_x)
    positions_a = np.concatenate(positions_a)
    log_q = np.concatenate(log_q)

    # Save results
    sample_dir = os.path.join(cfg.training.save_dir, f"samples")
    pathlib.Path(sample_dir).mkdir(exist_ok=True)
    np.savez(os.path.join(sample_dir, f"sample_%04i.npz" % seed),
             positions_x=positions_x, positions_a=positions_a, log_q=log_q)



if __name__ == '__main__':
    run()
