from typing import Tuple

import matplotlib.pyplot as plt
from omegaconf import DictConfig
import yaml
import pickle
import jax.numpy as jnp
import jax

from examples.create_train_config import create_flow_config, AugmentedFlow, AugmentedFlowParams
from examples.create_train_config import build_flow
from examples.dw4 import to_local_config

def load_flow(cfg: DictConfig, checkpoint_path: str) -> Tuple[AugmentedFlow, AugmentedFlowParams]:
    flow_config = create_flow_config(cfg)
    flow = build_flow(flow_config)
    params = pickle.load(open(checkpoint_path, "rb"))
    return flow, params



if __name__ == '__main__':
    # Simple example where we load a checkpoint from dw4 and plot the trained model.
    # To run this example, first run `examples/dw4.py` with checkpointing turned on.
    checkpoint_path = 'dw4_results/2023-03-16T16:59:40.861716/model_checkpoints/iter_199/state.pkl'

    # I used the `to_local_config` in train.dw4.py for the run, so I use it to also load the checkpoint.
    cfg = to_local_config(DictConfig(yaml.safe_load(open(f"examples/config/dw4.yaml"))))
    flow, params = load_flow(cfg, checkpoint_path)

    graph_features = jnp.zeros((4, 1))
    n_samples = 64
    key = jax.random.PRNGKey(0)
    samples, log_prob, extra = flow.sample_and_log_prob_with_extra_apply(params, graph_features, key, (n_samples,))

    fig, ax = plt.subplots()
    plot_sample_hist(samples.positions, ax)
    plt.show()
