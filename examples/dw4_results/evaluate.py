from functools import partial

import jax.random
from omegaconf import DictConfig
import yaml
import pandas as pd

from target.double_well import log_prob_fn
from molboil.targets.data import load_dw4
from molboil.train.base import eval_fn
from examples.load_flow_and_checkpoint import load_flow
from train.max_lik_train_and_eval import eval_non_batched, get_eval_on_test_batch


def evaluate_dw4(flow,
                 state,
                 test_data,
                 K: int = 50,
                 n_samples_eval: int = int(1e4),
                 eval_batch_size=100):
    key = jax.random.PRNGKey(0)

    eval_on_test_batch_fn = partial(get_eval_on_test_batch,
                                    flow=flow, K=K, test_invariances=True)
    eval_batch_free_fn = partial(
        eval_non_batched,
        single_feature=test_data.features[0],
        flow=flow,
        n_samples=n_samples_eval,
        inner_batch_size=min(eval_batch_size, n_samples_eval),
        target_log_prob=log_prob_fn)

    def evaluation_fn(state, key) -> dict:
        eval_fn_no_jit = eval_fn._fun
        eval_info = eval_fn_no_jit(test_data, key, state.params,
                            eval_on_test_batch_fn=eval_on_test_batch_fn,
                            eval_batch_free_fn=eval_batch_free_fn,
                            batch_size=eval_batch_size)
        return eval_info

    info = evaluation_fn(state, key)

    return info


if __name__ == '__main__':
    flow_types = ['spherical']  # , 'along_vector', 'proj', 'non_equivariant']
    seeds = [0]

    train_data, valid_data, test_data = load_dw4(train_set_size=1000,
                                                 test_set_size=100,
                                                 val_set_size=1000)

    data = pd.DataFrame()

    for flow_type in flow_types:
        for seed in seeds:
            checkpoint_path = f"examples/dw4_results/models/{flow_type}_seed0.pkl"
            cfg = DictConfig(yaml.safe_load(open(f"examples/config/dw4.yaml")))

            flow, state = load_flow(cfg, checkpoint_path)

            info = evaluate_dw4(flow, state, test_data, K=8, n_samples_eval=10)







