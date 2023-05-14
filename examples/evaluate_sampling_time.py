import jax.random
from omegaconf import DictConfig
import yaml
import time

from molboil.targets.data import load_dw4
from examples.create_train_config import create_flow_config, AugmentedFlow, TrainingState
from examples.create_train_config import build_flow


def sampling_time_dw4():
    key = jax.random.PRNGKey(0)
    cfg = DictConfig(yaml.safe_load(open(f"examples/config/dw4.yaml")))
    train_data, valid_data, test_data = load_dw4(train_set_size=1,
                                                 test_set_size=1,
                                                 val_set_size=1)
    flow_config = create_flow_config(cfg)
    flow = build_flow(flow_config)

    params = flow.init(key, train_data[0])

    @jax.jit
    def sample(params, key):
        return flow.sample_apply(params, train_data[0].features, key, ())

    dummy_sample = sample(params, key)
    start = time.time()
    dummy_sample = sample(params, key)
    dummy_sample.positions.block_until_ready()
    time_elapsed = time.time() - start
    return time_elapsed




if __name__ == '__main__':
    print(sampling_time_dw4())
