import jax.random
from omegaconf import DictConfig
import yaml
import time

from molboil.targets.data import load_dw4, load_lj13, load_qm9
from examples.create_train_config import create_flow_config, AugmentedFlow, TrainingState
from examples.create_train_config import build_flow


def sampling_time(problem = "dw4"):
    key = jax.random.PRNGKey(0)
    if problem == "dw4":
        cfg = DictConfig(yaml.safe_load(open(f"examples/config/dw4.yaml")))
        train_data, _, _ = load_dw4(train_set_size=1,
                                                     test_set_size=1,
                                                     val_set_size=1)
    elif problem == "lj13":
        cfg = DictConfig(yaml.safe_load(open(f"examples/config/lj13.yaml")))
        train_data, _, _ = load_lj13(train_set_size=1)
    else:
        assert problem == "qm9"
        cfg = DictConfig(yaml.safe_load(open(f"examples/config/qm9.yaml")))
        train_data, _, _ = load_qm9(train_set_size=1)

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



#TODO: Add flow type, time properly, format in latex table
if __name__ == '__main__':
    for problem in ["dw4"]: # , "lj13", "qm9"]:
        print(sampling_time())
