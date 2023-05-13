import jax.random
from omegaconf import DictConfig
import yaml

from target.double_well import log_prob_fn
from molboil.targets.data import load_dw4
from examples.load_flow_and_checkpoint import load_flow
import jax.numpy as jnp


if __name__ == '__main__':
    flow_type = 'spherical'
    seed = 0
    key = jax.random.PRNGKey(0)

    small = True
    if small:
        test_set_size = 10
        K = 2
        n_samples_eval = 10
        eval_batch_size = 10
    else:
        test_set_size = 1000
        K = 50
        n_samples_eval = 10_000
        eval_batch_size = 200

    train_data, valid_data, test_data = load_dw4(train_set_size=1000,
                                                 test_set_size=test_set_size,
                                                 val_set_size=1000)

    checkpoint_path = f"examples/dw4_results/models/{flow_type}_seed0.pkl"
    cfg = DictConfig(yaml.safe_load(open(f"examples/config/dw4.yaml")))

    flow, state = load_flow(cfg, checkpoint_path)

    key, subkey = jax.random.split(key)
    x_augmented, log_p_a = flow.aux_target_sample_n_and_log_prob_apply(state.params.aux_target, test_data,
                                                                       subkey, K)

    x_test = jax.tree_map(lambda x: jnp.repeat(x[None, ...], K, axis=0), test_data)
    joint_sample = flow.separate_samples_to_joint(x_test.features, x_test.positions, x_augmented)

    z, log_det, extra = jax.vmap(flow.bijector_inverse_and_log_det_with_extra_apply, in_axes=(None, 0))(
        state.params.bijector, joint_sample)

    log_prob_base = jax.vmap(flow.base_log_prob, in_axes=(None, 0))(state.params.base, z)

    lower_bound = log_det + log_prob_base - log_p_a
