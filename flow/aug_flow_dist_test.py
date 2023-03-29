import chex
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from utils.test import test_fn_is_invariant
from utils.test import get_minimal_nets_config
from flow.build_flow import build_flow, ConditionalAuxDistConfig, FlowDistConfig, BaseConfig
from flow.aug_flow_dist import FullGraphSample
from flow.distrax_with_extra import Extra


_N_FLOW_LAYERS = 3
_N_NODES = 7
_FLOW_TYPE = "nice"
_FAST_COMPILE_FLOW = False
_IDENTITY_INIT = False
_FEATURE_DIM = 2


def test_distribution(dim: int = 3, n_aug: int = 3):
    """Visualise samples from the distribution, and check that it's log prob is invariant to
    translation and rotation."""
    if jnp.ones(()).dtype == jnp.float64:
        rtol = 1e-5
    else:
        rtol = 1e-4

    n_nodes = _N_NODES
    batch_size = 5
    key = jax.random.PRNGKey(0)
    base_aux_config = BaseConfig()
    target_aux_config = ConditionalAuxDistConfig(trainable_augmented_scale=False)


    config = FlowDistConfig(
        dim=dim, n_layers=_N_FLOW_LAYERS,
        nodes=_N_NODES,
        identity_init=_IDENTITY_INIT,
        type=_FLOW_TYPE,
        compile_n_unroll=2,
        nets_config=get_minimal_nets_config('egnn'),
        base=base_aux_config,
        target_aux_config=target_aux_config,
        n_aug=n_aug
    )

    flow = build_flow(config)


    # Init params.
    key, subkey = jax.random.split(key)
    dummy_samples = FullGraphSample(positions=jax.random.normal(subkey, (batch_size, n_nodes, dim)),
                                    features=jnp.zeros((batch_size, n_nodes, _FEATURE_DIM)))
    key, subkey = jax.random.split(key)
    params = flow.init(subkey, dummy_samples)
    params_check = flow.init(subkey, dummy_samples[0])
    chex.assert_trees_all_equal(params, params_check)  # Check that params aren't effected by batch-ing.

    info = flow.get_base_and_target_info(params)

    # Test aux-target distribution.
    key, subkey = jax.random.split(key)
    n_aug_samples = 2
    aug_samples, aux_log_probs = flow.aux_target_sample_n_and_log_prob_apply(
        params.aux_target, dummy_samples, subkey, n_aug_samples)
    chex.assert_shape(aug_samples, (n_aug_samples, batch_size, n_nodes, n_aug, dim))
    chex.assert_shape(aux_log_probs, (n_aug_samples, batch_size))
    aux_log_prob_check = flow.aux_target_log_prob_apply(
        params.aux_target, dummy_samples, aug_samples[0])
    chex.assert_trees_all_close(aux_log_prob_check, aux_log_probs[0])
    aug_samples_check = flow.aux_target_sample_n_apply(
        params.aux_target, dummy_samples, subkey, n_aug_samples)
    chex.assert_trees_all_close(aug_samples_check, aug_samples_check)

    # Test flow.
    key, subkey = jax.random.split(key)
    sample, log_prob = flow.sample_and_log_prob_apply(params, dummy_samples.features[0], subkey, (batch_size,))
    sample_old = sample  # save for later in this test.
    chex.assert_tree_all_finite(log_prob)
    chex.assert_shape(log_prob, (batch_size,))
    chex.assert_shape(sample.positions, (batch_size, n_nodes, n_aug+1, dim))
    chex.assert_shape(sample.features, (batch_size, n_nodes, 1, _FEATURE_DIM))

    log_prob_check = flow.log_prob_apply(params, sample)
    log_prob_single_sample_check = flow.log_prob_apply(params, sample[0])
    chex.assert_trees_all_equal(log_prob_check[0], log_prob_single_sample_check)

    chex.assert_trees_all_close(log_prob, log_prob_check, rtol=rtol)

    def fake_loss_fn(params, use_extra=True, x=sample):
        if use_extra:
            log_prob, extra = flow.log_prob_with_extra_apply(params, x)
        else:
            log_prob = flow.log_prob_apply(params, x)
            extra = Extra()
        loss = jnp.mean(log_prob)  # + jnp.mean(extra.aux_loss)
        return loss, extra

    (fake_loss, extra), grads = jax.value_and_grad(fake_loss_fn, has_aux=True)(params, True)
    (fake_loss_check, extra_check), grads_check = jax.value_and_grad(fake_loss_fn, has_aux=True)(params, False)
    chex.assert_tree_all_finite(grads)
    chex.assert_trees_all_equal((fake_loss, grads), (fake_loss_check, grads_check))

    plt.plot(sample.positions[0, :, 0, 0], sample.positions[0, :, 0, 1], 'o')
    plt.title("Single sample first and second dim of x.")
    plt.show()

    # Test with-extra.
    sample, log_prob, extra = flow.sample_and_log_prob_with_extra_apply(params, jnp.zeros((n_nodes, _FEATURE_DIM)),
                                                                        subkey, (batch_size,))
    chex.assert_trees_all_equal(sample, sample_old)  # Haven't change source of randomness, so should match.
    log_prob_check, extra_chex = flow.log_prob_with_extra_apply(params, sample)
    chex.assert_trees_all_close(log_prob, log_prob_check, rtol=rtol)
    chex.assert_trees_all_close(extra.aux_info, extra_chex.aux_info, rtol=rtol)


    # Test log prob function is invariant.
    def invariant_log_prob(x):
        x = FullGraphSample(positions=x, features=jnp.zeros((n_nodes, 1, _FEATURE_DIM)))
        log_probs = flow.log_prob_apply(params, x)
        return log_probs

    test_fn_is_invariant(invariant_log_prob, subkey, event_shape=(n_nodes, n_aug+1, dim))


def test_target_reparam(dim: int = 3, n_aug: int = 3):
    """Visualise samples from the distribution, and check that it's log prob is invariant to
    translation and rotation."""
    if jnp.ones(()).dtype == jnp.float64:
        rtol = 1e-5
    else:
        rtol = 1e-4

    n_nodes = _N_NODES
    batch_size = 5
    key = jax.random.PRNGKey(0)
    base_aux_config = BaseConfig()
    target_aux_config = ConditionalAuxDistConfig(trainable_augmented_scale=True)


    config = FlowDistConfig(
        dim=dim, n_layers=_N_FLOW_LAYERS,
        nodes=_N_NODES,
        identity_init=_IDENTITY_INIT,
        type=_FLOW_TYPE,
        compile_n_unroll=2,
        nets_config=get_minimal_nets_config('egnn'),
        base=base_aux_config,
        target_aux_config=target_aux_config,
        n_aug=n_aug
    )

    flow = build_flow(config)


    # Init params.
    key, subkey = jax.random.split(key)
    dummy_samples = FullGraphSample(positions=jax.random.normal(subkey, (batch_size, n_nodes, dim)),
                                    features=jnp.zeros((batch_size, n_nodes, _FEATURE_DIM)))
    key, subkey = jax.random.split(key)
    params = flow.init(subkey, dummy_samples)

    def dummy_loss(params, key):
        samples = flow.aux_target_sample_n_apply(params, dummy_samples, key, 2)
        return jnp.sum(samples)

    print(jax.grad(dummy_loss)(params.aux_target, key))
    pass





if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)

    test_target_reparam(dim=3)


    test_distribution(dim=3)
    test_distribution(dim=2)
