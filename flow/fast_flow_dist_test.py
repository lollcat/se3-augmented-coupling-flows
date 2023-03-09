import chex
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from flow.test_utils import test_fn_is_invariant
from flow.test_utils import get_minimal_nets_config
from flow.build_flow import build_flow, BaseConfig, FlowDistConfig
from flow.distrax_with_extra import Extra


_N_FLOW_LAYERS = 4
_N_NODES = 16
_FLOW_TYPE = "proj"
_FAST_COMPILE_FLOW = False
_IDENTITY_INIT = False


def test_distribution(dim = 3):
    """Visualise samples from the distribution, and check that it's log prob is invariant to
    translation and rotation."""
    if jnp.ones(()).dtype == jnp.float64:
        rtol = 1e-5
    else:
        rtol = 1e-4

    n_nodes = _N_NODES
    batch_size = 5
    key = jax.random.PRNGKey(0)
    base_config = BaseConfig(double_centered_gaussian=False, global_centering=False,
                             trainable_augmented_scale=False)


    config = FlowDistConfig(
        dim=dim, n_layers=_N_FLOW_LAYERS, nodes=_N_NODES, identity_init=_IDENTITY_INIT,
        type=_FLOW_TYPE, fast_compile=_FAST_COMPILE_FLOW, compile_n_unroll=2,
        nets_config=get_minimal_nets_config('egnn'),
        base_config=base_config,
        act_norm=False
    )

    flow = build_flow(config)


    # Init params.
    dummy_samples = jnp.zeros((batch_size, n_nodes, dim*2))
    key, subkey = jax.random.split(key)
    params = flow.init(subkey, dummy_samples)
    params_check = flow.init(subkey, dummy_samples[0])
    chex.assert_trees_all_equal(params, params_check)  # Check that params aren't effected by batch-ing.

    key, subkey = jax.random.split(key)
    sample, log_prob = flow.sample_and_log_prob_apply(params, subkey, (batch_size,))
    sample_old = sample  # save for later in this test.
    chex.assert_tree_all_finite(log_prob)
    chex.assert_shape(log_prob, (batch_size,))
    chex.assert_shape(sample, (batch_size, n_nodes, dim*2))

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

    plt.plot(sample[0, :, 0], sample[0, :, 1], 'o')
    plt.title("Samples marginal in first and second dim")
    plt.show()

    # Test log prob function is invariant.
    test_fn_is_invariant(lambda x: flow.log_prob_apply(params, x), subkey, dim=dim, n_nodes=n_nodes)


    # Test with-extra.
    sample, log_prob, extra = flow.sample_and_log_prob_with_extra_apply(params, subkey, (batch_size,))
    chex.assert_trees_all_equal(sample, sample_old)  # Haven't change source of randomness, so should match.
    log_prob_check, extra_chex = flow.log_prob_with_extra_apply(params, sample)
    chex.assert_trees_all_close(log_prob, log_prob_check, rtol=rtol)
    chex.assert_trees_all_close(extra.aux_info, extra_chex.aux_info, rtol=rtol)



if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)

    test_distribution(dim=2)
    test_distribution(dim=3)
