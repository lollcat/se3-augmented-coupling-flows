import chex
import matplotlib.pyplot as plt
import jax
import haiku as hk

from flow.test_utils import test_fn_is_invariant, bijector_test
from flow.distribution import make_equivariant_augmented_flow_dist, FlowDistConfig, BaseConfig
from nets.en_gnn import HConfig, EgnnConfig, TransformerConfig


_N_FLOW_LAYERS = 4
_N_NODES = 16
_FLOW_TYPE = "nice"
_FAST_COMPILE_FLOW = False
_IDENTITY_INIT = False


def test_distribution(dim = 3):
    """Visualise samples from the distribution, and check that it's log prob is invariant to
    translation and rotation."""
    n_nodes = _N_NODES
    batch_size = 5
    key = jax.random.PRNGKey(0)
    base_config = BaseConfig(double_centrered_gaussian=False, global_centering=False,
                             trainable_augmented_scale=False)


    config = FlowDistConfig(
        dim=dim, n_layers=_N_FLOW_LAYERS, nodes=_N_NODES, identity_init=_IDENTITY_INIT,
        type=_FLOW_TYPE, fast_compile=_FAST_COMPILE_FLOW, compile_n_unroll=2,
        egnn_config=EgnnConfig(name="", mlp_units=(4,), n_layers=2, h_config=HConfig()._replace(
            linear_softmax=True, share_h=True)),
        transformer_config=TransformerConfig(mlp_units=(4,), n_layers=2),
        base_config=base_config,
        act_norm=True
    )


    @hk.transform
    def sample_and_log_prob_fn(sample_shape=()):
        distribution = make_equivariant_augmented_flow_dist(config)
        return distribution.sample_and_log_prob(seed=hk.next_rng_key(), sample_shape=sample_shape)


    @hk.without_apply_rng
    @hk.transform
    def log_prob_fn(x):
        distribution = make_equivariant_augmented_flow_dist(config)
        return distribution.log_prob(x)


    # Init params.
    key, subkey = jax.random.split(key)
    params = sample_and_log_prob_fn.init(subkey)

    key, subkey = jax.random.split(key)
    sample, log_prob = sample_and_log_prob_fn.apply(params, subkey, (batch_size,))
    chex.assert_tree_all_finite(log_prob)

    log_prob_check = log_prob_fn.apply(params, sample)

    chex.assert_trees_all_close(log_prob, log_prob_check)

    plt.plot(sample[0, :, 0], sample[0, :, 1], 'o')
    plt.title("Samples marginal in first and second dim")
    plt.show()

    # Test log prob function is invariant.
    key, subkey = jax.random.split(key)
    test_fn_is_invariant(lambda x: log_prob_fn.apply(params, x), subkey, dim=dim, n_nodes=n_nodes)


def test_flow(dim=3):
    n_nodes = _N_NODES

    base_config = BaseConfig(double_centrered_gaussian=False, global_centering=False,
                             trainable_augmented_scale=False)

    config = FlowDistConfig(
        dim=dim, n_layers=_N_FLOW_LAYERS, nodes=_N_NODES, identity_init=_IDENTITY_INIT,
        type=_FLOW_TYPE, fast_compile=_FAST_COMPILE_FLOW, compile_n_unroll=2,
        egnn_config=EgnnConfig(name="", mlp_units=(4,), n_layers=2, h_config=HConfig()._replace(
            linear_softmax=True, share_h=True)),
        transformer_config=TransformerConfig(mlp_units=(4,), n_layers=2),
        base_config=base_config
    )

    @hk.without_apply_rng
    @hk.transform
    def forward_and_log_det(x):
        distribution = make_equivariant_augmented_flow_dist(config)
        return distribution.bijector.forward_and_log_det(x)

    @hk.without_apply_rng
    @hk.transform
    def inverse_and_log_det(x):
        distribution = make_equivariant_augmented_flow_dist(config)
        return distribution.bijector.inverse_and_log_det(x)


    bijector_test(forward_and_log_det, inverse_and_log_det, dim=dim, n_nodes=n_nodes)



if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)

    test_flow(dim=2)
    test_distribution(dim=2)
    test_flow(dim=3)
    test_distribution(dim=3)

