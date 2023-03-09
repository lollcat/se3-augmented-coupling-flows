import haiku as hk
import jax
import chex
import jax.numpy as jnp

from flow.bijectors.bijector_proj_real_nvp import make_se_equivariant_split_coupling_with_projection
from flow.bijectors.bijector_nice import make_se_equivariant_nice
from flow.distrax_with_extra import ChainWithExtra, TransformedWithExtra, Extra
from flow.fast_hk_chain import Chain as FastChain
from flow.base_dist import CentreGravitryGaussianAndCondtionalGuassian
from flow.test_utils import get_minimal_nets_config


def test_dist_with_info(
        test_bijector_only: bool = False,
        test_fast_chain: bool = True,
        dim: int = 2, n_nodes = 5,
                        batch_size: int = 2,
                        gram_schmidt: bool = False,
                        global_frame: bool = False,
                        process_flow_params_jointly: bool = False,
                        type: str = 'egnn'):
    nets_config = get_minimal_nets_config(type)

    def make_dist(bijector_type='proj'):
        def bijector_fn():
            bijectors = []
            for swap in (True, False):
                if bijector_type == "proj":
                    bijector = make_se_equivariant_split_coupling_with_projection(
                        layer_number=0, dim=dim, swap=swap,
                        identity_init=False,
                        nets_config=nets_config,
                        global_frame=global_frame,
                        gram_schmidt=gram_schmidt,
                        process_flow_params_jointly=process_flow_params_jointly,
                    )
                else:
                    bijector = make_se_equivariant_nice(layer_number=0, dim=dim, swap=swap,
                        identity_init=False, nets_config=nets_config)
                bijectors.append(bijector)
            bijector_block = ChainWithExtra(bijectors)
            if test_bijector_only:
                return bijectors[0]
            else:
                return bijector_block
        if test_fast_chain:
            flow = FastChain(bijector_fn=bijector_fn, n_layers=4)
        else:
            flow = bijector_fn()
        base = CentreGravitryGaussianAndCondtionalGuassian(
            dim=dim, n_nodes=n_nodes, trainable_augmented_scale=True,
            augmented_scale_init=True
        )
        distribution = TransformedWithExtra(base, flow)
        return distribution

    @hk.without_apply_rng
    @hk.transform
    def log_prob_with_info_fn(x):
        dist = make_dist()
        log_prob, info = dist.log_prob_with_extra(x)
        return log_prob, info

    @hk.without_apply_rng
    @hk.transform
    def log_prob_fn(x):
        dist = make_dist()
        log_prob = dist.log_prob(x)
        return log_prob

    @hk.transform
    def sample_n_and_log_prob_with_info_fn(n: int):
        dist = make_dist()
        sample, log_prob, info = dist.sample_n_and_log_prob_with_extra(hk.next_rng_key(), n)
        return sample, log_prob, info

    @hk.transform
    def sample_n_and_log_prob_fn(n: int):
        dist = make_dist()
        sample, log_prob = dist._sample_n_and_log_prob(hk.next_rng_key(), n)
        return sample, log_prob

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x_dummy = jax.random.normal(key=key, shape=(batch_size, n_nodes, dim*2))
    params = log_prob_with_info_fn.init(key, x_dummy)
    params_check = log_prob_fn.init(key, x_dummy)
    chex.assert_trees_all_equal(params, params_check)

    # test grads
    def fake_loss_fn(params, use_extra=True, x=x_dummy):
        if use_extra:
            log_prob, extra = log_prob_with_info_fn.apply(params, x)
        else:
            log_prob = log_prob_fn.apply(params, x)
            extra = Extra()
        loss = jnp.mean(log_prob)  # + jnp.mean(extra.aux_loss)
        return loss, extra

    # Test log-probing
    log_prob, extra = log_prob_with_info_fn.apply(params, x_dummy)
    log_prob_check = log_prob_fn.apply(params, x_dummy)
    chex.assert_trees_all_close(log_prob_check, log_prob)


    (fake_loss, extra), grads = jax.value_and_grad(fake_loss_fn, has_aux=True)(params, True)
    (fake_loss_check, extra_check), grads_check = jax.value_and_grad(fake_loss_fn, has_aux=True)(params, False)
    chex.assert_tree_all_finite(grads)
    chex.assert_trees_all_equal((fake_loss, grads), (fake_loss_check, grads_check))


    # Test sample-n-and-log-prob
    sample, log_prob, info = sample_n_and_log_prob_with_info_fn.apply(params, subkey, 5)
    sample_check, log_prob_check = sample_n_and_log_prob_fn.apply(params, subkey, 5)
    chex.assert_trees_all_close((sample, log_prob), (sample_check, log_prob_check))



if __name__ == '__main__':
    test_dist_with_info()
