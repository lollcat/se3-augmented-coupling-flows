import chex
import jax.numpy as jnp
import distrax
import haiku as hk
import jax

from flow.aug_flow_dist import FullGraphSample


def get_broadcasted_loc_and_scalediag(x: chex.Array, n_aux: int, global_centering: bool, scale: chex.Array):
    chex.assert_shape(scale, (n_aux,))
    chex.assert_rank(x, 2)

    scale_diag = jnp.zeros((x.shape[0], n_aux, x.shape[-1])) + scale[None, :, None]
    loc = jnp.zeros((x.shape[0], n_aux, x.shape[-1]))
    if global_centering:
        loc = loc + jnp.mean(x, axis=-2, keepdims=True)[:, None, :]
    else:
        loc = loc + x[:, None, :]

    chex.assert_equal_shape((loc, scale_diag))
    return loc, scale_diag

def get_conditional_gaussian_augmented_dist(x: chex.Array, n_aux: int, global_centering: bool, scale: chex.Array):
    if len(x.shape) == 2:
        loc, scale_diag = get_broadcasted_loc_and_scalediag(x, n_aux, global_centering, scale)
    else:
        assert len(x.shape) == 3
        loc, scale_diag = jax.vmap(get_broadcasted_loc_and_scalediag, in_axes=(0, None, None, None))(
            x, n_aux, global_centering, scale)

    dist = distrax.Independent(distrax.MultivariateNormalDiag(loc=loc,
                                                              scale_diag=scale_diag), reinterpreted_batch_ndims=2)
    return dist


def build_aux_dist(n_aug: int,
                   name: str,
                   global_centering: bool = False,
                   augmented_scale_init: float = 1.0,
                   trainable_scale: bool = True):
    def make_aux_target(sample: FullGraphSample):
        if trainable_scale:
            log_scale = hk.get_parameter(name=name + '_augmented_scale_logit', shape=(n_aug,),
                                         init=hk.initializers.Constant(jnp.log(augmented_scale_init)))
            scale = jnp.exp(log_scale)
        else:
            scale = jnp.ones(n_aug) * augmented_scale_init
        dist = get_conditional_gaussian_augmented_dist(sample.positions,
                                                       n_aux=n_aug,
                                                       global_centering=global_centering,
                                                       scale=scale)
        return dist
    return make_aux_target
