import chex
import jax
import jax.numpy as jnp
import haiku as hk
from utils.nets import LayerNormMLP


_LAYER_NORM = True
_EQUI_NORM = False


def _se_equivariant_fn(x, mlp_units, zero_init, layer_norm: bool = _LAYER_NORM, equi_norm: bool = _EQUI_NORM
                       ):
    mlp = LayerNormMLP if layer_norm else hk.nets.MLP
    chex.assert_rank(x, 2)

    diff_combos = x - x[:, None]   # [n_nodes, n_nodes, dim]

    # Need to add 1e-10 to prevent nan grads, but we overwrite this anyway.
    norms = jnp.linalg.norm(diff_combos + 1e-10, ord=2, axis=-1)
    norms = norms * (jnp.ones_like(norms) - jnp.eye(norms.shape[0]))
    net = hk.Sequential([mlp(mlp_units, activate_final=True),
                         hk.Linear(1, w_init=jnp.zeros, b_init=jnp.zeros) if zero_init else
                         hk.Linear(1)])
    m = jnp.squeeze(net(norms[..., None]), axis=-1)
    if not equi_norm:
        equivariant_shift = jnp.einsum('ijd,ij->id', diff_combos, m)
    else:
        equivariant_shift = jnp.einsum('ijd,ij->id', diff_combos / (norms + 1)[..., None], m)
    return x + equivariant_shift


def se_equivariant_fn(x, mlp_units=(5, 5), zero_init: bool = False):
    if len(x.shape) == 2:
        return _se_equivariant_fn(x, mlp_units, zero_init)
    else:
        return jax.vmap(_se_equivariant_fn, in_axes=(0, None, None))(x, mlp_units, zero_init)


def _se_invariant_fn(x, n_vals, mlp_units, zero_init, layer_norm: bool = _LAYER_NORM):
    chex.assert_rank(x, 2)
    mlp = LayerNormMLP if layer_norm else hk.nets.MLP

    diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]

    # Need to add 1e-10 to prevent nan grads, but we overwrite this anyway.
    norms = jnp.linalg.norm(diff_combos + 1e-10, ord=2, axis=-1)
    norms = norms * (jnp.ones_like(norms) - jnp.eye(norms.shape[0]))

    net = hk.Sequential([mlp(mlp_units, activate_final=True),
                         hk.Linear(n_vals, w_init=jnp.zeros, b_init=jnp.zeros) if zero_init else
                         hk.Linear(n_vals)])
    net_out = net(norms[..., None])
    return jnp.sum(net_out, axis=-2)


def se_invariant_fn(x, n_vals, zero_init: bool = False, mlp_units=(5, 5)):
    if len(x.shape) == 2:
        return _se_invariant_fn(x, n_vals, mlp_units, zero_init)
    else:
        return jax.vmap(_se_invariant_fn, in_axes=(0, None, None, None))(x, n_vals, mlp_units, zero_init)


if __name__ == '__main__':
    from test_utils import test_fn_is_invariant, test_fn_is_equivariant

    key = jax.random.PRNGKey(0)
    equivariant_fn_hk = hk.without_apply_rng(hk.transform(se_equivariant_fn))
    invariant_fn_hk = hk.without_apply_rng(hk.transform(se_invariant_fn))

    x = jnp.zeros((4, 2))
    key, subkey = jax.random.split(key)
    params_eq = equivariant_fn_hk.init(subkey, x, zero_init=False)
    key, subkey = jax.random.split(key)
    params_in = invariant_fn_hk.init(subkey, x, n_vals=2, zero_init=False)

    key, subkey = jax.random.split(key)
    test_fn_is_equivariant(lambda x: equivariant_fn_hk.apply(params_eq, x, zero_init=False), subkey)
    key, subkey = jax.random.split(key)
    test_fn_is_invariant(lambda x: invariant_fn_hk.apply(params_in, x, n_vals=2, zero_init=False), subkey)
