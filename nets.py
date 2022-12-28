import chex
import jax
import jax.numpy as jnp
import haiku as hk


def equivariant_fn(x, mlp_units=(5,), zero_init: bool = True):
    chex.assert_rank(x, 2)
    diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]
    norms = jnp.linalg.norm(diff_combos, ord=2, axis=-1)
    net = hk.Sequential([hk.nets.MLP((mlp_units), activation=jax.nn.elu),
                         hk.Linear(1, w_init=jnp.zeros, b_init=jnp.zeros) if zero_init else
                         hk.Linear(1)])
    m = jnp.squeeze(net(norms[..., None]), axis=-1)
    return x + jnp.einsum('ijd,ij->id', diff_combos / (norms + 1)[..., None], m)

def invariant_fn(x, n_vals, zero_init: bool = True):
    chex.assert_rank(x, 2)
    equivariant_x = jnp.stack([equivariant_fn(x, zero_init=zero_init) for _ in range(n_vals)], axis=-1)
    return jnp.linalg.norm(x[..., None] - equivariant_x, ord=2, axis=-2)


if __name__ == '__main__':
    from test_utils import test_fn_is_invariant, test_fn_is_equivariant

    key = jax.random.PRNGKey(0)
    equivariant_fn_hk = hk.without_apply_rng(hk.transform(equivariant_fn))
    invariant_fn_hk = hk.without_apply_rng(hk.transform(invariant_fn))

    x = jnp.zeros((4, 2))
    key, subkey = jax.random.split(key)
    params_eq = equivariant_fn_hk.init(subkey, x, zero_init=False)
    key, subkey = jax.random.split(key)
    params_in = invariant_fn_hk.init(subkey, x, n_vals=2, zero_init=False)

    key, subkey = jax.random.split(key)
    test_fn_is_equivariant(lambda x: equivariant_fn_hk.apply(params_eq, x, zero_init=False), subkey)
    key, subkey = jax.random.split(key)
    test_fn_is_invariant(lambda x: invariant_fn_hk.apply(params_in, x, n_vals=2, zero_init=False), subkey)
