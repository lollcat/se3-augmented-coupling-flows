import chex
import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial

from utils.nets import LayerNormMLP

# Typically need one of these to be True to stack lots of layers.
_LAYER_NORM = True
_EQUI_NORM = True


class se_equivariant_net(hk.Module):
    def __init__(self, name, mlp_units, zero_init, layer_norm: bool = _LAYER_NORM, equi_norm: bool = _EQUI_NORM):
        super().__init__(name=name + "equivariant")
        self.mlp_units = mlp_units
        self.zero_init = zero_init
        self.layer_norm = layer_norm
        self.equi_norm = equi_norm


    def __call__(self, x):
        if len(x.shape) == 2:
            return self.forward_single(x)
        else:
            return jax.vmap(self.forward_single)(x)
    
    def forward_single(self, x):
        mlp = LayerNormMLP if self.layer_norm else partial(hk.nets.MLP, activation=jax.nn.elu)
        chex.assert_rank(x, 2)
    
        diff_combos = x - x[:, None]   # [n_nodes, n_nodes, dim]
        diff_combos = diff_combos.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # prevents nan grads
        norms = jnp.linalg.norm(diff_combos, ord=2, axis=-1)

        net = hk.Sequential([mlp(self.mlp_units, activate_final=True),
                             hk.Linear(1, w_init=jnp.zeros, b_init=jnp.zeros) if self.zero_init else
                             hk.Linear(1)])
        m = jnp.squeeze(net(norms[..., None]), axis=-1)
        m = m.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # explicitly set diagonal to 0

        if not self.equi_norm:
            equivariant_shift = jnp.einsum('ijd,ij->id', diff_combos, m)
        else:
            equivariant_shift = jnp.einsum('ijd,ij->id', diff_combos / (norms + 1)[..., None], m)
        return x + equivariant_shift



class se_invariant_net(hk.Module):
    def __init__(self, name, n_vals, mlp_units, zero_init, layer_norm: bool = _LAYER_NORM, equi_norm: bool = _EQUI_NORM):
        super().__init__(name=name + "invariant_net")
        self.n_vals = n_vals
        self.mlp_units = mlp_units
        self.zero_init = zero_init
        self.layer_norm = layer_norm
        self.equi_norm = equi_norm

    def __call__(self, x):
        if len(x.shape) == 2:
            return self.forward_single(x)
        else:
            return jax.vmap(self.forward_single)(x)

    def forward_single(self, x):
        chex.assert_rank(x, 2)
        mlp = LayerNormMLP if self.layer_norm else hk.nets.MLP

        diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]
        diff_combos = diff_combos.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # prevents nan grads
        norms = jnp.linalg.norm(diff_combos, ord=2, axis=-1)

        net = hk.Sequential([mlp(self.mlp_units, activate_final=True),
                             hk.Linear(self.n_vals, w_init=jnp.zeros, b_init=jnp.zeros) if self.zero_init else
                             hk.Linear(self.n_vals)])
        net_out = net(norms[..., None])
        net_out = net_out.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # explicitly set diagonal to 0
        return jnp.sum(net_out, axis=-2)


if __name__ == '__main__':
    from test_utils import test_fn_is_invariant, test_fn_is_equivariant

    name = "layer1"
    mlp_units = (5,)
    zero_init = False

    key = jax.random.PRNGKey(0)
    equivariant_fn_hk = hk.without_apply_rng(hk.transform(lambda x: se_equivariant_net(name, mlp_units, zero_init)(x)))
    invariant_fn_hk = hk.without_apply_rng(hk.transform(lambda x: se_invariant_net(name, n_vals=2,
                                                                                   mlp_units=mlp_units,
                                                                                   zero_init=zero_init)(x)))

    x = jnp.zeros((5, 4, 2))
    key, subkey = jax.random.split(key)
    params_eq = equivariant_fn_hk.init(subkey, x)
    key, subkey = jax.random.split(key)
    params_in = invariant_fn_hk.init(subkey, x, n_vals=2, zero_init=False)

    key, subkey = jax.random.split(key)
    test_fn_is_equivariant(lambda x: equivariant_fn_hk.apply(params_eq, x), subkey)
    key, subkey = jax.random.split(key)
    test_fn_is_invariant(lambda x: invariant_fn_hk.apply(params_in, x), subkey)
