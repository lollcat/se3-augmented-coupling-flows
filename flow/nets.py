import chex
import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial

from utils.nets import LayerNormMLP


# Typically need one of these to be True (esp layer norm) to stack lots of layers.
_LAYER_NORM = True
_EQUI_NORM = True
_EGNN_N_LAYERS = 2


# TODO: Can use h for invariant features. Can clump h and x for scale and shift into a single NN architecture.
# TODO: is mean instead of sum okay? Could make it a parameter fn of the number of nodes.

class EGNN(hk.Module):
    # Following notation of E(n) normalizing flows paper: https://arxiv.org/pdf/2105.09016.pdf
    def __init__(self, name, mlp_units, zero_init, layer_norm: bool, equi_norm: bool):
        super().__init__(name=name + "equivariant")
        mlp = LayerNormMLP if layer_norm else partial(hk.nets.MLP, activation=jax.nn.elu)
        self.phi_e = mlp(mlp_units)
        self.phi_inf = lambda x: jax.nn.sigmoid(hk.Linear(1)(x))
        self.phi_x = hk.Sequential([mlp(mlp_units, activate_final=True),
                             hk.Linear(1, w_init=jnp.zeros, b_init=jnp.zeros) if zero_init else
                             hk.Linear(1)])
        self.equi_norm = equi_norm
        self.phi_h = mlp(mlp_units)

    def __call__(self, x, h = None):
        if h is None:
            h_embedding_dim = 3
            h = jnp.ones((*x.shape[:-1], h_embedding_dim))
        if len(x.shape) == 2:
            return self.forward_single(x, h)
        else:
            return hk.vmap(self.forward_single, split_rng=False)(x, h)

    def forward_single(self, x, h):
        chex.assert_rank(x, 2)
        chex.assert_rank(h, 2)
        h_dim = h.shape[-1]


        h_combos = jnp.concatenate([jnp.repeat(h[None, ...], h.shape[0], axis=0),
                                    jnp.repeat(h[:, None, ...], h.shape[0], axis=1)], axis=-1)

        diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]
        diff_combos = diff_combos.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # prevents nan grads
        norms = jnp.linalg.norm(diff_combos, ord=2, axis=-1)

        m_ij = self.phi_e(jnp.concatenate([norms[..., None], h_combos], axis=-1))
        m_ij = m_ij.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # explicitly set diagonal to 0

        e = self.phi_inf(m_ij)
        e = e.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # explicitly set diagonal to 0
        m_i = jnp.einsum('ijd,ij->id', m_ij, jnp.squeeze(e, axis=-1))

        phi_x_out = jnp.squeeze(self.phi_x(m_ij), axis=-1)
        phi_x_out = phi_x_out.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # explicitly set diagonal to 0

        # Get updated x.
        C = x.shape[0] - 1
        if not self.equi_norm:
            equivariant_shift = jnp.einsum('ijd,ij->id', diff_combos, phi_x_out) / C
        else:
            equivariant_shift = jnp.einsum('ijd,ij->id', diff_combos / (norms + 1)[..., None], phi_x_out) / C

        # Get updated h
        h_new = self.phi_h(jnp.concatenate([m_i, h], axis=-1))
        h_new = hk.Linear(h_dim)(h_new)

        return x + equivariant_shift, h_new



class se_equivariant_net(hk.Module):
    def __init__(self, name, mlp_units, zero_init, n_layers=_EGNN_N_LAYERS, layer_norm: bool = _LAYER_NORM, equi_norm: bool = _EQUI_NORM):
        super().__init__(name=name + "equivariant")
        self.egnn_layers = [EGNN(name + f"_{i}", mlp_units, zero_init, layer_norm, equi_norm) for i in range(n_layers)]


    def __call__(self, x):
        if len(x.shape) == 2:
            return self.forward_single(x)
        else:
            return hk.vmap(self.forward_single, split_rng=False)(x)
    
    def forward_single(self, x):
        x, h = self.egnn_layers[0](x)
        for layer in self.egnn_layers[1:]:
            x, h = layer(x, h)
        return x



class se_invariant_net(hk.Module):
    def __init__(self, name, n_vals, mlp_units, zero_init, n_layers=_EGNN_N_LAYERS, layer_norm: bool = _LAYER_NORM, equi_norm: bool = _EQUI_NORM):
        super().__init__(name=name + "invariant_net")
        self.egnn_layers = [EGNN(name + f"_{i}", mlp_units, zero_init, layer_norm, equi_norm) for i in range(n_layers)]
        self.final_mlp = hk.Sequential([hk.nets.MLP(mlp_units, activate_final=True),
                             hk.Linear(n_vals, w_init=jnp.zeros, b_init=jnp.zeros) if zero_init else
                             hk.Linear(n_vals)])

    def __call__(self, x):
        if len(x.shape) == 2:
            return self.forward_single(x)
        else:
            return hk.vmap(self.forward_single, split_rng=False)(x)

    def forward_single(self, x):
        chex.assert_rank(x, 2)
        x, h = self.egnn_layers[0](x)
        for layer in self.egnn_layers[1:]:
            x, h = layer(x, h)

        diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]
        diff_combos = diff_combos.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # prevents nan grads
        norms = jnp.linalg.norm(diff_combos, ord=2, axis=-1)

        net_out = self.final_mlp(norms[..., None])
        net_out = net_out.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # explicitly set diagonal to 0
        return jnp.mean(net_out, axis=-2)


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
