from typing import NamedTuple, Sequence, Callable, Optional

import chex
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

def get_norms_sqrd(x):
    diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]
    diff_combos = diff_combos.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # prevents nan grads
    sq_norms = jnp.sum(diff_combos ** 2, axis=-1)
    return sq_norms


class EGCL_Multi(hk.Module):
    def __init__(self, name: str,
                 mlp_units: Sequence[int],
                 residual: bool = True,
                 normalize_by_x_norm: bool = True, activation_fn: Callable = jax.nn.silu,
                 normalization_constant: float = 1.0,
                 agg='mean',
                 stop_gradient_for_norm: bool = False,
                 variance_scaling_init: float = 0.001):
        super().__init__(name=name + "equivariant")
        self.variance_scaling_init = variance_scaling_init
        self.phi_e = hk.nets.MLP(mlp_units, activation=activation_fn, activate_final=True)
        self.phi_inf = lambda x: jax.nn.sigmoid(hk.Linear(1)(x))

        self.phi_x = hk.nets.MLP(mlp_units, activate_final=True, activation=activation_fn)

        self.phi_h_mlp = hk.nets.MLP(mlp_units, activate_final=True, activation=activation_fn)
        self.residual_h = residual
        self.normalize_by_x_norm = normalize_by_x_norm
        self.normalization_constant = normalization_constant
        self.agg = agg
        self.stop_gradient_for_norm = stop_gradient_for_norm

    def __call__(self, x, h):
        if len(x.shape) == 3:
            return self.forward_single(x, h)
        else:
            chex.assert_rank(x, 4)
            return hk.vmap(self.forward_single, split_rng=False)(x, h)

    def forward_single(self, x, h):
        """Forward pass for a single graph (no batch dimension)."""
        chex.assert_rank(x, 3)
        chex.assert_rank(h, 2)
        n_nodes, n_heads = x.shape[0:2]
        avg_num_neighbours = n_nodes - 1

        # Equation 4(a)
        # Append norms between all heads to h.
        diff_combos_heads = jax.vmap(lambda x: x - x[:, None], in_axes=0)(x)  # [n_nodes, n_heads, n_heads, dim]

        diff_combos_heads = diff_combos_heads.at[:, jnp.arange(n_heads), jnp.arange(n_heads)].set(0.0)  # prevents nan grads
        sq_norms_heads = jnp.sum(diff_combos_heads**2, axis=-1)
        sq_norms_heads_flat = sq_norms_heads.reshape(sq_norms_heads.shape[0], np.prod(sq_norms_heads.shape[1:]))  # Flatten.
        h_concat = jnp.concatenate([h, sq_norms_heads_flat], axis=-1)

        # create h_combos.
        h_combos = jnp.concatenate([jnp.repeat(h_concat[None, ...], h.shape[0], axis=0),
                                    jnp.repeat(h_concat[:, None, ...], h.shape[0], axis=1)], axis=-1)

        diff_combos_nodes = jax.vmap(lambda x: x - x[:, None], in_axes=-2, out_axes=-2)(x)  # [n_nodes, n_nodes, n_heads, dim]
        diff_combos_nodes = diff_combos_nodes.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # prevents nan grads
        sq_norms_nodes = jnp.sum(diff_combos_nodes**2, axis=-1)


        m_ij = self.phi_e(jnp.concatenate([sq_norms_nodes, h_combos], axis=-1))
        m_ij = m_ij.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # explicitly set diagonal to 0

        # Equation 4(b)
        e = self.phi_inf(m_ij)
        e = e.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # explicitly set diagonal to 0
        m_i = jnp.einsum('ijd,ij->id', m_ij, jnp.squeeze(e, axis=-1))
        m_i = m_i / jnp.sqrt(avg_num_neighbours)

        # Get vectors.
        # Equation 5(a)
        phi_x_out = self.phi_x(m_ij)
        phi_x_out = hk.Linear(n_heads, w_init=hk.initializers.VarianceScaling(self.variance_scaling_init,
                                                                               "fan_avg", "uniform"))(phi_x_out)
        phi_x_out = phi_x_out.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # explicitly set diagonal to 0

        if self.normalize_by_x_norm:
            # Get norm in safe way that prevents nans.
            norm = jnp.sqrt(jnp.where(sq_norms_nodes == 0., 1., sq_norms_nodes)) + self.normalization_constant
            if self.stop_gradient_for_norm:
                norm = jax.lax.stop_gradient(norm)
            norm_diff_combo = diff_combos_nodes / norm[..., None]
            norm_diff_combo = norm_diff_combo.at[jnp.arange(n_nodes), jnp.arange(n_nodes)].set(0.0)
            equivariant_shift = jnp.einsum('ijhd,ijh->ihd', norm_diff_combo, phi_x_out)

        else:
            equivariant_shift = jnp.einsum('ijhd,ijh->id', diff_combos_nodes, phi_x_out)

        if self.agg == 'mean':
            equivariant_shift = equivariant_shift / avg_num_neighbours
        else:
            assert self.agg == 'sum'

        x_new = x + equivariant_shift

        # Get feature updates.
        # Equation 5(b)
        phi_h = hk.Sequential([self.phi_h_mlp, hk.Linear(h.shape[-1])])
        phi_h_in = jnp.concatenate([m_i, h], axis=-1)
        h_new = phi_h(phi_h_in)
        if self.residual_h:
            h_new = h + h_new

        return x_new, h_new


class EgnnTorsoConfig(NamedTuple):
    mlp_units: Sequence[int] = (3,)
    identity_init_x: bool = False
    zero_init_h: int = False
    h_embedding_dim: int = 3   # Dimension of h embedding in the EGCL.
    h_linear_softmax: bool = True    # Linear layer followed by softmax for improving stability.
    h_residual: bool = True
    n_layers: int = 3  # Number of EGCL layers.
    normalize_by_norms: bool = True
    activation_fn: Callable = jax.nn.silu
    agg: str = 'mean'
    stop_gradient_for_norm: bool = False
    variance_scaling_init: float = 0.001
    normalization_constant: float = 1.0


class MultiEgnnConfig(NamedTuple):
    """Config of the EGNN."""
    name: str
    n_invariant_feat_out: int
    n_equivariant_vectors_out: int
    torso_config: EgnnTorsoConfig
    invariant_feat_zero_init: bool = True


class multi_se_equivariant_net(hk.Module):
    def __init__(self, config: MultiEgnnConfig):
        super().__init__(name=config.name + "_multi_x_egnn")
        self.egnn_layers = [EGCL_Multi(config.name + str(i),
                                             mlp_units=config.torso_config.mlp_units,
                                             normalize_by_x_norm=config.torso_config.normalize_by_norms,
                                             residual=config.torso_config.h_residual,
                                             activation_fn=config.torso_config.activation_fn,
                                             agg=config.torso_config.agg,
                                             stop_gradient_for_norm=config.torso_config.stop_gradient_for_norm,
                                             variance_scaling_init=config.torso_config.variance_scaling_init,
                                             normalization_constant=config.torso_config.normalization_constant
                           ) for i in range(config.torso_config.n_layers)]
        self.egnn_config = config.torso_config
        self.n_heads = config.n_equivariant_vectors_out
        self.config = config


    def __call__(self, x: chex.Array, h: chex.Array):
        if len(x.shape) == 3:
            return self.forward_single(x, h)
        else:
            return hk.vmap(self.forward_single, split_rng=False)(x, h)


    def forward_single(self, x: chex.Array, h: chex.Array):
        """Compute forward pass of EGNN for a single x (no batch dimension)."""
        assert x.shape[0:2] == h.shape[0:2]
        n_nodes, multiplicity_in = x.shape[:2]

        # Perform forward pass of EGNN.

        # Project to number of heads
        x = jnp.repeat(x, repeats=self.n_heads, axis=1)
        x_original = x
        h = h.reshape(h.shape[0], np.prod(h.shape[1:]))  # flatten along last 2 axes.

        x_out, h_egnn = x, h
        for layer in self.egnn_layers:
            x_out, h_egnn = layer(x_out, h_egnn)

        # Use h_egnn output from the EGNN as a feature for h-out.
        if self.egnn_config.h_linear_softmax:
            h_egnn = jax.nn.softmax(h_egnn, axis=-1)


        h_out = hk.Linear(self.config.n_invariant_feat_out*multiplicity_in, w_init=jnp.zeros, b_init=jnp.zeros) \
            if self.config.invariant_feat_zero_init else hk.Linear(self.config.n_invariant_feat_out*multiplicity_in)(h_egnn)
        x_out = x_out - x_original
        x_out = x_out.reshape((n_nodes, multiplicity_in, self.n_heads, x_out.shape[-1]))
        h_out = h_out.reshape((n_nodes, multiplicity_in, self.config.n_invariant_feat_out))
        return x_out, h_out
