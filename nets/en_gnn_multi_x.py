from typing import NamedTuple, Sequence, Callable

import chex
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from nets.en_gnn import EgnnTorsoConfig


def get_norms_sqrd(x):
    diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]
    diff_combos = diff_combos.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # prevents nan grads
    sq_norms = jnp.sum(diff_combos ** 2, axis=-1)
    return sq_norms


class EGCL_Multi(hk.Module):
    def __init__(self, name: str, n_heads, mlp_units: Sequence[int], identity_init_x: bool, residual: bool = True,
                 normalize_by_x_norm: bool = True, activation_fn: Callable = jax.nn.silu,
                 normalization_constant: float = 1.0, tanh: bool = False, phi_x_max: float = 10.0,
                 agg='mean', stop_gradient_for_norm: bool = False,
                 variance_scaling_init: float =0.001,
                 cross_attention: bool = False):


        super().__init__(name=name + "equivariant")
        self.phi_e = hk.nets.MLP(mlp_units, activation=activation_fn, activate_final=True)
        self.phi_inf = lambda x: jax.nn.sigmoid(hk.Linear(1)(x))

        self.phi_x = hk.Sequential([
            hk.nets.MLP(mlp_units, activate_final=True, activation=activation_fn),
             hk.Linear(n_heads, w_init=jnp.zeros, b_init=jnp.zeros) if identity_init_x else
             hk.Linear(n_heads, w_init=hk.initializers.VarianceScaling(variance_scaling_init, "fan_avg", "uniform")),
             lambda x: jax.nn.tanh(x)*phi_x_max if tanh else x])

        if cross_attention:
            self.phi_x_cross = hk.Sequential([
                hk.nets.MLP(mlp_units, activate_final=True, activation=activation_fn),
                 hk.Linear(n_heads*n_heads, w_init=jnp.zeros, b_init=jnp.zeros) if identity_init_x else
                 hk.Linear(n_heads*n_heads, w_init=hk.initializers.VarianceScaling(variance_scaling_init, "fan_avg",
                                                                                   "uniform")),
                 lambda x: jax.nn.tanh(x)*phi_x_max if tanh else x])

        self.phi_h_mlp = hk.nets.MLP(mlp_units, activate_final=True, activation=activation_fn)
        self.residual_h = residual
        self.normalize_by_x_norm = normalize_by_x_norm
        self.normalization_constant = normalization_constant
        self.agg = agg
        self.stop_gradient_for_norm = stop_gradient_for_norm
        self.n_heads = n_heads
        self.cross_attention = cross_attention

    def __call__(self, x, h):
        assert x.shape[-2] == self.n_heads
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
        phi_x_out = phi_x_out.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # explicitly set diagonal to 0


        if self.cross_attention:
            # Get phi_x_out_cross_attention
            phi_x_cross_out = self.phi_x_cross(m_i)
            phi_x_cross_out = jnp.reshape(phi_x_cross_out, (n_nodes, n_heads, n_heads))

        if self.normalize_by_x_norm:
            # Get norm in safe way that prevents nans.
            norm = jnp.sqrt(jnp.where(sq_norms_nodes == 0., 1., sq_norms_nodes)) + self.normalization_constant
            if self.stop_gradient_for_norm:
                norm = jax.lax.stop_gradient(norm)
            norm_diff_combo = diff_combos_nodes / norm[..., None]
            norm_diff_combo = norm_diff_combo.at[jnp.arange(n_nodes), jnp.arange(n_nodes)].set(0.0)
            equivariant_shift = jnp.einsum('ijhd,ijh->ihd', norm_diff_combo, phi_x_out)

            if self.cross_attention:
                # cross attention
                norm = jnp.sqrt(jnp.where(sq_norms_heads == 0., 1., sq_norms_heads)) + self.normalization_constant
                if self.stop_gradient_for_norm:
                    norm = jax.lax.stop_gradient(norm)
                norm_diff_combo = diff_combos_heads / norm[..., None]
                norm_diff_combo = norm_diff_combo.at[:, jnp.arange(n_heads), jnp.arange(n_heads)].set(0.0)
                equivariant_shift_cross_attention = jnp.einsum('nijd,nij->njd', norm_diff_combo, phi_x_cross_out)
                equivariant_shift = equivariant_shift + equivariant_shift_cross_attention
                chex.assert_equal_shape((equivariant_shift, equivariant_shift_cross_attention))
        else:
            equivariant_shift = jnp.einsum('ijhd,ijh->id', diff_combos_nodes, phi_x_out)
            if self.cross_attention:
                equivariant_shift_cross_attention = jnp.einsum('nijd,nij->njd', diff_combos_heads, phi_x_cross_out)
                equivariant_shift = equivariant_shift + equivariant_shift_cross_attention
                chex.assert_equal_shape((equivariant_shift, equivariant_shift_cross_attention))

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
        if config.torso_config.hk_layer_stack:
            self.egnn_layer_fn = lambda x, h: EGCL_Multi(config.name,
                                                 n_heads=config.n_equivariant_vectors_out,
                                                 mlp_units=config.torso_config.mlp_units,
                                                 identity_init_x=config.torso_config.identity_init_x,
                                                 normalize_by_x_norm=config.torso_config.normalize_by_norms,
                                                 residual=config.torso_config.h_residual,
                                                 activation_fn=config.torso_config.activation_fn,
                                                 tanh=config.torso_config.tanh,
                                                 phi_x_max=config.torso_config.phi_x_max,
                                                 agg=config.torso_config.agg,
                                                 stop_gradient_for_norm=config.torso_config.stop_gradient_for_norm,
                                                 variance_scaling_init=config.torso_config.variance_scaling_init,
                                                 normalization_constant=config.torso_config.normalization_constant,
                                                 cross_attention=config.torso_config.cross_attention
                                                 )(x, h)
        else:
            self.egnn_layers = [EGCL_Multi(config.name,
                                                 n_heads=config.n_equivariant_vectors_out,
                                                 mlp_units=config.torso_config.mlp_units,
                                                 identity_init_x=config.torso_config.identity_init_x,
                                                 normalize_by_x_norm=config.torso_config.normalize_by_norms,
                                                 residual=config.torso_config.h_residual,
                                                 activation_fn=config.torso_config.activation_fn,
                                                 tanh=config.torso_config.tanh,
                                                 phi_x_max=config.torso_config.phi_x_max,
                                                 agg=config.torso_config.agg,
                                                 stop_gradient_for_norm=config.torso_config.stop_gradient_for_norm,
                                                 variance_scaling_init=config.torso_config.variance_scaling_init,
                                                 normalization_constant=config.torso_config.normalization_constant
                               ) for _ in range(config.torso_config.n_layers)]

        self.h_final_layer = hk.Linear(config.n_invariant_feat_out, w_init=jnp.zeros, b_init=jnp.zeros) \
            if config.invariant_feat_zero_init else hk.Linear(config.n_invariant_feat_out)
        self.egnn_config = config.torso_config
        self.n_heads = config.n_equivariant_vectors_out


    def __call__(self, x):
        if len(x.shape) == 2:
            return self.forward_single(x)
        else:
            return hk.vmap(self.forward_single, split_rng=False)(x)
    
    def forward_single(self, x):
        """Compute forward pass of EGNN for a single x (no batch dimension)."""
        x_original = x
        # Perform forward pass of EGNN.

        # No node feature, so initialise them zeros.
        h = jnp.zeros((*x.shape[:-1], self.egnn_config.h_embedding_dim))

        # Project to number of heads
        x = jnp.repeat(x[:, None, ...], repeats=self.n_heads, axis=1)

        if self.egnn_config.hk_layer_stack:
            # Use layer_stack to speed up compilation time.
            stack = hk.experimental.layer_stack(self.egnn_config.n_layers, with_per_layer_inputs=False, name="EGCL_layer_stack",
                                                unroll=self.egnn_config.compile_n_unroll)
            x_out, h_egnn = stack(self.egnn_layer_fn)(x, h)
        else:
            x_out, h_egnn = x, h
            for layer in self.egnn_layers:
                x_out, h_egnn = layer(x_out, h_egnn)

        # Use h_egnn output from the EGNN as a feature for h-out.
        if self.egnn_config.h_linear_softmax:
            h_egnn = jax.nn.softmax(h_egnn, axis=-1)

        h_out = self.h_final_layer(h_egnn)
        return x_out - x_original[:, None], h_out
