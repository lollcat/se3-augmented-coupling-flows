from typing import NamedTuple, Sequence, Callable

import chex
import jax
import jax.numpy as jnp
import haiku as hk

# TODO: need to be careful of mean if number of nodes is varying? Could normalisation a parameter function of
#  the number of nodes?


def get_norms_sqrd(x):
    diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]
    diff_combos = diff_combos.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # prevents nan grads
    sq_norms = jnp.sum(diff_combos ** 2, axis=-1)
    return sq_norms


class EGCL(hk.Module):
    """Single layer of Equivariant Graph Convolutional layer.
    Following notation of E(n) normalizing flows paper (section 2.1): https://arxiv.org/pdf/2105.09016.pdf.
    See also https://github.com/ehoogeboom/e3_diffusion_for_molecules/egnn/egnn_new.py"""
    def __init__(self, name: str, mlp_units: Sequence[int], identity_init_x: bool, residual: bool = True,
                 normalize_by_x_norm: bool = True, activation_fn: Callable = jax.nn.silu,
                 normalization_constant: float = 1.0, tanh: bool = False, phi_x_max: float = 10.0,
                 agg='mean', stop_gradient_for_norm: bool = False,
                 variance_scaling_init: float =0.001):
        """

        Args:
            name: Layer name.
            mlp_units: MLP units for phi_e, phi_x and phi_h.
            identity_init_x: Whether to initialise the transform of x to the identity function.
            residual: Whether to include a residual connection for h.
            normalize_by_x_norm: See divisor in Equation 12 of https://arxiv.org/pdf/2203.17003.pdf.
        """
        super().__init__(name=name + "equivariant")
        self.phi_e = hk.nets.MLP(mlp_units, activation=activation_fn, activate_final=True)
        self.phi_inf = lambda x: jax.nn.sigmoid(hk.Linear(1)(x))


        self.phi_x = hk.Sequential([
            hk.nets.MLP(mlp_units, activate_final=True, activation=activation_fn),
             hk.Linear(1, w_init=jnp.zeros, b_init=jnp.zeros) if identity_init_x else
             hk.Linear(1, w_init=hk.initializers.VarianceScaling(variance_scaling_init, "fan_avg", "uniform")),
             lambda x: jax.nn.tanh(x)*phi_x_max if tanh else x])

        self.phi_h_mlp = hk.nets.MLP(mlp_units, activate_final=False, activation=activation_fn)
        self.residual_h = residual
        self.normalize_by_x_norm = normalize_by_x_norm
        self.normalization_constant = normalization_constant
        self.agg = agg
        self.stop_gradient_for_norm = stop_gradient_for_norm

    def __call__(self, x, h):
        if len(x.shape) == 2:
            return self.forward_single(x, h)
        else:
            chex.assert_rank(x, 3)
            return hk.vmap(self.forward_single, split_rng=False)(x, h)

    def forward_single(self, x, h):
        """Forward pass for a single graph (no batch dimension)."""
        chex.assert_rank(x, 2)
        chex.assert_rank(h, 2)
        n_nodes = x.shape[0]

        # Equation 4(a)
        h_combos = jnp.concatenate([jnp.repeat(h[None, ...], h.shape[0], axis=0),
                                    jnp.repeat(h[:, None, ...], h.shape[0], axis=1)], axis=-1)

        diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]
        diff_combos = diff_combos.at[jnp.arange(n_nodes), jnp.arange(n_nodes)].set(0.0)  # prevents nan grads
        sq_norms = jnp.sum(diff_combos**2, axis=-1)

        m_ij = self.phi_e(jnp.concatenate([sq_norms[..., None], h_combos], axis=-1))
        m_ij = m_ij.at[jnp.arange(n_nodes), jnp.arange(n_nodes)].set(0.0)  # explicitly set diagonal to 0


        # Get vector updates.
        # Equation 5(a)
        phi_x_out = jnp.squeeze(self.phi_x(m_ij), axis=-1)
        phi_x_out = phi_x_out.at[jnp.arange(n_nodes), jnp.arange(n_nodes)].set(0.0)  # explicitly set diagonal to 0


        if self.normalize_by_x_norm:
            # Get norm in safe way that prevents nans.
            norm = jnp.sqrt(jnp.where(sq_norms == 0., 1., sq_norms)) + self.normalization_constant
            if self.stop_gradient_for_norm:
                norm = jax.lax.stop_gradient(norm)
            norm_diff_combo = diff_combos / norm[..., None]
            norm_diff_combo = norm_diff_combo.at[jnp.arange(n_nodes), jnp.arange(n_nodes)].set(0.0)
            equivariant_shift = jnp.einsum('ijd,ij->id', norm_diff_combo, phi_x_out)
        else:
            equivariant_shift = jnp.einsum('ijd,ij->id', diff_combos, phi_x_out)


        if self.agg == 'mean':
            equivariant_shift = equivariant_shift / (n_nodes - 1)
        else:
            assert self.agg == 'sum'

        x_new = x + equivariant_shift

        # Get feature updates.
        # Equation 4(b)
        e = self.phi_inf(m_ij)
        e = e.at[jnp.arange(n_nodes), jnp.arange(n_nodes)].set(0.0)  # explicitly set diagonal to 0
        m_i = jnp.einsum('ijd,ij->id', m_ij, jnp.squeeze(e, axis=-1))
        if self.agg == 'mean':
            m_i = m_i / (n_nodes - 1)

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
    hk_layer_stack: bool = True  # To lower compile time.
    compile_n_unroll: int = 1
    normalize_by_norms: bool = True
    activation_fn: Callable = jax.nn.silu
    tanh: bool = False
    phi_x_max: float = 2.0
    agg: str = 'mean'
    stop_gradient_for_norm: bool = False
    variance_scaling_init: float = 0.001
    normalization_constant: float = 1.0
    cross_attention: bool = False  # Only for if multiple x are output.

class EgnnConfig(NamedTuple):
    """Config of the EGNN."""
    name: str
    torso_config: EgnnTorsoConfig
    n_invariant_feat_out: int
    invariant_feat_zero_init: bool = True


class EnGNN(hk.Module):
    def __init__(self, config: EgnnConfig):
        super().__init__(name=config.name + "_egnn")
        if config.torso_config.hk_layer_stack:
            self.egnn_layer_fn = lambda x, h: EGCL(config.name,
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
                                                   )(x, h)
        else:
            self.egnn_layers = [EGCL(config.name,
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
        self.config = config

    def __call__(self, x):
        if len(x.shape) == 2:
            return self.forward_single(x)
        else:
            return hk.vmap(self.forward_single, split_rng=False)(x)
    
    def forward_single(self, x):
        """Compute forward pass of EGNN for a single x (no batch dimension)."""
        x_in = x
        # Perform forward pass of EGNN.

        # No node feature, so initialise them zeros.
        h = jnp.zeros((*x.shape[:-1], self.config.torso_config.h_embedding_dim))

        if self.config.torso_config.hk_layer_stack:
            # Use layer_stack to speed up compilation time.
            stack = hk.experimental.layer_stack(self.config.torso_config.n_layers, with_per_layer_inputs=False,
                                                name="EGCL_layer_stack",
                                                unroll=self.config.torso_config.compile_n_unroll)
            x_out, h_egnn = stack(self.egnn_layer_fn)(x, h)
        else:
            x_out, h_egnn = x, h
            for layer in self.egnn_layers:
                x_out, h_egnn = layer(x_out, h_egnn)

        # Use h_egnn output from the EGNN as a feature for h-out.
        if self.config.torso_config.h_linear_softmax:
            h_egnn = jax.nn.softmax(h_egnn, axis=-1)

        h_out = self.h_final_layer(h_egnn)
        return x_out - x_in, h_out
