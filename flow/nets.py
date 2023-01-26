from typing import NamedTuple, Sequence

import chex
import jax
import jax.numpy as jnp
import haiku as hk


# TODO: need to be careful of mean if number of nodes is varying? Could normalisation a parameter function of
#  the number of nodes?


class EGCL(hk.Module):
    """Single layer of Equivariant Graph Convolutional layer.
    Following notation of E(n) normalizing flows paper (section 2.1): https://arxiv.org/pdf/2105.09016.pdf"""
    def __init__(self, name: str, mlp_units: Sequence[int], identity_init_x: bool, recurrent_h: bool = True):
        """

        Args:
            name: Layer name.
            mlp_units: MLP units for phi_e, phi_x and phi_h.
            identity_init_x: Whether to initialise the transform of x to the identity function.
            recurrent_h: Whether to include a residual connection for h.
        """
        super().__init__(name=name + "equivariant")
        self.phi_e = hk.nets.MLP(mlp_units)
        self.phi_inf = lambda x: jax.nn.sigmoid(hk.Linear(1)(x))
        self.phi_x = hk.Sequential([hk.nets.MLP(mlp_units, activate_final=True),
                             hk.Linear(1, w_init=jnp.zeros, b_init=jnp.zeros) if identity_init_x else
                             hk.Linear(1)])
        self.phi_h_mlp = hk.nets.MLP(mlp_units, activate_final=False)
        self.recurrent_h = recurrent_h

    def __call__(self, x, h):
        if len(x.shape) == 2:
            return self.forward_single(x, h)
        else:
            return hk.vmap(self.forward_single, split_rng=False)(x, h)

    def forward_single(self, x, h):
        """Forward pass for a single graph (no batch dimension)."""
        chex.assert_rank(x, 2)
        chex.assert_rank(h, 2)

        # Equation 4(a)
        h_combos = jnp.concatenate([jnp.repeat(h[None, ...], h.shape[0], axis=0),
                                    jnp.repeat(h[:, None, ...], h.shape[0], axis=1)], axis=-1)

        diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]
        diff_combos = diff_combos.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # prevents nan grads
        sq_norms = jnp.sum(diff_combos**2, axis=-1)

        m_ij = self.phi_e(jnp.concatenate([sq_norms[..., None], h_combos], axis=-1))
        m_ij = m_ij.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # explicitly set diagonal to 0

        # Equation 4(b)
        e = self.phi_inf(m_ij)
        e = e.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # explicitly set diagonal to 0
        m_i = jnp.einsum('ijd,ij->id', m_ij, jnp.squeeze(e, axis=-1))

        # Equation 5(a)
        phi_x_out = jnp.squeeze(self.phi_x(m_ij), axis=-1)
        phi_x_out = phi_x_out.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # explicitly set diagonal to 0
        C = x.shape[0] - 1  # This C is from the E(n) GNN paper: https://arxiv.org/pdf/2102.09844.pdf.
        equivariant_shift = jnp.einsum('ijd,ij->id', diff_combos / (sq_norms + 1)[..., None], phi_x_out) / C
        x_new = x + equivariant_shift

        # Equation 5(b)
        phi_h = hk.Sequential([self.phi_h_mlp, hk.Linear(h.shape[-1])])
        phi_h_in = jnp.concatenate([m_i, h], axis=-1)
        h_new = phi_h(phi_h_in)
        if self.recurrent_h:
            h_new = h + h_new

        return x_new, h_new


class HConfig(NamedTuple):
    """Config for h (node features), see https://arxiv.org/pdf/2105.09016.pdf."""
    h_embedding_dim: int = 3   # Dimension of h embedding in the EGCL.
    h_out: bool = False  # Whether to output h (it may be used as an invariant scale parameter for example).
    h_out_dim: int = 1  # Number of dimensions of h output by the EGNN.
    share_h: bool = False   # Whether to use the h from the EGCL for the computation of h-out.
    linear_softmax: bool = False    # Linear layer followed by softmax for improving stability.


class EgnnConfig(NamedTuple):
    """Config of the EGNN."""
    name: str
    mlp_units: Sequence[int] = (3,)
    identity_init_x: bool = False
    zero_init_h: int = False
    n_layers: int = 3  # Number of EGCL layers.
    h_config: HConfig = HConfig()


class se_equivariant_net(hk.Module):
    def __init__(self, config: EgnnConfig):
        super().__init__(name=config.name + "_egnn")
        self.egnn_layer_fn = lambda x, h: EGCL(config.name, config.mlp_units, config.identity_init_x
                                               )(x, h)
        if config.h_config.h_out:
            self.h_final_layer = hk.Linear(config.h_config.h_out_dim, w_init=jnp.zeros, b_init=jnp.zeros) \
                if config.zero_init_h else hk.Linear(1)
        self.config = config

    def __call__(self, x):
        if len(x.shape) == 2:
            return self.forward_single(x)
        else:
            return hk.vmap(self.forward_single, split_rng=False)(x)
    
    def forward_single(self, x):
        """Compute forward pass of EGNN for a single x (no batch dimension)."""

        # Perform forward pass of EGNN.
        # No node feature, so initialise them with zeros.
        h = jnp.zeros((*x.shape[0:-1], self.config.h_config.h_embedding_dim))
        # Use layer_stack to speed up compilation time.
        stack = hk.experimental.layer_stack(self.config.n_layers, with_per_layer_inputs=False)
        x_out, h_egnn = stack(self.egnn_layer_fn)(x, h)


        if not self.config.h_config.h_out:
            return x_out
        else:  # Extra processing to get h_out.

            # Pass square norms of x as a feature.
            diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]
            diff_combos = diff_combos.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)
            sq_norms = jnp.sum(diff_combos ** 2, axis=-1, keepdims=True)

            if self.config.h_config.linear_softmax:
                sq_norms = jax.nn.softmax(hk.Linear(self.config.h_config.h_embedding_dim)(sq_norms))

            mlp_out = hk.nets.MLP((*self.config.mlp_units, self.config.h_config.h_embedding_dim),
                                  activate_final=True)(sq_norms)
            h_out = jnp.mean(mlp_out, axis=(-2))

            if self.config.h_config.share_h:
                # Use h_egnn output from the EGNN as a feature for h-out.
                if self.config.h_config.linear_softmax:
                    h_egnn = jax.nn.softmax(h_egnn)
                h_out = jnp.concatenate([h_out, h_egnn], axis=-1)

            h_out = self.h_final_layer(h_out)
            return x_out, h_out