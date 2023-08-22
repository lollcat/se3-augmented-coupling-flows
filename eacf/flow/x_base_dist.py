from typing import Tuple, Iterable, Optional, List

import chex
import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey, Array
import distrax
import haiku as hk
import mdtraj

from eacf.utils.coordinate_transform.internal import CompleteInternalCoordinateTransform


class CentreGravityGaussian(distrax.Distribution):
    """Gaussian distribution over nodes in space, with a zero centre of gravity.
    Following https://github.com/vgsatorras/en_flows/blob/main/flows/utils.py,
    see also https://arxiv.org/pdf/2105.09016.pdf."""
    def __init__(self, dim: int, n_nodes: int):

        self.dim = dim
        self.n_nodes = n_nodes

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        shape = (n, self.n_nodes, self.dim)
        return sample_center_gravity_zero_gaussian(key, shape)

    def log_prob(self, value: Array) -> Array:
        value = remove_mean(value)
        base_log_prob = center_gravity_zero_gaussian_log_likelihood(value)
        return base_log_prob

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.n_nodes, self.dim)


class HarmonicPotential(distrax.Distribution):
    """Distribution having a harmonic potential based on a graph,
    similar to base distribution used in https://arxiv.org/abs/2304.02198"""
    def __init__(self, dim: int, n_nodes: int, edges: Iterable = [],
                 a: float = 1., mode_scale: Optional[Array] = None,
                 trainable_mode_scale: bool = False):
        self.dim = dim
        self.n_nodes = n_nodes
        H = np.zeros((n_nodes, n_nodes))
        for i, j in edges:
            H[i, j] = -a
            H[j, i] = -a
            H[i, i] += a
            H[j, j] += a
        self.D, self.P = jnp.linalg.eigh(jnp.array(H))
        self.std = jnp.concatenate([jnp.zeros(1), 1 / jnp.sqrt(self.D[1:])])
        self.log_det = dim * jnp.sum(jnp.log(self.std[1:]))
        self.log_Z = 0.5 * dim * (n_nodes - 1) * np.log(2 * np.pi)
        if mode_scale is None:
            mode_scale = jnp.ones(n_nodes - 1)
        if trainable_mode_scale:
            self.log_mode_scale = hk.get_parameter('x_base_dist_log_mode_scale', shape=(n_nodes - 1,),
                                                   init=hk.initializers.Constant(jnp.log(mode_scale)),
                                                   dtype=float)
        else:
            self.log_mode_scale = jnp.log(mode_scale)

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        shape = (n, self.n_nodes, self.dim)
        eps = jax.random.normal(key, shape)
        scale = jnp.concatenate([jnp.zeros(1), jnp.exp(self.log_mode_scale)])
        x = scale[:, None] * eps
        x = self.P @ (self.std[:, None] * x)
        chex.assert_shape(x, shape)
        return remove_mean(x)

    def log_prob(self, value: Array) -> Array:
        value = remove_mean(value)
        value = self.P.T @ value
        value = value / self.std[:, None]
        value = value[..., 1:, :] / jnp.exp(self.log_mode_scale)[:, None]
        log_p = - 0.5 * jnp.sum(value**2, axis=(-2, -1)) - self.log_det \
                - self.log_Z - self.dim * jnp.sum(self.log_mode_scale)
        return log_p

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.n_nodes, self.dim)


class AldpTransformedInternals(distrax.Distribution):
    def __init__(self, data_path: str):
        traj = mdtraj.load(data_path)

        ndim = 66
        if jax.config.jax_enable_x64:
            dtype = jnp.float64
        else:
            dtype = jnp.float32
        transform_data = jnp.array(np.array(traj.xyz).reshape(-1, ndim),
                                   dtype=dtype)
        z_matrix = [
            (0, [1, 4, 6]),
            (1, [4, 6, 8]),
            (2, [1, 4, 0]),
            (3, [1, 4, 0]),
            (4, [6, 8, 14]),
            (5, [4, 6, 8]),
            (7, [6, 8, 4]),
            (9, [8, 6, 4]),
            (10, [8, 6, 4]),
            (11, [10, 8, 6]),
            (12, [10, 8, 11]),
            (13, [10, 8, 11]),
            (15, [14, 8, 16]),
            (16, [14, 8, 6]),
            (17, [16, 14, 15]),
            (18, [16, 14, 8]),
            (19, [18, 16, 14]),
            (20, [18, 16, 19]),
            (21, [18, 16, 19])
        ]
        cart_indices = [8, 6, 14]
        ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]
        self.transform = CompleteInternalCoordinateTransform(ndim, z_matrix, cart_indices,
                                                             transform_data, ind_circ_dih=ind_circ_dih)

        ncarts = self.transform.len_cart_inds
        permute_inv = self.transform.permute_inv
        dih_ind_ = self.transform.ic_transform.dih_indices
        std_dih = self.transform.ic_transform.std_dih

        ind = jnp.arange(60)
        ind = jnp.concatenate([ind[:3 * ncarts - 6], -np.ones(6, dtype=np.int), ind[3 * ncarts - 6:]])
        ind = ind[permute_inv]
        dih_ind = ind[dih_ind_]

        ind_circ_dih_ = jnp.array(ind_circ_dih)
        ind_circ = dih_ind[ind_circ_dih_]
        bound_circ = np.pi / std_dih[ind_circ_dih_]

        scale = jnp.ones(60)
        scale = scale.at[ind_circ].set(bound_circ)
        self.dist_internal = UniformGaussian(dim=60, ind_uniform=ind_circ, scale=scale)

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        z = self.dist_internal._sample_n(key, n)
        x, _ = self.transform.inverse(z)
        x = x.reshape(-1, 22, 3)
        return remove_mean(x)

    def log_prob(self, value: Array) -> Array:
        z, log_det = self.transform.forward(value.reshape(*value.shape[:-2], 66))
        return self.dist_internal.log_prob(z) + log_det

    @property
    def event_shape(self) -> Tuple[int]:
        return (22, 3)



class UniformGaussian(distrax.Distribution):
    def __init__(self, dim: int, ind_uniform: List[int], scale: Optional[chex.Array] = None):
        self.dim = dim
        self.ind_uniform = jnp.array(ind_uniform)
        self.ind_gaussian = jnp.setdiff1d(jnp.arange(dim), self.ind_uniform,
                                          assume_unique=True, size=dim - len(ind_uniform))
        perm = jnp.concatenate([self.ind_uniform, self.ind_gaussian])
        self.inv_perm = jnp.zeros_like(perm).at[perm].set(jnp.arange(dim))
        if scale is None:
            self.scale = jnp.ones((dim,))
        else:
            self.scale = scale

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        key_u, key_g = jax.random.split(key)
        eps_u = jax.random.uniform(key_u, shape=(n, self.ind_uniform.shape[0]), minval=-1, maxval=1)
        eps_g = jax.random.normal(key_g, shape=(n, self.ind_gaussian.shape[0]))
        z = jnp.concatenate([eps_u, eps_g], axis=-1)
        z = z[:, self.inv_perm]
        return z * self.scale

    def log_prob(self, value: Array) -> Array:
        log_prob_u = jnp.broadcast_to(-jnp.log(2 * self.scale[self.ind_uniform]),
                                      value.shape[:-1] + self.ind_uniform.shape[:1])
        log_prob_g = - 0.5 * np.log(2 * np.pi) \
                     - jnp.log(self.scale[self.ind_gaussian]) \
                     - 0.5 * (value[..., self.ind_gaussian] / self.scale[self.ind_gaussian]) ** 2
        return jnp.sum(log_prob_u, -1) + jnp.sum(log_prob_g, -1)

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.dim,)


def assert_mean_zero(x: chex.Array, node_axis=-2):
    mean = jnp.mean(x, axis=node_axis, keepdims=True)
    chex.assert_trees_all_close(1 + jnp.zeros_like(mean), 1 + mean)

def remove_mean(x: chex.Array) -> chex.Array:
    mean = jnp.mean(x, axis=-2, keepdims=True)
    x = x - mean
    return x

def center_gravity_zero_gaussian_log_likelihood(x: chex.Array) -> chex.Array:
    try:
        chex.assert_rank(x, 3)  # [batch, nodes, x]
    except:
        chex.assert_rank(x, 2)  # [nodes, x]

    N, D = x.shape[-2:]

    # assert_mean_zero(x)

    # r is invariant to a basis change in the relevant hyperplane.
    r2 = jnp.sum(x**2, axis=(-1, -2))

    # The relevant hyperplane is (N-1) * D dimensional.
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * jnp.log(2*jnp.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px



def sample_center_gravity_zero_gaussian(key: chex.PRNGKey, shape: chex.Shape) -> chex.Array:
    x = jax.random.normal(key, shape)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected
