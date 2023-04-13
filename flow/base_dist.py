"""Following https://github.com/vgsatorras/en_flows/blob/main/flows/utils.py."""
from typing import Tuple, List, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey, Array
import distrax
import haiku as hk
import torch
from boltzgen.flows import CoordinateTransform
import mdtraj

from flow.conditional_dist import get_conditional_gaussian_augmented_dist
from flow.distrax_with_extra import DistributionWithExtra, Extra


class CentreGravitryGaussianAndCondtionalGuassian(DistributionWithExtra):
    """x ~ CentreGravityGaussian, a ~ Normal(x, I)"""
    def __init__(self, dim, n_nodes: int, n_aux: int,
                 global_centering: bool = False,
                 x_scale_init: float = 1.0,
                 trainable_x_scale: bool = False,
                 augmented_scale_init: float = 1.0,
                 trainable_augmented_scale: bool = True):
        self.n_aux = n_aux
        self.dim = dim
        self.n_nodes = n_nodes
        self.global_centering = global_centering
        if trainable_augmented_scale:
            self.augmented_log_scale = hk.get_parameter(name='augmented_log_scale', shape=(n_aux,),
                                                          init=hk.initializers.Constant(jnp.log(augmented_scale_init)))
        else:
            self.augmented_log_scale = jnp.zeros((n_aux,))
        if trainable_x_scale:
            self.x_log_scale = hk.get_parameter(name='x_log_scale', shape=(),
                                                          init=hk.initializers.Constant(jnp.log(x_scale_init)))
        else:
            self.x_log_scale = jnp.log(x_scale_init)
        self.x_dist = CentreGravityGaussian(dim=dim, n_nodes=n_nodes, log_scale=self.x_log_scale)


    def get_augmented_dist(self, x: chex.Array) -> distrax.Distribution:
        dist = get_conditional_gaussian_augmented_dist(
            x, n_aux=self.n_aux, scale=jnp.exp(self.augmented_log_scale), global_centering=self.global_centering)
        return dist


    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        key1, key2 = jax.random.split(key)
        original_coords = self.x_dist._sample_n(key1, n)
        augmented_coords = self.get_augmented_dist(original_coords).sample(seed=key2)
        joint_coords = jnp.concatenate([jnp.expand_dims(original_coords, axis=-2), augmented_coords], axis=-2)
        return joint_coords

    def log_prob(self, value: Array) -> Array:
        original_coords, augmented_coords = jnp.split(value, indices_or_sections=[1, ], axis=-2)
        assert original_coords.shape[-2] == 1
        assert augmented_coords.shape[-2] == self.n_aux
        original_coords = jnp.squeeze(original_coords, axis=-2)

        log_prob_a_given_x = self.get_augmented_dist(original_coords).log_prob(augmented_coords)
        log_p_x = self.x_dist.log_prob(original_coords)
        chex.assert_equal_shape((log_p_x, log_prob_a_given_x))
        return log_p_x + log_prob_a_given_x

    def get_extra(self) -> Extra:
        extra = Extra(aux_info={"base_x_scale": jnp.exp(self.x_log_scale)},
                      info_aggregator={"base_x_scale": jnp.mean})
        for i in range(self.n_aux):
            extra.aux_info[f"base_a{i}_scale"] = jnp.exp(self.augmented_log_scale[i])
            extra.info_aggregator[f"base_a{i}_scale"] = jnp.mean
        return extra

    def log_prob_with_extra(self, value: Array) -> Tuple[Array, Extra]:
        extra = self.get_extra()
        log_prob = self.log_prob(value)
        return log_prob, extra

    def sample_n_and_log_prob_with_extra(self, key: PRNGKey, n: int) -> Tuple[Array, Array, Extra]:
        x, log_prob = self._sample_n_and_log_prob(key, n)
        extra = self.get_extra()
        return x, log_prob, extra

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.n_nodes, self.n_aux+1, self.dim)


class CentreGravityGaussian(distrax.Distribution):
    """Gaussian distribution over nodes in space, with a zero centre of gravity.
    See https://arxiv.org/pdf/2105.09016.pdf."""
    def __init__(self, dim: int, n_nodes: int, log_scale: chex.Array = jnp.zeros(())):

        self.dim = dim
        self.n_nodes = n_nodes
        self.log_scale = log_scale

    @property
    def scale(self):
        return jnp.exp(self.log_scale)

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        shape = (n, self.n_nodes, self.dim)
        return sample_center_gravity_zero_gaussian(key, shape)*self.scale

    def log_prob(self, value: Array) -> Array:
        value = remove_mean(value)
        value = value / self.scale
        inv_log_det = - self.log_scale*np.prod(self.event_shape)
        base_log_prob = center_gravity_zero_gaussian_log_likelihood(value)
        return base_log_prob + inv_log_det

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.n_nodes, self.dim)


class AldpTransformedInternalsAndConditionalGaussian(distrax.Distribution):
    """x ~ AldpTransformedInternals, a ~ Normal(x, I)"""
    def __init__(self, data_path: str, n_aux: int,
                 global_centering: bool = False,
                 augmented_scale_init: float = 1.0,
                 trainable_augmented_scale: bool = True):
        self.n_aux = n_aux
        self.dim = 3
        self.n_nodes = 22
        self.global_centering = global_centering
        if trainable_augmented_scale:
            self.augmented_log_scale = hk.get_parameter(name='augmented_log_scale', shape=(n_aux,),
                                                          init=hk.initializers.Constant(jnp.log(augmented_scale_init)))
        else:
            self.augmented_log_scale = jnp.zeros((n_aux,))
        self.x_dist = AldpTransformedInternals(data_path)


    def get_augmented_dist(self, x: chex.Array) -> distrax.Distribution:
        dist = get_conditional_gaussian_augmented_dist(
            x, n_aux=self.n_aux, scale=jnp.exp(self.augmented_log_scale), global_centering=self.global_centering)
        return dist


    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        key1, key2 = jax.random.split(key)
        original_coords = self.x_dist._sample_n(key1, n)
        augmented_coords = self.get_augmented_dist(original_coords).sample(seed=key2)
        joint_coords = jnp.concatenate([jnp.expand_dims(original_coords, axis=-2), augmented_coords], axis=-2)
        return joint_coords

    def log_prob(self, value: Array) -> Array:
        original_coords, augmented_coords = jnp.split(value, indices_or_sections=[1, ], axis=-2)
        assert original_coords.shape[-2] == 1
        assert augmented_coords.shape[-2] == self.n_aux
        original_coords = jnp.squeeze(original_coords, axis=-2)

        log_prob_a_given_x = self.get_augmented_dist(original_coords).log_prob(augmented_coords)
        log_p_x = self.x_dist.log_prob(original_coords)
        chex.assert_equal_shape((log_p_x, log_prob_a_given_x))
        return log_p_x + log_prob_a_given_x

    def get_extra(self) -> Extra:
        extra = Extra(aux_info={},
                      info_aggregator={})
        for i in range(self.n_aux):
            extra.aux_info[f"base_a{i}_scale"] = jnp.exp(self.augmented_log_scale[i])
            extra.info_aggregator[f"base_a{i}_scale"] = jnp.mean
        return extra

    def log_prob_with_extra(self, value: Array) -> Tuple[Array, Extra]:
        extra = self.get_extra()
        log_prob = self.log_prob(value)
        return log_prob, extra

    def sample_n_and_log_prob_with_extra(self, key: PRNGKey, n: int) -> Tuple[Array, Array, Extra]:
        x, log_prob = self._sample_n_and_log_prob(key, n)
        extra = self.get_extra()
        return x, log_prob, extra

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.n_nodes, self.n_aux+1, self.dim)


class AldpTransformedInternals(distrax.Distribution):
    def __init__(self, data_path: str):
        traj = mdtraj.load(data_path)

        ndim = 66
        transform_data = torch.tensor(np.array(traj.xyz).reshape(-1, ndim),
                                      dtype=torch.float64)
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
        self.transform = CoordinateTransform(transform_data, ndim, z_matrix, cart_indices,
                                             mode="internal", ind_circ_dih=ind_circ_dih)

        ncarts = self.transform.transform.len_cart_inds
        permute_inv = self.transform.transform.permute_inv.cpu().numpy()
        dih_ind_ = self.transform.transform.ic_transform.dih_indices.cpu().numpy()
        std_dih = self.transform.transform.ic_transform.std_dih.cpu().numpy()

        ind = np.arange(60)
        ind = np.concatenate([ind[:3 * ncarts - 6], -np.ones(6, dtype=np.int), ind[3 * ncarts - 6:]])
        ind = ind[permute_inv]
        dih_ind = ind[dih_ind_]

        ind_circ = dih_ind[ind_circ_dih]
        bound_circ = np.pi / std_dih[ind_circ_dih]

        scale = jnp.ones(60)
        scale = scale.at[ind_circ].set(bound_circ)
        self.dist_internal = UniformGaussian(dim=ndim, ind_uniform=ind_circ, scale=scale)

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        z = self.dist_internal.sample(key, n)
        z_torch = torch.tensor(np.array(z).reshape(-1, 60),
                               dtype=torch.float64)
        x_np = self.transform(z_torch)[0].detach().numpy()
        x = jnp.array(x_np.reshape(-1, 22, 3))
        return remove_mean(x)

    def log_prob(self, value: Array) -> Array:
        x_torch = torch.tensor(np.array(value).reshape(-1, 66),
                               dtype=torch.float64)
        z_torch, log_det_torch = self.transform.inverse(x_torch)
        z_np, log_det_np = z_torch.detach().numpy(), log_det_torch.detach().numpy()
        return self.dist_internal.log_prob(jnp.array(z_np)) + jnp.array(log_det_np)

    @property
    def event_shape(self) -> Tuple[int]:
        return (22, 3)



class UniformGaussian(distrax.Distribution):
    def __init__(self, dim: int, ind_uniform: List[int], scale: Optional[chex.Array] = None):
        self.dim = dim
        self.ind_uniform = jnp.array(ind_uniform)
        self.ind_gaussian = jnp.array([i for i in range(dim) if i not in ind_uniform])
        perm = jnp.concatenate([self.ind_uniform, self.ind_gaussian])
        inv_perm = jnp.zeros_like(perm)
        for i in range(dim):
            inv_perm[perm[i]] = i
        self.inv_perm = inv_perm
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
        log_prob_u = jnp.broadcast_to(-jnp.log(2 * self.scale[self.ind_uniform]), (value.shape[0], -1))
        log_prob_g = - 0.5 * np.log(2 * np.pi) \
                     - jnp.log(self.scale[self.ind_gaussian]) \
                     - 0.5 * (value[:, self.ind_gaussian] / self.scale[self.ind_gaussian]) ** 2
        return jnp.sum(log_prob_u, -1) + jnp.sum(log_prob_g, -1)

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.dim,)




def assert_mean_zero(x: chex.Array):
    mean = jnp.mean(x, axis=-2, keepdims=True)
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
