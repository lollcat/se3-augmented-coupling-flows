import chex
import jax
from jax import numpy as jnp
import numpy as np
import boltzgen as bg
import torch

from eacf.targets.data import load_aldp
from eacf.utils.coordinate_transform import internal


def tesst_internal_transform():
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)

    # Load aldp data
    train_set, _, _ = load_aldp(train_path='eacf/targets/data/aldp_500K_train_mini.h5')
    # Get positions
    ndim = 66
    if jax.config.jax_enable_x64:
        dtype = jnp.float64
    else:
        dtype = jnp.float32
    data_jax = jnp.array(train_set.positions.reshape(-1, ndim), dtype=dtype)

    # Reference transform
    data_torch = torch.tensor(np.array(data_jax).reshape(-1, ndim),
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
    transform_bg = bg.flows.internal.CompleteInternalCoordinateTransform(ndim, z_matrix,
                                                                         cart_indices, data_torch,
                                                                         ind_circ_dih=ind_circ_dih)

    # jax transform
    transform_jax = internal.CompleteInternalCoordinateTransform(
        ndim, z_matrix, cart_indices, data_jax, ind_circ_dih=ind_circ_dih)

    # Check forward consistency
    batch_size = 10
    x_torch = data_torch[:batch_size]
    x_jax = data_jax[:batch_size]
    z_torch, log_det_torch = transform_bg.forward(x_torch)
    z_jax, log_det_jax = transform_jax.forward(x_jax)
    rtol = 5e-3
    chex.assert_tree_all_close(jnp.array(z_torch.numpy()), z_jax, rtol=rtol)
    chex.assert_tree_all_close(jnp.array(log_det_torch.numpy()), log_det_jax, rtol=rtol)

    # Check inverse consistency
    x_jax_, log_det_jax_ = transform_jax.inverse(z_jax)
    chex.assert_tree_all_close(log_det_jax + log_det_jax_,
                               jnp.zeros_like(log_det_jax), atol=1e-4)
    x_jax__ = transform_jax.inverse(transform_jax.forward(x_jax_)[0])[0]
    chex.assert_tree_all_close(x_jax_, x_jax__, rtol=rtol)
    x_torch_, log_det_torch_ = transform_bg.inverse(torch.as_tensor(np.array(z_jax)))
    chex.assert_tree_all_close(jnp.array(x_torch_.numpy()), x_jax_, rtol=rtol, atol=3e-4)
    chex.assert_tree_all_close(jnp.array(log_det_torch_.numpy()), log_det_jax_, rtol=rtol)

    # Test vmap
    z_vmap, log_det_vmap = jax.vmap(transform_jax.forward)(x_jax)
    chex.assert_tree_all_close(z_vmap, z_jax, rtol=rtol)
    chex.assert_tree_all_close(log_det_vmap, log_det_jax, rtol=rtol)
    x_vmap, log_det_vmap_ = jax.vmap(transform_jax.inverse)(z_jax)
    chex.assert_tree_all_close(x_vmap, x_jax_, rtol=rtol)
    chex.assert_tree_all_close(log_det_vmap_, log_det_jax_, rtol=rtol)

    # Test jit
    fwd_jit = jax.jit(transform_jax.forward)
    z_jit, log_det_jit = fwd_jit(x_jax)
    chex.assert_tree_all_close(z_jit, z_jax, rtol=rtol)
    chex.assert_tree_all_close(log_det_jit, log_det_jax, rtol=rtol)
    inv_jit = jax.jit(transform_jax.inverse)
    x_jit, log_det_jit_ = inv_jit(z_jax)
    chex.assert_tree_all_close(x_jit, x_jax_, rtol=rtol)
    chex.assert_tree_all_close(log_det_jit_, log_det_jax_, rtol=rtol)


if __name__ == '__main__':
    tesst_internal_transform()

