import chex
import jax.numpy as jnp
import jax

from flow.test_utils import rotate_translate_x_and_a_2d, rotate_translate_x_and_a_3d
from flow.aug_flow_dist import FullGraphSample, separate_samples_to_full_joint, joint_to_separate_samples
from flow.conditional_dist import build_aux_dist

_N_NODES = 8
_FEATURE_DIM = 1


def test_conditional_dist(dim: int = 3, n_aug: int = 7):
    n_nodes = _N_NODES
    batch_size = 5
    key = jax.random.PRNGKey(0)

    # Init params.
    key, subkey = jax.random.split(key)
    samples = jax.random.normal(subkey, (batch_size, n_nodes, n_aug, dim))

    key, subkey = jax.random.split(key)
    key1, key2, key3 = jax.random.split(subkey, 3)

    theta = jax.random.uniform(key1, shape=(batch_size,)) * 2 * jnp.pi
    translation = jax.random.normal(key2, shape=(batch_size, dim))
    phi = jax.random.uniform(key3, shape=(batch_size,)) * 2 * jnp.pi

    def group_action(x_and_a):
        if dim == 2:
            x_and_a_rot = jax.vmap(rotate_translate_x_and_a_2d)(x_and_a, theta, translation)
        else:  # dim == 3:
            x_and_a_rot = jax.vmap(rotate_translate_x_and_a_3d)(x_and_a, theta, phi, translation)
        return x_and_a_rot

    a = group_action(samples)
    b = group_action(samples[:, :, :n_aug-1])
    c = group_action(samples[:, :, :n_aug - 2])
    a[:, :, :n_aug-1] == b
    a[:, :, :n_aug - 2] == c
    b[:,:,:n_aug - 2] == c



if __name__ == '__main__':
    test_conditional_dist(dim=3)
