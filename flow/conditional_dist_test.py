import chex
import jax.numpy as jnp
import distrax
import haiku as hk
import jax

from flow.test_utils import rotate_translate_x_and_a_2d, rotate_translate_x_and_a_3d
from flow.aug_flow_dist import FullGraphSample, separate_samples_to_full_joint, joint_to_separate_samples
from flow.conditional_dist import build_aux_dist

_N_NODES = 7
_FEATURE_DIM = 1


def test_conditional_dist(dim: int = 3, n_aug: int = 3):
    n_nodes = _N_NODES
    batch_size = 5
    key = jax.random.PRNGKey(0)

    make_aux_target = build_aux_dist(name='target', n_aug=n_aug, global_centering=False, augmented_scale_init=1.,
                                     trainable_scale=False)

    # Init params.
    key, subkey = jax.random.split(key)
    dummy_samples = FullGraphSample(positions=jax.random.normal(subkey, (batch_size, n_nodes, dim)),
                                    features=jnp.zeros((batch_size, n_nodes, _FEATURE_DIM)))
    dist_v1 = make_aux_target(dummy_samples)

    # Test aux-target distribution.
    key, subkey = jax.random.split(key)
    aug_samples, aux_log_probs = dist_v1.sample_and_log_prob(seed=subkey)

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

    positions_rot = jnp.squeeze(group_action(jnp.expand_dims(dummy_samples.positions, axis=-2)), axis=-2)
    aug_samples_rot = group_action(aug_samples)
    dist_g = make_aux_target(FullGraphSample(features=dummy_samples.features, positions=positions_rot))

    log_prob = dist_v1.log_prob(aug_samples)
    log_prob_alt = dist_g.log_prob(aug_samples_rot)
    chex.assert_trees_all_close(log_prob, log_prob_alt)


if __name__ == '__main__':
    test_conditional_dist(dim=2)
