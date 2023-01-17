import chex
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from flow.base import CentreGravityGaussian


def load_dataset(path, batch_size, train_test_split_ratio: float = 0.8, seed = 0):
    """Load dataset and add augmented dataset N(0, 1). """
    # Make length divisible by batch size also.
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))

    dataset = np.load(path)
    augmented_dataset = get_target_augmented_variables(dataset, key1)
    dataset = jnp.concatenate((dataset, augmented_dataset), axis=-1)

    dataset = jax.random.permutation(key2, dataset, axis=0)

    train_index = int(dataset.shape[0] * train_test_split_ratio)
    train_set = dataset[:train_index]
    test_set = dataset[train_index:]

    train_set = train_set[:train_set.shape[0] - (train_set.shape[0] % batch_size)]
    test_set = test_set[:train_set.shape[0] - (test_set.shape[0] % batch_size)]
    return train_set, test_set


def get_target_augmented_variables(x_original, key):
    B, N, D = x_original.shape
    x_augmented = CentreGravityGaussian(n_nodes=N, dim=D)._sample_n(key=key, n=B)
    augmented_mean = jnp.mean(x_original, axis=(1, 2), keepdims=True)
    return x_augmented + augmented_mean


# def get_augmented_dist(x_original, scale: float = 1.0):
#     chex.assert_rank(x_original, 3)
#     if AUG_DIST_GLOBAL_MEAN:
#         augmented_mean = jnp.mean(x_original, axis=(1, 2), keepdims=True)
#     else:
#         augmented_mean = jnp.mean(x_original, axis=2, keepdims=True)
#     augmented_mean = jnp.broadcast_to(augmented_mean, x_original.shape)
#     scale_vector = jnp.ones_like(augmented_mean) * scale
#     augmented_dist = distrax.Independent(distrax.MultivariateNormalDiag(
#         loc=augmented_mean, scale_diag=scale_vector),
#         reinterpreted_batch_ndims=1)
#     return augmented_dist

def get_augmented_sample_and_log_prob(x_original, key, K):
    B, N, D = x_original.shape
    x_augmented, log_p_a = CentreGravityGaussian(n_nodes=N, dim=D).sample_and_log_prob(seed=key, sample_shape=(K, B))
    augmented_mean = jnp.mean(x_original, axis=(1, 2), keepdims=True)
    x_augmented = x_augmented + augmented_mean
    return x_augmented, log_p_a

def get_augmented_log_prob(x_augmented):
    B, N, D = x_augmented.shape
    log_p_a = CentreGravityGaussian(n_nodes=N, dim=D).log_prob(x_augmented)
    return log_p_a


def get_marginal_log_lik(log_prob_fn, x_original, key, K: int = 10):
    x_augmented, log_p_a = get_augmented_sample_and_log_prob(x_original, key, K)
    x_original = jnp.stack([x_original]*K, axis=0)
    log_q = jax.vmap(log_prob_fn)(jnp.concatenate((x_original, x_augmented), axis=-1))
    chex.assert_equal_shape((log_p_a, log_q))
    return jnp.mean(jax.nn.logsumexp(log_q - log_p_a, axis=0) - jnp.log(jnp.array(K)))


@partial(jax.jit, static_argnums=(3, 4, 5))
def eval_fn(params, x, key, flow_log_prob_fn, flow_sample_and_log_prob_fn, target_log_prob):
    dim = x.shape[-1] // 2
    key1, key2 = jax.random.split(key)

    log_prob = flow_log_prob_fn.apply(params, x)
    marginal_log_lik = get_marginal_log_lik(log_prob_fn=lambda x: flow_log_prob_fn.apply(params, x),
                                            x_original=x[..., :dim], key=key1)

    # Calculate ESS
    x_flow, log_prob_flow = flow_sample_and_log_prob_fn.apply(params, key2, (x.shape[0]))
    x_flow_original, x_flow_aug = jnp.split(x_flow, axis=-1, indices_or_sections=2)
    log_w = target_log_prob(x_flow_original) + get_augmented_log_prob(x_flow_aug) - log_prob_flow
    ess = 1 / jnp.sum(jax.nn.softmax(log_w) ** 2) / log_w.shape[0]

    info = {"eval_marginal_log_lik": marginal_log_lik,
            "eval_log_lik": jnp.mean(log_prob),
            "eval_kl": jnp.mean(target_log_prob(x) - log_prob),
            "ess": ess
            }
    return info