import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import chex
import optax
from tqdm import tqdm
from functools import partial

from flow.distribution import make_equivariant_augmented_flow_dist
from target import double_well as dw
from utils.loggers import ListLogger



def get_target_augmented_variables(x_original, key):
    x_augmented = jnp.mean(x_original, axis=(1, 2), keepdims=True) + \
    jax.random.normal(key, shape=x_original.shape)
    return x_augmented


def get_marginal_log_lik(log_prob_fn, x_original, key, K: int = 10):
    augmented_mean = jnp.mean(x_original, axis=(1, 2), keepdims=True)
    augmented_mean = jnp.broadcast_to(augmented_mean, x_original.shape)
    augmented_dist = distrax.Independent(distrax.MultivariateNormalDiag(loc=augmented_mean),
                                         reinterpreted_batch_ndims=1)
    augmented_var, log_p_a = augmented_dist._sample_n_and_log_prob(n=K, key=key)
    x_original = jnp.stack([x_original]*K, axis=0)
    log_q = jax.vmap(log_prob_fn)(jnp.concatenate((x_original, augmented_var), axis=-1))
    return jnp.mean(jax.nn.logsumexp(log_q - log_p_a, axis=0) - jnp.log(jnp.array(K)))



def load_dataset(batch_size, train_test_split_ratio: float = 0.8, seed = 0):
    """Load dataset and add augmented dataset N(0, 1). """
    # Make length divisible by batch size also.
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))

    dataset = np.load('target/data/lj_data_vertices13_dim3.npy')
    augmented_dataset = get_target_augmented_variables(dataset, key1)
    dataset = jnp.concatenate((dataset, augmented_dataset), axis=-1)

    dataset = jax.random.permutation(key2, dataset, axis=0)

    train_index = int(dataset.shape[0] * train_test_split_ratio)
    train_set = dataset[:train_index]
    test_set = dataset[train_index:]

    train_set = train_set[:-(train_set.shape[0] % batch_size)]
    test_set = test_set[:-(test_set.shape[0] % batch_size)]
    return train_set, test_set


@partial(jax.jit, static_argnums=(2,))
def eval(params, x, log_prob_fn, key):
    log_prob = log_prob_fn.apply(params, x)
    marginal_log_lik = get_marginal_log_lik(log_prob_fn=lambda x: log_prob_fn.apply(params, x),
                                            x_original=x[..., :2], key=key)
    info = {"eval_marginal_log_lik": marginal_log_lik,
            "eval_log_lik": jnp.mean(log_prob),
            "eval_kl": jnp.mean(dw.log_prob_fn(x) - log_prob),
            }
    return info


def loss_fn(params, x, log_prob_fn):
    log_prob = log_prob_fn.apply(params, x)
    loss = - jnp.mean(log_prob)
    info = {"loss": loss}
    return loss, info



@partial(jax.jit, static_argnums=(3, 4))
def step(params, x, opt_state, log_prob_fn, optimizer):
    grad, info = jax.grad(loss_fn, has_aux=True)(params, x, log_prob_fn)
    updates, new_opt_state = optimizer.update(grad, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    info.update(grad_norm=optax.global_norm(grad))
    return new_params, new_opt_state, info


def plot_sample_hist(samples, ax, dim=(0, 1), vertices=(0, 1), *args, **kwargs):
    d = jnp.linalg.norm(samples[:, vertices[0], dim] - samples[:, vertices[1], dim], axis=-1)
    ax.hist(d, bins=50, density=True, alpha=0.4, *args, **kwargs)



def train():
    n_epoch = int(128)
    dim = 3
    lr = 1e-3
    n_nodes = 13
    n_layers = 4
    batch_size = 32
    max_global_norm = 100  # jnp.inf
    mlp_units = (16,)
    key = jax.random.PRNGKey(0)
    flow_type = "vector_scale_shift"  # "nice", "proj", "vector_scale_shift"
    identity_init = True

    n_plots = 3

    logger = ListLogger()


    @hk.without_apply_rng
    @hk.transform
    def log_prob_fn(x):
        distribution = make_equivariant_augmented_flow_dist(
            dim=dim, nodes=n_nodes, n_layers=n_layers,
            flow_identity_init=identity_init, type=flow_type, mlp_units=mlp_units)
        return distribution.log_prob(x)

    @hk.transform
    def sample_and_log_prob_fn(sample_shape=()):
        distribution = make_equivariant_augmented_flow_dist(
            dim=dim, nodes=n_nodes, n_layers=n_layers,
            flow_identity_init=identity_init, type=flow_type, mlp_units=mlp_units)
        return distribution.sample_and_log_prob(seed=hk.next_rng_key(), sample_shape=sample_shape)

    key, subkey = jax.random.split(key)
    params = log_prob_fn.init(rng=subkey, x=jnp.zeros((1, n_nodes, dim*2)))

    optimizer = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(max_global_norm), optax.adam(lr))
    opt_state = optimizer.init(params)

    train_data, test_data = load_dataset(batch_size)

    print(f"training data size of {train_data.shape[0]}")

    def plot(n_samples=512):
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        samples = \
        jax.jit(sample_and_log_prob_fn.apply, static_argnums=(2,))(params, jax.random.PRNGKey(0), (n_samples,))[0]

        for i in range(3):
            plot_sample_hist(samples, axs[i, 0], dim=(0, 1), vertices=(0, i+1), label="flow samples")
            plot_sample_hist(train_data, axs[i, 0], dim=(0, 1), vertices=(0, i+1), label="ground truth samples")
            plot_sample_hist(samples, axs[i, 1], dim=(2, 3), vertices=(0, i+1), label="flow samples")
            plot_sample_hist(train_data, axs[i, 1], dim=(2, 3), vertices=(0, i+1),
                             label="ground truth samples")
            axs[i, 0].set_title(f"norm dim0-{i + 1} original coordinates")
            axs[i, 1].set_title(f"norm dim0-{i + 1} augmented coordinates")
        axs[0, 0].legend()
        plt.tight_layout()
        plt.show()

    plot()
    # key, subkey = jax.random.split(key)
    # eval_info = eval(params, test_data, log_prob_fn, subkey)
    # logger.write(eval_info)

    pbar = tqdm(range(n_epoch))
    for i in pbar:
        key, subkey = jax.random.split(key)
        train_data = jax.random.permutation(subkey, train_data, axis=0)

        for x in jnp.reshape(train_data, (-1, batch_size, *train_data.shape[1:])):
            params, opt_state, info = step(params, x, opt_state, log_prob_fn, optimizer)
            logger.write(info)
            if jnp.isnan(info["grad_norm"]):
                print("nan grad")
                # raise Exception("nan grad encountered")



        if i % (n_epoch // n_plots) == 0 or i == (n_epoch - 1):
            plot()
            key, subkey = jax.random.split(key)
            eval_info = eval(params, test_data, log_prob_fn, subkey)
            pbar.write(str(eval_info))
            logger.write(eval_info)


    plot_history(logger.history)
    plt.show()

    return logger, params, log_prob_fn, sample_and_log_prob_fn



if __name__ == '__main__':
    from utils.plotting import plot_history
    import matplotlib.pyplot as plt

    USE_64_BIT = False
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)

    logger, params, log_prob_fn, sample_and_log_prob_fn = train()





