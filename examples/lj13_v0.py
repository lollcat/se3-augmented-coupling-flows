import haiku as hk
import jax
import numpy as np
import jax.numpy as jnp
import optax
from tqdm.autonotebook import tqdm
from functools import partial
import matplotlib.pyplot as plt

from flow.distribution import make_equivariant_augmented_flow_dist
from target import leonard_jones as lj
from utils.loggers import ListLogger
from utils.plotting import plot_history
from utils.train_and_eval import eval_fn, get_target_augmented_variables


def load_dataset(batch_size, train_set_size: int = 1000, val_set_size:int = 1000, seed: int = 0):
    # dataset from https://github.com/vgsatorras/en_flows
    # Loading following https://github.com/vgsatorras/en_flows/blob/main/dw4_experiment/dataset.py.

    # Train data
    data = np.load("target/data/holdout_data_LJ13.npy")
    idx = np.load("target/data/idx_LJ13.npy")
    train_set = data[idx[:train_set_size]]
    train_set = jnp.reshape(train_set, (-1, 13, 3))
    augmented_dataset_train = get_target_augmented_variables(train_set, jax.random.PRNGKey(seed))
    train_set = jnp.concatenate((train_set, augmented_dataset_train), axis=-1)
    train_set = train_set[:train_set_size - (train_set.shape[0] % batch_size)]

    # Test set
    test_data_path = 'target/data/all_data_LJ13.npy'  # target/data/lj_data_vertices13_dim3.npy
    dataset = np.load(test_data_path)
    dataset = jnp.reshape(dataset, (-1, 13, 3))
    augmented_dataset = get_target_augmented_variables(dataset, jax.random.PRNGKey(seed))
    dataset = jnp.concatenate((dataset, augmented_dataset), axis=-1)
    test_set = dataset[:val_set_size]
    return train_set, test_set



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


def plot_sample_hist(samples, ax, dim=(0, 1, 2), vertices=(0, 1), *args, **kwargs):
    d = jnp.linalg.norm(samples[:, vertices[0], dim] - samples[:, vertices[1], dim], axis=-1)
    ax.hist(d, bins=50, density=True, alpha=0.4, *args, **kwargs)



def train(
    n_epoch = int(32),
    dim = 3,
    lr = 1e-3,
    n_nodes = 13,
    n_layers = 4,
    batch_size = 32,
    max_global_norm = 100,  # jnp.inf
    mlp_units = (16,),
    seed = 0,
    flow_type= "nice",  # "nice", "proj", "vector_scale_shift"
    identity_init = True,
    n_plots: int = 3,
):
    key = jax.random.PRNGKey(seed)


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
            plot_sample_hist(samples, axs[i, 0], dim=(0, 1, 2), vertices=(0, i+1), label="flow samples")
            plot_sample_hist(train_data, axs[i, 0], dim=(0, 1, 2), vertices=(0, i+1), label="ground truth samples")
            plot_sample_hist(samples, axs[i, 1], dim=(3, 4, 5), vertices=(0, i+1), label="flow samples")
            plot_sample_hist(train_data, axs[i, 1], dim=(3, 4, 5), vertices=(0, i+1),
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
            eval_info = eval_fn(params=params, x=test_data, flow_log_prob_fn=log_prob_fn,
                                flow_sample_and_log_prob_fn=sample_and_log_prob_fn,
                                target_log_prob=lj.log_prob_fn,
                                key=subkey)
            pbar.write(str(eval_info))
            logger.write(eval_info)


    plot_history(logger.history)
    plt.show()

    return logger, params, log_prob_fn, sample_and_log_prob_fn



if __name__ == '__main__':

    USE_64_BIT = False
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)

    logger, params, log_prob_fn, sample_and_log_prob_fn = train()





