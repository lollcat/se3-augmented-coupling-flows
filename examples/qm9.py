import haiku as hk
import jax
import numpy as np
import jax.numpy as jnp
import optax
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

from flow.distribution import make_equivariant_augmented_flow_dist, EquivariantFlowDistConfig
from utils.loggers import ListLogger
from utils.plotting import plot_history
from utils.train_and_eval import eval_fn, original_dataset_to_joint_dataset
from utils.numerical import get_pairwise_distances
from flow.nets import EgnnConfig, HConfig


def load_dataset(batch_size, train_data_n_points = None, test_data_n_points = None, seed=0):
    # First need to run `qm9.download_data`
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))

    data_dir = "target/data/qm9_"
    train_data = np.load(data_dir + "train.npy")
    test_data = np.load(data_dir + "test.npy")
    valid_data = np.load(data_dir + "valid.npy")

    if train_data_n_points is not None:
        train_data = train_data[:train_data_n_points]
    if test_data_n_points is not None:
        test_data = test_data[:test_data_n_points]

    train_data = train_data[:train_data.shape[0] - (train_data.shape[0] % batch_size)]

    train_data = original_dataset_to_joint_dataset(train_data, key1)
    test_data = original_dataset_to_joint_dataset(test_data, key2)

    return train_data, test_data



def loss_fn(params, x, log_prob_fn):
    log_prob = log_prob_fn.apply(params, x)
    loss = - jnp.mean(log_prob)
    info = {"loss": loss}
    return loss, info



# @partial(jax.jit, static_argnums=(3, 4))
def step(params, x, opt_state, log_prob_fn, optimizer):
    grad, info = jax.grad(loss_fn, has_aux=True)(params, x, log_prob_fn)
    updates, new_opt_state = optimizer.update(grad, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    info.update(grad_norm=optax.global_norm(grad))
    return new_params, new_opt_state, info


def plot_sample_hist(samples, ax, dim=(0, 1, 3), n_vertices = 5, *args, **kwargs):
    """n_vertices argument allows us to look at pairwise distances for subset of vertices,
    to prevent plotting taking too long"""
    differences = jax.vmap(get_pairwise_distances)(samples[:, :n_vertices, dim])
    d = differences.flatten()
    d = d[jnp.isfinite(d)]
    d = d.clip(max=10)  # Clip keep plot reasonable.
    d = d[d != 0.0]
    ax.hist(d, bins=50, density=True, alpha=0.4, *args, **kwargs)


_DEFAULT_FLOW_CONFIG = EquivariantFlowDistConfig(
        dim=3, n_layers=4, nodes=29,  flow_identity_init=True,
        type="vector_scale", fast_compile=True, compile_n_unroll=1,
        egnn_config = EgnnConfig(name="", mlp_units=(4,), n_layers=2, h_config=HConfig()._replace(
                linear_softmax=True, share_h=True))
    )


def train(
    n_epoch = int(10),
    flow_dist_config: EquivariantFlowDistConfig = _DEFAULT_FLOW_CONFIG,
    lr: float = 1e-4,
    batch_size: int = 4,
    max_global_norm: float = jnp.inf,  # jnp.inf
    key: int = jax.random.PRNGKey(0),
    n_plots: int = 3,
    reload_aug_per_epoch: bool = True,
    train_data_n_points = None,  # set to None to use full set
    test_data_n_poins = None,  # set to None to use full set,
    plotting_n_nodes = 5,
    K: int = 20,
):
    n_nodes = 29
    dim = 3
    assert flow_dist_config.dim == dim
    assert flow_dist_config.nodes == n_nodes

    logger = ListLogger()


    @hk.without_apply_rng
    @hk.transform
    def log_prob_fn(x):
        distribution = make_equivariant_augmented_flow_dist(flow_dist_config)
        return distribution.log_prob(x)

    @hk.transform
    def sample_and_log_prob_fn(sample_shape=()):
        distribution = make_equivariant_augmented_flow_dist(flow_dist_config)
        return distribution.sample_and_log_prob(seed=hk.next_rng_key(), sample_shape=sample_shape)


    key, subkey = jax.random.split(key)
    params = log_prob_fn.init(rng=subkey, x=jnp.zeros((1, n_nodes, dim*2)))

    optimizer = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(max_global_norm), optax.adam(lr))
    opt_state = optimizer.init(params)

    train_data, test_data = load_dataset(batch_size, train_data_n_points=train_data_n_points, test_data_n_points=test_data_n_poins)

    print(f"training data size of {train_data.shape[0]}")

    def scan_fn(carry, xs):
        params, opt_state = carry
        x = xs
        params, opt_state, info = step(params, x, opt_state, log_prob_fn, optimizer)
        return (params, opt_state), info

    def plot(n_samples=batch_size):
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        samples = \
        jax.jit(sample_and_log_prob_fn.apply, static_argnums=(2,))(params, jax.random.PRNGKey(0), (n_samples,))[0]

        plot_sample_hist(samples, axs[0], dim=(0, 1, 2), label="flow samples", n_vertices=plotting_n_nodes)
        plot_sample_hist(train_data[:n_samples], axs[0], dim=(0, 1, 2), label="train samples",
                         n_vertices=plotting_n_nodes)
        plot_sample_hist(test_data[:n_samples], axs[0], dim=(0, 1, 2), label="test samples",
                         n_vertices=plotting_n_nodes)
        plot_sample_hist(samples, axs[1], dim=(3, 4, 5), label="flow samples",
                         n_vertices=plotting_n_nodes)
        plot_sample_hist(train_data[:n_samples], axs[1], dim=(3, 4, 5),
                         label="train samples", n_vertices=plotting_n_nodes)
        axs[0].set_title(f"norms between original coordinates")
        axs[1].set_title(f"norms between augmented coordinates")
        axs[0].legend()
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
        if reload_aug_per_epoch:
            key, subkey = jax.random.split(key)
            train_data = original_dataset_to_joint_dataset(train_data[..., :dim], subkey)

        batched_data = jnp.reshape(train_data, (-1, batch_size, *train_data.shape[1:]))
        (params, opt_state), infos = jax.lax.scan(scan_fn, (params, opt_state), batched_data, unroll=1)

        for batch_index in range(batched_data.shape[0]):
            info = jax.tree_map(lambda x: x[batch_index], infos)
            logger.write(info)
            if jnp.isnan(info["grad_norm"]):
                print("nan grad")


        if i % (n_epoch // n_plots) == 0 or i == (n_epoch - 1):
            plot()
            key, subkey = jax.random.split(key)
            eval_info = eval_fn(params=params, x=test_data, flow_log_prob_fn=log_prob_fn,
                                flow_sample_and_log_prob_fn=sample_and_log_prob_fn,
                                key=subkey,
                                batch_size=max(100, batch_size),
                                K=K)
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

    logger, params, log_prob_fn, sample_and_log_prob_fn = train(train_data_n_points=100, test_data_n_poins=100)


