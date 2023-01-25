import haiku as hk
import jax
import jax.numpy as jnp
import optax
from tqdm.autonotebook import tqdm
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

from flow.distribution import make_equivariant_augmented_flow_dist
from target import double_well as dw
from utils.loggers import ListLogger
from utils.plotting import plot_history
from utils.train_and_eval import eval_fn, original_dataset_to_joint_dataset
from utils.numerical import get_pairwise_distances
from flow.nets import EgnnConfig



def load_dataset(batch_size, train_set_size: int = 1000, test_set_size:int = 1000, seed: int = 0):
    # dataset from https://github.com/vgsatorras/en_flows
    # Loading following https://github.com/vgsatorras/en_flows/blob/main/dw4_experiment/dataset.py.

    data_path = 'target/data/dw4-dataidx.npy'  # 'target/data/dw_data_vertices4_dim2.npy'
    dataset = np.asarray(np.load(data_path, allow_pickle=True)[0])
    dataset = jnp.reshape(dataset, (-1, 4, 2))
    dataset = original_dataset_to_joint_dataset(dataset, jax.random.PRNGKey(seed))

    train_set = dataset[:train_set_size]
    train_set = train_set[:train_set_size - (train_set.shape[0] % batch_size)]

    test_set = dataset[-test_set_size:]
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


def plot_sample_hist(samples, ax, dim=(0, 1), *args, **kwargs):
    differences = jax.vmap(get_pairwise_distances)(samples[..., dim])
    d = differences.flatten()
    d = d[d != 0.0]
    ax.hist(d, bins=50, density=True, alpha=0.4, *args, **kwargs)



def train(
    n_epoch = int(100),
    dim = 2,
    lr = 5e-4,
    n_nodes = 4,
    n_layers = 8,
    batch_size = 16,
    max_global_norm: int = jnp.inf,  # 100, jnp.inf
    key = jax.random.PRNGKey(0),
    flow_type = "vector_scale_shift",  # "nice", "proj", "vector_scale_shift"
    identity_init = True,
    n_plots = 4,
    reload_aug_per_epoch: bool = True,
    egnn_config: EgnnConfig = EgnnConfig(name="dummy", mlp_units=(4,), n_layers=1, h_embedding_dim=3,
                                         share_h=True)
):

    logger = ListLogger()


    @hk.without_apply_rng
    @hk.transform
    def log_prob_fn(x):
        distribution = make_equivariant_augmented_flow_dist(
            dim=dim, nodes=n_nodes, n_layers=n_layers,
            flow_identity_init=identity_init, type=flow_type,
            egnn_conifg=egnn_config
        )
        return distribution.log_prob(x)

    @hk.transform
    def sample_and_log_prob_fn(sample_shape=()):
        distribution = make_equivariant_augmented_flow_dist(
            dim=dim, nodes=n_nodes, n_layers=n_layers,
            flow_identity_init=identity_init, type=flow_type,
            egnn_conifg=egnn_config
        )
        return distribution.sample_and_log_prob(seed=hk.next_rng_key(), sample_shape=sample_shape)

    key, subkey = jax.random.split(key)
    params = log_prob_fn.init(rng=subkey, x=jnp.zeros((1, n_nodes, dim*2)))

    optimizer = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(max_global_norm), optax.adamw(lr))
    opt_state = optimizer.init(params)

    train_data, test_data = load_dataset(batch_size=batch_size, train_set_size=1028, test_set_size=128)

    print(f"training data size of {train_data.shape[0]}")

    def plot(n_samples=512):
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        samples = \
        jax.jit(sample_and_log_prob_fn.apply, static_argnums=(2,))(params, jax.random.PRNGKey(0), (n_samples,))[0]

        plot_sample_hist(samples, axs[0], dim=(0, 1), label="flow samples")
        plot_sample_hist(train_data, axs[0], dim=(0, 1), label="ground truth samples")
        plot_sample_hist(samples, axs[1], dim=(2, 3), label="flow samples")
        plot_sample_hist(train_data, axs[1], dim=(2, 3),
                         label="ground truth samples")
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
                                target_log_prob=dw.log_prob_fn,
                                key=subkey, batch_size=max(16, batch_size),
                                K=2)
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





