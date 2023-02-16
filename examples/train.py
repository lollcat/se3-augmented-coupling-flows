from typing import Callable, Tuple, Optional, List, NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import numpy as np
import wandb
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import pickle
import os
import pathlib
from datetime import datetime
from omegaconf import DictConfig
import matplotlib as mpl

from flow.distribution import make_equivariant_augmented_flow_dist, EquivariantFlowDistConfig, BaseConfig
from nets.en_gnn import EgnnConfig, HConfig
from nets.transformer import TransformerConfig
from utils.plotting import plot_history
from utils.train_and_eval import eval_fn, original_dataset_to_joint_dataset, ml_step
from utils.numerical import get_pairwise_distances
from utils.loggers import Logger, WandbLogger, ListLogger



mpl.rcParams['figure.dpi'] = 150
TestData = chex.Array
TrainData = chex.Array
FlowSampleFn = Callable[[hk.Params, chex.PRNGKey, chex.Shape], chex.Array]
Plotter = Callable[[hk.Params, FlowSampleFn, TestData, TrainData], List[plt.Figure]]


def plot_orig_aug_centre_mass_diff_hist(samples,
                                        ax, max_distance=10,
                                        *args, **kwargs):
    dim = samples.shape[-1] // 2
    centre_mass_original = jnp.mean(samples[..., :dim], axis=-2)
    centre_mass_augmented = jnp.mean(samples[..., dim:], axis=-2)
    d = jnp.linalg.norm(centre_mass_original - centre_mass_augmented, axis=-1)
    d = d[jnp.isfinite(d)]
    d = d.clip(max=max_distance)  # Clip keep plot reasonable.
    ax.hist(d, bins=50, density=True, alpha=0.4, *args, **kwargs)


def plot_sample_hist(samples,
                     ax,
                     original_coords,  # or augmented
                     n_vertices: Optional[int] = None,
                     max_distance = 10, *args, **kwargs):
    """n_vertices argument allows us to look at pairwise distances for subset of vertices,
    to prevent plotting taking too long"""
    dim = samples.shape[-1] // 2
    dims = jnp.arange(dim) + (0 if original_coords else dim)
    n_vertices = samples.shape[1] if n_vertices is None else n_vertices
    n_vertices = min(samples.shape[1], n_vertices)
    differences = jax.jit(jax.vmap(get_pairwise_distances))(samples[:, :n_vertices, dims])
    mask = jnp.ones_like(differences, dtype=bool).at[:, jnp.arange(n_vertices), jnp.arange(n_vertices)].set(False)
    d = differences.flatten()
    d = d[mask.flatten()]
    d = d[jnp.isfinite(d)]
    d = d.clip(max=max_distance)  # Clip keep plot reasonable.
    ax.hist(d, bins=50, density=True, alpha=0.4, *args, **kwargs)

def plot_original_aug_norms_sample_hist(samples, ax, max_distance=10, *args, **kwargs):
    dim = samples.shape[-1] // 2
    norms = jnp.linalg.norm(samples[..., :dim] - samples[..., dim:], axis=-1).flatten()
    norms = norms.clip(max=max_distance)  # Clip keep plot reasonable.
    ax.hist(norms, bins=50, density=True, alpha=0.4, *args, **kwargs)



def default_plotter(params, flow_sample_fn, key, n_samples, train_data, test_data,
                    plotting_n_nodes: Optional[int] = None):

    # Plot interatomic distance histograms.
    fig1, axs = plt.subplots(2, 3, figsize=(15, 10))
    samples = flow_sample_fn(params, key, (n_samples,))

    for i, og_coords in enumerate([True, False]):
        plot_sample_hist(samples, axs[0, i], original_coords=og_coords, label="flow samples",
                         n_vertices=plotting_n_nodes)
        plot_sample_hist(samples, axs[1, i], original_coords=og_coords, label="flow samples",
                         n_vertices=plotting_n_nodes)
        plot_sample_hist(train_data[:n_samples], axs[0, i], original_coords=og_coords, label="train samples",
                         n_vertices=plotting_n_nodes)
        plot_sample_hist(test_data[:n_samples], axs[1, i], original_coords=og_coords, label="test samples",
                         n_vertices=plotting_n_nodes)

    plot_original_aug_norms_sample_hist(samples, axs[0, 2], label='flow samples')
    plot_original_aug_norms_sample_hist(train_data, axs[0, 2], label='train samples')
    plot_original_aug_norms_sample_hist(samples, axs[1, 2], label='flow samples')
    plot_original_aug_norms_sample_hist(test_data, axs[1, 2], label='test samples')

    axs[0, 0].set_title(f"norms between original coordinates")
    axs[0, 1].set_title(f"norms between augmented coordinates")
    axs[0, 2].set_title(f"norms between original-aug pairs")
    axs[0, 0].legend()
    axs[1, 0].legend()
    fig1.tight_layout()

    # Plot histogram for centre of mean
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
    plot_orig_aug_centre_mass_diff_hist(samples, ax=axs2[0], label='flow samples')
    plot_orig_aug_centre_mass_diff_hist(train_data, ax=axs2[0], label='train samples')
    plot_orig_aug_centre_mass_diff_hist(samples, ax=axs2[1], label='flow samples')
    plot_orig_aug_centre_mass_diff_hist(test_data, ax=axs2[1], label='test samples')
    axs2[0].legend()
    axs2[1].legend()
    axs2[0].set_title("norms between original - aug centre of mass histogram")
    axs2[1].set_title("norms between original - aug centre of mass histogram")
    fig2.tight_layout()

    return [fig1, fig2]


def plot_and_maybe_save(plotter, params, sample_fn, key, plot_batch_size, train_data, test_data, epoch_n,
                        save: bool,
                        plots_dir):
    figures = plotter(params, sample_fn, key, plot_batch_size, train_data, test_data)
    for j, figure in enumerate(figures):
        if save:
            figure.savefig(os.path.join(plots_dir, f"{j}_iter_{epoch_n}.png"))
        else:
            plt.show()
        plt.close(figure)



class TrainConfig(NamedTuple):
    n_epoch: int
    dim: int
    n_nodes: int
    flow_dist_config: EquivariantFlowDistConfig
    aug_target_global_centering: bool
    aug_target_scale: float
    load_datasets: Callable[[int, int, int], Tuple[chex.Array, chex.Array]]
    lr: float
    batch_size: int
    max_global_norm: float
    K_marginal_log_lik: int
    logger: Logger
    seed: int
    n_plots: int
    n_eval: int
    n_checkpoints: int
    plot_batch_size: int
    plotter: Plotter = default_plotter
    reload_aug_per_epoch: bool = True
    train_set_size: Optional[int] = None
    test_set_size: Optional[int] = None
    save: bool = True
    save_dir: str = "/tmp"
    wandb_upload_each_time: bool = True
    scan_run: bool = True  # Set to False is useful for debugging.
    optimizer_name: str = 'adam'
    use_64_bit: bool = False


def setup_logger(cfg: DictConfig) -> Logger:
    if hasattr(cfg.logger, "wandb"):
        logger = WandbLogger(**cfg.logger.wandb, config=dict(cfg))
    elif hasattr(cfg.logger, "list_logger"):
        logger = ListLogger()
    else:
        raise Exception("No logger specified, try adding the wandb or "
                        "pandas logger to the config file.")
    return logger


def create_flow_config(flow_cfg: DictConfig):
    print(f"creating flow of type {flow_cfg.type}")
    flow_cfg = dict(flow_cfg)
    egnn_cfg = dict(flow_cfg.pop("egnn"))
    h_cfg = dict(egnn_cfg.pop("h"))
    transformer_cfg = dict(flow_cfg.pop("transformer"))
    transformer_config = TransformerConfig(**dict(transformer_cfg))
    egnn_cfg = EgnnConfig(**egnn_cfg, h_config=HConfig(**h_cfg))
    base_config = dict(flow_cfg.pop("base"))
    base_config = BaseConfig(**base_config)


    flow_dist_config = EquivariantFlowDistConfig(
        **flow_cfg,
        egnn_config=egnn_cfg,
        transformer_config=transformer_config,
        base_config=base_config
    )
    return flow_dist_config

def create_train_config(cfg: DictConfig, load_dataset, dim, n_nodes) -> TrainConfig:
    logger = setup_logger(cfg)

    training_config = dict(cfg.training)
    save_path = os.path.join(training_config.pop("save_dir"), str(datetime.now().isoformat()))
    if cfg.training.save:
        if hasattr(cfg.logger, "wandb"):
            # if using wandb then save to wandb path
            save_path = os.path.join(wandb.run.dir, save_path)
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    else:
        save_path = ''


    flow_config = create_flow_config(cfg.flow)

    experiment_config = TrainConfig(
        dim=dim,
        n_nodes=n_nodes,
        flow_dist_config=flow_config,
        load_datasets=load_dataset,
        **training_config,
        logger=logger,
        save_dir=save_path,
        aug_target_global_centering=cfg.target.aug_global_centering,
        aug_target_scale=cfg.target.aug_scale
    )
    return experiment_config


def train(config: TrainConfig):
    """Generic Training script."""
    if config.use_64_bit:
        jax.config.update("jax_enable_x64", True)

    assert config.flow_dist_config.dim == config.dim
    assert config.flow_dist_config.nodes == config.n_nodes

    if config.save:
        plots_dir = os.path.join(config.save_dir, f"plots")
        pathlib.Path(plots_dir).mkdir(exist_ok=False)
        checkpoints_dir = os.path.join(config.save_dir, f"model_checkpoints")
        pathlib.Path(checkpoints_dir).mkdir(exist_ok=False)
    else:
        plots_dir = None
        checkpoints_dir = None

    checkpoint_iter = list(np.linspace(0, config.n_epoch - 1, config.n_checkpoints, dtype="int"))
    eval_iter = list(np.linspace(0, config.n_epoch - 1, config.n_eval, dtype="int"))
    plot_iter = list(np.linspace(0, config.n_epoch - 1, config.n_plots, dtype="int"))

    @hk.without_apply_rng
    @hk.transform
    def log_prob_fn(x):
        distribution = make_equivariant_augmented_flow_dist(config.flow_dist_config)
        return distribution.log_prob(x)

    @hk.transform
    def sample_and_log_prob_fn(sample_shape=()):
        distribution = make_equivariant_augmented_flow_dist(config.flow_dist_config)
        return distribution.sample_and_log_prob(seed=hk.next_rng_key(), sample_shape=sample_shape)


    sample_fn = jax.jit(lambda params, key, shape: sample_and_log_prob_fn.apply(params, key, shape)[0],
                        static_argnums=2)


    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    params = log_prob_fn.init(rng=subkey, x=jnp.zeros((1, config.n_nodes, config.dim*2)))

    optimizer = optax.chain(optax.zero_nans(),
                            optax.clip_by_global_norm(config.max_global_norm if config.max_global_norm else jnp.inf),
                            getattr(optax, config.optimizer_name)(config.lr))
    opt_state = optimizer.init(params)

    train_data_original, test_data_original = config.load_datasets(config.batch_size, config.train_set_size,
                                                                   config.test_set_size)

    print(f"training data shape of {train_data_original.shape}")
    chex.assert_tree_shape_suffix(train_data_original, (config.n_nodes, config.dim))

    # Load augmented coordinates.
    key, subkey = jax.random.split(key)
    train_data = original_dataset_to_joint_dataset(train_data_original, subkey,
                                                   global_centering=config.aug_target_global_centering,
                                                   aug_scale=config.aug_target_scale)
    test_data = original_dataset_to_joint_dataset(test_data_original, subkey,
                                                  global_centering=config.aug_target_global_centering,
                                                  aug_scale=config.aug_target_scale)

    plot_and_maybe_save(config.plotter, params, sample_fn, key, config.plot_batch_size, train_data, test_data, 0,
                        config.save, plots_dir)

    if config.scan_run:
        def scan_fn(carry, xs):
            params, opt_state = carry
            x = xs
            params, opt_state, info = ml_step(params, x, opt_state, log_prob_fn, optimizer)
            return (params, opt_state), info

    pbar = tqdm(range(config.n_epoch))

    for i in pbar:
        key, subkey = jax.random.split(key)
        train_data = jax.random.permutation(subkey, train_data, axis=0)
        if config.reload_aug_per_epoch:
            key, subkey = jax.random.split(key)
            train_data = original_dataset_to_joint_dataset(train_data[..., :config.dim], subkey,
                                                           global_centering=config.aug_target_global_centering,
                                                           aug_scale=config.aug_target_scale)

        if config.scan_run:
            batched_data = jnp.reshape(train_data, (-1, config.batch_size, *train_data.shape[1:]))
            (params, opt_state), infos = jax.lax.scan(scan_fn, (params, opt_state), batched_data, unroll=1)

            for batch_index in range(batched_data.shape[0]):
                info = jax.tree_map(lambda x: x[batch_index], infos)
                info.update(epoch=i)
                config.logger.write(info)
                if jnp.isnan(info["grad_norm"]):
                    print("nan grad")
        else:
            for x in jnp.reshape(train_data, (-1, config.batch_size, *train_data.shape[1:])):
                params, opt_state, info = ml_step(params, x, opt_state, log_prob_fn, optimizer)
                config.logger.write(info)
                info.update(epoch=i)
                if jnp.isnan(info["grad_norm"]):
                    print("nan grad")

        if i in plot_iter:
            plot_and_maybe_save(config.plotter, params, sample_fn, key, config.plot_batch_size, train_data, test_data,
                                i + 1, config.save,
                                plots_dir)

        if i in eval_iter:
            key, subkey = jax.random.split(key)
            eval_info = eval_fn(params=params, x=test_data, flow_log_prob_fn=log_prob_fn,
                                flow_sample_and_log_prob_fn=sample_and_log_prob_fn,
                                global_centering=config.aug_target_global_centering,
                                aug_scale=config.aug_target_scale,
                                key=subkey,
                                batch_size=config.batch_size,
                                K=config.K_marginal_log_lik)
            pbar.write(str(eval_info))
            eval_info.update(epoch=i)
            config.logger.write(eval_info)

        if i in checkpoint_iter and config.save:
            checkpoint_path = os.path.join(checkpoints_dir, f"iter_{i}/")
            pathlib.Path(checkpoint_path).mkdir(exist_ok=False)
            with open(os.path.join(checkpoint_path, "state.pkl"), "wb") as f:
                pickle.dump(params, f)

    if isinstance(config.logger, ListLogger):
        plot_history(config.logger.history)
        plt.show()

    return config.logger, params, log_prob_fn, sample_and_log_prob_fn
