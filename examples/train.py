from typing import Callable, Tuple, Optional, List, NamedTuple

import chex
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
from functools import partial

from molboil.train.base import get_shuffle_and_batchify_data_fn, create_scan_epoch_fn, training_step, eval_fn

from flow.build_flow import build_flow, FlowDistConfig, ConditionalAuxDistConfig
from flow.aug_flow_dist import FullGraphSample, AugmentedFlow, AugmentedFlowParams
from nets.base import NetsConfig, MLPHeadConfig, EnTransformerTorsoConfig, E3GNNTorsoConfig, EgnnTorsoConfig, \
    MACETorsoConfig
from nets.transformer import TransformerConfig
from utils.plotting import plot_history
from utils.aug_flow_train_and_eval import general_ml_loss_fn, get_eval_on_test_batch
from utils.graph import get_senders_and_receivers_fully_connected
from utils.loggers import Logger, WandbLogger, ListLogger, PandasLogger


mpl.rcParams['figure.dpi'] = 150
TestData = chex.Array
TrainData = chex.Array
PlottingBatchSize = int
Plotter = Callable[[AugmentedFlowParams, AugmentedFlow, chex.PRNGKey, PlottingBatchSize,
                    TestData, TrainData], List[plt.Figure]]

def plot_sample_hist(samples,
                     ax,
                     n_vertices: Optional[int] = None,
                     max_distance = 10, *args, **kwargs):
    """n_vertices argument allows us to look at pairwise distances for subset of vertices,
    to prevent plotting taking too long"""
    n_vertices = samples.shape[1] if n_vertices is None else n_vertices
    n_vertices = min(samples.shape[1], n_vertices)
    senders, receivers = get_senders_and_receivers_fully_connected(n_nodes=n_vertices)
    norms = jnp.linalg.norm(samples[:, senders] - samples[:, receivers], axis=-1)
    d = norms.flatten()
    d = d[jnp.isfinite(d)]
    d = d.clip(max=max_distance)  # Clip keep plot reasonable.
    ax.hist(d, bins=50, density=True, alpha=0.4, *args, **kwargs)


def default_plotter(params: AugmentedFlowParams,
                    flow: AugmentedFlow,
                    key: chex.PRNGKey,
                    n_samples: int,
                    train_data: FullGraphSample,
                    test_data: FullGraphSample,
                    plotting_n_nodes: Optional[int] = None):

    # Plot interatomic distance histograms.
    key1, key2 = jax.random.split(key)
    joint_samples_flow = jax.jit(flow.sample_apply, static_argnums=3)(params, train_data.features[0], key1,
                                                                      (n_samples,))
    features, positions_x, positions_a = jax.jit(flow.joint_to_separate_samples)(joint_samples_flow)
    positions_x_target = test_data.positions[:n_samples]
    positions_a_target = jax.jit(flow.aux_target_sample_n_apply)(params.aux_target, test_data[:n_samples], key2)

    # Plot original coords
    fig1, axs = plt.subplots(1, 2, figsize=(10, 5))
    plot_sample_hist(positions_x, axs[0], label="flow samples", n_vertices=plotting_n_nodes)
    plot_sample_hist(positions_x, axs[1], label="flow samples", n_vertices=plotting_n_nodes)
    plot_sample_hist(train_data.positions[:n_samples], axs[0],  label="train samples", n_vertices=plotting_n_nodes)
    plot_sample_hist(test_data.positions[:n_samples], axs[1],  label="test samples", n_vertices=plotting_n_nodes)

    axs[0].set_title(f"norms between original coordinates train")
    axs[1].set_title(f"norms between original coordinates test")
    axs[0].legend()
    axs[1].legend()
    fig1.tight_layout()

    # Augmented info.
    fig2, axs2 = plt.subplots(1, flow.n_augmented, figsize=(5*flow.n_augmented, 5))
    axs2 = [axs2] if isinstance(axs2, plt.Subplot) else axs2
    for i in range(flow.n_augmented):
        positions_a_single = positions_a[:, :, i]  # get single group of augmented coordinates
        positions_a_target_single = positions_a_target[:, :, i]  # Get first set of aux variables.
        chex.assert_equal_shape((positions_x, positions_a_single, positions_a_target_single))
        plot_sample_hist(positions_a_single, axs2[i], label="flow samples", n_vertices=plotting_n_nodes)
        plot_sample_hist(positions_a_target_single, axs2[i], label="test samples", n_vertices=plotting_n_nodes)
        axs2[i].set_title(f"norms between augmented coordinates (aug group {i})")
    axs2[0].legend()
    fig2.tight_layout()

    # Plot histogram (x - a)

    fig3, axs3 = plt.subplots(1, flow.n_augmented, figsize=(5*flow.n_augmented, 5))
    axs3 = [axs3] if isinstance(axs3, plt.Subplot) else axs3
    for i in range(flow.n_augmented):
        positions_a_single = positions_a[:, :, i]  # get single group of augmented coordinates
        positions_a_target_single = positions_a_target[:, :, i]  # Get first set of aux variables.
        plot_sample_hist(positions_x - positions_a_single, axs3[i], label="flow samples", n_vertices=plotting_n_nodes)
        plot_sample_hist(positions_x_target - positions_a_target_single, axs3[i], label="test samples", n_vertices=plotting_n_nodes)
        axs3[i].set_title(f"norms between graph of x - a (aug group {i}). ")
    axs3[0].legend()
    fig3.tight_layout()

    return [fig1, fig2, fig3]


def plot_and_maybe_save(plotter,
                        params: AugmentedFlowParams,
                        flow: AugmentedFlow,
                        key: chex.PRNGKey,
                        plot_batch_size: int,
                        train_data: FullGraphSample,
                        test_data: FullGraphSample,
                        epoch_n: int,
                        save: bool,
                        plots_dir: str
                        ):
    figures = plotter(params, flow, key, plot_batch_size, train_data, test_data)
    for j, figure in enumerate(figures):
        if save:
            figure.savefig(os.path.join(plots_dir, f"{j}_iter_{epoch_n}.png"))
        else:
            plt.show()
        plt.close(figure)


class OptimizerConfig(NamedTuple):
    init_lr: float
    use_schedule: bool
    optimizer_name: str = "adam"
    max_global_norm: Optional[float] = None
    peak_lr: Optional[float] = None
    end_lr: Optional[float] = None
    warmup_n_epoch: Optional[int] = None


def get_optimizer_and_step_fn(optimizer_config: OptimizerConfig, n_iter_per_epoch, total_n_epoch):
    if optimizer_config.use_schedule:
        lr = optax.warmup_cosine_decay_schedule(
            init_value=float(optimizer_config.init_lr),
            peak_value=float(optimizer_config.peak_lr),
            end_value=float(optimizer_config.end_lr),
            warmup_steps=optimizer_config.warmup_n_epoch * n_iter_per_epoch,
            decay_steps=n_iter_per_epoch*total_n_epoch
                                                     )
    else:
        lr = float(optimizer_config.init_lr)

    grad_transforms = [optax.zero_nans()]

    if optimizer_config.max_global_norm:
        clipping_fn = optax.clip_by_global_norm(float(optimizer_config.max_global_norm))
        grad_transforms.append(clipping_fn)
    else:
        pass

    grad_transforms.append(getattr(optax, optimizer_config.optimizer_name)(lr))
    optimizer = optax.chain(*grad_transforms)
    return optimizer, lr


class TrainConfig(NamedTuple):
    n_epoch: int
    dim: int
    n_nodes: int
    flow_dist_config: FlowDistConfig
    load_datasets: Callable[[int, int, int], Tuple[FullGraphSample, FullGraphSample]]
    optimizer_config: OptimizerConfig
    batch_size: int
    K_marginal_log_lik: int
    logger: Logger
    seed: int
    n_plots: int
    n_eval: int
    n_checkpoints: int
    plot_batch_size: int
    use_flow_aux_loss: bool = False
    aux_loss_weight: float = 1.0
    plotter: Plotter = default_plotter
    train_set_size: Optional[int] = None
    test_set_size: Optional[int] = None
    save: bool = True
    save_dir: str = "/tmp"
    wandb_upload_each_time: bool = True
    scan_run: bool = True  # Set to False is useful for debugging.
    use_64_bit: bool = False
    with_train_info: bool = True  # Grab info from the flow during each forward pass.
    # Only log the info from the last iteration within each epoch. Reduces runtime a lot if an epoch has many iter.
    last_iter_info_only: bool = True


def setup_logger(cfg: DictConfig) -> Logger:
    if hasattr(cfg.logger, "wandb"):
        logger = WandbLogger(**cfg.logger.wandb, config=dict(cfg))
    elif hasattr(cfg.logger, "list_logger"):
        logger = ListLogger()
    elif hasattr(cfg.logger, 'pandas_logger'):
        logger = PandasLogger(save_path=cfg.training.save_dir, save_period=cfg.logger.pandas_logger.save_period,
                              save=cfg.training.save)
    else:
        raise Exception("No logger specified, try adding the wandb or "
                        "pandas logger to the config file.")
    return logger

def create_nets_config(nets_config: DictConfig):
    """Configure nets (MACE, EGNN, Transformer, MLP)."""
    nets_config = dict(nets_config)
    egnn_cfg = EgnnTorsoConfig(**dict(nets_config.pop("egnn"))) if "egnn" in nets_config.keys() else None
    e3gnn_config = E3GNNTorsoConfig(**dict(nets_config.pop("e3gnn"))) if "e3gnn" in nets_config.keys() else None
    mace_config = MACETorsoConfig(**dict(nets_config.pop("mace"))) if "mace" in nets_config.keys() else None
    e3transformer_cfg = EnTransformerTorsoConfig(**dict(nets_config.pop("e3transformer"))) if "e3transformer" in nets_config.keys() else None
    transformer_cfg = dict(nets_config.pop("transformer")) if "transformer" in nets_config.keys() else None
    transformer_config = TransformerConfig(**dict(transformer_cfg)) if transformer_cfg else None
    mlp_head_config = MLPHeadConfig(**nets_config.pop('mlp_head_config')) if "mlp_head_config" in \
                                                                             nets_config.keys() else None
    type = nets_config['type']
    nets_config = NetsConfig(type=type,
                             egnn_torso_config=egnn_cfg,
                             e3gnn_torso_config=e3gnn_config,
                             mace_torso_config=mace_config,
                             e3transformer_lay_config=e3transformer_cfg,
                             transformer_config=transformer_config,
                             mlp_head_config=mlp_head_config)
    return nets_config

def create_flow_config(cfg: DictConfig):
    """Create config for building the flow."""
    flow_cfg = cfg.flow
    print(f"creating flow of type {flow_cfg.type}")
    flow_cfg = dict(flow_cfg)
    nets_config = create_nets_config(flow_cfg.pop("nets"))
    base_aux_config = dict(flow_cfg.pop("base").aux)
    base_aux_config = ConditionalAuxDistConfig(**base_aux_config)
    target_aux_config = ConditionalAuxDistConfig(**dict(cfg.target.aux))
    flow_dist_config = FlowDistConfig(
        **flow_cfg,
        nets_config=nets_config,
        base_aux_config=base_aux_config,
        target_aux_config=target_aux_config
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


    flow_config = create_flow_config(cfg)

    optimizer_config = OptimizerConfig(**dict(training_config.pop("optimizer")))

    experiment_config = TrainConfig(
        dim=dim,
        n_nodes=n_nodes,
        flow_dist_config=flow_config,
        load_datasets=load_dataset,
        optimizer_config=optimizer_config,
        **training_config,
        logger=logger,
        save_dir=save_path,
    )
    return experiment_config


def train(config: TrainConfig):
    """Generic Training script."""
    if config.use_64_bit:
        jax.config.update("jax_enable_x64", True)

    assert config.flow_dist_config.dim == config.dim
    assert config.flow_dist_config.nodes == config.n_nodes

    if config.save:
        pathlib.Path(config.save_dir).mkdir(exist_ok=True)
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

    train_data, test_data = config.load_datasets(config.batch_size, config.train_set_size,
                                                                   config.test_set_size)

    # Define flow, and initialise params.
    flow_dist = build_flow(config.flow_dist_config)

    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    params = flow_dist.init(subkey, train_data[0:2])
    params_test = flow_dist.init(subkey, train_data[0])
    chex.assert_trees_all_equal_shapes(params, params_test)



    print(f"training data position shape of {train_data.positions.shape}, "
          f"feature shape of {train_data.features.shape}")
    chex.assert_tree_shape_suffix(train_data.positions, (config.n_nodes, config.dim))

    plot_and_maybe_save(config.plotter, params, flow_dist, key, config.plot_batch_size, train_data, test_data, 0,
                        config.save, plots_dir)

    optimizer, lr = get_optimizer_and_step_fn(config.optimizer_config,
                                              n_iter_per_epoch=train_data.positions.shape[0] // config.batch_size,
                                              total_n_epoch=config.n_epoch)
    opt_state = optimizer.init(params)


    pbar = tqdm(range(config.n_epoch))


    shuffle_and_batchify_data = get_shuffle_and_batchify_data_fn(train_data, batch_size=config.batch_size)


    loss_fn = partial(general_ml_loss_fn,
                      flow=flow_dist,
                      use_flow_aux_loss=config.use_flow_aux_loss,
                      aux_loss_weight=config.aux_loss_weight)
    training_step_fn = partial(training_step, optimizer=optimizer, loss_fn=loss_fn)
    scan_epoch_fn = create_scan_epoch_fn(training_step_fn, config.last_iter_info_only)

    eval_on_test_batch_fn = partial(get_eval_on_test_batch,
                                    flow=flow_dist, K=config.K_marginal_log_lik, test_invariances=True)
    eval_batch_free_fn = None


    for i in pbar:
        key, subkey = jax.random.split(key)
        # TODO: In boiler plate we can move this to happen internal to scan_epoch_fn.
        batched_data = shuffle_and_batchify_data(subkey)

        if config.scan_run:
            key, subkey = jax.random.split(key)
            params, opt_state, key, info_out = scan_epoch_fn(params, opt_state, subkey, batched_data)
            if config.last_iter_info_only:
                info = info_out
                info.update(epoch=i)
                info.update(n_optimizer_steps=opt_state[-1][0].count)
                if hasattr(lr, "__call__"):
                    info.update(lr=lr(info["n_optimizer_steps"]))
                config.logger.write(info)
                if jnp.isnan(info["grad_norm"]):
                    print("nan grad")
            else:
                for batch_index in range(batched_data.positions.shape[0]):
                    info = jax.tree_map(lambda x: x[batch_index], info_out)
                    info.update(epoch=i)
                    info.update(n_optimizer_steps=opt_state[-1][0].count)
                    if hasattr(lr, "__call__"):
                        info.update(lr=lr(info["n_optimizer_steps"]))
                    config.logger.write(info)
                    if jnp.isnan(info["grad_norm"]):
                        print("nan grad")
        else:
            for i in range(batched_data.positions.shape[0]):
                x = batched_data[i]
                key, subkey = jax.random.split(key)
                params, opt_state, info = training_step_fn(params, x, opt_state, subkey)
                config.logger.write(info)
                info.update(epoch=i)
                if jnp.isnan(info["grad_norm"]):
                    print("nan grad")

        if i in plot_iter:
            plot_and_maybe_save(config.plotter, params, flow_dist, key, config.plot_batch_size, train_data, test_data,
                                i + 1, config.save,
                                plots_dir)

        if i in eval_iter:
            key, subkey = jax.random.split(key)
            eval_info = eval_fn(test_data, subkey, params,
                                eval_on_test_batch_fn=eval_on_test_batch_fn,
                                eval_batch_free_fn=eval_batch_free_fn,
                                batch_size=config.batch_size)
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

    config.logger.close()
    return config.logger, params, flow_dist
