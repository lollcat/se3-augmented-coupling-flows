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
from molboil.train.train import TrainConfig

from flow.build_flow import build_flow, FlowDistConfig, ConditionalAuxDistConfig, BaseConfig
from flow.aug_flow_dist import FullGraphSample, AugmentedFlow, AugmentedFlowParams
from nets.base import NetsConfig, MLPHeadConfig, EnTransformerTorsoConfig, E3GNNTorsoConfig, EgnnTorsoConfig, \
    MACETorsoConfig
from nets.transformer import TransformerConfig
from utils.aug_flow_train_and_eval import general_ml_loss_fn, get_eval_on_test_batch
from utils.loggers import Logger, WandbLogger, ListLogger, PandasLogger
from examples.default_plotter import default_plotter, FLowPlotter

mpl.rcParams['figure.dpi'] = 150


class TrainingState(NamedTuple):
    params: AugmentedFlowParams
    opt_state: optax.OptState
    key: chex.PRNGKey


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


class FlowTrainConfig(NamedTuple):
    n_epoch: int
    dim: int
    n_nodes: int
    flow_dist_config: FlowDistConfig
    load_datasets: Callable[[int, int], Tuple[FullGraphSample, FullGraphSample]]
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
    train_set_size: Optional[int] = None
    test_set_size: Optional[int] = None
    save: bool = True
    save_dir: str = "/tmp"
    wandb_upload_each_time: bool = True
    plotter: FLowPlotter = default_plotter
    debug: bool = False  # Set to False is useful for debugging.
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
    base_config = dict(flow_cfg.pop("base"))
    base_aux_config = base_config.pop("aux")
    base_aux_config = ConditionalAuxDistConfig(**base_aux_config)
    base_config = BaseConfig(**base_config, aug=base_aux_config)
    target_aux_config = ConditionalAuxDistConfig(**dict(cfg.target.aux))
    flow_dist_config = FlowDistConfig(
        **flow_cfg,
        nets_config=nets_config,
        base=base_config,
        target_aux_config=target_aux_config
    )
    return flow_dist_config


def create_flow_train_config(cfg: DictConfig, load_dataset, dim, n_nodes, plotter: FLowPlotter = default_plotter) -> \
        FlowTrainConfig:
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

    config = FlowTrainConfig(
        dim=dim,
        n_nodes=n_nodes,
        flow_dist_config=flow_config,
        load_datasets=load_dataset,
        optimizer_config=optimizer_config,
        plotter=plotter,
        **training_config,
        logger=logger,
        save_dir=save_path,
    )
    return config


def create_train_config(cfg: DictConfig, load_dataset, dim, n_nodes, plotter: Optional[FLowPlotter] = default_plotter):
    config = create_flow_train_config(cfg, load_dataset, dim, n_nodes, plotter)
    assert config.flow_dist_config.dim == config.dim
    assert config.flow_dist_config.nodes == config.n_nodes

    train_data, test_data = config.load_datasets(config.train_set_size, config.test_set_size)
    optimizer, lr = get_optimizer_and_step_fn(config.optimizer_config,
                                              n_iter_per_epoch=train_data.positions.shape[0] // config.batch_size,
                                              total_n_epoch=config.n_epoch)
    flow = build_flow(config.flow_dist_config)

    # Setup training functions.
    loss_fn = partial(general_ml_loss_fn,
                      flow=flow,
                      use_flow_aux_loss=config.use_flow_aux_loss,
                      aux_loss_weight=config.aux_loss_weight)
    training_step_fn = partial(training_step, optimizer=optimizer, loss_fn=loss_fn)

    scan_epoch_fn = create_scan_epoch_fn(training_step_fn,
                                         data=train_data,
                                         last_iter_info_only=config.last_iter_info_only,
                                         batch_size=config.batch_size)

    # Setup eval functions
    eval_on_test_batch_fn = partial(get_eval_on_test_batch,
                                    flow=flow, K=config.K_marginal_log_lik, test_invariances=True)
    eval_batch_free_fn = None


    def init_fn(key: chex.PRNGKey) -> TrainingState:
        key1, key2 = jax.random.split(key)
        params = flow.init(key1, train_data[0])
        opt_state = optimizer.init(params)
        return TrainingState(params, opt_state, key2)

    def update_fn(state: TrainingState) -> Tuple[TrainingState, dict]:
        if not config.debug:
            params, opt_state, key, info = scan_epoch_fn(state.params, state.opt_state, state.key)
        else:  # If we want to debug.
            batchify_data = get_shuffle_and_batchify_data_fn(train_data, config.batch_size)
            params = state.params
            opt_state = state.opt_state
            key, subkey = jax.random.split(state.key)
            batched_data = batchify_data(subkey)
            for i in range(batched_data.positions.shape[0]):
                x = batched_data[i]
                key, subkey = jax.random.split(key)
                params, opt_state, info = training_step_fn(params, x, opt_state, subkey)
        return TrainingState(params, opt_state, key), info

    def evaluation_fn(state: TrainingState, key: chex.PRNGKey) -> dict:
        eval_info = eval_fn(test_data, key, state.params,
                eval_on_test_batch_fn=eval_on_test_batch_fn,
                eval_batch_free_fn=eval_batch_free_fn,
                batch_size=config.batch_size)
        return eval_info

    def plotter_fn(state: TrainingState, key: chex.PRNGKey) -> List[plt.Figure]:
        return config.plotter(state.params, flow, key, config.plot_batch_size, train_data,
                                   test_data)

    return TrainConfig(
        n_iteration=config.n_epoch,
        logger=config.logger,
        seed=config.seed,
        n_checkpoints=config.n_checkpoints,
        n_eval=config.n_eval,
        n_plot=config.n_plots,
        plotter=plotter_fn,
        init_state=init_fn,
        update_state=update_fn,
        eval_state=evaluation_fn,
        save=config.save,
        save_dir=config.save_dir,
        use_64_bit=config.use_64_bit
                       )