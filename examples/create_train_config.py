from typing import Tuple, Optional

import chex
import jax
import wandb
import matplotlib.pyplot as plt
import os
import pathlib
import haiku as hk
from datetime import datetime
from omegaconf import DictConfig
import matplotlib as mpl
from functools import partial
import jax.numpy as jnp

from molboil.train.base import get_shuffle_and_batchify_data_fn, create_scan_epoch_fn, eval_fn
from train.custom_step import training_step
from molboil.train.train import TrainConfig
from molboil.eval.base import get_eval_and_plot_fn

from flow.build_flow import build_flow, FlowDistConfig, ConditionalAuxDistConfig, BaseConfig
from flow.aug_flow_dist import FullGraphSample, AugmentedFlow, AugmentedFlowParams
from nets.base import NetsConfig, MLPHeadConfig, E3GNNTorsoConfig, EGNNTorsoConfig, TransformerConfig, E3TransformerTorsoConfig
from train.max_lik_train_and_eval import general_ml_loss_fn, get_eval_on_test_batch, eval_non_batched
from molboil.utils.loggers import Logger, WandbLogger, ListLogger, PandasLogger
from examples.default_plotter import make_default_plotter
from examples.configs import TrainingState
from utils.optimize import get_optimizer, OptimizerConfig
from utils.pmap import get_from_first_device

mpl.rcParams['figure.dpi'] = 150

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
    egnn_cfg = EGNNTorsoConfig(**dict(nets_config.pop("egnn"))) if "egnn" in nets_config.keys() else None
    e3gnn_config = E3GNNTorsoConfig(**dict(nets_config.pop("e3gnn"))) if "e3gnn" in nets_config.keys() else None
    e3transformer_config = E3TransformerTorsoConfig(**dict(nets_config.pop("e3transformer")))\
        if "e3transformer" in nets_config.keys() else None
    mlp_head_config = MLPHeadConfig(**nets_config.pop('mlp_head_config')) if "mlp_head_config" in \
                                                                             nets_config.keys() else None
    non_equivariant_transformer_config = TransformerConfig(**nets_config.pop('non_equivariant_transformer_config')) \
        if 'non_equivariant_transformer_config' in nets_config.keys() else None
    nets_config = NetsConfig(
                             **nets_config,
                             egnn_torso_config=egnn_cfg,
                             e3gnn_torso_config=e3gnn_config,
                             e3transformer_config=e3transformer_config,
                             mlp_head_config=mlp_head_config,
                             non_equivariant_transformer_config=non_equivariant_transformer_config
                             )
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

def create_train_config(cfg: DictConfig, load_dataset, dim, n_nodes,
                        plotter: Optional = None,
                        evaluation_fn: Optional = None,
                        eval_and_plot_fn: Optional = None,
                        date_folder: bool = True,
                        target_log_prob_fn: Optional = None) -> TrainConfig:
    devices = jax.devices()
    if len(devices) > 1:
        print(f"Running with pmap using {len(devices)} devices.")
        return create_train_config_pmap(cfg, load_dataset, dim, n_nodes, plotter, evaluation_fn, eval_and_plot_fn,
                                        date_folder, target_log_prob_fn)
    else:
        print(f"Running on one device only.")
        return create_train_config_non_pmap(cfg, load_dataset, dim, n_nodes, plotter, evaluation_fn, eval_and_plot_fn,
                                        date_folder, target_log_prob_fn)

def create_train_config_non_pmap(cfg: DictConfig, load_dataset, dim, n_nodes,
                        plotter: Optional = None,
                        evaluation_fn: Optional = None,
                        eval_and_plot_fn: Optional = None,
                        date_folder: bool = True,
                        target_log_prob_fn: Optional = None
    ) -> TrainConfig:
    """Creates `mol_boil` style train config"""
    assert cfg.flow.dim == dim
    assert cfg.flow.nodes == n_nodes
    assert (plotter is None or evaluation_fn is None) or (eval_and_plot_fn is None)


    training_config = dict(cfg.training)
    if date_folder:
        save_path = os.path.join(training_config.pop("save_dir"), str(datetime.now().isoformat()))
    else:
        save_path = training_config.pop("save_dir")
    pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)
    logger = setup_logger(cfg)

    train_data, test_data = load_dataset(cfg.training.train_set_size, cfg.training.test_set_size)
    batch_size = min(cfg.training.batch_size, train_data.positions.shape[0])
    flow_config = create_flow_config(cfg)
    flow = build_flow(flow_config)

    n_epoch = cfg.training.n_epoch
    if cfg.flow.type == 'non_equivariant' or 'non_equivariant' in cfg.flow.type:
        n_epoch = n_epoch * cfg.training.factor_to_train_non_eq_flow

    # Setup Optimizer.
    opt_cfg = dict(training_config.pop("optimizer"))
    n_iter_per_epoch = train_data.positions.shape[0] // batch_size
    n_iter_warmup = opt_cfg.pop('warmup_n_epoch')*n_iter_per_epoch
    n_iter_total = n_epoch * n_iter_per_epoch
    optimizer_config = OptimizerConfig(**opt_cfg,
                                       n_iter_total=n_iter_total,
                                       n_iter_warmup=n_iter_warmup)
    optimizer, lr = get_optimizer(optimizer_config)


    if plotter is None and eval_and_plot_fn is None:
        plotter = make_default_plotter(train_data,
                                       test_data,
                                       flow=flow,
                                       n_samples_from_flow=cfg.training.plot_batch_size,
                                       max_n_samples=1000,
                                       plotting_n_nodes=None
                                       )

    # Setup training functions.
    data_augmentation = (cfg.flow.type == 'non_equivariant' or 'non_equivariant' in
                         cfg.flow.type) and cfg.training.data_augmentation_for_non_eq
    if data_augmentation:
        print("using data augmentation")
    loss_fn = partial(general_ml_loss_fn,
                      flow=flow,
                      use_flow_aux_loss=cfg.training.use_flow_aux_loss,
                      aux_loss_weight=cfg.training.aux_loss_weight,
                      apply_random_rotation=data_augmentation)
    training_step_fn = partial(training_step, optimizer=optimizer, loss_fn=loss_fn,
                               verbose_info=cfg.training.verbose_info)
    scan_epoch_fn = create_scan_epoch_fn(training_step_fn,
                                         data=train_data,
                                         last_iter_info_only=cfg.training.last_iter_info_only,
                                         batch_size=batch_size)

    def init_fn(key: chex.PRNGKey) -> TrainingState:
        key1, key2 = jax.random.split(key)
        params = flow.init(key1, train_data[0])
        opt_state = optimizer.init(params)
        return TrainingState(params, opt_state, key2)

    def update_fn(state: TrainingState) -> Tuple[TrainingState, dict]:
        if not cfg.training.debug:
            params, opt_state, key, info = scan_epoch_fn(state.params, state.opt_state, state.key)
        else:  # If we want to debug.
            batchify_data = get_shuffle_and_batchify_data_fn(train_data, cfg.training.batch_size)
            params = state.params
            opt_state = state.opt_state
            key, subkey = jax.random.split(state.key)
            batched_data = batchify_data(subkey)
            for i in range(batched_data.positions.shape[0]):
                x = batched_data[i]
                key, subkey = jax.random.split(key)
                params, opt_state, info = training_step_fn(params, x, opt_state, subkey)
        return TrainingState(params, opt_state, key), info

    if evaluation_fn is None and eval_and_plot_fn is None:
        # Setup eval functions
        eval_on_test_batch_fn = partial(get_eval_on_test_batch,
                                        flow=flow, K=cfg.training.K_marginal_log_lik, test_invariances=True)
        if target_log_prob_fn:
            eval_batch_free_fn = partial(
                                    eval_non_batched,
                                    single_feature=test_data.features[0],
                                    flow=flow,
                                    n_samples=cfg.training.eval_model_samples,
                                    inner_batch_size=cfg.training.eval_batch_size,
                                    target_log_prob=target_log_prob_fn)
        else:
            eval_batch_free_fn = None

        def evaluation_fn(state: TrainingState, key: chex.PRNGKey) -> dict:
            eval_info = eval_fn(test_data, key, state.params,
                    eval_on_test_batch_fn=eval_on_test_batch_fn,
                    eval_batch_free_fn=eval_batch_free_fn,
                    batch_size=cfg.training.eval_batch_size)
            return eval_info

    if eval_and_plot_fn is None and (plotter is not None or evaluation_fn is not None):
        eval_and_plot_fn = get_eval_and_plot_fn(evaluation_fn, plotter)


    return TrainConfig(
        n_iteration=n_epoch,
        logger=logger,
        seed=cfg.training.seed,
        n_checkpoints=cfg.training.n_checkpoints,
        n_eval=cfg.training.n_eval,
        init_state=init_fn,
        update_state=update_fn,
        eval_and_plot_fn=eval_and_plot_fn,
        save=cfg.training.save,
        save_dir=save_path,
        resume=cfg.training.resume,
        use_64_bit=cfg.training.use_64_bit,
        runtime_limit=cfg.training.runtime_limit
                       )


def create_train_config_pmap(cfg: DictConfig, load_dataset, dim, n_nodes,
                        plotter: Optional = None,
                        evaluation_fn: Optional = None,
                        eval_and_plot_fn: Optional = None,
                        date_folder: bool = True,
                        target_log_prob_fn: Optional = None
                             ) -> TrainConfig:
    """Creates `mol_boil` style train config"""
    if target_log_prob_fn is not None:
        raise NotImplementedError # TODO still need to implement this
    devices = jax.devices()
    n_devices = len(devices)
    pmap_axis_name = 'data'

    assert cfg.flow.dim == dim
    assert cfg.flow.nodes == n_nodes
    assert (plotter is None or evaluation_fn is None) or (eval_and_plot_fn is None)

    training_config = dict(cfg.training)
    if date_folder:
        save_path = os.path.join(training_config.pop("save_dir"), str(datetime.now().isoformat()))
    else:
        save_path = training_config.pop("save_dir")
    pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)
    logger = setup_logger(cfg)

    train_data, test_data = load_dataset(cfg.training.train_set_size, cfg.training.test_set_size)
    flow_config = create_flow_config(cfg)
    flow = build_flow(flow_config)

    n_epoch = cfg.training.n_epoch
    if cfg.flow.type == 'non_equivariant' or 'non_equivariant' in cfg.flow.type:
        n_epoch = n_epoch * cfg.training.factor_to_train_non_eq_flow

    opt_cfg = dict(training_config.pop("optimizer"))
    n_iter_per_epoch = train_data.positions.shape[0] // (cfg.training.batch_size * n_devices)
    n_iter_warmup = opt_cfg.pop('warmup_n_epoch')*n_iter_per_epoch
    n_iter_total = n_epoch * n_iter_per_epoch
    optimizer_config = OptimizerConfig(**opt_cfg,
                                       n_iter_total=n_iter_total,
                                       n_iter_warmup=n_iter_warmup)
    optimizer, lr = get_optimizer(optimizer_config)


    if plotter is None and eval_and_plot_fn is None:
        plotter = make_default_plotter(train_data,
                                       test_data,
                                       flow=flow,
                                       n_samples_from_flow=cfg.training.plot_batch_size,
                                       max_n_samples=1000,
                                       plotting_n_nodes=None
                                       )


    # Setup training functions.
    loss_fn = partial(general_ml_loss_fn,
                      flow=flow,
                      use_flow_aux_loss=cfg.training.use_flow_aux_loss,
                      aux_loss_weight=cfg.training.aux_loss_weight)
    training_step_fn = partial(training_step, optimizer=optimizer, loss_fn=loss_fn, use_pmap=True,
                               pmap_axis_name=pmap_axis_name)


    def init_fn_single_devices(common_key: chex.PRNGKey, per_device_key: chex.PRNGKey) -> TrainingState:
        """Initialise the state. `common_key` ensures that the same initialisation is used for params on all devices."""
        params = flow.init(common_key, train_data[0])
        opt_state = optimizer.init(params)
        return TrainingState(params, opt_state, per_device_key)

    def init_fn(key: chex.PRNGKey) -> TrainingState:
        common_key, per_device_key = jax.random.split(key)
        common_keys = jnp.repeat(common_key[None, ...], n_devices, axis=0)
        per_device_keys = jax.random.split(per_device_key, n_devices)
        init_state = jax.pmap(init_fn_single_devices)(common_keys, per_device_keys)
        # Run check to ensure params are synched.
        chex.assert_trees_all_equal(jax.tree_map(lambda x: x[0], init_state.params),
                                    jax.tree_map(lambda x: x[1], init_state.params))
        assert (init_state.key[0] != init_state.key[1]).all()  # Check rng per state is different.
        return init_state

    data_rng_key_generator = hk.PRNGSequence(cfg.training.seed)

    def step_function(state: TrainingState, x: chex.ArrayTree) -> Tuple[TrainingState, dict]:
        key, subkey = jax.random.split(state.key)
        params, opt_state, info = training_step_fn(state.params, x, state.opt_state, subkey)
        return TrainingState(params=params, opt_state=opt_state, key=key), info

    def eval_function(state: TrainingState, x: chex.Array, key: chex.PRNGKey) -> dict:
        info = get_eval_on_test_batch(params=state.params,
                                        x_test=x,
                                        key=key,
                                        flow=flow,
                                        K = cfg.training.K_marginal_log_lik,
                                        test_invariances=True)
        return info


    def update_fn(state: TrainingState) -> Tuple[TrainingState, dict]:
        batchify_data = get_shuffle_and_batchify_data_fn(train_data, cfg.training.batch_size * n_devices)
        data_shuffle_key = next(data_rng_key_generator)  # Use separate key gen to avoid grabbing from state.
        batched_data = batchify_data(data_shuffle_key)

        for i in range(batched_data.positions.shape[0]):
            x = batched_data[i]
            # Reshape to [n_devices, batch_size]
            x = jax.tree_map(lambda x: jnp.reshape(x, (n_devices, cfg.training.batch_size, *x.shape[1:])), x)
            state, info = jax.pmap(step_function, axis_name=pmap_axis_name)(state, x)
        return state, get_from_first_device(info, as_numpy=False)

    if eval_and_plot_fn is not None:
        print("Running evaluation on 1 device only.")
        eval_and_plot_fn_single_device = eval_and_plot_fn
        def eval_and_plot_fn(state: chex.Array, *args) -> dict:
            # Perform evaluation on single device.
            return eval_and_plot_fn_single_device(get_from_first_device(state, as_numpy=False), *args)

    else:
        if evaluation_fn is None:
            def evaluation_fn(state: TrainingState, key: chex.PRNGKey) -> dict:
                batchify_data = get_shuffle_and_batchify_data_fn(test_data, cfg.training.eval_batch_size*n_devices)
                key, subkey = jax.random.split(key)
                batched_data = batchify_data(subkey)

                infos = []
                for i in range(batched_data.positions.shape[0]):
                    x = batched_data[i]
                    # Reshape to [n_devices, batch_size]
                    x = jax.tree_map(lambda x: jnp.reshape(x, (n_devices, cfg.training.eval_batch_size, *x.shape[1:])),
                                     x)
                    key, subkey = jax.random.split(key)
                    key_per_device = jax.random.split(subkey, n_devices)
                    info = jax.pmap(eval_function, axis_name=pmap_axis_name)(state, x, key_per_device)
                    infos.append(get_from_first_device(info))
                info = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs)), *infos)  # Aggregate over whole test set.
                return info

        # Plot single device.
        plotter_single_device = plotter
        def plotter(state: TrainingState, key: chex.PRNGKey):
            return plotter_single_device(get_from_first_device(state, as_numpy=False), key)

        eval_and_plot_fn = get_eval_and_plot_fn(evaluation_fn, plotter)


    return TrainConfig(
        n_iteration=n_epoch,
        logger=logger,
        seed=cfg.training.seed,
        n_checkpoints=cfg.training.n_checkpoints,
        n_eval=cfg.training.n_eval,
        init_state=init_fn,
        update_state=update_fn,
        eval_and_plot_fn=eval_and_plot_fn,
        save=cfg.training.save,
        save_dir=save_path,
        resume=cfg.training.resume,
        use_64_bit=cfg.training.use_64_bit,
        runtime_limit=cfg.training.runtime_limit
                       )
