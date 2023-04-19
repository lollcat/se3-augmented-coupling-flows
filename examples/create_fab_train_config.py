from typing import Tuple, Optional, Union

import chex
import wandb
import os
import pathlib
import jax
from datetime import datetime
from omegaconf import DictConfig
from functools import partial

from molboil.train.base import eval_fn
from molboil.train.train import TrainConfig
from molboil.eval.base import get_eval_and_plot_fn


from flow.build_flow import build_flow
from examples.default_plotter_fab import make_default_plotter
from train.max_lik_train_and_eval import get_eval_on_test_batch
from examples.create_train_config import setup_logger, create_flow_config
from utils.optimize import get_optimizer, OptimizerConfig

from fabjax.sampling import build_smc, build_blackjax_hmc, build_metropolis
from fabjax.buffer import build_prioritised_buffer
from train.fab_train_no_buffer import build_fab_no_buffer_init_step_fns, TrainStateNoBuffer
from train.fab_train_with_buffer import build_fab_with_buffer_init_step_fns, TrainStateWithBuffer
from train.fab_eval import fab_eval_function


def create_train_config(cfg: DictConfig, target_log_p_x_fn, load_dataset, dim, n_nodes,
                        plotter: Optional = None,
                        evaluation_fn: Optional = None,
                        eval_and_plot_fn: Optional = None,
                        date_folder: bool = True) -> TrainConfig:
    """Creates `mol_boil` style train config"""
    # AIS.
    use_hmc = cfg.fab.use_hmc
    hmc_n_outer_steps = cfg.fab.transition_operator.hmc.n_outer_steps
    hmc_init_step_size = cfg.fab.transition_operator.hmc.init_step_size
    target_p_accept = cfg.fab.transition_operator.hmc.target_p_accept
    tune_step_size = cfg.fab.transition_operator.hmc.tune_step_size

    metro_n_outer_steps = cfg.fab.transition_operator.metropolis.n_outer_steps
    metro_init_step_size = cfg.fab.transition_operator.metropolis.init_step_size
    alpha = cfg.fab.alpha
    n_intermediate_distributions = cfg.fab.n_intermediate_distributions
    spacing_type = cfg.fab.spacing_type



    assert cfg.flow.dim == dim
    assert cfg.flow.nodes == n_nodes
    assert (plotter is None or evaluation_fn is None) or (eval_and_plot_fn is None)

    logger = setup_logger(cfg)
    training_config = dict(cfg.training)
    if date_folder:
        save_path = os.path.join(training_config.pop("save_dir"), str(datetime.now().isoformat()))
    else:
        save_path = training_config.pop("save_dir")
    if cfg.training.save:
        if hasattr(cfg.logger, "wandb"):
            # if using wandb then save to wandb path
            save_path = os.path.join(wandb.run.dir, save_path)
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    else:
        save_path = ''
    train_data, test_data = load_dataset(cfg.training.train_set_size, cfg.training.test_set_size)
    flow_config = create_flow_config(cfg)
    flow = build_flow(flow_config)


    opt_cfg = dict(training_config.pop("optimizer"))
    n_iter_warmup = opt_cfg.pop('warmup_n_epoch')
    optimizer_config = OptimizerConfig(**opt_cfg,
                                       n_iter_total=cfg.training.n_epoch,
                                       n_iter_warmup=n_iter_warmup)
    optimizer, lr = get_optimizer(optimizer_config)


    # Setup training functions.
    dim_total = int(flow.dim_x*(flow.n_augmented+1)*train_data.features.shape[-2])
    if use_hmc:
        transition_operator = build_blackjax_hmc(dim=dim_total, n_outer_steps=hmc_n_outer_steps,
                                                     init_step_size=hmc_init_step_size, target_p_accept=target_p_accept,
                                                     adapt_step_size=tune_step_size)
    else:
        transition_operator = build_metropolis(dim_total, metro_n_outer_steps, metro_init_step_size,
                                               tune_step_size=tune_step_size, target_p_accept=target_p_accept)
    smc = build_smc(transition_operator=transition_operator,
                    n_intermediate_distributions=n_intermediate_distributions, spacing_type=spacing_type,
                    alpha=alpha, use_resampling=cfg.fab.use_resampling)

    if plotter is None and eval_and_plot_fn is None:
        plotter = make_default_plotter(
                test_data=test_data,
                flow=flow,
                ais=smc,
                log_p_x=target_log_p_x_fn,
                n_samples_from_flow=cfg.training.plot_batch_size,
                max_n_samples=1000,
                max_distance=20.)

    features = train_data.features[0]

    if cfg.fab.with_buffer:
        print("running fab with buffer")
        assert cfg.fab.n_updates_per_smc_forward_pass*cfg.training.batch_size <= cfg.fab.buffer_min_length
        buffer = build_prioritised_buffer(dim=dim_total, max_length=cfg.fab.buffer_max_length,
                                          min_length_to_sample=cfg.fab.buffer_min_length)
        init_fn, update_fn = build_fab_with_buffer_init_step_fns(
            flow=flow, log_p_x=target_log_p_x_fn, features=features,
            smc=smc, optimizer=optimizer,
            batch_size=cfg.training.batch_size,
            n_updates_per_smc_forward_pass=cfg.fab.n_updates_per_smc_forward_pass,
            buffer=buffer
        )
    else:
        print("running fab without buffer")
        init_fn, update_fn = build_fab_no_buffer_init_step_fns(
            flow=flow, log_p_x=target_log_p_x_fn, features=features,
            smc=smc, optimizer=optimizer, batch_size=cfg.training.batch_size,
        )

    if evaluation_fn is None and eval_and_plot_fn is None:
        # Setup eval functions
        eval_on_test_batch_fn = partial(get_eval_on_test_batch,
                                        flow=flow, K=cfg.training.K_marginal_log_lik, test_invariances=True)

        # AIS with p as the target. Note that step size params will have been tuned for alpha=2.
        smc_eval = build_smc(transition_operator=transition_operator,
                        n_intermediate_distributions=n_intermediate_distributions, spacing_type=spacing_type,
                        alpha=1., use_resampling=False)

        @jax.jit
        def evaluation_fn(state: Union[TrainStateNoBuffer, TrainStateWithBuffer], key: chex.PRNGKey) -> dict:
            eval_info = eval_fn(test_data, key, state.params,
                    eval_on_test_batch_fn=eval_on_test_batch_fn,
                    eval_batch_free_fn=None,
                    batch_size=cfg.training.eval_batch_size)
            eval_info_fab = fab_eval_function(
                state=state, key=key, flow=flow,
                smc=smc_eval,
                log_p_x=target_log_p_x_fn,
                features=train_data.features[0],
                batch_size=cfg.fab.eval_total_batch_size,
                inner_batch_size=cfg.fab.eval_inner_batch_size
            )
            eval_info.update(eval_info_fab)
            return eval_info

    if eval_and_plot_fn is None and (plotter is not None or evaluation_fn is not None):
        eval_and_plot_fn = get_eval_and_plot_fn(evaluation_fn, plotter)


    return TrainConfig(
        n_iteration=cfg.training.n_epoch,
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
