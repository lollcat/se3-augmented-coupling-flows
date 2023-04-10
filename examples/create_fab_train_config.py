from typing import Tuple, Optional

import chex
import wandb
import os
import pathlib
from datetime import datetime
from omegaconf import DictConfig
from functools import partial

from molboil.train.base import eval_fn
from molboil.train.train import TrainConfig
from molboil.eval.base import get_eval_and_plot_fn


from flow.build_flow import build_flow
from examples.default_plotter import make_default_plotter
from examples.configs import TrainingState, OptimizerConfig
from train.max_lik_train_and_eval import get_eval_on_test_batch
from examples.create_train_config import setup_logger, create_flow_config, get_optimizer

from fabjax.sampling import build_ais, build_blackjax_hmc, build_metropolis
from train.fab_train import build_fab_no_buffer_init_step_fns


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
    optimizer_config = OptimizerConfig(**dict(training_config.pop("optimizer")))
    optimizer, lr = get_optimizer(optimizer_config,
                                  n_iter_per_epoch=1,
                                  total_n_epoch=cfg.training.n_epoch)


    if plotter is None and eval_and_plot_fn is None:
        plotter = make_default_plotter(train_data,
                                       test_data,
                                       flow=flow,
                                       n_samples_from_flow=cfg.training.plot_batch_size,
                                       max_n_samples=1000,
                                       plotting_n_nodes=None
                                       )

    # Setup training functions.
    dim_total = int(flow.dim_x*(flow.n_augmented+1)*train_data.features.shape[-2])
    if use_hmc:
        transition_operator = build_blackjax_hmc(dim=dim_total, n_outer_steps=hmc_n_outer_steps,
                                                     init_step_size=hmc_init_step_size, target_p_accept=target_p_accept,
                                                     adapt_step_size=tune_step_size)
    else:
        transition_operator = build_metropolis(dim_total, metro_n_outer_steps, metro_init_step_size)
    ais = build_ais(transition_operator=transition_operator,
                    n_intermediate_distributions=n_intermediate_distributions, spacing_type=spacing_type,
                    alpha=alpha)

    features = train_data.features[0]
    init_fn, update_fn = build_fab_no_buffer_init_step_fns(
        flow=flow,log_p_x=target_log_p_x_fn, features=features,
        ais=ais, optimizer=optimizer, batch_size=cfg.training.batch_size,
    )

    if evaluation_fn is None and eval_and_plot_fn is None:
        # Setup eval functions
        eval_on_test_batch_fn = partial(get_eval_on_test_batch,
                                        flow=flow, K=cfg.training.K_marginal_log_lik, test_invariances=True)
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
