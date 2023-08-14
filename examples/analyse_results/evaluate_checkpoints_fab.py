import jax.random
from hydra import compose, initialize
import hydra
import pathlib

from examples.create_train_config import setup_logger
from examples.load_flow_and_checkpoint import load_flow
from examples.analyse_results.get_wandb_runs import download_checkpoint
from examples.dw4_fab import load_dataset_original as load_ds_dw4
from examples.dw4_fab import log_prob_fn as log_prob_fn_dw4
from examples.lj13_fab import load_dataset as load_ds_lj13
from examples.lj13_fab import log_prob_fn as log_prob_fn_lj13


from typing import Union

import chex
import jax
from functools import partial
import jax.numpy as jnp

from molboil.train.base import eval_fn
from train.max_lik_train_and_eval import get_eval_on_test_batch_with_further

from fabjax.sampling import build_smc, build_blackjax_hmc, build_metropolis
from train.fab_train_no_buffer import TrainStateNoBuffer
from train.fab_train_with_buffer import TrainStateWithBuffer
from train.fab_eval import fab_eval_function

_BASE_DIR = '../..'

problems = ["dw4_fab", "lj13_fab"]
hydra_configs = ["dw4_fab.yaml", "lj13_fab.yaml"]


def get_setup_info(problem: str):
    tags = ["cblgpu", "fab"]
    hydra_config = problem + ".yaml"
    if problem == "dw4_fab":
        tags.append("dw4")
        tags.append("post_sub1")
        max_iter = 20000
    else:
        assert problem == "lj13_fab"
        tags.append("lj13")
        tags.append("post_sub")
        max_iter = 14000
    return tags, hydra_config, max_iter


def setup_eval(cfg, flow, target_log_p_x_fn, test_data):
    use_hmc = cfg.fab.use_hmc
    n_intermediate_distributions = cfg.fab.n_intermediate_distributions
    spacing_type = cfg.fab.spacing_type

    # Setup training functions.
    dim_total = int(flow.dim_x*(flow.n_augmented+1)*test_data.features.shape[-2])
    if use_hmc:
        transition_operator = build_blackjax_hmc(dim=dim_total, **cfg.fab.transition_operator.hmc)
    else:
        transition_operator = build_metropolis(dim_total, **cfg.fab.transition_operator.metropolis)

    eval_on_test_batch_fn = partial(get_eval_on_test_batch_with_further,
                                    flow=flow, K=cfg.training.K_marginal_log_lik, test_invariances=True,
                                    target_log_prob=target_log_p_x_fn)

    # AIS with p as the target. Note that step size params will have been tuned for alpha=2.
    smc_eval = build_smc(transition_operator=transition_operator,
                         n_intermediate_distributions=n_intermediate_distributions, spacing_type=spacing_type,
                         alpha=1., use_resampling=False)

    def further_fn(log_w: chex.Array, mask: chex.Array) -> dict:
        chex.assert_equal_shape((log_w, mask))
        log_w = jnp.where(mask, log_w, jnp.zeros_like(log_w))  # make sure log_w finite
        log_z_inv = jax.nn.logsumexp(-log_w, b=mask) - jnp.log(jnp.sum(mask))
        log_z_expectation_p_over_q = jax.nn.logsumexp(log_w, b=mask) - jnp.log(jnp.sum(mask))
        log_forward_ess = - log_z_inv - log_z_expectation_p_over_q
        forward_ess = jnp.exp(log_forward_ess)
        info = {}
        info.update(forward_ess=forward_ess)
        return info

    def evaluation_fn(state: Union[TrainStateNoBuffer, TrainStateWithBuffer], key: chex.PRNGKey) -> dict:
        eval_info = eval_fn(test_data, key, state.params,
                            eval_on_test_batch_fn=eval_on_test_batch_fn,
                            eval_batch_free_fn=None,
                            batch_size=cfg.training.eval_batch_size,
                            further_processing_fn=further_fn)
        eval_info_fab = fab_eval_function(
            state=state, key=key, flow=flow,
            smc=smc_eval,
            log_p_x=target_log_p_x_fn,
            features=test_data.features[0],
            batch_size=cfg.fab.eval_total_batch_size,
            inner_batch_size=cfg.fab.eval_inner_batch_size
        )
        eval_info.update(eval_info_fab)
        return eval_info
    return evaluation_fn


def download_checkpoint_and_eval(problem, seed, flow_type):
    print(f"evaluating checkpoint for {problem} {seed} {flow_type}")

    # Setup
    tags, hydra_config, max_iter = get_setup_info(problem)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=f"{_BASE_DIR}/examples/config/")
    cfg = compose(config_name=hydra_config)
    cfg.flow.type = flow_type
    cfg.training.seed = seed

    load_dataset = load_ds_dw4 if problem == "dw4_fab" else load_ds_lj13
    target_log_p_x_fn = log_prob_fn_dw4 if problem == "dw4_fab" else log_prob_fn_lj13

    base_dir = f'./examples/analyse_results/{hydra_config[:-4]}/models'
    pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

    # Download checkpoint from WANDB.
    download_checkpoint(flow_type=flow_type, tags=tags, seed=seed, max_iter=max_iter,
                        base_path=base_dir)
    print("checkpoint downloaded")

    checkpoint_path = f"examples/analyse_results/{hydra_config[:-4]}/models/{flow_type}_seed{seed}.pkl"

    flow, state = load_flow(cfg, checkpoint_path)
    if problem == "lj13_fab":
        state = jax.tree_map(lambda x: x[0], state)  # for lj13 we used multiple devices.
        print("loaded checkpoint from first device")
    else:
        print("loaded checkpoint (single device)")

    debug = False
    if debug:
        cfg.training.test_set_size = 10
        cfg.training.eval_model_samples = 100
        cfg.training.eval_batch_size = 10
        cfg.training.K_marginal_log_lik = 2

    train_data, test_data = load_dataset(cfg.training.train_set_size, cfg.training.test_set_size)

    eval_fn = setup_eval(cfg=cfg, flow=flow, target_log_p_x_fn=target_log_p_x_fn, test_data=test_data)

    key = jax.random.PRNGKey(0)

    cfg.logger.wandb.tags = [problem, "evaluation", "eval_3", "with_fwd_ess"]
    cfg.logger.wandb.name = problem + "_evaluation"
    logger = setup_logger(cfg)
    info = eval_fn(state, key)
    logger.write(info)
    print(info)
    logger.close()
    print("evaluation complete")


if __name__ == '__main__':
    # download_checkpoint_and_eval(problem="dw4_fab", seed=0, flow_type="spherical")
    for flow_type in ["spherical"]: # ["along_vector", "spherical", "proj", "non_equivariant"]:
        for seed in [0, 1, 2]:
            download_checkpoint_and_eval(problem="lj13_fab", seed=seed, flow_type=flow_type)
