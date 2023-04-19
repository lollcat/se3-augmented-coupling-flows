from typing import Union

import chex
from fabjax.sampling.smc import SequentialMonteCarloSampler

from flow.aug_flow_dist import AugmentedFlow, GraphFeatures
from train.fab_train_no_buffer import TrainStateNoBuffer, build_smc_forward_pass
from train.fab_train_with_buffer import TrainStateWithBuffer


def fab_eval_function(state: Union[TrainStateNoBuffer, TrainStateWithBuffer],
                      key: chex.PRNGKey,
                      flow: AugmentedFlow,
                      smc: SequentialMonteCarloSampler,
                      log_p_x,
                      features: GraphFeatures,
                      batch_size: int) -> dict:
    """Evaluate the ESS of the flow, and AIS. """
    assert smc.alpha == 1.  # Make sure target is set to p.
    assert smc.use_resampling is False  # Make sure we are doing AIS, not SMC.

    smc_forward = build_smc_forward_pass(flow, log_p_x, features, smc, batch_size)


    sample_flow, x_smc, log_w, log_q, smc_state, smc_info = smc_forward(state.params, state.smc_state, key)
    info = {}
    info.update(eval_ess_flow=smc_info['ess_q_p'],
                eval_ess_ais=smc_info['ess_smc_final'])
    return info
