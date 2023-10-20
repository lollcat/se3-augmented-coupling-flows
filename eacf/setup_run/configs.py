from typing import NamedTuple

import chex
import optax

from eacf.flow.aug_flow_dist import AugmentedFlowParams

class TrainingState(NamedTuple):
    params: AugmentedFlowParams
    opt_state: optax.OptState
    key: chex.PRNGKey

