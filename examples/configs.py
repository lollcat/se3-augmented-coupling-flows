from typing import Optional, NamedTuple

import chex
import optax

from flow.aug_flow_dist import AugmentedFlowParams

class TrainingState(NamedTuple):
    params: AugmentedFlowParams
    opt_state: optax.OptState
    key: chex.PRNGKey


class OptimizerConfig(NamedTuple):
    init_lr: float
    use_schedule: bool
    optimizer_name: str = "adam"
    max_global_norm: Optional[float] = None
    peak_lr: Optional[float] = None
    end_lr: Optional[float] = None
    warmup_n_epoch: Optional[int] = None

