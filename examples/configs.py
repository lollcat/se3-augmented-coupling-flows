from typing import Callable, Tuple, Optional, NamedTuple

import chex
import optax

from flow.build_flow import FlowDistConfig
from flow.aug_flow_dist import FullGraphSample, AugmentedFlowParams
from utils.loggers import Logger

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
    debug: bool = False  # Set to False is useful for debugging.
    use_64_bit: bool = False
    with_train_info: bool = True  # Grab info from the flow during each forward pass.
    # Only log the info from the last iteration within each epoch. Reduces runtime a lot if an epoch has many iter.
    last_iter_info_only: bool = True
