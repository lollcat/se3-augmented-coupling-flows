from typing import Tuple

from omegaconf import DictConfig
import pickle
from examples.create_train_config import create_flow_config, AugmentedFlow, TrainingState
from examples.create_train_config import build_flow

def load_flow(cfg: DictConfig, checkpoint_path: str) -> Tuple[AugmentedFlow, TrainingState]:
    flow_config = create_flow_config(cfg)
    flow = build_flow(flow_config)
    state = pickle.load(open(checkpoint_path, "rb"))
    return flow, state