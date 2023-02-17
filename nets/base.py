from typing import NamedTuple, Optional

from nets.mace import MACELayerConfig
from nets.en_gnn import EgnnConfig
from nets.transformer import TransformerConfig

class NetsConfig(NamedTuple):
    use_mace: bool
    mace_lay_config: Optional[MACELayerConfig] = None
    egnn_lay_config: Optional[EgnnConfig] = None
    transformer_config: Optional[TransformerConfig] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_mace and self.mace_lay_config is None:
            raise Exception
        if not self.use_mace and self.egnn_lay_config is None:
            raise Exception