from typing import NamedTuple, Optional

import chex

class DataPoint(NamedTuple):
    positions: chex.Array
    features: Optional[chex.Array] = None
