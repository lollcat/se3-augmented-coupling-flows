import jax
import chex

def get_from_first_device(nest: chex.ArrayTree, as_numpy: bool = True) -> chex.ArrayTree:
    # Copied from https://github.com/deepmind/acme/blob/d1e69c92000079b118b868ce9303ee6d39c4a0b6/acme/jax/utils.py#L368
    zeroth_nest = jax.tree_map(lambda x: x[0], nest)
    return jax.device_get(zeroth_nest) if as_numpy else zeroth_nest
