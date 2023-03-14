import chex
import jax.numpy as jnp

from flow.aug_flow_dist import FullGraphSample

def positional_dataset_only_to_full_graph(positions: chex.Array) -> FullGraphSample:
    """Convert positional dataset into full graph by using zeros for features. Assumes data is only for x, and not
    augmented coordinates."""
    chex.assert_rank(positions, 3)  # [n_data_points, n_nodes, dim]
    features = jnp.zeros((*positions.shape[:-1], 1))
    return FullGraphSample(positions=positions, features=features)
