from typing import Protocol, Tuple

import chex

class EquivariantForwardFunction(Protocol):
    """A callable type"""

    def __call__(
        self,
        positions: chex.Array,
        node_features: chex.Array,
        senders: chex.Array,
        receivers: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """The `init` function.
        Args:
          positions: [n_nodes, multiplicity, 3]-chex.Array containing the euclidean coordinates of each node.
          node_features: [n_nodes, n_node_features]-chex.Array containing invariant features for each node.
          senders: [n_messages,]-chex.Array containing the integer sender indices for each message
          receivers: [n_messages,]-chex.Array containing the integer receiver indices for each message
        Returns:
          vectors: [n_nodes, n_vectors, 3]-chex.Array containing rotation equivariant but translation invariant vectors.
          scalars: [n_nodes, n_scalars]-chex.Array containing invariant features.
        """


