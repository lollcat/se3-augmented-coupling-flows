import distrax
import haiku as hk
import jax.random

from base import CentreGravityGaussian
from bijector_with_proj import make_se_equivariant_split_coupling


def make_equivariant_augmented_flow_dist(dim, nodes, n_layers):
    base = CentreGravityGaussian(dim=int(dim*2), n_nodes=nodes)

    bijectors = []
    for i in range(n_layers):
        bijector = make_se_equivariant_split_coupling(dim, swap=i % 2 == 0)
        bijectors.append(bijector)

    flow = distrax.Chain(bijectors)
    distribution = distrax.Transformed(base, flow)
    return distribution