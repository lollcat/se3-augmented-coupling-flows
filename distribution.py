import distrax
import haiku as hk
import jax.random

from base import CentreGravityGaussian
from bijector_proj_real_nvp import make_se_equivariant_split_coupling_with_projection
from bijector_simple_real_nvp import make_se_equivariant_split_coupling_simple


def make_equivariant_augmented_flow_dist(dim, nodes, n_layers, type="proj"):
    base = CentreGravityGaussian(dim=int(dim*2), n_nodes=nodes)

    bijectors = []
    for i in range(n_layers):
        if type == "proj":
            bijector = make_se_equivariant_split_coupling_with_projection(dim, swap=i % 2 == 0)
        else:
            bijector = make_se_equivariant_split_coupling_simple(dim, swap=i % 2 == 0)
        bijectors.append(bijector)

    flow = distrax.Chain(bijectors)
    distribution = distrax.Transformed(base, flow)
    return distribution
