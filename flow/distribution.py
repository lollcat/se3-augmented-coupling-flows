import jax.numpy as jnp
import distrax
import haiku as hk

from flow.base import CentreGravityGaussian
from flow.bijector_proj_real_nvp import make_se_equivariant_split_coupling_with_projection
from flow.bijector_nice import make_se_equivariant_nice
from flow.bijector_scale_and_shift_along_vector import make_se_equivariant_vector_scale_shift


def make_equivariant_augmented_flow_dist(dim,
                                         nodes,
                                         n_layers,
                                         type="nice",
                                         flow_identity_init: bool = True,
                                         mlp_units = (10, 10)):
    base = CentreGravityGaussian(dim=int(dim*2), n_nodes=nodes)

    bijectors = []
    # bijectors.append(distrax.ScalarAffine(log_scale=hk.get_parameter("base_scale", shape=(), init=jnp.zeros),
    #                                       shift=jnp.zeros(dim*2)))
    for i in range(n_layers):
        swap = i % 2 == 0
        if type == "proj":
            assert dim == 2
            bijector = make_se_equivariant_split_coupling_with_projection(layer_number=i,
                                                dim=dim, swap=swap, identity_init=flow_identity_init,
                                                mlp_units=mlp_units)
        elif type == "nice":
            bijector = make_se_equivariant_nice(layer_number=i,
                                                dim=dim, swap=swap, identity_init=flow_identity_init,
                                                mlp_units=mlp_units)
        elif type == "vector_scale_shift":
            bijector = make_se_equivariant_vector_scale_shift(layer_number=i, dim=dim, swap=swap, identity_init=flow_identity_init,
                                                              mlp_units=mlp_units)
        else:
            raise NotImplemented
        bijectors.append(bijector)

    flow = distrax.Chain(bijectors)
    distribution = distrax.Transformed(base, flow)
    return distribution

