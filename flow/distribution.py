import distrax

from flow.base import CentreGravityGaussian
from flow.bijector_proj_real_nvp import make_se_equivariant_split_coupling_with_projection
from flow.bijector_nice import make_se_equivariant_nice


def make_equivariant_augmented_flow_dist(dim,
                                         nodes,
                                         n_layers,
                                         type="proj",
                                         flow_identity_init: bool = True,
                                         mlp_units = (10, 10)):
    base = CentreGravityGaussian(dim=int(dim*2), n_nodes=nodes)

    bijectors = []
    for i in range(n_layers):
        swap = i % 2 == 0
        if type == "proj":
            bijector = make_se_equivariant_split_coupling_with_projection(dim, swap=swap,
                                                                          identity_init=flow_identity_init,
                                                                          mlp_units=mlp_units)
        elif type == "nice":
            bijector = make_se_equivariant_nice(dim, swap=swap, identity_init=flow_identity_init, mlp_units=mlp_units)
        else:
            raise NotImplemented
        bijectors.append(bijector)

    flow = distrax.Chain(bijectors)
    distribution = distrax.Transformed(base, flow)
    return distribution
