import distrax

from flow.bijectors.bijector_coupled_global_scaling import make_coupled_global_scaling
from flow.bijectors.bijector_coupled_global_shift import make_global_shift_layer
from flow.bijectors.bijector_coupled_particle_pair_shift import make_per_particle_pair_shift_layer


def make_pseudo_act_norm_bijector(layer_number, dim, flow_identity_init):
    bijectors = []
    bijectors.append(make_per_particle_pair_shift_layer(layer_number=layer_number, swap=True, dim=dim,
                                             identity_init=flow_identity_init))
    bijectors.append(make_per_particle_pair_shift_layer(layer_number=layer_number, swap=False, dim=dim,
                                             identity_init=flow_identity_init))
    bijectors.append(make_coupled_global_scaling(layer_number=layer_number, swap=True, dim=dim,
                                                 identity_init=flow_identity_init))
    bijectors.append(make_coupled_global_scaling(layer_number=layer_number, swap=False, dim=dim,
                                                 identity_init=flow_identity_init))
    bijectors.append(make_global_shift_layer(layer_number=layer_number, swap=True, dim=dim,
                                             identity_init=flow_identity_init))
    bijectors.append(make_global_shift_layer(layer_number=layer_number, swap=False, dim=dim,
                                             identity_init=flow_identity_init))
    return distrax.Chain(bijectors)