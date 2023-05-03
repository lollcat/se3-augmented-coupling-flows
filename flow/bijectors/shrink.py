from typing import Callable, Union, Optional, Tuple

import chex
import distrax
import jax.nn
import jax.numpy as jnp
import haiku as hk

from nets.conditioner_mlp import ConditionerHead
from flow.distrax_with_extra import BijectorWithExtra
from flow.bijectors.batch import BatchBijector
from utils.numerical import inverse_softplus, safe_norm

BijectorParams = Union[chex.Array]

class IsotropicScalingFlow(BijectorWithExtra):
    """A flow that scales the zero-mean sets of coordinates. I.e. acts on the x coords, and momentum coords."""
    def __init__(self,
                 conditioner: Callable[[chex.Array], BijectorParams],
                 n_aug: int,
                 dim: int,
                 ):
        super().__init__(event_ndims_in=3,
                         event_ndims_out=3)
        self.conditioner = conditioner
        self.dim = dim
        self.n_aug = n_aug

    def get_scale_from_logit(self, logit: chex.Array) -> chex.Array:
        """Compute the scaling factor (forced to be positive)."""
        chex.assert_shape(logit, (self.n_aug + 1,))

        # If params is initialised to 0 then this initialises the flow to the identity.
        log_scaling = logit + inverse_softplus(jnp.array(1.))
        scale = jax.nn.softplus(log_scaling)

        return scale

    def split_to_zero_mean_point_clouds_and_centre_of_masses(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        chex.assert_rank(x, 3)
        n_nodes, multiplicity, dim = x.shape
        n_augmented = multiplicity - 1

        x0 = x[:, 0:1]
        chex.assert_shape(x0, (n_nodes, 1, dim))

        # Pull out the centre of masses as separate.
        augmented_variables = x[:, 1:]
        centre_of_masses_augmented = jnp.mean(augmented_variables, axis=0)
        chex.assert_shape(centre_of_masses_augmented, (n_augmented, dim))
        centred_augmented = augmented_variables - centre_of_masses_augmented[None]

        centred_point_clouds = jnp.concatenate([x0, centred_augmented], axis=-2)
        return centred_point_clouds, centre_of_masses_augmented

    def recombine(self, centred_point_clouds: chex.Array, centre_of_masses_augmented: chex.Array) -> chex.Array:
        """Recombine `centred_point_clouds` and `centre_of_masses_augmented` into a single array."""
        chex.assert_rank(centred_point_clouds, 3)
        n_nodes, multiplicity, dim = centred_point_clouds.shape
        n_augmented = multiplicity - 1
        chex.assert_shape(centre_of_masses_augmented, (n_augmented, dim))

        # Add the centre of masses back into the array.
        x = centred_point_clouds.at[:, 1:].add(centre_of_masses_augmented[None])
        return x

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        chex.assert_rank(x, 3)
        chex.assert_rank(x, 3)
        n_nodes, multiplicity, dim = x.shape
        n_augmented = multiplicity - 1

        # Split off centre of masses into separate representation.
        centred_point_clouds, centre_of_masses_augmented = \
            self.split_to_zero_mean_point_clouds_and_centre_of_masses(x)

        # Get scale.
        scale_logit = self.conditioner(centre_of_masses_augmented)
        scale = self.get_scale_from_logit(scale_logit)

        # Pass through bijection and compute log det.
        # Note `scale` is of shape [multiplicity,],
        # while `centred_point_clouds` is of shape [n_nodes, multiplicity, dim]
        # hence the broadcasting.
        centred_point_clouds_new = centred_point_clouds * scale[None, :, None]
        log_det = (n_nodes - 1) * self.dim * jnp.sum(jnp.log(scale))

        # Return to original representation.
        y = self.recombine(centred_point_clouds=centred_point_clouds_new,
                           centre_of_masses_augmented=centre_of_masses_augmented)
        return y, log_det


    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        chex.assert_rank(y, 3)
        chex.assert_rank(y, 3)
        n_nodes, multiplicity, dim = y.shape
        n_augmented = multiplicity - 1

        # Split off centre of masses into separate representation.
        centred_point_clouds, centre_of_masses_augmented = \
            self.split_to_zero_mean_point_clouds_and_centre_of_masses(y)

        # Get scale.
        scale_logit = self.conditioner(centre_of_masses_augmented)
        scale = self.get_scale_from_logit(scale_logit)

        # Pass through bijection and compute log det.
        # Note `scale` is of shape [multiplicity,],
        # while `centred_point_clouds` is of shape [n_nodes, multiplicity, dim]
        # hence the broadcasting.
        centred_point_clouds_new = centred_point_clouds / scale[None, :, None]
        log_det = - (n_nodes - 1) * self.dim * jnp.sum(jnp.log(scale))

        # Return to original representation.
        y = self.recombine(centred_point_clouds=centred_point_clouds_new,
                           centre_of_masses_augmented=centre_of_masses_augmented)
        return y, log_det



def build_shrink_layer(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        identity_init: bool,
        condition: bool = True
) -> BijectorWithExtra:
    """Perform isotropic scaling of the zero-mean variables, conditioning on the augmented centre of masses."""
    # TODO: Could use graph features by passing them through a transformer.

    mlp_units: Tuple[int, ...] = (16, 16)  # Fix as a small MLP.

    def conditioner(centre_of_masses_augmented: chex.Array) -> chex.Array:
        chex.assert_shape(centre_of_masses_augmented, (n_aug, dim))

        if condition:
            norms = safe_norm(centre_of_masses_augmented, axis=-1)  # Use norms to stay equivariant.
            mlp = ConditionerHead(name=f'shrinkage_lay_{layer_number}', n_output_params=n_aug+1,
                                  zero_init=identity_init, mlp_units=mlp_units)
            log_scale = jnp.squeeze(mlp(norms[None, :]), axis=0)
        else:
            log_scale = hk.get_parameter(name=f'shrinkage_lay_{layer_number}',
                                         shape=(n_aug + 1,),
                                         init=jnp.zeros if identity_init else hk.initializers.VarianceScaling(1.),
                                         dtype=float
                                         )
        chex.assert_shape(log_scale, (n_aug+1,))
        return log_scale

    bijector = IsotropicScalingFlow(conditioner=conditioner, n_aug=n_aug, dim=dim)
    return BatchBijector(bijector)
