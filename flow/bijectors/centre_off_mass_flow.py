from typing import Callable, Union, Optional, Tuple

import chex
import distrax
import jax.numpy as jnp
import haiku as hk

from flow.distrax_with_extra import BijectorWithExtra
from flow.bijectors.batch import BatchBijector

BijectorParams = Union[chex.Array]

class CentreOfMassFlow(BijectorWithExtra):
    """A flow that acts on only the centre of masses."""
    def __init__(self,
                 conditioner: Callable[[chex.Array], BijectorParams],
                 bijector: Callable[[BijectorParams], Union[BijectorWithExtra, distrax.Bijector]]
                 ):
        super().__init__(event_ndims_in=3,
                         event_ndims_out=3)
        self.conditioner = conditioner
        self.bijector = bijector

    def split_to_zero_mean_point_clouds_and_centre_of_masses(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        chex.assert_rank(x, 3)
        n_nodes, multiplicity, dim = x.shape
        n_augmented = multiplicity - 1

        centre_of_mass_x0 = jnp.mean(x[:, 0], axis=0, keepdims=True)[:, None, :]
        x = x - centre_of_mass_x0  # Make sure everything is relative to x0 (the non-augmented variable).
        x0 = x[:, 0:1]
        chex.assert_shape(x0, (n_nodes, 1, dim))

        # Now pull out the centre of masses as separate.
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
        x = centred_point_clouds.at[:, 1:].add(centre_of_masses_augmented)
        return x

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        chex.assert_rank(x, 3)
        chex.assert_rank(x, 3)
        n_nodes, multiplicity, dim = x.shape
        n_augmented = multiplicity - 1

        # Split off centre of masses into separate representation.
        centred_point_clouds, centre_of_masses_augmented = \
            self.split_to_zero_mean_point_clouds_and_centre_of_masses(x)

        # Get bijector.
        bijector_params = self.conditioner(centred_point_clouds)
        bijector = self.bijector(bijector_params)

        # Pass through bijector.
        centre_of_masses_augmented_new, log_det = bijector.forward_and_log_det(centre_of_masses_augmented)
        chex.assert_shape(log_det, (n_augmented, dim))
        log_det = jnp.sum(log_det)

        # Return to original representation.
        y = self.recombine(centred_point_clouds=centred_point_clouds,
                           centre_of_masses_augmented=centre_of_masses_augmented_new)
        return y, log_det



    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        chex.assert_rank(y, 3)
        chex.assert_rank(y, 3)
        n_nodes, multiplicity, dim = y.shape
        n_augmented = multiplicity - 1

        # Split off centre of masses into separate representation.
        centred_point_clouds, centre_of_masses_augmented = \
            self.split_to_zero_mean_point_clouds_and_centre_of_masses(y)

        # Get bijector.
        bijector_params = self.conditioner(centred_point_clouds)
        bijector = self.bijector(bijector_params)

        # Pass through bijector.
        centre_of_masses_augmented_new, log_det = bijector.inverse_and_log_det(centre_of_masses_augmented)
        chex.assert_shape(log_det, (n_augmented, dim))
        log_det = jnp.sum(log_det)

        # Return to original representation.
        x = self.recombine(centred_point_clouds=centred_point_clouds,
                           centre_of_masses_augmented=centre_of_masses_augmented_new)
        return x, log_det



def build_unconditional_centre_of_mass_layer(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        identity_init: bool) -> BijectorWithExtra:

    def bijector_fn(params):
        chex.assert_shape(params, (n_aug, dim))
        log_scaling = params
        inner_bijector = distrax.ScalarAffine(log_scale=log_scaling, shift=jnp.zeros_like(log_scaling))
        return inner_bijector

    def conditioner(zero_mean_point_clouds: chex.Array) -> chex.Array:
        del(zero_mean_point_clouds)  # Unconditioned.

        log_scale = hk.get_parameter(name=f'centre_of_mass_shifting_lay{layer_number}',
                                            shape=(n_aug,),
                                            init=jnp.zeros if identity_init else hk.initializers.Constant(1.))
        log_scale = jnp.repeat(log_scale[:, None], dim, axis=-1)
        return log_scale

    bijector = CentreOfMassFlow(conditioner=conditioner, bijector=bijector_fn)
    return BatchBijector(bijector)
