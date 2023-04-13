import distrax
import jax.numpy as jnp

from flow.bijectors.zero_mean_bijector import ZeroMeanBijector


def tesst_zero_mean_bijector():
    """Check that bijector that transforms ONLY the centre of mass with a non-0 log det, has a 0 log det when
    we join it to the zero-mean bijector.
    """
    pass
    # TODO
