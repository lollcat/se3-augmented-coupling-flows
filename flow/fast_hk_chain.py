"""Copy distrax Bijector chain but use hk scan to make it fast to compile."""

from typing import List, Sequence, Tuple

from distrax._src.bijectors import bijector as base
from distrax._src.utils import conversion

Array = base.Array
BijectorLike = base.BijectorLike
BijectorT = base.BijectorT


class Chain(base.Bijector):

  def __init__(self, bijector: BijectorLike, n_layers):
    self._bijector = bijector
    self._n_layers = n_layers

    # Check that neighboring bijectors in the chain have compatible dimensions
    for i, (outer, inner) in enumerate(zip(self._bijectors[:-1],
                                           self._bijectors[1:])):
      if outer.event_ndims_in != inner.event_ndims_out:
        raise ValueError(
            f"The chain of bijector event shapes are incompatible. Bijector "
            f"{i} ({outer.name}) expects events with {outer.event_ndims_in} "
            f"dimensions, while Bijector {i+1} ({inner.name}) produces events "
            f"with {inner.event_ndims_out} dimensions.")

    is_constant_jacobian = all(b.is_constant_jacobian for b in self._bijectors)
    is_constant_log_det = all(b.is_constant_log_det for b in self._bijectors)
    super().__init__(
        event_ndims_in=self._bijectors[-1].event_ndims_in,
        event_ndims_out=self._bijectors[0].event_ndims_out,
        is_constant_jacobian=is_constant_jacobian,
        is_constant_log_det=is_constant_log_det)

  @property
  def bijectors(self) -> List[BijectorT]:
    """The list of bijectors in the chain."""
    return self._bijectors

  def forward(self, x: Array) -> Array:
    """Computes y = f(x)."""
    for bijector in reversed(self._bijectors):
      x = bijector.forward(x)
    return x

  def inverse(self, y: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    for bijector in self._bijectors:
      y = bijector.inverse(y)
    return y

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    x, log_det = self._bijectors[-1].forward_and_log_det(x)
    for bijector in reversed(self._bijectors[:-1]):
      x, ld = bijector.forward_and_log_det(x)
      log_det += ld
    return x, log_det

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    y, log_det = self._bijectors[0].inverse_and_log_det(y)
    for bijector in self._bijectors[1:]:
      y, ld = bijector.inverse_and_log_det(y)
      log_det += ld
    return y, log_det

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    if type(other) is Chain:  # pylint: disable=unidiomatic-typecheck
      if len(self.bijectors) != len(other.bijectors):
        return False
      for bij1, bij2 in zip(self.bijectors, other.bijectors):
        if not bij1.same_as(bij2):
          return False
      return True
    elif len(self.bijectors) == 1:
      return self.bijectors[0].same_as(other)

    return False

