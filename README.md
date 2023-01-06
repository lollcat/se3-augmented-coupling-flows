# Equivariant Augmented Normalizing Flows
Exploring whether the idea works.
See `examples/dw4_v0.ipynb` for a training run on the double well problem with 4 particles. 

## Flows
The below flows are working
- Perform equivariant shift (like in the NICE paper). See `flow/bijector_nice.py`
- Perform equivariant shift, and an invariant scale along an equivariant vector. See `flow/bijector_scale_and_shift_along_vector`.

### Tests
Currently implemented in 2D.

- See `distribution_test.py` where we test
(1) the distribution log prob is invariant to rotation and translation, and
(2) the bijector composed of multiple flow layers is equivariant to rotation and translation. 
- additionally `base_test.py` tests the base distribution and each bijector `.py` file has its own tests for equivariance. 


## Further notes
- projected flow currently is numerically unstable


## TODO
 - Get proper metrics on dw4 problem
 - Clean
 - For NICE flow we will need some scaling transforms. This could be done with a scaling layer applied to the base distribution. 
 - Test jacobian determinant (test normalizing constant using IS?)
 - Think of how to make projected flow stable (can we projection axis to be orthogonal unit vectors?)
 - Generalise tests to more than 2D
