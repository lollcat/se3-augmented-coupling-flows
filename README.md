# Equivariant Augmented Normalizing Flows
Currently TODO: Big update to networks to make them more expressive (following equivariant flow paper more closely)


Exploring whether the idea works.
See `example/dw4_v0.py` for a training run on the double well problem with 4 particles - this can be run locally in under
a minute to get some prelim results. 
See `examples/dw4.ipynb` for training a bit longer, I run this on GPU. 

See `examples/lj13.ipynb` for the LJ 13 problem (I also run this on GPU). 

## Flows
The below flows are working
- Perform equivariant shift (like in the NICE paper). See `flow/bijector_nice.py`
- Perform equivariant shift, and an invariant scale along an equivariant vector. See `flow/bijector_scale_and_shift_along_vector`.


The `flow/bijector_proj_realnvp.py` flow is currently numerically unstable and not fit for use. 

### Tests
Currently implemented in 2D.

- See `distribution_test.py` where we test
(1) the distribution log prob is invariant to rotation and translation, and
(2) the bijector composed of multiple flow layers is equivariant to rotation and translation. 
- additionally `base_test.py` tests the base distribution and each bijector `.py` file has its own tests for equivariance.


## TODO
 - Improve flow expresiveness: more augmented coordinates, transformer in EGNN
 - For NICE flow we will need some scaling transforms. This could be done with a scaling layer applied to the base distribution. 
 - Test jacobian determinant (test normalizing constant using IS?)
 - Add equivariant ActNorm type layers
 - Make projected flow stable. 
 - QM9
