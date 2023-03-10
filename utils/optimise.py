import jax

def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]


def line_search_by_jvp(f,
                       func_params,
                       grad_direction,
                       opt_update,
                       damping=1e-6):

    # this is the raw NN gradient
    ravelled_grad_direction, _ = jax.flatten_util.ravel_pytree(grad_direction)
    # this is the Adam proposed direction
    ravelled_update, _ = jax.flatten_util.ravel_pytree(opt_update)

    # HVP against Adam solution
    hvp_direction = hvp(f, (func_params,), (opt_update,))

    # flattening HVP
    ravelled_hvp_direction, _ = jax.flatten_util.ravel_pytree(
        hvp_direction)

    # solving for multiplicative constant on the direction vector
    step_size = -((ravelled_grad_direction.T @ ravelled_update)
                  / (ravelled_update.T @ ravelled_hvp_direction
                     + damping * ravelled_update.T @ ravelled_update))
    return step_size