from typing import Callable, Tuple, Optional

import chex
import jax
import optax

from molboil.base import FullGraphSample
from molboil.train.base import get_tree_leaf_norm_info
from utils.optimize import IgnoreNanOptState

Params = chex.ArrayTree

# TODO: Port optimizer stuff to Molboil

def training_step(
    params: Params,
    x: FullGraphSample,
    opt_state: optax.OptState,
    key: chex.PRNGKey,
    optimizer: optax.GradientTransformation,
    loss_fn: Callable[
        [chex.PRNGKey, chex.ArrayTree, FullGraphSample], Tuple[chex.Array, dict]
    ],
    verbose_info: Optional[bool] = False,
    use_pmap: bool = False,
    pmap_axis_name: str = 'data'
) -> Tuple[Params, optax.OptState, dict]:
    """Compute loss and gradients and update model parameters.

    Args:
        params (AugmentedFlowParams): _description_
        x (FullGraphSample): _description_
        opt_state (optax.OptState): _description_
        key (chex.PRNGKey): _description_
        optimizer (optax.GradientTransformation): _description_
        loss_fn
        verbose_info
        use_pmap: whether the training step function is pmapped, such that gradient aggregation is needed.
        pmap_axis_name: name of axis for gradient aggregation across devices.


    Returns:
        Tuple[AugmentedFlowParams, optax.OptState, dict]: _description_
    """

    grad, info = jax.grad(loss_fn, has_aux=True, argnums=1)(
        key, params, x, verbose_info
    )
    if use_pmap:
        grad = jax.lax.pmean(grad, axis_name=pmap_axis_name)
    updates, new_opt_state = optimizer.update(grad, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    info.update(
        grad_norm=optax.global_norm(grad),
        update_norm=optax.global_norm(updates),
        param_norm=optax.global_norm(params),
    )

    if verbose_info:
        info.update(
            {
                "grad_" + key: value
                for key, value in get_tree_leaf_norm_info(grad).items()
            }
        )
        info.update(
            {
                "update_" + key: value
                for key, value in get_tree_leaf_norm_info(updates).items()
            }
        )
    if isinstance(opt_state, IgnoreNanOptState):
        info.update(ignored_grad_count=opt_state.ignored_grads_count,
                    total_optimizer_steps=opt_state.total_steps)
    return new_params, new_opt_state, info
