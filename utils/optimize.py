from typing import NamedTuple, Tuple, Optional

import chex
import jax.lax
import optax
import jax.numpy as jnp


class IgnoreNanOptState(NamedTuple):
    opt_state: optax.OptState
    grad_norms: chex.Array
    ignored_grads_count: chex.Array = jnp.array(0, dtype=int)

def dynamic_update_ignore_and_grad_norm_clip(optimizer: optax.GradientTransformation,
                                             window_length: int = 100,
                                             factor_clip_norm: float = 5.,
                                             factor_allowable_norm: float = 20.) -> \
        optax.GradientTransformation:

    def init(params: chex.ArrayTree) -> IgnoreNanOptState:
        opt_state = optimizer.init(params)
        grad_norms = jnp.zeros(window_length)
        grad_norms = grad_norms.at[0:3].set(1e30)
        return IgnoreNanOptState(opt_state=opt_state, grad_norms=grad_norms)

    def update(grad: chex.ArrayTree, opt_state: IgnoreNanOptState, params: chex.ArrayTree) ->\
            Tuple[chex.ArrayTree, IgnoreNanOptState]:
        grad_norm = optax.global_norm(grad)
        grad_median = jnp.nanmedian(opt_state.grad_norms)
        skip_update = (grad_norm > grad_median * factor_allowable_norm) | (~jnp.isfinite(grad_norm))

        # Dynamic global norm clipping.
        global_norm_clip = optax.clip_by_global_norm(grad_median*factor_clip_norm)
        global_norm_clip_state = global_norm_clip.init(params)
        grad = global_norm_clip.update(grad, global_norm_clip_state)[0]

        updates, new_opt_state = optimizer.update(grad, opt_state.opt_state, params=params)

        # Update rolling window of gradient norms
        grad_norms = opt_state.grad_norms.at[:-1].set(opt_state.grad_norms[1:])
        grad_norms = grad_norms.at[-1].set(grad_norm)

        # If grad norm is too big then ignore update.
        updates, new_opt_state, ignored_grad_count = jax.lax.cond(skip_update,
                              lambda: (jax.tree_map(jnp.zeros_like, updates), opt_state.opt_state,
                                       opt_state.ignored_grads_count + 1),
                              lambda: (updates, new_opt_state, opt_state.ignored_grads_count))

        state = IgnoreNanOptState(opt_state=new_opt_state, ignored_grads_count=ignored_grad_count,
                                  grad_norms=grad_norms)
        return updates, state


    return optax.GradientTransformation(init=init, update=update)



class OptimizerConfig(NamedTuple):
    init_lr: float
    optimizer_name: str = "adam"
    use_schedule: bool = False
    n_iter_total: Optional[int] = None
    n_iter_warmup: Optional[int] = None
    peak_lr: Optional[float] = None
    end_lr: Optional[float] = None
    max_global_norm: Optional[float] = None
    max_param_grad: Optional[float] = None
    dynamic_grad_ignore_and_clip: bool = False
    dynamic_grad_ignore_factor: float = 20.
    dynamic_grad_norm_factor: float = 2.
    dynamic_grad_norm_window: int = 100


def get_optimizer(optimizer_config: OptimizerConfig):
    if optimizer_config.use_schedule:
        lr = optax.warmup_cosine_decay_schedule(
            init_value=float(optimizer_config.init_lr),
            peak_value=float(optimizer_config.peak_lr),
            end_value=float(optimizer_config.end_lr),
            warmup_steps=optimizer_config.n_iter_warmup,
            decay_steps=optimizer_config.n_iter_total
                                                     )
    else:
        lr = float(optimizer_config.init_lr)

    main_grad_tranform = getattr(optax, optimizer_config.optimizer_name)(lr) # e.g. adam.

    if optimizer_config.dynamic_grad_ignore_and_clip:
        optimizer = dynamic_update_ignore_and_grad_norm_clip(
            optimizer=main_grad_tranform,
            window_length=optimizer_config.dynamic_grad_norm_window,
            factor_clip_norm=optimizer_config.dynamic_grad_norm_factor,
            factor_allowable_norm=optimizer_config.dynamic_grad_ignore_factor
        )
    else:
        grad_transforms = [optax.zero_nans()]
        if optimizer_config.max_param_grad:
            clipping_fn = optax.clip(float(optimizer_config.max_param_grad))
            grad_transforms.append(clipping_fn)
        if optimizer_config.max_global_norm:
            clipping_fn = optax.clip_by_global_norm(float(optimizer_config.max_global_norm))
            grad_transforms.append(clipping_fn)
        grad_transforms.append(main_grad_tranform)
        optimizer = optax.chain(*grad_transforms)
    return optimizer, lr

