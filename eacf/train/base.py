from typing import Callable, Tuple, Optional, Any, Union

import chex
import jax
import jax.numpy as jnp
import optax

from eacf.utils.base import FullGraphSample

Params = chex.ArrayTree



def setup_padded_reshaped_data(data: chex.ArrayTree,
                               interval_length: int,
                               reshape_axis=0) -> Tuple[chex.ArrayTree, chex.Array]:
    test_set_size = jax.tree_util.tree_flatten(data)[0][0].shape[0]
    chex.assert_tree_shape_prefix(data, (test_set_size, ))

    padding_amount = (interval_length - test_set_size % interval_length) % interval_length
    test_data_padded_size = test_set_size + padding_amount
    test_data_padded = jax.tree_map(
        lambda x: jnp.concatenate([x, jnp.zeros((padding_amount, *x.shape[1:]), dtype=x.dtype)], axis=0), data
    )
    mask = jnp.zeros(test_data_padded_size, dtype=int).at[jnp.arange(test_set_size)].set(1)


    if reshape_axis == 0:  # Used for pmap.
        test_data_reshaped, mask = jax.tree_map(
            lambda x: jnp.reshape(x, (interval_length, test_data_padded_size // interval_length, *x.shape[1:])),
            (test_data_padded, mask)
        )
    else:
        assert reshape_axis == 1  # for minibatching
        test_data_reshaped, mask = jax.tree_map(
            lambda x: jnp.reshape(x, (test_data_padded_size // interval_length, interval_length, *x.shape[1:])),
            (test_data_padded, mask)
        )
    return test_data_reshaped, mask


def maybe_masked_mean(array: chex.Array, mask: Optional[chex.Array]):
    chex.assert_rank(array, 1)
    if mask is None:
        return jnp.mean(array)
    else:
        chex.assert_equal_shape([array, mask])
        array = jnp.where(mask, array, jnp.zeros_like(array))
        divisor = jnp.sum(mask)
        divisor = jnp.where(divisor == 0, jnp.ones_like(divisor), divisor)  # Prevent nan when fully masked.
        return jnp.sum(array) / divisor

def maybe_masked_max(array: chex.Array, mask: Optional[chex.Array]):
    chex.assert_rank(array, 1)
    if mask is None:
        return jnp.max(array)
    else:
        chex.assert_equal_shape([array, mask])
        array = jnp.where(mask, array, jnp.zeros_like(array)-jnp.inf)
        return jnp.max(array)


def get_tree_leaf_norm_info(tree):
    """Returns metrics about contents of PyTree leaves.

    Args:
        tree (_type_): _description_

    Returns:
        _type_: _description_
    """
    norms = jax.tree_util.tree_map(jnp.linalg.norm, tree)
    norms = jnp.stack(jax.tree_util.tree_flatten(norms)[0])
    max_norm = jnp.max(norms)
    min_norm = jnp.min(norms)
    mean_norm = jnp.mean(norms)
    median_norm = jnp.median(norms)
    info = {}
    info.update(
        per_layer_max_norm=max_norm,
        per_layer_min_norm=min_norm,
        per_layer_mean_norm=mean_norm,
        per_layer_median_norm=median_norm,
    )
    return info


def batchify_array(data: chex.Array, batch_size: int):
    num_datapoints = get_leading_axis_tree(data, 1)[0]
    batch_size = min(batch_size, num_datapoints)
    x = data[: num_datapoints - num_datapoints % batch_size]
    return jnp.reshape(
        x,
        (-1, batch_size, *x.shape[1:]),
    )


def batchify_data(data: chex.ArrayTree, batch_size: int):
    return jax.tree_map(lambda x: batchify_array(x, batch_size), data)


def get_leading_axis_tree(tree: chex.ArrayTree, n_dims: int = 1):
    flat_tree, _ = jax.tree_util.tree_flatten(tree)
    leading_shape = flat_tree[0].shape[:n_dims]
    chex.assert_tree_shape_prefix(tree, leading_shape)
    return leading_shape


def get_shuffle_and_batchify_data_fn(train_data: chex.ArrayTree, batch_size: int):
    def shuffle_and_batchify_data(train_data_array, key):
        key, subkey = jax.random.split(key)
        permutted_train_data = jax.random.permutation(subkey, train_data_array, axis=0)
        batched_data = batchify_array(permutted_train_data, batch_size)
        return batched_data

    return lambda key: jax.tree_map(
        lambda x: shuffle_and_batchify_data(x, key), train_data
    )


# Training


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
    pmap_axis_name: str = "data",
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
    # ΤΟDO: log learning rate

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
    return new_params, new_opt_state, info


def create_scan_fn(
    training_step: Callable[
        [Params, FullGraphSample, chex.ArrayTree, chex.PRNGKey],
        Tuple[Params, optax.OptState, dict],
    ],
    last_iter_info_only: Optional[bool] = True,
) -> Callable[
    [Tuple[chex.ArrayTree, optax.OptState, chex.PRNGKey], FullGraphSample],
    Tuple[Tuple[chex.ArrayTree, optax.OptState, chex.PRNGKey], dict],
]:
    def scan_fn(
        carry: Tuple[chex.ArrayTree, optax.OptState, chex.PRNGKey], xs: FullGraphSample
    ) -> Tuple[Tuple[chex.ArrayTree, optax.OptState, chex.PRNGKey], dict]:
        params, opt_state, key = carry
        key, subkey = jax.random.split(key)
        params, opt_state, info = training_step(params, xs, opt_state, subkey)
        if last_iter_info_only:
            info = None
        return (params, opt_state, key), info

    return scan_fn


def create_scan_epoch_fn(
    training_step: Callable[
        [Params, FullGraphSample, chex.ArrayTree, chex.PRNGKey],
        Tuple[Params, optax.OptState, dict],
    ],
    data,
    batch_size: int,
    last_iter_info_only: Optional[bool] = True,
):
    scan_fn = create_scan_fn(training_step, last_iter_info_only)

    shuffle_and_batchify_data = get_shuffle_and_batchify_data_fn(data, batch_size)

    def scan_epoch(params, opt_state, key):
        batched_data = shuffle_and_batchify_data(key)

        if last_iter_info_only:
            final_batch = batched_data[-1]
            batched_data = batched_data[:-1]

        (params, opt_state, key), info = jax.lax.scan(
            scan_fn, (params, opt_state, key), batched_data, unroll=1
        )

        if last_iter_info_only:
            key, subkey = jax.random.split(key)
            params, opt_state, info = training_step(
                params,
                final_batch,
                opt_state,
                subkey,
            )
        return params, opt_state, key, info

    return jax.jit(scan_epoch)


# Evaluation

Mask = chex.Array

FurtherData = Any

def eval_fn(
    x: FullGraphSample,
    key: chex.PRNGKey,
    params: Params,
        eval_on_test_batch_fn: Optional[
            Callable[[Params, chex.ArrayTree, chex.PRNGKey, Mask], Union[Tuple[FurtherData, dict], dict]]
        ] = None,
    eval_batch_free_fn: Optional[Callable[[Params, chex.PRNGKey], dict]] = None,
    batch_size: Optional[int] = None,
    mask: Optional[Mask] = None,
    further_processing_fn: Optional[Callable[[FurtherData, Mask], dict]] = None,
) -> dict:
    info = {}
    key1, key2 = jax.random.split(key)

    if mask is None:
        mask = jnp.ones(x.positions.shape[0], dtype=int)

    if eval_on_test_batch_fn is not None:

        def scan_fn(carry, xs):
            # Scan over data in the test set. Vmapping all at once causes memory issues I think?
            x_batch, mask, key = xs
            info = eval_on_test_batch_fn(
                params,
                x_batch,
                key=key,
                mask=mask
            )
            return None, info

        (x_batched, mask_batched), mask_batched_new = \
            setup_padded_reshaped_data((x, mask), interval_length=batch_size, reshape_axis=1)
        mask_batched = mask_batched * mask_batched_new

        _, batched_info = jax.lax.scan(
            scan_fn,
            None,
            (x_batched, mask_batched, jax.random.split(key1, get_leading_axis_tree(x_batched, 1)[0])),
        )
        if isinstance(batched_info, dict):
            # Aggregate test set info across batches.
            per_batch_weighting = jnp.sum(mask_batched, axis=-1) / jnp.sum(jnp.sum(mask_batched, axis=-1))
            info.update(jax.tree_map(lambda x: jnp.sum(per_batch_weighting*x), batched_info))
        else:
            assert further_processing_fn is not None
            per_batch_weighting = jnp.sum(mask_batched, axis=-1) / jnp.sum(jnp.sum(mask_batched, axis=-1))
            info.update(jax.tree_map(lambda x: jnp.sum(per_batch_weighting * x), batched_info[1]))

            flat_mask, processing_in = jax.tree_map(lambda x: x.reshape(x.shape[0]*x.shape[1],
                                                                        *x.shape[2:]), (mask_batched, batched_info[0]))
            further_info = further_processing_fn(processing_in, flat_mask)
            info.update(further_info)


    if eval_batch_free_fn is not None:
        non_batched_info = eval_batch_free_fn(
            params,
            key=key2,
        )
        info.update(non_batched_info)

    return info
