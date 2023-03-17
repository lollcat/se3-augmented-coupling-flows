import chex
import jax
import jax.numpy as jnp

def get_leading_axis_tree(tree: chex.ArrayTree, n_dims: int = 1):
    flat_tree, tree_struct = jax.tree_util.tree_flatten(tree)
    leading_shape = flat_tree[0].shape[:n_dims]
    chex.assert_tree_shape_prefix(tree, leading_shape)
    return leading_shape


if __name__ == '__main__':
    print(get_leading_axis_tree(jnp.ones((10, 10))))

