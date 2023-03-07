import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import chex


from flow.test_utils import bijector_test
from utils.numerical import rotate_translate_permute_2d, rotate_translate_permute_3d
from flow.bijectors.bijector_proj_real_nvp import make_se_equivariant_split_coupling_with_projection, \
    affine_transform_in_new_space, inverse_affine_transform_in_new_space
from nets.base import NetsConfig, TransformerConfig, EgnnTorsoConfig, MLPHeadConfig, MACETorsoConfig
from flow.distrax_with_info import ChainWithInfo
from flow.fast_hk_chain import Chain as FastChain


def test_matmul_transform_in_new_space(n_nodes: int = 5, dim: int = 3):
    """Test equivariance and invertibility."""
    if jnp.ones(()).dtype == jnp.float64:
        rtol = 1e-6
    else:
        rtol = 1e-4


    key = jax.random.PRNGKey(0)


    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (n_nodes, dim))
    origin = x + jax.random.normal(subkey, (n_nodes, dim))*0.1
    change_of_basis_matrix = jnp.repeat(jnp.eye(dim)[None, ...], n_nodes, axis=0)
    key, subkey = jax.random.split(key)
    shift = jax.random.normal(subkey, (n_nodes, dim))

    key, subkey = jax.random.split(key)
    scale = jnp.exp(jax.random.normal(subkey, (n_nodes, dim)))

    x_out = jax.vmap(affine_transform_in_new_space)(x, change_of_basis_matrix, origin, scale, shift)
    x_original = jax.vmap(inverse_affine_transform_in_new_space)(x_out, change_of_basis_matrix, origin, scale, shift)

    # Test invertibility.
    chex.assert_trees_all_close(x_original, x, rtol=rtol)
    assert jnp.sum(jnp.abs(x - x_out)) > 1e-3

    # Test equivariance.
    # Get rotated version of x_and_a.
    key, subkey = jax.random.split(key)
    theta = jax.random.uniform(subkey) * 2 * jnp.pi
    key, subkey = jax.random.split(key)
    translation = jax.random.normal(subkey, shape=(dim,))
    key, subkey = jax.random.split(key)
    phi = jax.random.uniform(subkey) * 2 * jnp.pi

    def group_action(x, theta=theta, translation=translation):
        if dim == 2:
            x_rot = rotate_translate_permute_2d(x, theta, translation, permute=False)
        else:  #  dim == 3:
            x_rot = rotate_translate_permute_3d(x, theta, phi, translation, permute=False)
        return x_rot

    def group_action_to_column_matrix_of_vecs(change_of_basis_matrix, theta=theta):
        if dim == 2:
            change_of_basis_matrix_rot = rotate_translate_permute_2d(
                change_of_basis_matrix.T, theta, jnp.zeros_like(translation),
                permute=False
            ).T
        else:  #  dim == 3:
            change_of_basis_matrix_rot = rotate_translate_permute_3d(
                change_of_basis_matrix.T, theta, phi, jnp.zeros_like(translation),
                permute=False).T
        return change_of_basis_matrix_rot

    # Apply group action to relevant things.
    x_g = group_action(x)
    origin = group_action(origin)

    change_of_basis_matrix = jax.vmap(group_action_to_column_matrix_of_vecs, in_axes=0)(change_of_basis_matrix)

    x_g_out = jax.vmap(affine_transform_in_new_space)(x_g, change_of_basis_matrix, origin, scale, shift)

    x_out_g = group_action(x_out)

    chex.assert_trees_all_close(x_g_out, x_out_g)


def test_bijector_with_proj(dim: int = 3, n_layers: int = 2,
                            gram_schmidt: bool = False,
                            global_frame: bool = False,
                            process_flow_params_jointly: bool = True,
                            use_mace: bool = True):
    nets_config = NetsConfig(type='mace' if use_mace else "egnn",
                             mace_torso_config=MACETorsoConfig(
                                    n_vectors_residual = 3,
                                    n_invariant_feat_residual = 3,
                                    n_vectors_hidden_readout_block = 3,
                                    n_invariant_hidden_readout_block = 3,
                                    hidden_irreps = '4x0e+4x1o'
                                 ),
                             egnn_torso_config=EgnnTorsoConfig() if not use_mace else None,
                             mlp_head_config=MLPHeadConfig((4,)) if not process_flow_params_jointly else None,
                             transformer_config=TransformerConfig() if process_flow_params_jointly else None
                             )


    def make_flow():
        bijectors = []
        for i in range(n_layers):
            swap = i % 2 == 0
            bijector = make_se_equivariant_split_coupling_with_projection(
                layer_number=i, dim=dim, swap=swap,
                identity_init=False,
                nets_config=nets_config,
                global_frame=global_frame,
                gram_schmidt=gram_schmidt,
                process_flow_params_jointly=process_flow_params_jointly,
                                                            )
            bijectors.append(bijector)
        flow = distrax.Chain(bijectors)
        return flow

    @hk.without_apply_rng
    @hk.transform
    def bijector_forward(x):
        return make_flow().forward_and_log_det(x)

    @hk.without_apply_rng
    @hk.transform
    def bijector_backward(x):
        return make_flow().inverse_and_log_det(x)

    bijector_test(bijector_forward, bijector_backward, dim=dim, n_nodes=4)

def test_bijector_with_info(dim: int = 3, n_nodes = 5,
                            gram_schmidt: bool = False,
                            global_frame: bool = False,
                            process_flow_params_jointly: bool = False,
                            use_mace: bool = True):

    nets_config = NetsConfig(type='mace' if use_mace else "egnn",
                             mace_torso_config=MACETorsoConfig(
                                    n_vectors_residual = 3,
                                    n_invariant_feat_residual = 3,
                                    n_vectors_hidden_readout_block = 3,
                                    n_invariant_hidden_readout_block = 3,
                                    hidden_irreps = '4x0e+4x1o'
                                 ),
                             egnn_torso_config=EgnnTorsoConfig() if not use_mace else None,
                             mlp_head_config=MLPHeadConfig((4,)) if not process_flow_params_jointly else None,
                             transformer_config=TransformerConfig() if process_flow_params_jointly else None
                             )

    def make_flow():
        def bijector_fn():
            bijectors = []
            for swap in (True, False):
                bijector = make_se_equivariant_split_coupling_with_projection(
                    layer_number=0, dim=dim, swap=swap,
                    identity_init=False,
                    nets_config=nets_config,
                    global_frame=global_frame,
                    gram_schmidt=gram_schmidt,
                    process_flow_params_jointly=process_flow_params_jointly,
                )
                bijectors.append(bijector)
            bijector_block = ChainWithInfo(bijectors)
            return bijector_block
        flow = FastChain(bijector_fn=bijector_fn, n_layers=2)
        return flow
    @hk.without_apply_rng
    @hk.transform
    def bijector_forward_with_info(x):
        y, log_det, info = make_flow().forward_and_log_det_with_extra(x)
        return y, log_det, info

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x_dummy = jax.random.normal(key=key, shape=(n_nodes, dim*2))
    params = bijector_forward_with_info.init(key, x_dummy)
    out = bijector_forward_with_info.apply(params, x_dummy)




if __name__ == '__main__':
    USE_64_BIT = True  # Fails for 32-bit
    gram_schmidt = False


    test_bijector_with_info()

    test_matmul_transform_in_new_space()
    print("affine transform passing tests")

    if USE_64_BIT:
        from jax.config import config

        config.update("jax_enable_x64", True)

    use_mace = False
    for global_frame in [False, True]:
        for process_jointly in [False, True]:
            print(f"running tests for global-frame={global_frame}, process jointly={process_jointly}")
            test_bijector_with_proj(dim=3, process_flow_params_jointly=process_jointly, global_frame=global_frame,
                                    gram_schmidt=gram_schmidt, use_mace=use_mace)
            print("passed 3D test")

            if not use_mace:
                test_bijector_with_proj(dim=2, process_flow_params_jointly=process_jointly, global_frame=global_frame,
                                        gram_schmidt=gram_schmidt, use_mace=use_mace)
                print("passed 2D test")



