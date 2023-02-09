import distrax
import haiku as hk


from flow.test_utils import bijector_test
from flow.bijector_proj_real_nvp import make_se_equivariant_split_coupling_with_projection
from flow.nets import EgnnConfig, TransformerConfig


def test_bijector_with_proj(dim: int = 2, n_layers: int = 2,
                            gram_schmidt: bool = False,
                            global_frame: bool = False,
                            process_flow_params_jointly: bool = True):

    egnn_config = EgnnConfig("", mlp_units=(2,), n_layers=2)
    transformer_config = TransformerConfig(mlp_units=(2,), n_layers=2) if process_flow_params_jointly else None
    mlp_function_units = (4,) if not process_flow_params_jointly else None

    def make_flow():
        bijectors = []
        for i in range(n_layers):
            swap = i % 2 == 0
            bijector = make_se_equivariant_split_coupling_with_projection(
                layer_number=i, dim=dim, swap=swap,
                identity_init=False,
                egnn_config=egnn_config,
                transformer_config=transformer_config,
                global_frame=global_frame,
                gram_schmidt=gram_schmidt,
                process_flow_params_jointly=process_flow_params_jointly,
                mlp_function_units=mlp_function_units,
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



if __name__ == '__main__':
    USE_64_BIT = True  # Fails for 32-bit
    gram_schmidt = False

    if USE_64_BIT:
        from jax.config import config

        config.update("jax_enable_x64", True)

    for global_frame in [False, True]:
        for process_jointly in [False, True]:
            print(f"running tests for global-frame={global_frame}, process jointly={process_jointly}")
            test_bijector_with_proj(dim=2, process_flow_params_jointly=process_jointly, global_frame=global_frame,
                                    gram_schmidt=gram_schmidt)
            print("passed 2D test")

            test_bijector_with_proj(dim=3, process_flow_params_jointly=process_jointly, global_frame=global_frame,
                                    gram_schmidt=gram_schmidt)
            print("passed 3D test")

