import haiku as hk
import jax
import chex

from flow.bijectors.bijector_proj_real_nvp import make_se_equivariant_split_coupling_with_projection
from nets.base import NetsConfig, TransformerConfig, EgnnTorsoConfig, MLPHeadConfig, MACETorsoConfig
from flow.distrax_with_extra import ChainWithExtra, TransformedWithExtra
from flow.fast_hk_chain import Chain as FastChain
from flow.base_dist import CentreGravitryGaussianAndCondtionalGuassian


def test_dist_with_info(dim: int = 3, n_nodes = 5,
                            gram_schmidt: bool = False,
                            global_frame: bool = False,
                            process_flow_params_jointly: bool = False,
                            use_mace: bool = False):

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

    def make_dist():
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
            bijector_block = ChainWithExtra(bijectors)
            return bijector_block
        flow = FastChain(bijector_fn=bijector_fn, n_layers=2)
        base = CentreGravitryGaussianAndCondtionalGuassian(
            dim=dim, n_nodes=n_nodes
        )
        distribution = TransformedWithExtra(base, flow)
        return distribution

    @hk.without_apply_rng
    @hk.transform
    def log_prob_with_info_fn(x):
        dist = make_dist()
        log_prob, info = dist.log_prob_with_extra(x)
        return log_prob, info

    @hk.without_apply_rng
    @hk.transform
    def log_prob_fn(x):
        dist = make_dist()
        log_prob = dist.log_prob(x)
        return log_prob

    @hk.transform
    def sample_n_and_log_prob_with_info_fn(n: int):
        dist = make_dist()
        sample, log_prob, info = dist.sample_n_and_log_prob_with_extra(hk.next_rng_key(), n)
        return sample, log_prob, info

    @hk.transform
    def sample_n_and_log_prob_fn(n: int):
        dist = make_dist()
        sample, log_prob = dist._sample_n_and_log_prob(hk.next_rng_key(), n)
        return sample, log_prob

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x_dummy = jax.random.normal(key=key, shape=(n_nodes, dim*2))
    params = log_prob_with_info_fn.init(key, x_dummy)

    # Test log-probing
    log_prob, info = log_prob_with_info_fn.apply(params, x_dummy)
    log_prob_check = log_prob_fn.apply(params, x_dummy)
    chex.assert_trees_all_close(log_prob_check, log_prob)

    # Test sample-n-and-log-prob
    sample, log_prob, info = sample_n_and_log_prob_with_info_fn.apply(params, subkey, 5)
    sample_check, log_prob_check = sample_n_and_log_prob_fn.apply(params, subkey, 5)
    chex.assert_trees_all_close((sample, log_prob), (sample_check, log_prob_check))




if __name__ == '__main__':
    test_dist_with_info()
