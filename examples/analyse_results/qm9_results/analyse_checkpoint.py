import jax.random
import jax.numpy as jnp
import optax
from hydra import compose, initialize
import hydra

from examples.load_flow_and_checkpoint import load_flow
from examples.default_plotter import *
from molboil.targets.data import load_qm9
from examples.analyse_results.get_wandb_runs import download_checkpoint
from examples.analyse_results.dw4_results.plot import make_get_data_for_plotting
from train.max_lik_train_and_eval import general_ml_loss_fn, masked_ml_loss_fn
from train.custom_step import training_step, training_step_with_masking


_BASE_DIR = '../../..'





def plot_qm9(ax: Optional = None):
    seed = 1

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=f"{_BASE_DIR}/examples/config/")
    cfg = compose(config_name="qm9.yaml")

    flow_type = 'proj'
    # download_checkpoint(flow_type=flow_type, tags=["qm9pos", "ml", "ema"], seed=seed, max_iter=200,
    #                     base_path='./examples/analyse_results/qm9_results/models')

    checkpoint_path = f"examples/analyse_results/qm9_results/models/{flow_type}_seed{seed}.pkl"
    cfg.flow.type = flow_type
    key = jax.random.PRNGKey(0)

    flow, state = load_flow(cfg, checkpoint_path)

    train_data, valid_data, test_data = load_qm9(train_set_size=1000)  # None)
    n_samples_from_flow_plotting = train_data.positions.shape[0]

    batch = train_data[0:8]
    a = flow.aux_target_sample_n_apply(state.params.aux_target, batch, key)
    joint_samples = flow.separate_samples_to_joint(batch.features, batch.positions, a)

    # loss_fn_with_mask = partial(masked_ml_loss_fn,
    #                             flow=flow,
    #                             use_flow_aux_loss=cfg.training.use_flow_aux_loss,
    #                             aux_loss_weight=cfg.training.aux_loss_weight,
    #                             apply_random_rotation=False)
    # optimizer = optax.adam(1e-4)
    # opt_state = optimizer.init(params=state.params)
    # training_step_fn = partial(training_step_with_masking, optimizer=optimizer,
    #                            loss_fn_with_mask=loss_fn_with_mask,
    #                            verbose_info=cfg.training.verbose_info)
    #
    # new_params, new_opt_state, info = training_step_fn(state.params, batch, opt_state, state.key)

    latent, log_det, extra = flow.bijector_inverse_and_log_det_with_extra_apply(
        state.params.bijector, joint_samples, layer_indices=(-5, None), regularise = True, base_params=state.params.base)

    log_q, extra = flow.log_prob_with_extra_apply(state.params, joint_samples)




    get_data_for_plotting, count_list, bins_x = make_get_data_for_plotting(train_data=train_data, test_data=test_data,
                                                                           flow=flow,
                                   n_samples_from_flow=n_samples_from_flow_plotting)

    key = jax.random.PRNGKey(0)
    counts_flow_x, bins_a, count_list_a, bins_a_minus_x, count_list_a_minus_x = get_data_for_plotting(state, key)

    # Plot original coords
    if ax is None:
        fig1, axs = plt.subplots(1, 1, figsize=(5, 5))
    else:
        axs = ax
    axs = [axs]
    plot_histogram(counts_flow_x, bins_x, axs[0], label="flow")
    plot_histogram(count_list[0], bins_x, axs[0],  label="data")
    # axs[0].legend(loc="upper left")
    axs[0].set_title("QM9 Positional")
    axs[0].set_xlim(0.6, 7.3)
    axs[0].set_xlabel("interatomic distance")
    if ax is None:
        axs[0].set_ylabel("normalized count")
        fig1.tight_layout()
        plt.show()
        print("done")



if __name__ == '__main__':
    plot_qm9()
