from typing import Optional, Callable, List

from matplotlib import pyplot as plt
import chex
import jax

from ..train.train import TrainingState
from ..utils.plotting import plot_and_maybe_save


def get_eval_and_plot_fn(
    eval_state: Optional[Callable[[TrainingState, chex.PRNGKey], dict]] = None,
    plotter: Optional[
        Callable[[TrainingState, chex.PRNGKey], List[plt.Figure]]
    ] = None,
):
    """Generates eval_and_plot_fn compatible with previous
        train loop API, where both steps were decoupled.

    Args:
        eval_state (Optional[Callable[[chex.ArrayTree, chex.PRNGKey], dict]], optional): _description_. Defaults to None.
        plotter (Optional[Callable[[chex.ArrayTree, chex.PRNGKey], List[plt.Figure]]], optional): _description_. Defaults to None.
    """

    def eval_and_plot_fn(state, key, iteration, save, plots_dir):
        if plotter is not None:
            key, subkey = jax.random.split(key)
            plot_and_maybe_save(
                plotter,
                state,
                subkey,
                iteration,
                save,
                plots_dir,
            )
        if eval_state is not None:
            return eval_state(state, key)

    return eval_and_plot_fn
