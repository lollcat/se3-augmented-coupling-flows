from pathlib import Path
import jax
import jax.numpy as jnp

from eacf.targets.data import load_aldp
from eacf.targets.target_energy.aldp import get_log_prob_fn


def test_aldp_log_prob():
    # Get log prob function
    log_prob_fn = get_log_prob_fn(temperature=500.)

    # Load sample data
    path = Path(__file__).parent.parent / "data" / "aldp_500K_train_mini.h5"
    aldp_data = load_aldp(train_path=path, train_n_points=100)[0]

    # Compute log prob
    log_prob = log_prob_fn(aldp_data.positions)

    # Check log prob
    assert jnp.shape(log_prob) == (100,)
    assert jnp.all(jnp.isfinite(log_prob))

if __name__ == "__main__":
    test_aldp_log_prob()