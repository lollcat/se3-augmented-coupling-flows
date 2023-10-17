from pathlib import Path
import jax
import jax.numpy as jnp

from eacf.targets.data import load_aldp
from eacf.targets.target_energy.aldp import get_log_prob_fn, get_multi_proc_log_prob_fn


def test_aldp_log_prob():
    # Get log prob function
    log_prob_fn = get_log_prob_fn(temperature=500.)

    # Load sample data
    path = Path(__file__).parent.parent / "data" / "aldp_500K_train_mini.h5"
    aldp_data = load_aldp(train_path=path, train_n_points=100)[0]

    # Compute log prob
    log_prob = log_prob_fn(aldp_data.positions)

    # Test vmap and jit
    log_prob_vmap_jit = jax.jit(jax.vmap(log_prob_fn))
    log_prob_ = log_prob_vmap_jit(aldp_data.positions)

    # Compute gradient
    log_prob_grad = jax.vmap(jax.grad(log_prob_fn))(aldp_data.positions)

    # Check log prob
    assert jnp.shape(log_prob) == (100,)
    assert jnp.all(jnp.isfinite(log_prob))
    assert jnp.allclose(log_prob, log_prob_)
    assert log_prob_grad.shape == aldp_data.positions.shape


def test_aldp_multi_proc_log_prob():
    # Get log prob function
    log_prob_mp_fn = get_multi_proc_log_prob_fn(temperature=500., n_threads=8)
    log_prob_fn = get_log_prob_fn(temperature=500.)

    # Load sample data
    path = Path(__file__).parent.parent / "data" / "aldp_500K_train_mini.h5"
    n_data = 1000
    aldp_data = load_aldp(train_path=path, train_n_points=n_data)[0]

    # Compute log prob
    log_prob_mp = log_prob_mp_fn(aldp_data.positions)
    log_prob = log_prob_fn(aldp_data.positions)

    # Test vmap and jit
    log_prob_vmap_jit = jax.jit(jax.vmap(log_prob_mp_fn))
    log_prob_ = log_prob_vmap_jit(aldp_data.positions)

    # Compute gradient
    log_prob_grad = jax.vmap(jax.grad(log_prob_fn))(aldp_data.positions)

    # Check log prob
    assert jnp.shape(log_prob_mp) == (n_data,)
    assert jnp.shape(log_prob) == (n_data,)
    assert jnp.all(jnp.isfinite(log_prob_mp))
    assert jnp.all(jnp.isfinite(log_prob))
    assert jnp.allclose(log_prob_mp, log_prob)
    assert jnp.allclose(log_prob_mp, log_prob_)
    assert log_prob_grad.shape == aldp_data.positions.shape


if __name__ == "__main__":
    test_aldp_log_prob()
    test_aldp_multi_proc_log_prob()