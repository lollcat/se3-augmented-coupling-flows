from typing import Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np

import openmm
from openmmtools import testsystems
import multiprocessing as mp


R = 8.31447e-3

def openmm_energy(x: np.ndarray, openmm_context, temperature: float = 800.):
    """Compute the energy of a single configuration using OpenMM.

    Args:
        x: A configuration of shape (n_atoms, 3).
        openmm_context: An OpenMM context.
        temperature: The temperature in Kelvin.

    Returns:
        The energy of the configuration.
    """
    kBT = R * temperature
    # Handle nans and infinities
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        return np.nan
    else:
        openmm_context.setPositions(x)
        state = openmm_context.getState(getForces=False, getEnergy=True)

        # get energy
        energy = state.getPotentialEnergy().value_in_unit(
            openmm.unit.kilojoule / openmm.unit.mole) / kBT
        return energy


def openmm_multi_proc_init(env, temp, plat):
    """
    Method to initialize temperature and openmm context for workers
    of multiprocessing pool
    """
    global temperature_g, openmm_context_g
    temperature_g = temp
    # System setup
    if env == 'vacuum':
        system = testsystems.AlanineDipeptideVacuum(constraints=None)
    elif env == 'implicit':
        system = testsystems.AlanineDipeptideImplicit(constraints=None)
    else:
        raise NotImplementedError('This environment is not implemented.')
    sim = openmm.app.Simulation(system.topology, system.system,
                                openmm.LangevinIntegrator(temp * openmm.unit.kelvin,
                                                          1.0 / openmm.unit.picosecond,
                                                          1.0 * openmm.unit.femtosecond),
                                platform=openmm.Platform.getPlatformByName(plat))
    openmm_context_g = sim.context


def openmm_energy_multi_proc(x: np.ndarray):
    """Compute the energy of a single configuration using OpenMM using global
    temperature and context.

    Args:
        x: A configuration of shape (n_atoms, 3).

    Returns:
        The energy of the configuration.
    """
    kBT = R * temperature_g
    # Handle nans and infinities
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        return np.nan
    else:
        openmm_context_g.setPositions(x)
        state = openmm_context_g.getState(getForces=False, getEnergy=True)

        # get energy
        energy = state.getPotentialEnergy().value_in_unit(
            openmm.unit.kilojoule / openmm.unit.mole) / kBT
        return energy


def openmm_energy_batched(x: np.ndarray, openmm_context, temperature: float = 800.):
    """Compute the energy of a batch of configurations using OpenMM.

    Args:
        x: A batch of configurations of shape (n_batch, n_atoms, 3).
        openmm_context: An OpenMM context.
        temperature: The temperature in Kelvin.

    Returns:
        The energy of each configuration in the batch.
    """
    n_batch = x.shape[0]

    energies = []
    for i in range(n_batch):
        # Get numpy array as input for OpenMM
        energies.append(openmm_energy(x[i, ...], openmm_context, temperature))
    return np.array(energies, dtype=x.dtype)


def openmm_energy_multi_proc_batched(x: np.ndarray, pool):
    energies_out = pool.map(openmm_energy_multi_proc, x)
    energies = np.array(energies_out, dtype=x.dtype)
    return energies


def get_log_prob_fn(temperature: float = 800, environment: str = 'implicit', platform: str = 'Reference',
                    scale: Optional[float] = None):
    """Get a function that computes the energy of a batch of configurations.

    Args:
        temperature (float, optional): The temperature in Kelvin. Defaults to 800.
        environment (str, optional): The environment in which the energy is computed. Can be implicit or vacuum.
        Defaults to 'implicit'.
        platform (str, optional): The compute platform that OpenMM shall use. Can be 'Reference', 'CUDA', 'OpenCL',
        and 'CPU'. Defaults to 'Reference'.
        scale (Optional[float], optional): A scaling factor applied to the input batch. Defaults to None.

    Returns:
        A function that computes the energy of a batch of configurations.
    """
    # System setup
    if environment == 'vacuum':
        system = testsystems.AlanineDipeptideVacuum(constraints=None)
    elif environment == 'implicit':
        system = testsystems.AlanineDipeptideImplicit(constraints=None)
    else:
        raise NotImplementedError('This environment is not implemented.')
    assert platform in ['Reference', 'CUDA', 'OpenCL', 'CPU']
    sim = openmm.app.Simulation(system.topology, system.system,
                                openmm.LangevinIntegrator(temperature * openmm.unit.kelvin,
                                                          1. / openmm.unit.picosecond,
                                                          1. * openmm.unit.femtosecond),
                                openmm.Platform.getPlatformByName(platform))

    def log_prob_fn_np(x: np.ndarray):
        if scale is not None:
            x = x * scale
        if len(x.shape) == 2:
            energy = openmm_energy_batched(x[None, ...], sim.context, temperature=temperature)
            return -energy[0, ...]
        elif len(x.shape) == 3:
            energies = openmm_energy_batched(x, sim.context, temperature=temperature)
            return -energies
        else:
            raise NotImplementedError('The OpenMM energy function only supports 2D and 3D inputs')

    def log_prob_fn(x: chex.Array):
        result_shape = jax.ShapedArray(x.shape[:-2], x.dtype)
        logp = jax.pure_callback(log_prob_fn_np, result_shape, x)
        return logp

    return log_prob_fn


def get_multi_proc_log_prob_fn(temperature: float = 800, environment: str = 'implicit', platform: str = 'Reference',
                               n_threads: Optional[int] = None):
    """Get a function that computes the energy of a batch of configurations via multiprocessing.

    Args:
        temperature (float, optional): The temperature in Kelvin. Defaults to 800.
        environment (str, optional): The environment in which the energy is computed. Can be implicit or vacuum.
        Defaults to 'implicit'.
        n_threads (Optional[int], optional): Number of threads for multiprocessing. If None, the number of CPUs is
        taken. Defaults to None.

    Returns:
        A function that computes the energy of a batch of configurations.
    """
    assert environment in ['implicit', 'vacuum'], 'Environment must be either implicit or vacuum'
    assert platform in ['Reference', 'CUDA', 'OpenCL', 'CPU'], 'Platform must be either Reference, CUDA, OpenCL, or CPU'
    if n_threads is None:
        n_threads = mp.cpu_count()
    # Initialize multiprocessing pool
    pool = mp.Pool(n_threads, initializer=openmm_multi_proc_init, initargs=(environment, temperature, platform))

    # Define function
    def log_prob_fn_np(x: np.ndarray):
        if len(x.shape) == 2:
            energy = openmm_energy_multi_proc_batched(x[None, ...], pool)
            return -energy[0, ...]
        elif len(x.shape) == 3:
            energies = openmm_energy_multi_proc_batched(x, pool)
            return -energies
        else:
            raise NotImplementedError('The OpenMM energy function only supports 2D and 3D inputs')

    def log_prob_fn(x: chex.Array):
        result_shape = jax.ShapedArray(x.shape[:-2], x.dtype)
        logp = jax.pure_callback(log_prob_fn_np, result_shape, x)
        return logp

    return log_prob_fn
