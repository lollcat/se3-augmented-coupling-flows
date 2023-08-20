import chex
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


def openmm_multi_proc_init(sys, temp):
    """
    Method to initialize temperature and openmm context for workers
    of multiprocessing pool
    """
    global temperature, openmm_context
    temperature = temp
    sim = openmm.app.Simulation(sys.topology, sys.system,
                                openmm.LangevinIntegrator(temp * openmm.unit.kelvin,
                                                          1.0 / openmm.unit.picosecond,
                                                          1.0 * openmm.unit.femtosecond),
                                platform=openmm.Platform.getPlatformByName('Reference'))
    openmm_context = sim.context


def openmm_energy_multi_proc(x: np.ndarray):
    """Compute the energy of a single configuration using OpenMM using global
    temperature and context.

    Args:
        x: A configuration of shape (n_atoms, 3).

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


def openmm_energy_batched(x: chex.Array, openmm_context, temperature: float = 800.):
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
        x_np = np.array(x[i, ...])
        energies.append(openmm_energy(x_np, openmm_context, temperature))
    return jnp.array(energies)


def openmm_energy_multi_proc_batched(x: chex.Array, pool):
    x_np = np.asarray(x).copy()
    energies_out = pool.map(openmm_energy_multi_proc, x_np)
    energies = jnp.array(energies_out)
    return energies


def get_log_prob_fn(temperature: float = 800, environment: str = 'implicit'):
    """Get a function that computes the energy of a batch of configurations.

    Args:
        temperature (float, optional): The temperature in Kelvin. Defaults to 800.
        environment (str, optional): The environment in which the energy is computed. Can be implicit or vacuum.
        Defaults to 'implicit'.

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
    sim = openmm.app.Simulation(system.topology, system.system,
                                openmm.LangevinIntegrator(temperature * openmm.unit.kelvin,
                                                          1. / openmm.unit.picosecond,
                                                          1. * openmm.unit.femtosecond),
                                openmm.Platform.getPlatformByName('Reference'))

    def log_prob_fn(x: chex.Array):
        if len(x.shape) == 2:
            energy = openmm_energy_batched(x[None, ...], sim.context, temperature=temperature)
            return -energy[0, ...]
        elif len(x.shape) == 3:
            energies = openmm_energy_batched(x, sim.context, temperature=temperature)
            return -energies
        else:
            raise NotImplementedError('The OpenMM energy function only supports 2D and 3D inputs')

    return log_prob_fn


def get_multi_proc_log_prob_fn(temperature: float = 800, environment: str = 'implicit', n_threads: int = 18):
    """Get a function that computes the energy of a batch of configurations via multiprocessing.

    Args:
        temperature (float, optional): The temperature in Kelvin. Defaults to 800.
        environment (str, optional): The environment in which the energy is computed. Can be implicit or vacuum.
        Defaults to 'implicit'.
        n_threads (int, optional): Number of threads for multiprocessing. Defaults to 18.

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

    # Initialize multiprocessing pool
    pool = mp.Pool(n_threads, initializer=openmm_multi_proc_init, initargs=(system, temperature))

    # Define function
    def log_prob_fn(x: chex.Array):
        if len(x.shape) == 2:
            energy = openmm_energy_multi_proc_batched(x[None, ...], pool)
            return -energy[0, ...]
        elif len(x.shape) == 3:
            energies = openmm_energy_multi_proc_batched(x, pool)
            return -energies
        else:
            raise NotImplementedError('The OpenMM energy function only supports 2D and 3D inputs')

    return log_prob_fn
