import chex
import jax
import jax.numpy as jnp
import numpy as np

import openmm
from openmmtools import testsystems


R = 8.31447e-3

def openmm_energy(x: chex.Array, openmm_context, temperature: float = 800.):
    """Compute the energy of a batch of configurations using OpenMM.

    Args:
        x: A batch of configurations of shape (n_batch, n_atoms, 3).
        openmm_context: An OpenMM context.
        temperature: The temperature in Kelvin.

    Returns:
        The energy of each configuration in the batch.
    """
    n_batch = x.shape[0]

    kBT = R * temperature
    energies = []
    for i in range(n_batch):
        # Get numpy array as input for OpenMM
        x_np = np.array(x[i, ...])
        # Handle nans and infinities
        if np.any(np.isnan(x_np)) or np.any(np.isinf(x_np)):
            energies.append(np.nan)
        else:
            openmm_context.setPositions(x_np)
            state = openmm_context.getState(getForces=False, getEnergy=True)

            # get energy
            energies.append(
                state.getPotentialEnergy().value_in_unit(
                    openmm.unit.kilojoule / openmm.unit.mole) / kBT
            )
    return jnp.array(energies)


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
            energy = openmm_energy(x[None, ...], sim.context, temperature=temperature)
            return -energy[0, ...]
        elif len(x.shape) == 3:
            energies = openmm_energy(x, sim.context, temperature=temperature)
            return -energies
        else:
            raise NotImplementedError('The OpenMM energy function only supports 2D and 3D inputs')

    return log_prob_fn
