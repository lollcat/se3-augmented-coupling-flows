import mdtraj
import jax.numpy as jnp


def get_atom_encoding(traj):
    # First encode all atoms as different.
    # list(traj.topology.atoms)
    return jnp.arange(traj.n_atoms)


if __name__ == '__main__':
    traj = mdtraj.load('data/aldp_500K_train.h5')
    for a in traj.topology.atoms:
        print(a.element.name)
    a1 = traj.topology.atom(3)
    print(a1)
    print(a1.element)



