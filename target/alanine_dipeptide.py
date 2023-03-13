import mdtraj
import jax.numpy as jnp


def get_atom_encoding(traj):
    # First encode all atoms as different.
    # list(traj.topology.atoms)
    return jnp.arange(traj.n_atoms)

def save_mini_datasets():
    traj_train = mdtraj.load('data/aldp_500K_train.h5')
    traj_train[:2000].save('data/aldp_500K_train_mini.h5')
    traj_test = mdtraj.load('data/aldp_500K_val.h5')  # use val here as it is smaller.
    traj_test[:2000].save('data/aldp_500K_test_mini.h5')


if __name__ == '__main__':
    save_mini_datasets()

    traj = mdtraj.load('data/aldp_500K_train.h5')
    for a in traj.topology.atoms:
        print(a.element.name)
    a1 = traj.topology.atom(3)
    print(a1)
    print(a1.element)



