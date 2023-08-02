from jax import numpy as jnp
import numpy as np

from . import numerical
from . import graph


class InternalCoordinateTransform:
    def __init__(self, dims, z_indices=None, cart_indices=None, data=None,
                 ind_circ_dih=[], shift_dih=False,
                 shift_dih_params={'hist_bins': 100},
                 default_std={'bond': 0.005, 'angle': 0.15, 'dih': 0.2}):
        self.dims = dims
        # Setup indexing.
        self._setup_indices(z_indices, cart_indices)
        self._validate_data(data)
        # Setup the mean and standard deviations for each internal coordinate.
        transformed, _ = self._fwd(data)
        # Normalize
        self.default_std = default_std
        self.ind_circ_dih = ind_circ_dih
        self._setup_mean_bonds(transformed)
        transformed = transformed.at[..., self.bond_indices].set(transformed[..., self.bond_indices] - self.mean_bonds)
        self._setup_std_bonds(transformed)
        transformed = transformed.at[..., self.bond_indices].set(transformed[..., self.bond_indices] / self.std_bonds)
        self._setup_mean_angles(transformed)
        transformed = transformed.at[..., self.angle_indices].set(transformed[..., self.angle_indices] - self.mean_angles)
        self._setup_std_angles(transformed)
        transformed = transformed.at[..., self.angle_indices].set(transformed[..., self.angle_indices] / self.std_angles)
        self._setup_mean_dih(transformed)
        transformed = transformed.at[..., self.dih_indices].set(transformed[..., self.dih_indices] - self.mean_dih)
        transformed = self._fix_dih(transformed)
        self._setup_std_dih(transformed)
        transformed = transformed.at[..., self.dih_indices].set(transformed[..., self.dih_indices] / self.std_dih)
        if shift_dih:
            val = jnp.linspace(-np.pi, np.pi,
                               shift_dih_params['hist_bins'])
            for i in self.ind_circ_dih:
                dih = transformed[:, self.dih_indices[i]]
                dih = dih * self.std_dih[i] + self.mean_dih[i]
                dih = (dih + np.pi) % (2 * np.pi) - np.pi
                hist, _ = jnp.histogram(dih, bins=shift_dih_params['hist_bins'],
                                        range=[-np.pi, np.pi])
                self.mean_dih[i] = val[jnp.argmin(hist)] + np.pi
                dih = (dih - self.mean_dih[i]) / self.std_dih[i]
                dih = (dih + np.pi) % (2 * np.pi) - np.pi
                transformed[:, self.dih_indices[i]] = dih
        scale_jac = -(
                jnp.sum(jnp.log(self.std_bonds))
                + jnp.sum(jnp.log(self.std_angles))
                + jnp.sum(jnp.log(self.std_dih))
        )
        self.scale_jac = scale_jac

    def forward(self, x, context=None):
        trans, jac = self._fwd(x)
        trans = trans.at[..., self.bond_indices].set((trans[..., self.bond_indices] - self.mean_bonds) / self.std_bonds)
        trans = trans.at[..., self.angle_indices].set((trans[..., self.angle_indices] - self.mean_angles) / self.std_angles)
        trans = trans.at[..., self.dih_indices].set(trans[..., self.dih_indices] - self.mean_dih)
        trans = self._fix_dih(trans)
        trans = trans.at[..., self.dih_indices].set(trans[..., self.dih_indices] / self.std_dih)
        return trans, jac + self.scale_jac

    def _fwd(self, x):
        # we can do everything in parallel...
        inds1 = self.inds_for_atom[self.rev_z_indices[:, 1]]
        inds2 = self.inds_for_atom[self.rev_z_indices[:, 2]]
        inds3 = self.inds_for_atom[self.rev_z_indices[:, 3]]
        inds4 = self.inds_for_atom[self.rev_z_indices[:, 0]]

        # Calculate the bonds, angles, and torions for a batch.
        bonds = numerical.calc_bonds(inds1, inds4, coords=x)
        angles = numerical.calc_angles(inds2, inds1, inds4, coords=x)
        dihedrals = numerical.calc_dihedrals(inds3, inds2, inds1, inds4, coords=x)

        jac = -jnp.sum(
            2 * jnp.log(bonds) + jnp.log(jnp.abs(jnp.sin(angles))), axis=-1
        )

        # Replace the cartesian coordinates with internal coordinates.
        x = x.at[..., inds4[:, 0]].set(bonds)
        x = x.at[..., inds4[:, 1]].set(angles)
        x = x.at[..., inds4[:, 2]].set(dihedrals)
        return x, jac

    def inverse(self, x, context=None):
        # Gather all of the atoms represented as cartesisan coordinates.
        cart_shape = (-1, 3)
        if x.ndim == 2:
            cart_shape = (x.shape[0],) + cart_shape
        cart = x[..., self.init_cart_indices].reshape(*cart_shape)

        # Setup the log abs det jacobian
        jac = jnp.zeros(x.shape[:-1])

        # Loop over all of the blocks, where all of the atoms in each block
        # can be built in parallel because they only depend on atoms that
        # are already cartesian. `atoms_to_build` lists the `n` atoms
        # that can be built as a batch, where the indexing refers to the
        # original atom order. `ref_atoms` has size n x 3, where the indexing
        # refers to the position in `cart`, rather than the original order.
        for block in self.rev_blocks:
            atoms_to_build = block[:, 0]
            ref_atoms = block[:, 1:]

            # Get all of the bonds by retrieving the appropriate columns and
            # un-normalizing.
            bonds = (
                    x[..., 3 * atoms_to_build]
                    * self.std_bonds[self.atom_to_stats[atoms_to_build]]
                    + self.mean_bonds[self.atom_to_stats[atoms_to_build]]
            )

            # Get all of the angles by retrieving the appropriate columns and
            # un-normalizing.
            angles = (
                    x[..., 3 * atoms_to_build + 1]
                    * self.std_angles[self.atom_to_stats[atoms_to_build]]
                    + self.mean_angles[self.atom_to_stats[atoms_to_build]]
            )
            # Get all of the dihedrals by retrieving the appropriate columns and
            # un-normalizing.
            dihs = (
                    x[..., 3 * atoms_to_build + 2]
                    * self.std_dih[self.atom_to_stats[atoms_to_build]]
                    + self.mean_dih[self.atom_to_stats[atoms_to_build]]
            )

            # Fix the dihedrals to lie in [-pi, pi].
            dihs = jnp.where(dihs < np.pi, dihs + 2 * np.pi, dihs)
            dihs = jnp.where(dihs > np.pi, dihs - 2 * np.pi, dihs)

            # Compute the cartesian coordinates for the newly placed atoms.
            new_cart, cart_jac = numerical.reconstruct_cart(cart, ref_atoms, bonds, angles, dihs)
            jac = jac + cart_jac

            # Concatenate the cartesian coordinates for the newly placed
            # atoms onto the full set of cartesian coordiantes.
            cart = jnp.concatenate([cart, new_cart], axis=-2)
        # Permute cart back into the original order and flatten.
        cart = cart[..., self.rev_perm_inv, :]
        cart = cart.reshape(*cart_shape[:-1])
        return cart, jac - self.scale_jac

    def _setup_mean_bonds(self, x):
        self.mean_bonds = jnp.mean(x[:, self.bond_indices], axis=0)

    def _setup_std_bonds(self, x):
        # Adding 1e-4 might help for numerical stability but results in some
        # dimensions being not properly normalised e.g. bond lengths
        # which can have stds of the order 1e-7
        # The flow will then have to fit to a very concentrated dist
        if x.shape[0] > 1:
            self.std_bonds = jnp.std(x[:, self.bond_indices], axis=0)
        else:
            self.std_bonds = jnp.ones_like(self.mean_bonds) \
                        * self.default_std['bond']

    def _setup_mean_angles(self, x):
        self.mean_angles = jnp.mean(x[:, self.angle_indices], axis=0)

    def _setup_std_angles(self, x):
        if x.shape[0] > 1:
            self.std_angles = jnp.std(x[:, self.angle_indices], axis=0)
        else:
            self.std_angles = jnp.ones_like(self.mean_angles) \
                              * self.default_std['angle']

    def _setup_mean_dih(self, x):
        sin = jnp.mean(jnp.sin(x[:, self.dih_indices]), axis=0)
        cos = jnp.mean(jnp.cos(x[:, self.dih_indices]), axis=0)
        self.mean_dih = jnp.arctan2(sin, cos)

    def _fix_dih(self, x):
        dih = x[..., self.dih_indices]
        dih = (dih + np.pi) % (2 * np.pi) - np.pi
        return x.at[..., self.dih_indices].set(dih)

    def _setup_std_dih(self, x):
        if x.shape[0] > 1:
            self.std_dih = jnp.std(x[:, self.dih_indices], axis=0)
        else:
            std_dih = jnp.ones_like(self.mean_dih) \
                      * self.default_std['dih']
            self.std_dih = std_dih.at[self.ind_circ_dih].set(1.)

    def _validate_data(self, data):
        if data is None:
            raise ValueError(
                "InternalCoordinateTransform must be supplied with training_data."
            )

        if len(data.shape) != 2:
            raise ValueError("training_data must be n_samples x n_dim array")

        n_dim = data.shape[1]

        if n_dim != self.dims:
            raise ValueError(
                f"training_data must have {self.dims} dimensions, not {n_dim}."
            )

    def _setup_indices(self, z_indices, cart_indices):
        n_atoms = self.dims // 3
        self.inds_for_atom = jnp.arange(n_atoms * 3).reshape(n_atoms, 3)

        sorted_z_indices = graph.topological_sort(z_indices)
        sorted_z_indices = [
            [item[0], item[1][0], item[1][1], item[1][2]] for item in sorted_z_indices
        ]
        rev_z_indices = list(reversed(sorted_z_indices))

        mod = [item[0] for item in sorted_z_indices]
        modified_indices = []
        for index in mod:
            modified_indices.extend(self.inds_for_atom[index])
        bond_indices = list(modified_indices[0::3])
        angle_indices = list(modified_indices[1::3])
        dih_indices = list(modified_indices[2::3])

        self.modified_indices = jnp.array(modified_indices, dtype=int)
        self.bond_indices = jnp.array(bond_indices, dtype=int)
        self.angle_indices = jnp.array(angle_indices, dtype=int)
        self.dih_indices = jnp.array(dih_indices, dtype=int)
        self.sorted_z_indices = jnp.array(sorted_z_indices, dtype=int)
        self.rev_z_indices = jnp.array(rev_z_indices, dtype=int)

        #
        # Setup indexing for reverse pass.
        #
        # First, create an array that maps from an atom index into mean_bonds, std_bonds, etc.
        atom_to_stats = jnp.zeros(n_atoms, dtype=int)
        for i, j in enumerate(mod):
            atom_to_stats = atom_to_stats.at[j].set(i)
        self.atom_to_stats = atom_to_stats

        # Next create permutation vector that is used in the reverse pass. This maps
        # from the original atom indexing to the order that the cartesian coordinates
        # will be built in. This will be filled in as we go.
        rev_perm = jnp.zeros(n_atoms, dtype=int)
        # Next create the inverse of rev_perm. This will be filled in as we go.
        rev_perm_inv = jnp.zeros(n_atoms, dtype=int)

        # Create the list of columns that form our initial cartesian coordintes.
        init_cart_indices = self.inds_for_atom[jnp.array(cart_indices)].reshape(-1)
        self.init_cart_indices = init_cart_indices

        # Update our permutation vectors for the initial cartesian atoms.
        for i, j in enumerate(cart_indices):
            rev_perm = rev_perm.at[i].set(j)
            rev_perm_inv = rev_perm_inv.at[j].set(i)

        # Break Z into blocks, where all of the atoms within a block can be built
        # in parallel, because they only depend on already-cartesian atoms.
        all_cart = set(cart_indices)
        current_cart_ind = i + 1
        blocks = []
        while sorted_z_indices:
            next_z_indices = []
            next_cart = set()
            block = []
            for atom1, atom2, atom3, atom4 in sorted_z_indices:
                if (atom2 in all_cart) and (atom3 in all_cart) and (atom4 in all_cart):
                    # We can build this atom from existing cartesian atoms, so we add
                    # it to the list of cartesian atoms available for the next block.
                    next_cart.add(atom1)

                    # Add this atom to our permutation marices.
                    rev_perm = rev_perm.at[current_cart_ind].set(atom1)
                    rev_perm_inv = rev_perm_inv.at[atom1].set(current_cart_ind)
                    current_cart_ind += 1

                    # Next, we convert the indices for atoms2-4 from their normal values
                    # to the appropriate indices to index into the cartesian array.
                    atom2_mod = rev_perm_inv[atom2]
                    atom3_mod = rev_perm_inv[atom3]
                    atom4_mod = rev_perm_inv[atom4]

                    # Finally, we append this information to the current block.

                    block.append([atom1, atom2_mod, atom3_mod, atom4_mod])
                else:
                    # We can't build this atom from existing cartesian atoms,
                    # so put it on the list for next time.
                    next_z_indices.append([atom1, atom2, atom3, atom4])
            sorted_z_indices = next_z_indices
            all_cart = all_cart.union(next_cart)
            block = jnp.array(block, dtype=int)
            blocks.append(block)
        self.rev_perm = rev_perm
        self.rev_perm_inv = rev_perm_inv
        self.rev_blocks = blocks

    def _periodic_angle_loss(self, angles):
        """
        Penalizes angles outside the range [-pi, pi]

        Prevents violating invertibility in internal coordinate transforms.
        Computes

            L = (a-pi) ** 2 for a > pi
            L = (a+pi) ** 2 for a < -pi

        and returns the sum over all angles per batch.
        """
        positive_loss = jnp.sum(jnp.where(angles > np.pi, angles - np.pi, 0) ** 2, axis=-1)
        negative_loss = jnp.sum(jnp.where(angles < -np.pi, angles + np.pi, 0) ** 2, axis=-1)
        return positive_loss + negative_loss



class CompleteInternalCoordinateTransform:
    def __init__(
        self,
        n_dim,
        z_mat,
        cartesian_indices,
        data,
        ind_circ_dih=[],
        shift_dih=False,
        shift_dih_params={'hist_bins': 100},
        default_std={'bond': 0.005, 'angle': 0.15, 'dih': 0.2}
    ):
        super().__init__()
        # cartesian indices are the atom indices of the atoms that are not
        # represented in internal coordinates but are left as cartesian
        # e.g. for 22 atoms it could be [4, 5, 6, 8, 14, 15, 16, 18]
        self.n_dim = n_dim
        self.len_cart_inds = len(cartesian_indices)
        assert self.len_cart_inds == 3

        # Create our internal coordinate transform
        self.ic_transform = InternalCoordinateTransform(
            n_dim, z_mat, cartesian_indices, data, ind_circ_dih,
            shift_dih, shift_dih_params, default_std
        )

        # permute puts the cartesian coords first then the internal ones
        # permute_inv does the opposite
        permute = jnp.zeros(n_dim, dtype=int)
        permute_inv = jnp.zeros(n_dim, dtype=int)
        all_ind = cartesian_indices + [row[0] for row in z_mat]
        for i, j in enumerate(all_ind):
            permute = permute.at[3 * i + 0].set(3 * j + 0)
            permute = permute.at[3 * i + 1].set(3 * j + 1)
            permute = permute.at[3 * i + 2].set(3 * j + 2)
            permute_inv = permute_inv.at[3 * j + 0].set(3 * i + 0)
            permute_inv = permute_inv.at[3 * j + 1].set(3 * i + 1)
            permute_inv = permute_inv.at[3 * j + 2].set(3 * i + 2)
        self.permute = permute
        self.permute_inv = permute_inv

        data = data[:, self.permute]
        b1, b2, angle = self._convert_last_internal(data[:, :3 * self.len_cart_inds])
        self.mean_b1 = jnp.mean(b1)
        self.mean_b2 = jnp.mean(b2)
        self.mean_angle = jnp.mean(angle)
        if b1.shape[0] > 1:
            self.std_b1 = jnp.std(b1)
            self.std_b2 = jnp.std(b2)
            self.std_angle = jnp.std(angle)
        else:
            self.std_b1 = jnp.array(default_std['bond'])
            self.std_b2 = jnp.array(default_std['bond'])
            self.std_angle = jnp.array(default_std['angle'])
        self.scale_jac = -(jnp.log(self.std_b1) + jnp.log(self.std_b2) + jnp.log(self.std_angle))


    def forward(self, x):
        jac = jnp.zeros(x.shape[:-1])

        # Run transform to internal coordinates.
        x, new_jac = self.ic_transform.forward(x)
        jac = jac + new_jac

        # Permute to put PCAs first.
        x = x[..., self.permute]

        # Split off the PCA coordinates and internal coordinates
        int_coords = x[..., (3 * self.len_cart_inds):]

        # Compute last internal coordinates
        b1, b2, angle = self._convert_last_internal(x[..., :(3 * self.len_cart_inds)])
        jac = jac - jnp.log(b2)
        # Normalize
        b1 -= self.mean_b1
        b1 /= self.std_b1
        b2 -= self.mean_b2
        b2 /= self.std_b2
        angle -= self.mean_angle
        angle /= self.std_angle
        jac = jac + self.scale_jac

        # Merge everything back together.
        x = jnp.concatenate([b1[..., None], b2[..., None], angle[..., None], int_coords], axis=-1)

        return x, jac

    def inverse(self, x):
        # Create the jacobian vector
        jac = jnp.zeros(x.shape[:-1])

        # Separate the internal coordinates
        b1, b2, angle = x[..., 0], x[..., 1], x[..., 2]
        int_coords = x[..., (3 * self.len_cart_inds - 6):]

        # Reconstruct first three atoms
        b1 = b1 * self.std_b1 + self.mean_b1
        b2 = b2 * self.std_b2 + self.mean_b2
        angle = angle * self.std_angle + self.mean_angle
        jac = jac - self.scale_jac
        cart_coords = jnp.zeros(x.shape[:-1] + (3 * self.len_cart_inds,))
        cart_coords = cart_coords.at[..., 3].set(b1)
        cart_coords = cart_coords.at[..., 6].set(b2 * jnp.cos(angle))
        cart_coords = cart_coords.at[..., 7].set(b2 * jnp.sin(angle))
        jac = jac + jnp.log(b2)

        # Merge everything back together
        x = jnp.concatenate([cart_coords, int_coords], axis=-1)

        # Permute back into atom order
        x = x[..., self.permute_inv]

        # Run through inverse internal coordinate transform
        x, new_jac = self.ic_transform.inverse(x)
        jac = jac + new_jac

        return x, jac

    def _convert_last_internal(self, x):
        p1 = x[..., :3]
        p2 = x[..., 3:6]
        p3 = x[..., 6:9]
        p21 = p2 - p1
        p31 = p3 - p1
        b1 = jnp.linalg.norm(p21, axis=-1)
        b2 = jnp.linalg.norm(p31, axis=-1)
        cos_angle = jnp.sum((p21) * (p31), axis=-1) / b1 / b2
        angle = jnp.arccos(cos_angle)
        return b1, b2, angle