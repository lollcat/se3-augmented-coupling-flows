from typing import Tuple, Optional, Union

from pathlib import Path
import jax.numpy as jnp
import numpy as np
import mdtraj

from eacf.utils.base import positional_dataset_only_to_full_graph, FullGraphSample
from eacf.targets.qm9_download_data.dataset import qm9pos_download_and_save_data


def load_dw4(
    train_set_size: int = 1000,
    val_set_size: int = 1000,
    test_set_size: int = 1000,
    path: Optional[Union[Path, str]] = None,
) -> Tuple[FullGraphSample, FullGraphSample, FullGraphSample]:
    # dataset from https://github.com/vgsatorras/en_flows
    # Loading following https://github.com/vgsatorras/en_flows/blob/main/dw4_experiment/dataset.py.

    if path is None:
        here = Path(__file__).parent
        path = here / "data"
    path = Path(path)
    fpath = path / "dw4-dataidx.npy"
    dataset = jnp.asarray(np.load(fpath, allow_pickle=True)[0], dtype=float)
    dataset = jnp.reshape(dataset, (-1, 4, 2))

    train_set = dataset[:train_set_size]
    val_set = dataset[-test_set_size - val_set_size : -test_set_size]
    test_set = dataset[-test_set_size:]
    return (
        positional_dataset_only_to_full_graph(train_set),
        positional_dataset_only_to_full_graph(val_set),
        positional_dataset_only_to_full_graph(test_set),
    )


def load_dw4_3d(
    train_set_size: int = 1000,
    val_set_size: int = 1000,
    test_set_size: int = 1000,
    path: Optional[Union[Path, str]] = None,
) -> Tuple[FullGraphSample, FullGraphSample, FullGraphSample]:
    if path is None:
        here = Path(__file__).parent
        path = here / "data"
    path = Path(path)
    fpath = path / "dw_data_vertices4_dim3_temperature0.1.npy"
    dataset = jnp.asarray(np.load(fpath, allow_pickle=True), dtype=float)

    train_set = dataset[:train_set_size]
    val_set = dataset[-test_set_size - val_set_size : -test_set_size]
    test_set = dataset[-test_set_size:]
    return (
        positional_dataset_only_to_full_graph(train_set),
        positional_dataset_only_to_full_graph(val_set),
        positional_dataset_only_to_full_graph(test_set),
    )


def load_lj13(
    train_set_size: int = 1000, path: Optional[Union[Path, str]] = None
) -> Tuple[FullGraphSample, FullGraphSample, FullGraphSample]:
    # dataset from https://github.com/vgsatorras/en_flows
    # Loading following https://github.com/vgsatorras/en_flows/blob/main/dw4_experiment/dataset.py.

    # Train data
    if path is None:
        here = Path(__file__).parent
        path = here / "data"
    path = Path(path)
    fpath_train = path / "holdout_data_LJ13.npy"
    fpath_idx = path / "idx_LJ13.npy"
    fpath_val_test = path / "all_data_LJ13.npy"

    train_data = jnp.asarray(np.load(fpath_train, allow_pickle=True), dtype=float)
    idxs = jnp.asarray(np.load(fpath_idx, allow_pickle=True), dtype=int)
    val_test_data = jnp.asarray(np.load(fpath_val_test, allow_pickle=True), dtype=float)

    val_data = val_test_data[1000:2000]
    test_data = val_test_data[:1000]

    assert train_set_size <= len(idxs)
    train_data = train_data[idxs[:train_set_size]]

    val_data = jnp.reshape(val_data, (-1, 13, 3))
    test_data = jnp.reshape(test_data, (-1, 13, 3))
    train_data = jnp.reshape(train_data, (-1, 13, 3))

    return (
        positional_dataset_only_to_full_graph(train_data),
        positional_dataset_only_to_full_graph(val_data),
        positional_dataset_only_to_full_graph(test_data),
    )


def load_qm9(
    train_set_size: Optional[int] = None, path: Optional[Union[Path, str]] = None
) -> Tuple[FullGraphSample, FullGraphSample, FullGraphSample]:
    if path is None:
        here = Path(__file__).parent
        path = here / "data"
    base_path = Path(path)
    fpath_train = base_path / "qm9pos_train.npy"
    fpath_val = base_path / "qm9pos_valid.npy"
    fpath_test = base_path / "qm9pos_test.npy"

    if not fpath_train.exists():
        qm9pos_download_and_save_data(base_path=str(base_path))
    train_data = np.load(str(fpath_train))

    if train_set_size is not None:
        assert train_set_size <= len(train_data)
    train_data = train_data[:train_set_size]
    test_data = np.load(str(fpath_test))
    valid_data = np.load(str(fpath_val))

    train_data = jnp.asarray(train_data, dtype=float)
    test_data = jnp.asarray(test_data, dtype=float)
    valid_data = jnp.asarray(valid_data, dtype=float)

    return (
        positional_dataset_only_to_full_graph(train_data),
        positional_dataset_only_to_full_graph(valid_data),
        positional_dataset_only_to_full_graph(test_data),
    )


def load_aldp(
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    train_n_points=None,
    val_n_points=None,
    test_n_points=None,
    # atom_type_encoding_only: bool = True
) -> Tuple[FullGraphSample, FullGraphSample, FullGraphSample]:
    paths = [train_path, val_path, test_path]
    n_points = [train_n_points, val_n_points, test_n_points]
    datasets = [None, None, None]

    for i in range(3):
        if paths[i] is not None:
            traj = mdtraj.load(paths[i])
            # if atom_type_encoding_only:
            #     atom_encodings = {"carbon": 0, "hydrogen": 1, "oxygen": 2, "nitrogen": 3}
            #     features = jnp.array([atom_encodings[atom.element.name] for atom in traj.topology._atoms],
            #                          dtype=int)[:, None]
            # else:
            features = jnp.arange(traj.n_atoms, dtype=int)[:, None]
            positions = traj.xyz
            if n_points[i] is not None:
                positions = positions[: n_points[i]]
            datasets[i] = FullGraphSample(
                positions=positions,
                features=jnp.repeat(features[None, :], positions.shape[0], axis=0),
            )
    return tuple(datasets)


if __name__ == "__main__":
    train, val, test = load_aldp(train_path='molboil/targets/data/aldp_500K_train_mini.h5')
    print(train.positions.shape, val.positions.shape, test.positions.shape)
