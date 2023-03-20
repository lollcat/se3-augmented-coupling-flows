from qm9.data.args import init_argparse
from qm9.data.utils import initialize_datasets
import numpy as np


def retrieve_dataloaders(remove_h = True,
                         dataset='qm9_download_data',
                         datadir='./'):
    # Initialize dataloader
    filter_n_atoms = 9 if remove_h else 19  # max 9 heavy atoms
    args = init_argparse('qm9_download_data')
    # data_dir = cfg.data_root_dir
    args, datasets, num_species, charge_scale = initialize_datasets(args, datadir, dataset,
                                                                    subtract_thermo=args.subtract_thermo,
                                                                    force_download=args.force_download,
                                                                    remove_h=remove_h)
    qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                 'lumo': 27.2114}

    for dataset in datasets.values():
        dataset.convert_units(qm9_to_eV)

    if filter_n_atoms is not None:
        print("Retrieving molecules with only %d atoms" % filter_n_atoms)
        datasets = filter_atoms(datasets, filter_n_atoms)

    # Construct PyTorch dataloaders from datasets
    return datasets, charge_scale


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets


def download_and_save_data(base_path, remove_h = False):
    for remove_h in (True, False):
        n_atoms = 9 if remove_h else 19  # max 9 heavy atoms

        datasets, charge_scale = retrieve_dataloaders(remove_h=remove_h)
        print(f"data has been downloaded for QM9 positional: remove h={remove_h}")

        train = datasets['train'].data['positions'][:, :n_atoms]
        test = datasets['test'].data['positions'][:, :n_atoms]
        valid = datasets['valid'].data['positions'][:, :n_atoms]

        if remove_h:
            np.save(f'{base_path}/qm9_train_no_h.npy', train)
            np.save(f'{base_path}/qm9_test_no_h.npy', test)
            np.save(f'{base_path}/qm9_valid_no_h.npy', valid)
        else:
            np.save(f'{base_path}/qm9_train.npy', train)
            np.save(f'{base_path}/qm9_test.npy', test)
            np.save(f'{base_path}/qm9_valid.npy', valid)

        print("data saved to target/data")


if __name__ == '__main__':
    download_and_save_data("../target/data")