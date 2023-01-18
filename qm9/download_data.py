import argparse
from qm9 import dataset_ as dataset
import numpy as np


parser = argparse.ArgumentParser(description='SE3')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | kernel_dynamics | egnn_dynamics | gnn_dynamics')
parser.add_argument('--data', type=str, default='qm9_only19',
                    help='dw4 | qm9_only19')
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
parser.add_argument('--exp_name', type=str, default='qm9pos_debug')
parser.add_argument('--wandb_usr', type=str, default='lollcat')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--test_epochs', type=int, default=1)
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--hutch_noise', type=str, default='gaussian',
                    help='gaussian | bernoulli')
parser.add_argument('--nf', type=int, default=64,
                    help='number of layers')
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--save_model', type=eval, default=False,
                    help='save model')
parser.add_argument('--data_augmentation', type=eval, default=False,
                    help='use attention in the EGNN')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--x_aggregation', type=str, default='sum',
                    help='sum | mean')

args, unparsed_args = parser.parse_known_args()
print(args)

n_particles = 19  # 19 nodes is the most common type of molecule in QM9
n_dims = 3
dim = n_dims * n_particles  # system dimensionality





if __name__ == "__main__":
    # Retrieve QM9 dataloaders
    datasets, charge_scale = dataset.retrieve_dataloaders(args.batch_size, args.num_workers,
                                                          filter_n_atoms=n_particles)
    print("data has been downloaded for QM9 positional")


    train = datasets['train'].data['positions']
    np.save('target/data/qm9_train.npy', train)

    test = datasets['test'].data['positions']
    np.save('target/data/qm9_test.npy', train)

    valid = datasets['valid'].data['positions']
    np.save('target/data/qm9_valid.npy', valid)

    print("data saved to target/data")






