from lj13 import load_dataset
from train import plot_sample_hist
import matplotlib.pyplot as plt


if __name__ == '__main__':
    train_ds, test_ds = load_dataset(100)
    fig, ax = plt.subplots()
    plot_sample_hist(train_ds.positions, ax, label='train')
    plot_sample_hist(test_ds.positions, ax, label='test')
    plt.legend()
    plt.show()
