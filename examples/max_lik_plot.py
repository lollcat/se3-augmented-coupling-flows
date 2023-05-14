import matplotlib.pyplot as plt

from examples.dw4_results.plot import plot_dw4
from examples.lj13_results.plot import plot_lj13
from examples.qm9_results.plot import plot_qm9


if __name__ == '__main__':
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    plot_qm9(axs[2])
    plot_dw4(axs[0])
    plot_lj13(axs[1])
    plt.tight_layout()
    fig.savefig("examples/plots/ml_simple.png")
