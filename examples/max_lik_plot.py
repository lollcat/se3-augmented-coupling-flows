import matplotlib.pyplot as plt
import matplotlib as mpl

font = {'family': 'serif',
        'serif': ['Times New Roman'],
        # 'weight' : 'bold',
        'size': 8.
        }
mpl.rc('font', **font)
mpl.rc('text', usetex='true')
mpl.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amsfonts}')
pw = 5.50107
lw = pw / 2
text_size = 8
# mpl.rcParams['figure.dpi'] = 300
mpl.rc('axes', titlesize=text_size, labelsize=text_size)  # fontsize of the axes title and labels
mpl.rc('legend', fontsize=text_size)
mpl.rc('xtick', labelsize=text_size - 1)
mpl.rc('ytick', labelsize=text_size - 1)
# mpl.rc("lines", linewidth=4)

from examples.dw4_results.plot import plot_dw4
from examples.lj13_results.plot import plot_lj13
from examples.qm9_results.plot import plot_qm9


if __name__ == '__main__':
    fig, axs = plt.subplots(1, 3, figsize=(1. * pw, pw/3.6), sharey=True)
    plot_qm9(axs[2])
    print("plotted qm9")
    plot_dw4(axs[0])
    print("plotted dw4")
    plot_lj13(axs[1])
    plt.tight_layout()
    fig.savefig("examples/plots/ml_simple.pdf", bbox_inches="tight")
