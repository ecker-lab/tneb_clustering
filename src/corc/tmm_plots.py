import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_logprob_lines(tmm, i, j, temps, logprobs, path=None):
    # the direct line
    direct_x = np.linspace(0, 1, num=128)[..., None]
    ms = (1 - direct_x) * tmm.location[i] + direct_x * tmm.location[j]
    direct_y = tmm.score_samples(ms)
    plt.plot(direct_x[:, 0], direct_y, label="direct")

    # and the nudged elastic band
    plt.plot(temps[(i, j)], logprobs[(i, j)], label="elastic band")
    plt.legend()

    if path is not None:
        plt.savefig(path)


''' Plots the TMM field and the optimized paths (if available).
stm: trained studenttmixture model
selection: selects which paths are included in the plot, by default, all paths are included.
  other typical options: MST through selection=zip(mst.row,mst.col) and individuals via e.g. [(0,1), (3,4)] 
'''


def plot_field(data_X, tmm, paths=None, levels=20, selection=None, path=None):
    n_components = len(tmm.location)

    # grid coordinates
    x = np.linspace(data_X[:, 0].min() - 0.1, data_X[:, 0].max() + 0.1, 128)
    y = np.linspace(data_X[:, 1].min() - 0.1, data_X[:, 1].max() + 0.1, 128)
    XY = np.stack(np.meshgrid(x, y), -1)

    # get scores for the grid values
    tmm_probs = tmm.score_samples(XY.reshape(-1, 2)).reshape(128, 128)
    # gmm_probs = gm.score_samples(XY.reshape(-1, 2)).reshape(128,128)

    # plot the tmm
    plt.contourf(x, y, tmm_probs, levels=levels, cmap="coolwarm", alpha=0.5)
    # plt.contourf(x, y, gmm_probs, levels=64, cmap="coolwarm", alpha=0.5)
    plt.scatter(data_X[:, 0], data_X[:, 1], s=10, label="raw data")

    # cluster centers and IDs
    plt.scatter(tmm.location[:, 0], tmm.location[:, 1], color="black", marker="X",
                label="Student's t-mixture", s=100)
    for i, location in enumerate(tmm.location):
        plt.annotate(f"{i}", xy=location - 1, color="black")

    # print paths between centers (by default: all)
    if selection == None and paths is not None:
        selection = itertools.product(range(n_components), range(n_components))
    for i, j in selection:
        path = paths[(i, j)]
        plt.plot(path[:, 0], path[:, 1], lw=3, alpha=0.5)

    if path is not None:
        plt.savefig(path)
