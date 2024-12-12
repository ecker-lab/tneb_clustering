from os.path import exists, join
import os
import pickle
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy as np
import subprocess
import sklearn
import studenttmixture
import itertools
import scipy

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

GRID_RESOLUTION = 128  # the resolution for the "heatmap" computation for TMM plotting


def cond_mkdir(path):
    """create a folder if there is none.

    Parameters
    ----------
    path : str
        path or foldername
    """
    if not exists(path):
        os.makedirs(path)


def save(data, filename, outdir):
    """save data using pickle in folder under given name

    Parameters
    ----------
    data : ndarray
        data that should be saved
    filename : str
        name under which data is saved
    outdir : str
        folder name where to save data
    """
    with open(join(outdir, f"{filename}.pkl"), "wb") as f:
        pickle.dump(data, f)
    with open(join(outdir, f"{filename}.txt"), "w") as f:
        f.write(f"{data}")


def generate_overview_lineplot(log_dir):
    with open(join(log_dir, "config.yaml")) as f:
        opt = yaml.safe_load(f)

    df = pd.DataFrame(columns=["std", "value", "metric"])

    for metric in opt["metric"]["type"]:
        df_metric = pd.read_pickle(join(log_dir, f"{metric}.pkl"))
        df_metric["metric"] = [metric] * len(opt["metric"]["stds"])
        df_metric = df_metric.rename(columns={metric: "value"})
        df = pd.concat([df, df_metric])

    fig = plt.figure()
    sns.lineplot(data=df[df["metric"] != "CH"], x="std", y="value", hue="metric")
    filename = "overview_lineplot"
    fig.savefig(join(log_dir, f"{filename}.png"))
    fig.clear(True)

    sns.lineplot(data=df[df["metric"] == "CH"], x="std", y="value", hue="metric")
    filename = "overview_lineplot_CH"
    fig.savefig(join(log_dir, f"{filename}.png"))
    fig.clear(True)


def generate_overview_visplot(log_dir):

    from mpl_toolkits.axes_grid1 import ImageGrid

    with open(join(log_dir, "config.yaml")) as f:
        opt = yaml.safe_load(f)

    png_dir = join(log_dir, "png")
    n_stds = len(opt["metric"]["stds"])

    plot_types = sorted(list(set([t.split("_")[0] for t in os.listdir(png_dir)])))
    n_plots = len(plot_types)

    imgs, titles = [], []

    for std in opt["metric"]["stds"]:
        for t in plot_types:
            imgs.append(plt.imread(join(png_dir, f"{t}_{std:.2f}.png")))
            titles.append(f"plot {t} with std={std}")

    axes_pad = 0.1

    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(n_stds, n_plots),  # creates 2x2 grid of axes
        axes_pad=axes_pad,  # pad between axes in inch.
    )

    for ax, im, title in zip(grid, imgs, titles):
        ax.set_title(title, fontdict={"fontsize": 7}, pad=-0.5)
        ax.imshow(im)
        ax.axis("off")
    plt.axis("off")
    plt.tight_layout()

    filename = "overview_visplot"
    fig.savefig(join(log_dir, f"{filename}.pdf"))
    fig.savefig(join(log_dir, f"{filename}.png"))
    fig.clear(True)


def reorder_colors(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)
    _, col_ind = linear_sum_assignment(
        -cm
    )  # col_ind returns how to reorder the columns (colors of y_pred)
    y_pred_permuted = np.argsort(col_ind)[
        y_pred
    ]  # we need the inverse mapping which we get through argsort
    return y_pred_permuted


def check_cuda():
    """
    Check if CUDA is available, True if CUDA is available, False otherwise.
    """
    return jax.devices()[0].platform == "gpu"


def get_TSNE_embedding(data_X, perplexity=30, seed=42):
    """
    checks cuda availability and selects the correct TSNE implementation based on that.
    Both implementations give very similar results.
    """
    if check_cuda():
        import tsnecuda

        tsne = tsnecuda.TSNE(
            n_components=2,
            random_seed=seed,
            perplexity=perplexity,
            metric="euclidean",
            init="random",  # nothing else is implemented
            learning_rate=200.0,
            early_exaggeration=12.0,
            pre_momentum=0.8,
            post_momentum=0.8,
            n_iter=500,
        )
        transformed_X = tsne.fit_transform(data_X)
    else:
        import openTSNE

        tsne = openTSNE.TSNE(
            n_components=2,
            random_state=seed,
            perplexity=perplexity,
            metric="euclidean",
            initialization="random",  # default: pca
            learning_rate=200.0,
            early_exaggeration=12.0,
            n_iter=500,
            initial_momentum=0.8,
            final_momentum=0.8,
            n_jobs=16,
        )
        transformed_X = tsne.fit(data_X)
    return transformed_X


def snap_points_to_TSNE(points, data_X, transformed_X):
    """
    pseudo-transforming a set of points by using the embedding of the
    closest point in the dataset. Used only to transform the cluster centres.
    It is reasonable  since they are in dense regions.
    """
    transformed_points = list()
    for point in points:
        # find closest point in data_X
        dists = np.linalg.norm(data_X - point, axis=1)
        closest_idx = np.argmin(dists)
        # select the corresponding embedding
        transformed_points.append(transformed_X[closest_idx])
    return np.array(transformed_points)


def compute_mst_edges(raw_adjacency):
    """
    Computes the edges of the minimum spanning tree of the given adjacency matrix.

    Parameters
    ----------
    raw_adjacency : scipy.sparse.csr_matrix
        The adjacency matrix of the graph.

    Returns
    -------
    entries : list of tuples
        Each tuple contains the row and column indices of a minimum spanning tree edge.
    """
    mst = -scipy.sparse.csgraph.minimum_spanning_tree(-raw_adjacency)
    rows, cols = mst.nonzero()
    entries = list(zip(rows, cols))
    return entries


def plot_field(
    data_X,
    mixture_model,
    levels=20,
    paths=None,  # storage of all paths
    selection=None,  # selection which paths to plot
    save_path=None,
    axis=None,
    plot_points=True,  # whether data_X is plotted
):
    """Plots the TMM/GMM field and the optimized paths (if available).
    selection: selects which paths are included in the plot, by default, all paths are included.
      other typical options: MST through selection=zip(mst.row,mst.col) and individuals via e.g. [(0,1), (3,4)]

    """
    if isinstance(mixture_model, sklearn.mixture.GaussianMixture):
        locations = mixture_model.means_
    elif isinstance(mixture_model, studenttmixture.EMStudentMixture):
        locations = mixture_model.location
    n_components = len(locations)

    if axis is None:
        figure, axis = plt.subplots(1, 1)

    # grid coordinates
    margin = 0.5
    x = np.linspace(
        data_X[:, 0].min() - margin, data_X[:, 0].max() + margin, GRID_RESOLUTION
    )
    y = np.linspace(
        data_X[:, 1].min() - margin, data_X[:, 1].max() + margin, GRID_RESOLUTION
    )
    XY = np.stack(np.meshgrid(x, y), -1)

    # get scores for the grid values
    mm_probs = mixture_model.score_samples(XY.reshape(-1, 2)).reshape(
        GRID_RESOLUTION, GRID_RESOLUTION
    )
    # plotting the energy landscape
    axis.contourf(
        x,
        y,
        mm_probs,
        levels=levels,
        cmap="coolwarm",
        alpha=0.5,
        zorder=-10,
    )

    # the raw data
    if plot_points:
        axis.scatter(data_X[:, 0], data_X[:, 1], s=10, label="raw data")

    # cluster centers and IDs
    axis.scatter(
        locations[:, 0],
        locations[:, 1],
        color="black",
        marker="X",
        label="mixture centers",
        s=100,
    )
    for i, location in enumerate(locations):
        axis.annotate(f"{i}", xy=location - 1, color="black")

    # print paths between centers (by default: all)
    if paths is not None:
        if selection is None:
            selection = (
                (i, j)
                for i, j in itertools.combinations(range(n_components), r=2)
                if i != j
            )
        for i, j in selection:
            path = paths[(i, j)]
            axis.plot(path[:, 0], path[:, 1], lw=3, alpha=0.5)

    if save_path is not None:
        plt.savefig(save_path)

    # not returning the axis object since it is modified in-place
