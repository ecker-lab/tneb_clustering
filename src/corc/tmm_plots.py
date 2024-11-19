import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
import studenttmixture
import sklearn

import corc.graph_metrics.neb
from corc.graph_metrics import tmm_gmm_neb

GRID_RESOLUTION = 128 # the resolution for the "heatmap" computation for TMM plotting

def plot_logprob_lines(mixture_model, i, j, temps, logprobs, path=None):
    """plotting probabilities between the direct line and the nudged elastic band"""
    if isinstance(mixture_model, sklearn.mixture.GaussianMixture):
        locations = mixture_model.means_
    elif isinstance(mixture_model, studenttmixture.EMStudentMixture):
        locations = mixture_model.location
    # the direct line
    direct_x = np.linspace(0, 1, num=128)[..., None]
    ms = (1 - direct_x) * locations[i] + direct_x * locations[j]
    direct_y = mixture_model.score_samples(ms)
    plt.plot(direct_x[:, 0], direct_y, label="direct")

    # and the nudged elastic band
    plt.plot(temps[(i, j)], logprobs[(i, j)], label="elastic band")
    plt.legend()

    if path is not None:
        plt.savefig(path)


def plot_field(
    data_X,
    mixture_model,
    paths=None,
    levels=20,
    selection=None,
    save_path=None,
    axis=None,
):
    """Plots the TMM/GMM field and the optimized paths (if available).
    tmm: trained studenttmixture model
    gmm: trained Gaussian mixture model
    selection: selects which paths are included in the plot, by default, all paths are included.
      other typical options: MST through selection=zip(mst.row,mst.col) and individuals via e.g. [(0,1), (3,4)]
      One needs to set tmm or gmm, matching the mixture_model parameter
    """
    if isinstance(mixture_model, sklearn.mixture.GaussianMixture):
        locations = mixture_model.means_
    elif isinstance(mixture_model, studenttmixture.EMStudentMixture):
        locations = mixture_model.location
    n_components = len(locations)

    # grid coordinates
    x = np.linspace(data_X[:, 0].min() - 0.1, data_X[:, 0].max() + 0.1, GRID_RESOLUTION)
    y = np.linspace(data_X[:, 1].min() - 0.1, data_X[:, 1].max() + 0.1, GRID_RESOLUTION)
    XY = np.stack(np.meshgrid(x, y), -1)

    # get scores for the grid values
    mm_probs = mixture_model.score_samples(XY.reshape(-1, 2)).reshape(GRID_RESOLUTION, GRID_RESOLUTION)

    if axis is None:
        figure, axis = plt.subplots(1, 1)
    # plot the mixture model field
    axis.contourf(x, y, mm_probs, levels=levels, cmap="coolwarm", alpha=0.5)
    # the raw data
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
    if selection is None and paths is not None:
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


def compute_mst_edges(raw_adjacency):
    mst = -scipy.sparse.csgraph.minimum_spanning_tree(-raw_adjacency)
    rows, cols = mst.nonzero()
    entries = list(zip(rows, cols))
    return entries


def computations_for_plot_row(
    data_X, overclustering_n, iterations=500, mixture_model="tmm"
):
    # main computation
    if mixture_model == "tmm":
        model = studenttmixture.EMStudentMixture(
            n_components=overclustering_n,
            n_init=5,
            fixed_df=True,
            df=1.0,
            init_type="k++",
            random_state=42,
        )
    elif mixture_model == "gmm":
        model = sklearn.mixture.GaussianMixture(
            n_components=overclustering_n,
            n_init=5,
            random_state=42,
            init_params="k-means++",
            covariance_type="spherical",
        )
    model.fit(np.array(data_X, dtype=np.float64))

    # compute elastic band paths
    adjacency, raw_adjacency, paths, temps, logprobs = tmm_gmm_neb.compute_neb_paths(
        model, iterations=iterations
    )

    # thresholds for the cluster-number plot
    thresholds_dict, clusterings_dict = tmm_gmm_neb.get_thresholds_and_cluster_numbers(
        adjacency
    )

    # extracting the smallest edges to only draw them in the heatmap
    mst_edges = compute_mst_edges(raw_adjacency)

    return model, adjacency, paths, thresholds_dict, mst_edges, clusterings_dict


def plot_row_with_computation(
    data_X,
    data_y,
    overclustering_n=15,
    iterations=500,
    levels=None,
    mixture_model="tmm",
):
    # main computation
    tmm_model = corc.graph_metrics.neb.NEB(
        latent_dim=data_X.shape[1],
        data=data_X,
        labels=data_y,
        optimization_iterations=iterations,
        mixture_model_type=mixture_model,
    )
    # this is the time-consuming step
    tmm_model.fit(data=data_X)

    plot_row(
        data_X=data_X,
        data_y=data_y,
        tmm_model=tmm_model,
    )

    if levels is None:
        target_cluster_n = len(np.unique(data_y))
        levels = [target_cluster_n - 1, target_cluster_n, target_cluster_n + 1]
    plot_cluster_levels(levels, tmm_model, data_X)


def plot_row(
    data_X,
    data_y,
    tmm_model,
):
    """
    Creates a plot consisting of
    1) the GT clustering,
    2) the TMM clustering including the heatmap and all MST edges - no joining of clusters
    3) threshold values against number of clusters
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # ground truth
    axes[0].scatter(data_X[:, 0], data_X[:, 1], c=data_y, cmap="viridis")
    axes[0].set_title("Ground Truth")

    # heatmap with arrows
    mst_edges = tmm_model.compute_mst_edges(tmm_model.raw_adjacency_)
    plot_field(
        data_X,
        tmm_model.mixture_model,
        paths=tmm_model.paths_,
        selection=mst_edges,
        axis=axes[1],
    )
    axes[1].set_title("Heatmap")

    # # threshold counts
    # axes[2].bar(thresholds, counts)
    # axes[2].set_title('Threshold Counts')
    # axes[2].set_xlabel('Threshold')

    # clusters vs thresholds
    thresholds, cluster_numbers_per_threshold, clusterings = (
        tmm_model.get_thresholds_and_cluster_numbers()
    )
    axes[2].plot(
        cluster_numbers_per_threshold[:, 1],
        cluster_numbers_per_threshold[:, 0],
        marker="o",
    )
    axes[2].axvline(x=len(np.unique(data_y)), color="r")
    axes[2].set_xlabel("Number of clusters")
    axes[2].set_ylabel("Threshold")
    axes[2].set_title("Clusters")
    axes[2].grid()


def plot_cluster_levels(levels, tmm_model, data_X, save_path=None):
    """
    creates a 1-line plot of the tmm clustering such but with different target_number_cluster, so with different merging strategies.
    levels: a list of integers of how many clusters should be in each plot, its length determines the size of the plot.
    """
    n_plots = len(levels)

    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 4, 6))
    threshold_dict, clusterings_dict = tmm_model.get_thresholds_and_cluster_numbers()

    # grid coordinates
    x = np.linspace(data_X[:, 0].min() - 0.1, data_X[:, 0].max() + 0.1, GRID_RESOLUTION)
    y = np.linspace(data_X[:, 1].min() - 0.1, data_X[:, 1].max() + 0.1, GRID_RESOLUTION)
    XY = np.stack(np.meshgrid(x, y), -1)
    tmm_probs = tmm_model.mixture_model.score_samples(XY.reshape(-1, 2)).reshape(
        GRID_RESOLUTION, GRID_RESOLUTION
    )

    cmap = plt.get_cmap("viridis")  # choose a colormap

    for index, level in enumerate(levels):
        axis = axes[index]
        if level not in threshold_dict.keys():
            axis.set_title(f"no threshold leads to {level} clusters")
            print(f"{level} clusters is not achievable.")
            continue

        threshold = threshold_dict[level]
        # all pairs that are merged (i.e. are below threshold)
        tmp_adj = np.array(tmm_model.adjacency_ >= threshold, dtype=int)
        pairs = np.transpose(np.nonzero(tmp_adj))
        _, component_labels = scipy.sparse.csgraph.connected_components(
            tmp_adj, directed=False
        )
        if np.max(component_labels) > 0:
            normalized_component_labels = component_labels / np.max(component_labels)
        else:
            normalized_component_labels = component_labels

        axis.contourf(x, y, tmm_probs, levels=20, cmap="coolwarm", alpha=0.5)
        axis.scatter(data_X[:, 0], data_X[:, 1], s=10, label="raw data")
        axis.scatter(
            tmm_model.mixture_model.location[:, 0],
            tmm_model.mixture_model.location[:, 1],
            c=component_labels,
            cmap="viridis",
            marker="X",
            label="Student's t-mixture",
            s=60,
        )

        for i, j in pairs:
            if i == j:
                continue
            path = tmm_model.paths_[(i, j)]
            axis.plot(
                path[:, 0],
                path[:, 1],
                lw=3,
                alpha=0.5,
                color=cmap(normalized_component_labels[i]),
            )

        axis.set_title(
            f"{level} target clusters ({len(tmm_model.mixture_model.location)} total)"
        )

    if save_path is not None:
        plt.savefig(save_path)
