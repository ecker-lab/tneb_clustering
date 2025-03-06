import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
import studenttmixture
import sklearn
import corc.utils
import time
import math
import corc.graph_metrics.neb
from corc.graph_metrics import tmm_gmm_neb

GRID_RESOLUTION = 128


def plot_logprob_lines(mixture_model, i, j, temps, logprobs, path=None):
    """plotting probabilities between the direct line and the nudged elastic band"""
    locations = corc.utils.mixture_center_locations(mixture_model)
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



def plot_row(data_X, data_y, tmm_model, transformed_points=None):
    """
    Creates a plot consisting of
    1) the GT clustering,
    2) the TMM clustering including the heatmap and all MST edges - no joining of clusters
    3) threshold values against number of clusters
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # ground truth
    if transformed_points is None:
        if data_X.shape[-1] == 2:
            transformed_points = data_X
        else:
            transformed_points = corc.vizualization.get_TSNE_embedding(data_X)

    axes[0].scatter(
        transformed_points[:, 0], transformed_points[:, 1], c=data_y, cmap="viridis"
    )
    axes[0].set_title("Ground Truth")

    # heatmap with arrows
    mst_edges = tmm_model.compute_mst_edges()
    tmm_model.plot_field(
        data_X,
        selection=mst_edges,
        axis=axes[1],
        transformed_points=transformed_points,
    )
    axes[1].set_title("Heatmap")

    # # threshold counts
    # axes[2].bar(thresholds, counts)
    # axes[2].set_title('Threshold Counts')
    # axes[2].set_xlabel('Threshold')

    # clusters vs thresholds
    threshold_dict, merging_strategy_dict = (
        tmm_model.get_thresholds_and_cluster_numbers()
    )
    axes[2].plot(
        threshold_dict.keys(),
        threshold_dict.values(),
        marker="o",
    )
    axes[2].axvline(x=len(np.unique(data_y)), color="r")
    axes[2].set_xlabel("Number of clusters")
    axes[2].set_ylabel("Threshold")
    axes[2].set_title("Clusters")
    axes[2].grid()


def plot_cluster_levels(
    levels, tmm_model, data_X, save_path=None, transformed_points=None
):
    """
    creates a 1-line plot of the tmm clustering such but with different target_number_cluster, so with different merging strategies.
    levels: a list of integers of how many clusters should be in each plot, its length determines the size of the plot.
    """
    n_plots = len(levels)
    n_cols = min(4, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, 2 + n_rows * 4))
    if n_rows > 1:
        axes = axes.flatten()

    # TSNE
    centers = tmm_model.get_centers()
    if data_X.shape[-1] == 2:
        transformed_points = data_X
    else:
        if transformed_points is None:
            transformed_points = corc.vizualization.get_TSNE_embedding(data_X)
        centers = corc.vizualization.snap_points_to_TSNE(centers, data_X, transformed_points)

    for index, level in enumerate(levels):
        axis = axes[index]

        y = tmm_model.predict_with_target(data_X, level)
        axis.scatter(
            transformed_points[:, 0],
            transformed_points[:, 1],
            c=y,
            s=10,
            label="raw data",
        )
        tmm_model.plot_graph(
            X2D=transformed_points, target_num_clusters=level, axis=axis
        )

        axis.set_title(f"{level} target clusters ({len(centers)} total)")

    if save_path is not None:
        plt.savefig(save_path)


def plot_tmm_models(
    tmm_models, data_X, data_y, dataset_name, tsne_transform=None, ground_truth=True
):
    # general setup for plotting
    plt.figure(figsize=(20, 10))
    num_tiles = len(tmm_models) + int(ground_truth)
    num_rows = 1 + (num_tiles // 5)
    num_cols = min(num_tiles, 5)

    # fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10), sharex=True, sharey=True)
    # axes = axes.flatten()

    # compute TSNE if necessary
    dimension = data_X.shape[1]
    if dimension > 2:
        if tsne_transform is not None:
            transformed_X = tsne_transform
        else:
            print("computing TSNE...", end="")
            start_tsne = time.time()
            transformed_X = corc.vizualization.get_TSNE_embedding(data_X)
            print(f"done. ({time.time() - start_tsne:.2f}s)")
    else:  # dimension == 2
        transformed_X = data_X

    if ground_truth:
        plt.subplot(num_rows, num_cols, 1)
        plt.scatter(
            transformed_X[:, 0],
            transformed_X[:, 1],
            s=10,
            c=data_y,
            cmap="tab20",
        )
        plt.title(f"{dataset_name}: Ground truth")

    for i, tmm_model in enumerate(tmm_models):

        plt.subplot(num_rows, num_cols, i + 1 + int(ground_truth))

        # draw background for MEP/NEB plots (if dim=2)
        if dimension == 2:
            image_resolution = 128
            linspace_x = np.linspace(
                data_X[:, 0].min() - 0.1, data_X[:, 0].max() + 0.1, image_resolution
            )
            linspace_y = np.linspace(
                data_X[:, 1].min() - 0.1, data_X[:, 1].max() + 0.1, image_resolution
            )
            XY = np.stack(np.meshgrid(linspace_x, linspace_y), -1)
            tmm_probs = tmm_model.mixture_model.score_samples(
                XY.reshape(-1, 2)
            ).reshape(image_resolution, image_resolution)
            plt.contourf(
                linspace_x,
                linspace_y,
                tmm_probs,
                levels=20,
                cmap="coolwarm",
                alpha=0.5,
            )

        # extracting the predictions
        num_classes = len(np.unique(data_y))
        y_pred = tmm_model.predict_with_target(
            data=data_X, target_number_classes=num_classes
        )

        # draw points
        y_pred_permuted = corc.vizualization.reorder_colors(y_pred, data_y)
        plt.scatter(
            transformed_X[:, 0],
            transformed_X[:, 1],
            s=10,
            c=y_pred_permuted,
            cmap="tab20",
        )

        tmm_model.labels = data_y
        tmm_model.plot_graph(X2D=transformed_X, target_num_clusters=num_classes)

        # Compute ARI score
        ari_score = sklearn.metrics.adjusted_rand_score(data_y, y_pred)
        plt.title(
            f"{dataset_name}: {tmm_model.n_components} clusters, ARI: {ari_score:.2f}"
        )

    # return the figure
    return plt.gcf()


def remove_border(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
