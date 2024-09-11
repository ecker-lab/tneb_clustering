import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
import studenttmixture

import src.corc.jax_neb as jax_neb

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
def plot_field(data_X, tmm, paths=None, levels=20, selection=None, save_path=None, axis=None):
    n_components = len(tmm.location)
    if axis is None:
        figure, axis = plt.subplots(1, 1)

    # grid coordinates
    x = np.linspace(data_X[:, 0].min() - 0.1, data_X[:, 0].max() + 0.1, 128)
    y = np.linspace(data_X[:, 1].min() - 0.1, data_X[:, 1].max() + 0.1, 128)
    XY = np.stack(np.meshgrid(x, y), -1)

    # get scores for the grid values
    tmm_probs = tmm.score_samples(XY.reshape(-1, 2)).reshape(128, 128)
    # gmm_probs = gm.score_samples(XY.reshape(-1, 2)).reshape(128,128)

    # plot the tmm
    axis.contourf(x, y, tmm_probs, levels=levels, cmap="coolwarm", alpha=0.5)
    # plt.contourf(x, y, gmm_probs, levels=64, cmap="coolwarm", alpha=0.5)
    axis.scatter(data_X[:, 0], data_X[:, 1], s=10, label="raw data")

    # cluster centers and IDs
    axis.scatter(tmm.location[:, 0], tmm.location[:, 1], color="black", marker="X",
                label="Student's t-mixture", s=100)
    for i, location in enumerate(tmm.location):
        axis.annotate(f"{i}", xy=location - 1, color="black")

    # print paths between centers (by default: all)
    if selection == None and paths is not None:
        selection = ((i, j) for i, j in itertools.combinations(range(n_components), r=2) if i != j)
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


def computations_for_plot_row(data_X, overclustering_n, iterations=500):
    # main computation
    tmm = studenttmixture.EMStudentMixture(
        n_components=overclustering_n,
        fixed_df=True,
        df=1.0,
        init_type="k++",
        random_state=42
    )
    tmm.fit(np.array(data_X, dtype=np.float64))

    # compute elastic band paths
    adjacency, raw_adjacency, paths, temps, logprobs = jax_neb.compute_neb_paths(tmm, iterations=iterations)

    # thresholds for the cluster-number plot
    thresholds, cluster_numbers, counts = jax_neb.get_thresholds_and_cluster_numbers(adjacency)

    # extracting the smallest edges to only draw them in the heatmap
    mst_edges = compute_mst_edges(raw_adjacency)

    return tmm, adjacency, paths, thresholds, cluster_numbers, counts, mst_edges


def plot_row_with_computation(data_X, data_y, overclustering_n=15, iterations=500, levels=None):
    # main computation
    tmm, adjacency, paths, thresholds, cluster_numbers, counts, mst_edges = (
        computations_for_plot_row(data_X, overclustering_n, iterations=iterations))

    plot_row(data_X, data_y, tmm, paths, thresholds, cluster_numbers, counts, mst_edges)


    if levels is None:
        target_cluster_n = len(np.unique(data_y))
        levels = [target_cluster_n-1, target_cluster_n, target_cluster_n+1]
    plot_cluster_levels(levels, tmm, data_X, adjacency, paths)


def plot_row(data_X, data_y, tmm, paths, thresholds, cluster_numbers, counts, mst_edges):
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))

    # ground truth
    axes[0].scatter(data_X[:, 0], data_X[:, 1], c=data_y, cmap='viridis')
    axes[0].set_title('Ground Truth')

    # heatmap with arrows
    plot_field(data_X, tmm, paths=paths, selection=mst_edges, axis=axes[1])
    axes[1].set_title('Heatmap')

    # threshold counts
    axes[2].bar(thresholds, counts)
    axes[2].set_title('Threshold Counts')
    axes[2].set_xlabel('Threshold')

    # clusters vs thresholds
    axes[3].plot(cluster_numbers[:, 1], cluster_numbers[:, 0], marker='o')
    axes[3].set_xlabel('Number of clusters')
    axes[3].set_ylabel('Threshold')
    axes[3].set_title('Clusters')
    axes[3].grid()


def plot_cluster_levels(levels, tmm, data_X, adjacency, paths, save_path=None):
    n_plots = len(levels)

    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 4, 6))
    thresholds, cluster_numbers, counts = jax_neb.get_thresholds_and_cluster_numbers(adjacency)

    # grid coordinates
    x = np.linspace(data_X[:, 0].min() - 0.1, data_X[:, 0].max() + 0.1, 128)
    y = np.linspace(data_X[:, 1].min() - 0.1, data_X[:, 1].max() + 0.1, 128)
    XY = np.stack(np.meshgrid(x, y), -1)
    tmm_probs = tmm.score_samples(XY.reshape(-1, 2)).reshape(128, 128)

    cmap = plt.get_cmap('viridis')  # choose a colormap

    for index, level in enumerate(levels):
        axis = axes[index]
        if level not in cluster_numbers:
            axis.set_title(f'no threshold leads to {level} clusters')
            print(f'{level} clusters is not achievable.')
            continue

        level_index = np.where(cluster_numbers == level)[0][0]
        threshold = thresholds[level_index]
        # all pairs that are merged (i.e. are below threshold)
        tmp_adj = np.array(adjacency >= threshold, dtype=int)
        pairs = np.transpose(np.nonzero(tmp_adj))
        _, component_labels = scipy.sparse.csgraph.connected_components(tmp_adj, directed=False)
        if np.max(component_labels) > 0:
            normalized_component_labels = component_labels / np.max(component_labels)
        else:
            normalized_component_labels = component_labels

        axis.contourf(x, y, tmm_probs, levels=20, cmap="coolwarm", alpha=0.5)
        axis.scatter(data_X[:, 0], data_X[:, 1], s=10, label="raw data")
        axis.scatter(tmm.location[:, 0], tmm.location[:, 1], c=component_labels, cmap='viridis', marker="X",
                     label="Student's t-mixture", s=60)

        for i, j in pairs:
            if i==j:
                continue
            path = paths[(i, j)]
            axis.plot(path[:, 0], path[:, 1], lw=3, alpha=0.5, color=cmap(normalized_component_labels[i]))

        axis.set_title(f'{level} target clusters ({len(tmm.location)} total)')

    if save_path is not None:
        plt.savefig(save_path)