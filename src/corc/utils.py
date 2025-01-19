from os.path import exists, join
import os
import pickle
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import corc.utils
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy as np
import subprocess
import sklearn
import studenttmixture
import itertools
import scipy
import corc.studentmixture
import diptest

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

GRID_RESOLUTION = 128  # the resolution for the "heatmap" computation for TMM plotting

def compute_projection(data, cluster1, cluster2, means, predictions):
        c = means[cluster1] - means[cluster2]
        unit_vector = c / np.linalg.norm(c)

        points1 = data[predictions == cluster1]
        points2 = data[predictions == cluster2]
        cluster1_proj = np.dot(points1, unit_vector)
        cluster2_proj = np.dot(points2, unit_vector)

        mean = (np.mean(cluster1_proj) + np.mean(cluster2_proj)) / 2

        cluster1_proj -= mean
        cluster2_proj -= mean

        return cluster1_proj, cluster2_proj


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
    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(
        -cm
    )  # col_ind returns how to reorder the columns (colors of y_pred)
    # Create a mapping based on the optimal column assignments
    permuted_indices = np.zeros_like(y_pred)

    for r, c in zip(row_ind, col_ind):
        permuted_indices[y_pred == c] = r
    return permuted_indices


def reorder_colors2(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)
    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(
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
    transformed_points=None,
):
    """Plots the TMM/GMM field and the optimized paths (if available).
    selection: selects which paths are included in the plot, by default, all paths are included.
      other typical options: MST through selection=zip(mst.row,mst.col) and individuals via e.g. [(0,1), (3,4)]

    """
    # extract cluster centers
    if isinstance(mixture_model, sklearn.mixture.GaussianMixture):
        locations = mixture_model.means_
    elif isinstance(mixture_model, studenttmixture.EMStudentMixture):
        locations = mixture_model.location
    elif isinstance(mixture_model, corc.studentmixture.StudentMixture):
        locations = mixture_model.centers
    n_components = len(locations)

    # Compute TSNE if necessary
    if data_X.shape[-1] > 2:
        if transformed_points is None:
            transformed_points = corc.utils.get_TSNE_embedding(data_X)
        locations = corc.utils.snap_points_to_TSNE(
            locations, data_X, transformed_points
        )
    else:
        transformed_points = data_X

    if axis is None:
        figure, axis = plt.subplots(1, 1)

    # plot the energy landscape if possible
    if data_X.shape[-1] == 2:
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

    # plot the raw data
    if plot_points:
        axis.scatter(
            transformed_points[:, 0], transformed_points[:, 1], s=10, label="raw data"
        )

    # plot cluster centers and IDs
    axis.scatter(
        locations[:, 0],
        locations[:, 1],
        color="black",
        marker="X",
        label="mixture centers",
        s=100,
    )
    for i, location in enumerate(locations):
        y_min, y_max = axis.get_ylim()
        scale = y_max - y_min
        axis.annotate(f"{i}", xy=location - 0.05 * scale, color="black")

    # plot paths between centers (by default: all)
    if paths is not None:
        if selection is None:
            selection = list(itertools.combinations(range(n_components), r=2))
        for i, j in selection:
            path = paths[(i, j)]
            axis.plot(path[:, 0], path[:, 1], lw=3, alpha=0.5, color="red")

    if save_path is not None:
        plt.savefig(save_path)

    # not returning the axis object since it is modified in-place


def best_possible_labels_from_overclustering(y_true, y_pred):
    confusion = sklearn.metrics.confusion_matrix(y_true, y_pred)

    # Create a mapping of predicted clusters to the majority true label
    best_labels = np.zeros(confusion.shape[1], dtype=int)

    for predicted_label in range(confusion.shape[1]):
        if (
            confusion[:, predicted_label].sum() > 0
        ):  # Check if there are any samples for this predicted label
            best_labels[predicted_label] = np.argmax(confusion[:, predicted_label])
        else:
            best_labels[predicted_label] = -1  # Handle cases where there's no count

    # Map the new labels back to the original predicted labels
    y_best = np.array(
        [best_labels[label] if label < len(best_labels) else -1 for label in y_pred]
    )

    return y_best


def predict_by_joining_closest_clusters(centers, y_pred, num_classes, data, dip_stat=False, debug=False):

    def find_root(mapping, class_index):
        if mapping[class_index] != class_index:
            mapping[class_index] = find_root(mapping, mapping[class_index])
        return mapping[class_index]

    def merge_classes(mapping, class_i, class_j):
        root_i = find_root(mapping, class_i)
        root_j = find_root(mapping, class_j)

        # Merge by attaching root_j to root_i
        if root_i != root_j:
            mapping[root_j] = root_i
        return mapping

    def update_centers_after_merge(centers, mapping, i, j):
        # Get all points that belong to either cluster
        mask_i = (y_pred == i)
        mask_j = (y_pred == j)
        combined_points = data[mask_i | mask_j]
        
        # Compute new center as center of mass
        new_center = np.mean(combined_points, axis=0)
        
        # Update the center for cluster i (which is the root)
        centers[i] = new_center
        return centers

    def update_distances_after_merge(distances, mapping, i, j, centers, data, y_pred, dip_stat):
        root_i = find_root(mapping, i)
        
        # Set distances between merged clusters to infinity
        distances[i, j] = distances[j, i] = np.inf
        
        if dip_stat:
            # Recompute dip statistics for the merged cluster with all others
            for k in range(len(centers)):
                if k != i and k != j and find_root(mapping, k) == k:  # Only update for active clusters
                    # Compute new projections with merged cluster
                    pr1, pr2 = compute_projection(data, root_i, k, centers, y_pred)
                    dip, _ = diptest.diptest(np.concatenate([pr1, pr2]))
                    distances[min(root_i, k), max(root_i, k)] = -dip
        else:
            # Recompute Euclidean distances from merged cluster to all others
            for k in range(len(centers)):
                if k != i and k != j and find_root(mapping, k) == k:  # Only update for active clusters
                    dist = np.linalg.norm(centers[root_i] - centers[k])
                    distances[min(root_i, k), max(root_i, k)] = dist
        
        # Set all distances involving j to infinity since it's now merged
        distances[j, :] = distances[:, j] = np.inf
        
        return distances

    mapping = np.array(range(len(centers)))
    centers = centers.copy()  # Create a copy to modify
    distances = np.ones((len(centers), len(centers))) * np.inf
    if dip_stat:
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                pr1, pr2 = compute_projection(data, i, j, centers, y_pred)
                dip, _ = diptest.diptest(
                    np.concatenate([pr1, pr2])
                )
                distances[i, j] = -dip
    else:
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                distances[i, j] = np.linalg.norm(centers[i] - centers[j])

    if debug:
        plt.figure(figsize=(8,8))
        sns.heatmap(distances.T * (-1), annot=True,  fmt='.2g')
        plt.show()
    num_classes_to_join = len(centers) - num_classes
    for _ in range(num_classes_to_join):
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        mapping = merge_classes(mapping, i, j)
        # Update centers and distances
        centers = update_centers_after_merge(centers, mapping, i, j)
        distances = update_distances_after_merge(distances, mapping, i, j, centers, data, y_pred, dip_stat)
        print(f"joined {i} and {j} (both now in class {find_root(mapping,i)})")
        if debug:
            plt.figure(figsize=(8,8))
            sns.heatmap(distances.T * (-1), annot=True,  fmt='.2g')
            plt.show()

    final_mapping = np.zeros(len(centers))
    for i in range(len(centers)):
        final_mapping[i] = find_root(mapping, i)

    joined_predictions = final_mapping[y_pred]
    return joined_predictions


def load_dataset(dataset_name, cache_path="../cache"):

    dataset_name = dataset_name.replace(" ", "_")
    dataset_filename = f"{cache_path}/{dataset_name}.pickle"
    with open(dataset_filename, "rb") as f:
        dataset_info = pickle.load(f)

    X, y = dataset_info["dataset"]
    dimension = X.shape[-1]
    if "X2D" in dataset_info.keys():
        transformed_points = dataset_info["X2D"]
    elif dimension > 2:
        transformed_points = corc.utils.get_TSNE_embedding(X)
    else:
        transformed_points = X
    return X, y, transformed_points
