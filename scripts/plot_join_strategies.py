import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import corc.complex_datasets
import corc.graph_metrics.neb
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import corc.graph_metrics.tmm_gmm_neb
import corc.studentmixture
import pickle
import configargparse


def plot_mask(transformed_points, mask, best_mask, ax, title=""):
    ax.scatter(
        transformed_points[mask, 0],
        transformed_points[mask, 1],
        c="black",
        s=10,
        label="Correctly Classified",
    )
    ax.scatter(
        transformed_points[~mask, 0],
        transformed_points[~mask, 1],
        c="red",
        s=10,
        label="Misclassified",
    )
    ax.scatter(
        transformed_points[~best_mask, 0],
        transformed_points[~best_mask, 1],
        c="orange",
        s=10,
        label="always Misclassified",
    )
    ax.set_title(title)


def main():
    p = configargparse.ArgumentParser()
    p.add_argument(
        "--cache_path",
        type=str,
        default="cache",
        help="Path to the compressed cached datasets and clustering results.",
    )
    p.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="densired_soft_8",
        help="dataset for the figure",
    )
    p.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="dataset for the figure",
    )
    parser.add_argument(
        "--n_components",
        help="Number of mixture model components. Only used for seed-stability plots",
        default=15,
        type=int,
    )
    opt = p.parse_args()

    # load dataset
    X, y, transformed_points = corc.utils.load_dataset(
        dataset_name=opt.dataset, cache_path=opt.cache_path
    )

    # load tmm model
    alg_name = "TMM-NEB"
    alg_filename = f"{opt.cache_path}/{opt.dataset}_{alg_name}.pickle"
    if not os.path.exists(alg_filename):
        print(f"File {alg_filename} not found. training a new model.")
        # return -1
        tmm_model = tmm_model = corc.graph_metrics.neb.NEB(
            data=X, labels=y, n_components=opt.n_components, optimization_iterations=500, seed=opt.seed,
        )
        tmm_model.fit(X)
        with open(alg_filename, "wb") as f:
            pickle.dump(tmm_model, f)
    else:
        with open(alg_filename, "rb") as f:
            tmm_model = pickle.load(f)
            print("successfully loaded model from disk")

    figure = create_plot(
        X=X, transformed_points=transformed_points, y=y, tmm_model=tmm_model, seed=opt.seed, n_components=opt.n_components
    )
    plt.savefig(f"figures/join_strategies_{opt.dataset}_seed_{opt.seed}_n_components_{opt.n_components}.pdf")


def create_plot(X, transformed_points, y, tmm_model):
    fig, axs = plt.subplots(5, 2, figsize=(24, 20))
    marker_size = 60

    # ground truth
    axs[0, 0].scatter(
        transformed_points[:, 0], transformed_points[:, 1], c=y, cmap="viridis", s=10
    )
    axs[0, 0].set_title("Ground Truth")
    y_best = corc.utils.best_possible_labels_from_overclustering(
        y, tmm_model.mixture_model.predict(X)
    )
    correct_mask = y == y_best
    ari_best = sklearn.metrics.adjusted_rand_score(y, y_best)
    plot_mask(
        transformed_points,
        correct_mask,
        correct_mask,
        axs[0, 1],
        title=f"Best (ARI: {ari_best:.2f})",
    )
    print("GT done.")

    # ours
    y_pred = tmm_model.predict_with_target(X, len(np.unique(y)))
    y_pred = corc.utils.reorder_colors(y_pred, y)
    ari_neb = sklearn.metrics.adjusted_rand_score(y, y_pred)
    axs[1, 0].scatter(
        transformed_points[:, 0],
        transformed_points[:, 1],
        c=y_pred,
        cmap="viridis",
        s=10,
    )
    centers = corc.utils.snap_points_to_TSNE(
        tmm_model.mixture_model.centers, X, transformed_points
    )
    axs[1, 0].scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=marker_size)
    axs[1, 0].set_title("TMM-MEP (ours)")
    plot_mask(
        transformed_points,
        (y == y_pred),
        correct_mask,
        axs[1, 1],
        title=f"Ours (ARI: {ari_neb:.2f})",
    )
    print("ours done.")

    # baseline
    y_pred_full = tmm_model.mixture_model.predict(X)
    y_baseline = corc.utils.predict_by_joining_closest_clusters(
        centers=tmm_model.mixture_model.centers,
        y_pred=y_pred_full,
        num_classes=len(np.unique(y)),
        dip_stat=False,
        data=X
    )
    y_baseline = corc.utils.reorder_colors(y_baseline, y)
    ari_baseline = sklearn.metrics.adjusted_rand_score(y, y_baseline)
    axs[2, 0].scatter(
        transformed_points[:, 0],
        transformed_points[:, 1],
        c=y_baseline,
        cmap="viridis",
        s=10,
    )
    axs[2, 0].scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=marker_size)
    axs[2, 0].set_title("Nearest Neighbor (Baseline)")
    plot_mask(
        transformed_points,
        (y == y_baseline),
        correct_mask,
        axs[2, 1],
        title=f"Nearest Neighbor Baseline (ARI: {ari_baseline:.2f})",
    )
    print("Closest done.")

    # kmeans
    kmeans = sklearn.cluster.KMeans(n_clusters=n_components, random_state=seed).fit(X)
    centers = kmeans.cluster_centers_
    kmeans_pred = kmeans.predict(X)
    kmeans_pred_joined = corc.utils.predict_by_joining_closest_clusters(
        centers=centers, y_pred=kmeans_pred, num_classes=len(np.unique(y)), dip_stat=False, data=X
    )
    kmeans_pred_joined = corc.utils.reorder_colors(kmeans_pred_joined, y)
    ari_kmeans = sklearn.metrics.adjusted_rand_score(y, kmeans_pred_joined)
    axs[3, 0].scatter(
        transformed_points[:, 0],
        transformed_points[:, 1],
        c=kmeans_pred_joined,
        cmap="viridis",
        s=10,
    )
    centers = corc.utils.snap_points_to_TSNE(centers, X, transformed_points)
    axs[3, 0].scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=marker_size)
    axs[3, 0].set_title("K-Means (Baseline)")
    best_mask_kmeans = y == corc.utils.best_possible_labels_from_overclustering(
        y, kmeans_pred
    )
    plot_mask(
        transformed_points,
        (y == kmeans_pred_joined),
        best_mask_kmeans,
        axs[3, 1],
        title=f"K-Means Baseline (ARI: {ari_kmeans:.2f})",
    )
    print("kmeans done.")

    # now lets add dip-stats merge
    dip_stat_pred_joined = corc.utils.predict_by_joining_closest_clusters(
        centers=tmm_model.mixture_model.centers, 
        y_pred=np.asarray(tmm_model.predict(X).tolist()), 
        num_classes=len(np.unique(y)), 
        dip_stat=True, 
        data=tmm_model.data
    )

    dip_stat_pred_joined = corc.utils.reorder_colors(dip_stat_pred_joined, y)
    ari_dip_stat = sklearn.metrics.adjusted_rand_score(y, dip_stat_pred_joined)

    axs[4, 0].scatter(
        transformed_points[:, 0],
        transformed_points[:, 1],
        c=dip_stat_pred_joined,
        cmap="viridis",
        s=10,
    )
    centers = corc.utils.snap_points_to_TSNE(
        tmm_model.mixture_model.centers, X, transformed_points
    )
    axs[4, 0].scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=marker_size)
    axs[4, 0].set_title("TMM-dip stats")
    plot_mask(
        transformed_points,
        (y == dip_stat_pred_joined),
        correct_mask,
        axs[4, 1],
        title=f"TMM-dip stats (ARI: {ari_dip_stat:.2f})",
    )
    print("TMM+dip stats done.")

    return plt.gcf()


if __name__ == "__main__":
    main()
