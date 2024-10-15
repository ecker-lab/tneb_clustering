import pickle

# import our_datasets
# import our_algorithms
import re
import numpy as np
import matplotlib.pyplot as plt
import itertools

import corc.utils

datasets = [
    # "noisy_circles",
    # "noisy_moons",
    # "varied",
    # "aniso",
    # "blobs",
    # "worms",
    # "bowtie",
    # "zigzag",
    # "zigzig",
    # "uniform_circle",
    # "clusterlab10",
    ###########################
    ##### fig 2 datasets ######
    ###########################
    "blobs1_0",
    # "blobs1_1",
    # "blobs1_2",
    # "blobs1_3",
    # "blobs2_0",
    # "blobs2_1",
    # "blobs2_2",
    # "blobs2_3",
    # "densired0",
    # "densired1",
    # "densired2",
    # "densired3",
    # "mnist0",
    # "mnist1",
    # "mnist2",
    # "mnist3",
    # "paul15",
]

algorithms = [
    "MiniBatch\nKMeans",
    "Agglomerative\nClustering",
    "HDBSCAN",
    "Gaussian\nMixture",
    "t-Student\nMixture",
    "Leiden",
    "PAGA",
    "GWG-dip",
    "GWG-pvalue",
    "Affinity\nPropagation",
    "MeanShift",
    "Spectral\nClustering",
    "Ward",
    "DBSCAN",
    "OPTICS",
    "BIRCH",
    "TMM-NEB",
    "GMM-NEB",
]


def get_color_scheme(y_pred, y):
    colors = np.array(
        list(
            itertools.islice(
                itertools.cycle(
                    [
                        "#377eb8",
                        "#ff7f00",
                        "#4daf4a",
                        "#f781bf",
                        "#a65628",
                        "#984ea3",
                        "#999999",
                        "#e41a1c",
                        "#dede00",
                    ]
                ),
                int(max(max(y_pred), max(y)) + 1),
            )
        )
    )
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    return colors


plt.figure(figsize=(len(algorithms) * 2 + 3, len(datasets) * 2))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
)
plot_num = 1


for i_dataset, dataset_orig_name in enumerate(datasets):
    # load dataset
    dataset_name = re.sub(" ", "_", dataset_orig_name)
    dataset_filename = f"cache/{dataset_name}.pickle"
    with open(dataset_filename, "rb") as f:
        dataset_info = pickle.load(f)

    X, y = dataset_info["dataset"]
    dim = (dataset_info["algo_params"])["dim"]
    if "X2D" in dataset_info.keys():
        X2D = dataset_info["X2D"]
    else:
        X2D = None

    points = X2D if X2D is not None else X

    # first column ist GT
    plt.subplot(len(datasets), len(algorithms) + 1, plot_num)
    if i_dataset == 0:
        plt.title("Ground Truth", size=18)
    colors = get_color_scheme(y, y)
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y])
    plt.xticks(())
    plt.yticks(())
    plot_num += 1

    # plotting the other algorithms
    for i_algorithm, algorithm_name in enumerate(algorithms):
        # load algorithm object
        alg_name = re.sub("\n", "", algorithm_name)
        alg_filename = f"cache/{dataset_name}_{alg_name}.pickle"
        with open(alg_filename, "rb") as f:
            algorithm = pickle.load(f)

        # extracting the predictions
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            if hasattr(algorithm, "predict_with_target"):
                y_pred = algorithm.predict_with_target(X, len(np.unique(y))).astype(int)
            else:
                y_pred = algorithm.predict(X)

        # create plotting area
        plt.subplot(len(datasets), len(algorithms) + 1, plot_num)
        if i_dataset == 0:
            plt.title(algorithm_name, size=18)

        if dim == 2 and algorithm_name in ["TMM-NEB", "GMM-NEB"]:
            linspace_x = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 128)
            linspace_y = np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 128)
            XY = np.stack(np.meshgrid(linspace_x, linspace_y), -1)
            tmm_probs = algorithm.mixture_model.score_samples(
                XY.reshape(-1, 2)
            ).reshape(128, 128)
            plt.contourf(
                linspace_x, linspace_y, tmm_probs, levels=20, cmap="coolwarm", alpha=0.5
            )

        colors = get_color_scheme(y_pred, y)
        y_pred_permuted = corc.utils.reorder_colors(y_pred, y)
        plt.scatter(points[:, 0], points[:, 1], s=10, color=colors[y_pred_permuted])

        if algorithm_name in ["GWG-dip", "GWG-pvalue", "PAGA", "TMM-NEB", "GMM-NEB"]:
            algorithm.plot_graph(X2D=X2D)
            print(f"min {min(X2D[:,0])}, max {max(X2D[:,1])}")

        plt.xticks(())
        plt.yticks(())
        plot_num += 1

plt.savefig(f"figures/fig1.pdf", bbox_inches="tight")
