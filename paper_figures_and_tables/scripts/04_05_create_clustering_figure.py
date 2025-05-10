import pickle
import configargparse
import corc.graph_metrics
import corc.graph_metrics.gwgmara
import corc.our_algorithms
import corc.our_datasets as our_datasets
import corc.our_algorithms as our_algorithms
import corc.tmm_plots
from corc.visualization import get_color_scheme
import corc.visualization
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tqdm
import os
import sys
import sklearn

import corc.utils
import corc.metrics

"""
Example call: python scripts/create_clustering_figure.py --algorithms  "MiniBatch\nKMeans, Agglomerative\nClustering" --datasets "blobs1_8, mnist64"
"""


def main(opt):

    # set fontsize
    title_fontsize = 21
    ari_fontsize = 19

    print(f"{opt.algorithms}")

    missing_files = compute_missing_files(opt)
    if missing_files:
        print("The following files are missing:")
        for file in missing_files:
            print(file)
        response = input("Continue anyway? (Y/n): ")
        if response != "" and response.lower() != "y":
            sys.exit(1)

    if opt.datasets_horizontal:
        fig, axs = plt.subplots(
            len(opt.datasets),
            len(opt.algorithms) + 1,
            figsize=(len(opt.algorithms) * 2 + 3, len(opt.datasets) * 2),
        )
    else:
        fig, axs = plt.subplots(
            len(opt.algorithms) + 1,
            len(opt.datasets),
            figsize=(len(opt.datasets) * 2, len(opt.algorithms) * 2 + 3),
            # figsize=(len(opt.algorithms) * 2 + 3, len(opt.datasets) * 2),
        )

    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.02, hspace=0.02
    )

    metrics = dict()

    # now start the computation
    for i_dataset, dataset_name in enumerate(tqdm.tqdm(opt.datasets)):
        # load dataset
        X, y, tsne = corc.utils.load_dataset(dataset_name, opt.cache_path)
        y = np.array(y, dtype=int)
        if X.shape[1] > 2:
            points = tsne
        else:
            points = X

        # first column/row ist GT

        if opt.datasets_horizontal:
            ax = axs[i_dataset, 0]
            ax.set_ylabel(
                our_datasets.dataset_displaynames[dataset_name], fontsize=title_fontsize
            )
            if i_dataset == 0:
                ax.set_title("Ground Truth", fontsize=title_fontsize)
        else:
            ax = axs[0, i_dataset]
            ax.set_title(
                our_datasets.dataset_displaynames[dataset_name], fontsize=title_fontsize
            )
            if i_dataset == 0:
                ax.set_ylabel("Ground Truth", fontsize=title_fontsize)

        colors = get_color_scheme(int(max(y) + 1))
        ax.scatter(points[:, 0], points[:, 1], s=10, color=colors[y])
        # ax.scatter(X[:, 0], X[:, 1], s=10, color=colors[y])
        ax.set_xticks([])
        ax.set_yticks([])

        corc.visualization.remove_border(ax)
        dataset_metrics = dict()

        # plotting the other algorithms
        for i_algorithm, algorithm_name in enumerate(opt.algorithms):
            if opt.datasets_horizontal:
                ax = axs[i_dataset, i_algorithm + 1]
            else:
                ax = axs[i_algorithm + 1, i_dataset]
            if i_dataset == 0:
                alg_displayname = algorithm_name.replace("\\n", "\n")
                # rename TMM-NEB and GMM-NEB
                if alg_displayname in corc.our_algorithms.ALG_DISPLAYNAMES.keys():
                    alg_displayname = corc.our_algorithms.ALG_DISPLAYNAMES[
                        alg_displayname
                    ]
                if opt.datasets_horizontal:
                    ax.set_title(
                        alg_displayname,
                        size=title_fontsize,
                    )
                else:
                    ax.set_ylabel(
                        alg_displayname,
                        size=title_fontsize,
                    )

            # load algorithm
            algorithm, y_pred, ari_score = get_algorithm_and_predictions(
                opt, dataset_name, algorithm_name, X, y
            )
            if algorithm is None:  # and the others as well
                continue

            # store success metrics
            dataset_metrics[algorithm_name] = get_scores(y_pred, y)

            # plot points
            colors = get_color_scheme(int(max(max(y_pred), max(y)) + 1))
            y_pred_permuted = corc.visualization.reorder_colors(y_pred, y)
            ax.scatter(points[:, 0], points[:, 1], s=10, color=colors[y_pred_permuted])

            # plot graph
            if algorithm_name in [
                "GWG-dip",
                "PAGA",
                "TMM-NEB",
                "GMM-NEB",
            ]:
                algorithm.plot_graph(
                    X2D=tsne,
                    target_num_clusters=len(np.unique(y)),
                    ax=ax,
                )

            # add ARI scores to the plot
            ax.text(
                0.02,
                0.88,
                f"{ari_score:.2f}",
                transform=ax.transAxes,
                fontweight="bold",
                size=ari_fontsize,
                bbox=dict(facecolor="white", alpha=0.5, lw=0),
            )

            corc.visualization.remove_border(ax)
            ax.set_xticks(())
            ax.set_yticks(())

        metrics[dataset_name] = dataset_metrics

    # plt.savefig(f"{opt.figure_path}/{opt.figure_name}.pdf", bbox_inches="tight")
    plt.savefig(
        f"{opt.figure_path}/{opt.figure_name}.png", bbox_inches="tight", dpi=200
    )

    with open(f"{opt.cache_path}/metrics/{opt.figure_name}.pkl", "wb") as f:
        pickle.dump(metrics, f)


def get_algorithm_and_predictions(opt, dataset_name, algorithm_name, X, y):
    num_classes = len(np.unique(y))
    alg_name = algorithm_name.replace("\\n", "\n").replace("\n", "")

    # load algorithm object
    alg_filename = os.path.join(opt.cache_path, f"{dataset_name}_{alg_name}.pickle")

    def get_prediction(algorithm, X, num_classes):
        if isinstance(algorithm, corc.graph_metrics.gwgmara.GWGMara):
            y_pred = algorithm.predict(X, target_number_clusters=num_classes)
        elif hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        elif hasattr(algorithm, "predict_with_target"):
            y_pred = algorithm.predict_with_target(X, num_classes).astype(int)
        else:
            y_pred = algorithm.predict(X)
        return y_pred

    # load algorithm object
    alg_name = algorithm_name.replace("\\n", "\n").replace("\n", "")
    alg_filename = f"{opt.cache_path}/{dataset_name}_{alg_name}.pickle"

    # skip if the file is not there
    if not os.path.exists(alg_filename):
        return None, None, None

    with open(alg_filename, "rb") as f:
        algorithm = pickle.load(f)

    if alg_name not in our_algorithms.DETERMINISTIC_ALGORITHMS:
        # then there are 10 random seeds
        best_one = (None, None, -1)
        for model in algorithm:  # algorithm is a list of models in this case
            y_pred = get_prediction(model, X, num_classes)
            ari = sklearn.metrics.adjusted_rand_score(y, y_pred)
            if ari > best_one[2]:
                best_one = (model, y_pred, ari)
        algorithm, y_pred, ari = best_one

    else:
        y_pred = get_prediction(algorithm, X, num_classes)
        ari = sklearn.metrics.adjusted_rand_score(y, y_pred)

    algorithm.data = X  # for tmm plot_graph function later

    return algorithm, y_pred, ari


def compute_missing_files(opt):
    # check for missing files before starting the computation
    missing_files = []
    for dataset_name in opt.datasets:
        dataset_filename = f"{opt.cache_path}/datasets/{dataset_name}.pickle"
        if not os.path.exists(dataset_filename):
            missing_files.append(dataset_filename)

        for algorithm_name in opt.algorithms:
            algorithm_name = algorithm_name.replace("\\n", "\n").replace("\n", "")
            alg_filename = f"{opt.cache_path}/{dataset_name}_{algorithm_name}.pickle"
            if not os.path.exists(alg_filename):
                missing_files.append(alg_filename)

    return missing_files


def get_scores(y_pred, y):
    ari = sklearn.metrics.adjusted_rand_score(y, y_pred)
    nmi = sklearn.metrics.normalized_mutual_info_score(y, y_pred)
    fowlkes_mallows = sklearn.metrics.fowlkes_mallows_score(y, y_pred)
    vi, _, _ = corc.metrics.variation_of_information(y, y_pred)

    return ari, nmi, fowlkes_mallows, vi


def parse_args():
    p = configargparse.ArgumentParser()
    p.add(
        "-c",
        "--config",
        required=False,
        is_config_file=True,
        help="Path to config file.",
    )
    p.add_argument(
        "--cache_path",
        type=str,
        default="cache",
        help="Path to the compressed cached datasets and clustering results.",
    )
    p.add_argument(
        "--figure_path",
        type=str,
        default="figures",
        help="Path were resulting figures are saved.",
    )
    p.add_argument(
        "--figure_name", type=str, default="my_figure", help="Name of the figure."
    )
    p.add_argument(
        "-a",
        "--algorithms",
        type=str,
        default=our_algorithms.ALGORITHM_SELECTOR,
        help="List of algorithms to include in figure. Default is our_algorithms.ALGORITHM_SELECTOR.",
    )
    p.add_argument(
        "-d",
        "--datasets",
        type=str,
        default=our_datasets.DATASET_SELECTOR,
        help="List of datasets to include in figure. Default is our_datasets.DATASET_SELECTOR. Predefined sets include 'fig1', 'fig2', 'main1', and 'main2'.",
    )
    p.add_argument(
        "--datasets_horizontal",
        action="store_true",
        help="If set, datasets are shown in horizontal direction.",
    )
    p.add_argument(
        "--bend_paths",
        action="store_true",
        help="If set, bend paths for TMM-NEB and GMM-NEB on 2d datasets.",
    )

    opt = p.parse_args()

    if isinstance(opt.algorithms, str):
        opt.algorithms = opt.algorithms.replace(" ", "").split(",")
    if isinstance(opt.datasets, str):
        if opt.datasets.lower() == "2d" or opt.datasets.lower() == "fig1":
            opt.datasets = our_datasets.DATASETS2D
            if opt.figure_name == "my_figure":  # the default
                opt.figure_name = "figure1"
        elif opt.datasets.lower() == "complex" or opt.datasets.lower() == "fig2":
            opt.datasets = our_datasets.COMPLEX_DATASETS
            if opt.figure_name == "my_figure":  # the default
                opt.figure_name = "figure2"
        elif opt.datasets.lower() == "main1":
            opt.datasets = our_datasets.DATASETS2D
            opt.algorithms = our_algorithms.CORE_SELECTOR
            title_fontsize = 22
            ari_fontsize = 18
            opt.datasets_horizontal = True
            if opt.figure_name == "my_figure":  # the default
                opt.figure_name = "figure1_main"
        elif opt.datasets.lower() == "main2":
            opt.datasets = our_datasets.CORE_HD_DATASETS
            opt.algorithms = our_algorithms.CORE_SELECTOR
            title_fontsize = 20
            ari_fontsize = 18
            if opt.figure_name == "my_figure":  # the default
                opt.figure_name = "figure2_main"
        elif opt.datasets.lower() == "ours":
            opt.datasets = our_datasets.DATASETS2D
            opt.algorithms = ["TMM-NEB"]
            if opt.figure_name == "my_figure":
                opt.figure_name = "ours_2d"
            opt.datasets_horizontal = False
            opt.bend_paths = True

        else:  # otherwise, handle it as a list of datasets
            opt.datasets = opt.datasets.replace(" ", "").split(",")

    # create cache and figure paths
    os.makedirs(opt.cache_path, exist_ok=True)
    os.makedirs(opt.figure_path, exist_ok=True)

    return opt


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
