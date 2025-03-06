import pickle
import configargparse
import corc.graph_metrics
import corc.graph_metrics.gwgmara
import corc.our_algorithms
import corc.our_datasets as our_datasets
import corc.our_algorithms as our_algorithms
import corc.tmm_plots
from corc.colors import get_color_scheme
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tqdm
import os
import sys
import sklearn

import corc.utils

"""
Example call: python scripts/create_clustering_figure.py --algorithms  "MiniBatch\nKMeans, Agglomerative\nClustering" --datasets "blobs1_8, mnist64"
"""


def main(opt):

    # set fontsize
    title_fontsize = 21
    ari_fontsize = 19
    # the following does not works as opt.datasets has been modified...
    # if isinstance(opt.datasets, str):
    #     if opt.datasets.lower() == "main1":
    #         title_fontsize = 32
    #         ari_fontsize = 20
    #     elif opt.datasets.lower() == "main2":
    #         title_fontsize = 36
    #         ari_fontsize = 32

    print(f"{opt.algorithms}")

    missing_files = compute_missing_files(opt)
    if missing_files:
        print("The following files are missing:")
        for file in missing_files:
            print(file)
        response = input("Continue anyway? (Y/n): ")
        if response != "" and response.lower() != "y":
            sys.exit(1)

    fig, axs = plt.subplots(
        len(opt.algorithms) + 1,
        len(opt.datasets),
        figsize=(len(opt.datasets) * 2, len(opt.algorithms) * 2 + 3),
        # figsize=(len(opt.algorithms) * 2 + 3, len(opt.datasets) * 2),
    )

    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.02, hspace=0.02
    )

    # now start the computation
    for i_dataset, dataset_name in enumerate(tqdm.tqdm(opt.datasets)):

        # load dataset
        dataset_filename = f"{opt.cache_path}/{dataset_name}.pickle"
        if dataset_filename in missing_files:
            print(f"Skipping dataset {dataset_name}. (file does not exist)")
            continue
        with open(dataset_filename, "rb") as f:
            dataset_info = pickle.load(f)

        X, y = dataset_info["dataset"]
        if "X2D" in dataset_info.keys():
            X2D = dataset_info["X2D"]
        else:
            X2D = None

        points = X2D if X2D is not None else X

        # first column ist GT

        ax = axs[0, i_dataset]
        if i_dataset == 0:
            ax.set_ylabel("Ground Truth", fontsize=title_fontsize)
        colors = get_color_scheme(int(max(y) + 1))
        ax.scatter(points[:, 0], points[:, 1], s=10, color=colors[y])
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(
            our_datasets.dataset_displaynames[dataset_name], fontsize=title_fontsize
        )
        corc.tmm_plots.remove_border(ax)

        # plotting the other algorithms
        for i_algorithm, algorithm_name in enumerate(opt.algorithms):
            ax = axs[i_algorithm + 1, i_dataset]
            if i_dataset == 0:
                alg_displayname = algorithm_name.replace("\\n", "\n")
                # rename TMM-NEB and GMM-NEB
                if alg_displayname in corc.our_algorithms.ALG_DISPLAYNAMES.keys():
                    alg_displayname = corc.our_algorithms.ALG_DISPLAYNAMES[
                        alg_displayname
                    ]
                ax.set_ylabel(
                    alg_displayname,
                    size=title_fontsize,
                )

            # load algorithm
            algorithm, y_pred, ari_score = get_algorithm_and_predictions(
                opt, dataset_name, algorithm_name, X, y
            )

            # plot points
            colors = get_color_scheme(int(max(max(y_pred), max(y)) + 1)) 
            y_pred_permuted = corc.utils.reorder_colors(y_pred, y)
            ax.scatter(points[:, 0], points[:, 1], s=10, color=colors[y_pred_permuted])

            # plot graph
            if algorithm_name in [
                "GWG-dip",
                "PAGA",
                "TMM-NEB",
                "GMM-NEB",
            ]:
                algorithm.plot_graph(
                    X2D=X2D, target_num_clusters=len(np.unique(y)), ax=ax
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

            corc.tmm_plots.remove_border(ax)
            ax.set_xticks(())
            ax.set_yticks(())

    # plt.savefig(f"{opt.figure_path}/{opt.figure_name}.pdf", bbox_inches="tight")
    plt.savefig(
        f"{opt.figure_path}/{opt.figure_name}.png", bbox_inches="tight", dpi=200
    )


def get_algorithm_and_predictions(opt, dataset_name, algorithm_name, X, y):
    num_classes = len(np.unique(y))
    alg_name = algorithm_name.replace("\\n", "\n").replace("\n", "")

    if alg_name in ["TMM-NEB", "GMM-NEB"]:
        # select the best NEB model out of 10
        n_components = 25 if X.shape[-1] > 2 else 15
        if "TMM" in alg_name:  # TMM-NEB
            alg_filename = (
                f"{opt.cache_path}/stability/seeds_{dataset_name}_{n_components}.pkl"
            )
        else:  # GMM-NEB
            alg_filename = f"{opt.cache_path}/stability/seeds_{dataset_name}_gmm_{n_components}.pkl"
        with open(alg_filename, "rb") as f:
            neb_models = pickle.load(f)

        # select the best model
        best_pair = (None, None, -1)
        for model in neb_models:
            y_pred = model.predict_with_target(X, num_classes).astype(int)
            ari = sklearn.metrics.adjusted_rand_score(y, y_pred)
            if ari > best_pair[2]:
                best_pair = (model, y_pred, ari)
        algorithm, y_pred, ari = best_pair

        # Warn if the predictions are surprisingly bad
        _, counts = np.unique(y_pred, return_counts=True)
        if max(counts) > 0.7 * len(X) and "NEB" in algorithm_name:
            print(
                f"WARNING: {max(counts)/len(X)*100:.2f}% of predictions are in one class. ({algorithm_name} on {dataset_name})"
            )

    else:  # baseline methods
        # load algorithm object
        alg_name = algorithm_name.replace("\\n", "\n").replace("\n", "")
        alg_filename = f"{opt.cache_path}/{dataset_name}_{alg_name}.pickle"

        # skip if the file is not there
        if not os.path.exists(alg_filename):
            return None, None

        with open(alg_filename, "rb") as f:
            algorithm = pickle.load(f)

        if alg_name in ["GWG-dip", "GaussianMixture", "t-StudentMixture"]:
            # then there are 10 random seeds
            best_pair = (None, None, -1)
            for model in algorithm:  # algorithm is a list of models in this case
                if alg_name == "GWG-dip":
                    y_pred = model.predict(
                        X, target_number_clusters=num_classes
                    ).astype(int)
                else:
                    y_pred = model.predict(X).astype(int)
                ari = sklearn.metrics.adjusted_rand_score(y, y_pred)
                if ari > best_pair[2]:
                    best_pair = (model, y_pred, ari)
            algorithm, y_pred, ari = best_pair

        else:
            # extracting the predictions
            if isinstance(algorithm, corc.graph_metrics.gwgmara.GWGMara):
                y_pred = algorithm.predict(X, target_number_clusters=num_classes)
            elif hasattr(algorithm, "labels_"):
                y_pred = algorithm.labels_.astype(int)
            elif hasattr(algorithm, "predict_with_target"):
                y_pred = algorithm.predict_with_target(X, num_classes).astype(int)
            else:
                y_pred = algorithm.predict(X)
            ari = sklearn.metrics.adjusted_rand_score(y, y_pred)

    algorithm.data = X  # for tmm plot_graph function later

    return algorithm, y_pred, ari


def compute_missing_files(opt):
    # check for missing files before starting the computation
    missing_files = []
    for dataset_name in opt.datasets:
        dataset_filename = f"{opt.cache_path}/{dataset_name}.pickle"
        if not os.path.exists(dataset_filename):
            missing_files.append(dataset_filename)

        for algorithm_name in opt.algorithms:
            algorithm_name = algorithm_name.replace("\\n", "\n").replace("\n", "")
            alg_filename = f"{opt.cache_path}/{dataset_name}_{algorithm_name}.pickle"
            if not os.path.exists(alg_filename):
                missing_files.append(alg_filename)

    return missing_files


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
        help="List of datasets to include in figure. Default is our_datasets.DATASET_SELECTOR.",
    )

    opt = p.parse_args()

    if isinstance(opt.algorithms, str):
        opt.algorithms = opt.algorithms.replace(" ", "").split(",")
    if isinstance(opt.datasets, str):
        if opt.datasets.lower() == "2d":
            opt.datasets = our_datasets.DATASETS2D
            if opt.figure_name == "my_figure":  # the default
                opt.figure_name = "figure1"
        elif opt.datasets.lower() == "complex":
            opt.datasets = our_datasets.COMPLEX_DATASETS
            if opt.figure_name == "my_figure":  # the default
                opt.figure_name = "figure2"
        elif opt.datasets.lower() == "main1":
            opt.datasets = our_datasets.DATASETS2D
            opt.algorithms = our_algorithms.CORE_SELECTOR
            title_fontsize = 22
            ari_fontsize = 18
            if opt.figure_name == "my_figure":  # the default
                opt.figure_name = "figure1_main"
        elif opt.datasets.lower() == "main2":
            opt.datasets = our_datasets.CORE_HD_DATASETS
            opt.algorithms = our_algorithms.CORE_SELECTOR
            title_fontsize = 20
            ari_fontsize = 18
            if opt.figure_name == "my_figure":  # the default
                opt.figure_name = "figure2_main"

        else:  # otherwise, handle it as a list of datasets
            opt.datasets = opt.datasets.replace(" ", "").split(",")

    # create cache and figure paths
    os.makedirs(opt.cache_path, exist_ok=True)
    os.makedirs(opt.figure_path, exist_ok=True)

    return opt


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
