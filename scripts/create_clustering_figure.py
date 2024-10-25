import pickle
import configargparse
import our_datasets
import our_algorithms
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tqdm
import os
import sys

import corc.utils

"""
Example call: python scripts/create_clustering_figure.py --algorithms  "MiniBatch\nKMeans, Agglomerative\nClustering" --datasets "blobs1_8, mnist64"
"""


def main():
    p = configargparse.ArgumentParser()
    p.add(
        "-c",
        "--config",
        required=False,
        is_config_file=True,
        help="Path to config file.",
    )
    # general
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

    print(f"{opt.algorithms}")

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
        else:  # otherwise, handle it as a list of datasets
            opt.datasets = opt.datasets.replace(" ", "").split(",")

    os.makedirs(opt.cache_path, exist_ok=True)
    os.makedirs(opt.figure_path, exist_ok=True)

    plt.figure(figsize=(len(opt.algorithms) * 2 + 3, len(opt.datasets) * 2))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
    )
    plot_num = 1

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

    if missing_files:
        print("The following files are missing:")
        for file in missing_files:
            print(file)
        response = input("Continue anyway? (Y/n): ")
        if response != "" and response.lower() != "y":
            sys.exit(1)

    # now start the computation
    for i_dataset, dataset_name in enumerate(tqdm.tqdm(opt.datasets)):
        # load dataset
        dataset_filename = f"{opt.cache_path}/{dataset_name}.pickle"
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
        ax = plt.subplot(len(opt.datasets), len(opt.algorithms) + 1, plot_num)
        if i_dataset == 0:
            plt.title("Ground Truth", size=18)
        colors = get_color_scheme(y, y)
        plt.scatter(points[:, 0], points[:, 1], s=10, color=colors[y])
        plt.xticks(())
        plt.yticks(())
        ax.set_ylabel(dataset_name, size=18)
        plot_num += 1

        # plotting the other algorithms
        for i_algorithm, algorithm_name in enumerate(opt.algorithms):
            # load algorithm object
            alg_name = algorithm_name.replace("\\n", "\n").replace("\n", "")
            alg_filename = f"{opt.cache_path}/{dataset_name}_{alg_name}.pickle"
            # print(alg_filename)

            # skip if the file is not there
            if alg_filename in missing_files:
                plot_num += 1
                continue
            with open(alg_filename, "rb") as f:
                algorithm = pickle.load(f)

            # extracting the predictions
            if hasattr(algorithm, "labels_"):
                y_pred = algorithm.labels_.astype(int)
            else:
                if hasattr(algorithm, "predict_with_target"):
                    y_pred = algorithm.predict_with_target(X, len(np.unique(y))).astype(
                        int
                    )
                else:
                    y_pred = algorithm.predict(X)

            # create plotting area
            plt.subplot(len(opt.datasets), len(opt.algorithms) + 1, plot_num)
            if i_dataset == 0:
                plt.title(
                    algorithm_name.replace("\\n", "\n").replace("\n", " "), size=18
                )

            # drawing the background for NEB in the 2D case
            if dim == 2 and algorithm_name in ["TMM-NEB", "GMM-NEB"]:
                image_resolution = 128
                linspace_x = np.linspace(
                    X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, image_resolution
                )
                linspace_y = np.linspace(
                    X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, image_resolution
                )
                XY = np.stack(np.meshgrid(linspace_x, linspace_y), -1)
                tmm_probs = algorithm.mixture_model.score_samples(
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

            colors = get_color_scheme(y_pred, y)
            y_pred_permuted = corc.utils.reorder_colors(y_pred, y)
            plt.scatter(points[:, 0], points[:, 1], s=10, color=colors[y_pred_permuted])

            if algorithm_name in [
                "GWG-dip",
                "GWG-pvalue",
                "PAGA",
                "Stavia",
                "TMM-NEB",
                "GMM-NEB",
            ]:
                algorithm.plot_graph(X2D=X2D)

            plt.xticks(())
            plt.yticks(())
            plot_num += 1

    plt.savefig(f"{opt.figure_path}/{opt.figure_name}.pdf", bbox_inches="tight")
    plt.savefig(
        f"{opt.figure_path}/{opt.figure_name}.png", bbox_inches="tight", dpi=100
    )


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


if __name__ == "__main__":
    main()
