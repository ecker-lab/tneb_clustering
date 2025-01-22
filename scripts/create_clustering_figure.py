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
import sklearn

import corc.utils

"""
Example call: python scripts/create_clustering_figure.py --algorithms  "MiniBatch\nKMeans, Agglomerative\nClustering" --datasets "blobs1_8, mnist64"
"""

dataset_displaynames = {
    "noisy_moons": "noisy\nmoons",
    "noisy_circles": "noisy\ncircles",
    "varied": "varied\ndensity",
    "aniso": "anisotropic\nblobs",
    "blobs": "Gaussian\nblobs",
    "clusterlab10": "clusterlab10",
    ###########################
    ##### fig 2 datasets ######
    ###########################
    "blobs1_8": "Gaussian\nblobs 8D",
    "blobs1_16": "Gaussian\nblobs 16D",
    "blobs1_32": "Gaussian\nblobs 32D",
    "blobs1_64": "Gaussian\nblobs 64D",
    "blobs2_8": "Gaussian\nblobs 8D",
    "blobs2_16": "Gaussian\nblobs 16D",
    "blobs2_32": "Gaussian\nblobs 32D",
    "blobs2_64": "Gaussian\nblobs 64D",
    "densired8": "Densired\n'circles' 8D",
    "densired16": "Densired\n'circles' 16D",
    "densired32": "Densired\n'circles' 32D",
    "densired64": "Densired\n'circles' 64D",
    "densired_soft_8": "Densired\n'Stud-t' 8D",
    "densired_soft_16": "Densired\n'Stud-t' 16D",
    "densired_soft_32": "Densired\n'Stud-t' 32D",
    "densired_soft_64": "Densired\n'Stud-t' 64D",
    "mnist8": "MNIST-Nd\n8D",
    "mnist16": "MNIST-Nd\n16D",
    "mnist32": "MNIST-Nd\n32D",
    "mnist64": "MNIST-Nd\n64D",
}


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
    title_fontsize = 18
    ari_fontsize = 14
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

    os.makedirs(opt.cache_path, exist_ok=True)
    os.makedirs(opt.figure_path, exist_ok=True)

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

    plt.figure(figsize=(len(opt.algorithms) * 2 + 3, len(opt.datasets) * 2))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
    )
    plot_num = 1

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
        ax = plt.subplot(len(opt.datasets), len(opt.algorithms) + 1, plot_num)
        if i_dataset == 0:
            plt.title("Ground Truth", size=title_fontsize)
        colors = get_color_scheme(y, y)
        plt.scatter(points[:, 0], points[:, 1], s=10, color=colors[y])
        plt.xticks(())
        plt.yticks(())
        ax.set_ylabel(dataset_displaynames[dataset_name], size=title_fontsize)
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
            algorithm.data = X

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
                    algorithm_name.replace("\\n", "\n"),
                    size=title_fontsize,
                )
                # algorithm_name.replace("\\n", "\n").replace("\n", " "), size=18

            # plot points
            colors = get_color_scheme(y_pred, y)
            y_pred_permuted = corc.utils.reorder_colors(y_pred, y)
            plt.scatter(points[:, 0], points[:, 1], s=10, color=colors[y_pred_permuted])

            # plot graph
            if algorithm_name in [
                "GWG-dip",
                "GWG-pvalue",
                "PAGA",
                "Stavia",
                "TMM-NEB",
                "GMM-NEB",
            ]:
                algorithm.plot_graph(X2D=X2D, target_num_clusters=len(np.unique(y)))

            # add ARI scores to the plot
            ari_score = sklearn.metrics.adjusted_rand_score(y, y_pred)
            plt.text(
                0.02,
                0.88,
                f"ARI {ari_score:.2f}",
                transform=plt.gca().transAxes,
                fontweight="bold",
                size=ari_fontsize,
            )

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
