import os

# make sure that jax does not try allocating the whole GPU
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
import warnings
import numpy as np
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import corc.our_datasets as our_datasets
import corc.our_algorithms as our_algorithms
from openTSNE import TSNE
import re
import pickle
import sys
import corc.utils
import argparse

"""
This file performs the clustering computation of all selected datasets and clustering algorithms. The results will be stored in the cache/*.pickle files.
Those files can then be used by the "partner" script "create_clustering_figure.py" to create the overview plots.
Note that for NEB also TSNE embeddings of the paths between all pairs of nodes are generated, even though this takes a lot of time.

one can call the script with the list of datasets that should be used.
"""


def main(args):
    # get the datasets and default parameters for them
    # if no datasets are given, all datasets will be used
    if args.datasets is None or len(args.datasets) == 0:
        dataset_selector = our_datasets.DATASET_SELECTOR
    elif args.datasets[0] == "2d":
        dataset_selector = our_datasets.DATASETS2D
    else:
        dataset_selector = args.datasets
    print(f"Datasets: {dataset_selector}")

    cache_path = "cache"
    corc.utils.create_folder(cache_path)
    if args.algorithms == "all":
        clustering_algorithm_selector = our_algorithms.ALGORITHM_SELECTOR
    elif args.algorithms == "core":
        clustering_algorithm_selector = our_algorithms.CORE_SELECTOR
    elif args.algorithms == "tneb":
        clustering_algorithm_selector = ["TMM-NEB"]
    elif args.algorithms == "ours":
        clustering_algorithm_selector = ["GMM-NEB", "TMM-NEB"]
    print(f"Algorithms: {clustering_algorithm_selector}")

    for i_dataset, dataset_name in enumerate(dataset_selector):
        X, y, tsne, params = corc.utils.load_dataset(
            dataset_name, cache_path=cache_path, return_params=True
        )

        clustering_algorithms = our_algorithms.get_clustering_objects(
            params, X, selector=clustering_algorithm_selector
        )

        for name, algorithm in clustering_algorithms:
            # check whether this was already computed
            alg_name = re.sub("\n", "", name)
            filename = os.path.join(cache_path, f"{dataset_name}_{alg_name}.pickle")
            if os.path.exists(filename):
                print(f"{filename} already exists. Skipping.")
                continue

            t0 = time.time()
            print(f"algorithm {alg_name}", end="")

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the "
                    + "connectivity matrix is [0-9]{1,2}"
                    + " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding"
                    + " may not work as expected.",
                    category=UserWarning,
                )
                if alg_name not in our_algorithms.DETERMINISTIC_ALGORITHMS:
                    # train 10 with different seeds
                    algorithms = list()
                    base_seed = params["random_state"]
                    for i in range(10):
                        params["random_state"] = base_seed + i
                        _, algorithm = corc.our_algorithms.get_clustering_objects(
                            params, X, selector=[name]
                        )[0]
                        algorithm.fit(X)
                        algorithms.append(algorithm)
                    algorithm = algorithms

                else:
                    algorithm.fit(X)

            t1 = time.time()
            print(f" (fit in {t1 - t0:.2f} seconds)")

            # saving algorithm object (containing everything we computed)
            print(f"saving to {filename}")
            with open(filename, "wb") as f:
                pickle.dump(algorithm, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        help="List of datasets to be used. If not provided, all datasets in our_datasets.DATASET_SELECTOR will be used.",
    )
    parser.add_argument(
        "-a",
        "--algorithms",
        choices=["all", "core", "tneb", "ours"],
        help="algorithms to be used.",
        default="all",
    )
    args = parser.parse_args()

    main(args)
