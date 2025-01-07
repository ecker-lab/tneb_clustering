import os

# make sure that jax does not try allocating the whole GPU
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
import warnings
import numpy as np
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import our_datasets
import our_algorithms
from openTSNE import TSNE
import re
import pickle
import sys
import corc.utils

"""
This file performs the clustering computation and TSNE conversion of all selected datasets and clustering algorithms. The results will be stored in the cache/*.pickle files.
Those files can then be used by the "partner" script "create_clustering_figure.py" to create the plots.
Note that for NEB also TSNE embeddings of the paths between all pairs of nodes are generated, even though this takes a lot of time.

one can call the script with the list of datasets that should be used.
"""


def main():
    # get the datasets and default parameters for them
    my_datasets = our_datasets.our_datasets()
    default_base = my_datasets.default_base
    dataset_selector = (
        sys.argv[1:] if len(sys.argv[1:]) > 0 else our_datasets.DATASET_SELECTOR
    )
    datasets = my_datasets.select_datasets(dataset_selector)

    path = "cache"
    os.makedirs(path, exist_ok=True)

    print(f"Datasets: {[algo_params['name'] for _, algo_params in datasets]}")

    for i_dataset, (dataset, params) in enumerate(datasets):

        print(f"Dataset: {i_dataset+1} of {len(datasets)}: {params['name']}")

        # try to load the dataset from disk
        dataset_name = re.sub(" ", "_", params["name"])
        dataset_filename = f"{path}/{dataset_name}.pickle"
        if os.path.exists(dataset_filename):
            with open(dataset_filename, "rb") as f:
                dataset_info = pickle.load(f)
                X, y = dataset_info["dataset"]
                X2D = dataset_info["X2D"]
                # not loading algo_params such that we can make changes there
        else:
            # We have to actually do something
            X, y = dataset
            y = [0] * len(X) if y is None else np.array(y, dtype="int")

            # normalize dataset for easier parameter selection
            X = StandardScaler().fit_transform(X)

            # dimensionality reduction for plotting results in 2D
            if params["dim"] > 2:
                starttime = time.time()
                perplexity = 100 if dataset in ["Paul15"] else 30
                print("computing TSNE", end="")
                X2D = corc.utils.get_TSNE_embedding(data_X=X, perplexity=perplexity)
                print(
                    f"finished TSNE fit for {params['name']} in {time.time()-starttime:.2f} seconds"
                )
            else:
                X2D = None

            # write dataset with TSNE to disk
            dataset_info = dict()
            dataset_info["dataset"] = (X, y)
            dataset_info["X2D"] = X2D
            dataset_info["algo_params"] = params
            with open(dataset_filename, "wb") as f:
                pickle.dump(dataset_info, f)

        clustering_algorithms = our_algorithms.get_clustering_objects(params, X)

        print(f"Algorithms: {[name for name,_ in clustering_algorithms]}")

        for name, algorithm in clustering_algorithms:
            # check whether this was already computed
            alg_name = re.sub("\n", "", name)
            filename = f"{path}/{dataset_name}_{alg_name}.pickle"
            if os.path.exists(filename):
                print(f"{filename} already exists. Skipping.")
                continue

            t0 = time.time()
            print(f"algorithm {name}", end="")

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
                algorithm.fit(X)

            t1 = time.time()
            print(f" (fit in {t1 - t0:.2f} seconds)")

            if hasattr(algorithm, "labels_"):
                y_pred = algorithm.labels_.astype(int)
            else:
                if hasattr(algorithm, "predict_with_target"):
                    y_pred = algorithm.predict_with_target(X, len(np.unique(y))).astype(
                        int
                    )
                else:
                    y_pred = algorithm.predict(X)

            # store dataset in NEB algorithm object
            if "NEB" in name:
                algorithm.data = X
                algorithm.labels = y

            # saving algorithm object (containing everything we computed)
            alg_name = re.sub("\n", "", name)
            filename = f"{path}/{dataset_name}_{alg_name}.pickle"
            print(f"saving to {filename}")
            with open(filename, "wb") as f:
                pickle.dump(algorithm, f)


if __name__ == "__main__":
    main()
