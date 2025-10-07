import os

# make sure that jax does not try allocating the whole GPU
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
import corc.our_algorithms as our_algorithms
import pickle
import corc.utils
import argparse
import numpy as np

"""
This file performs the clustering computation of all selected datasets and clustering algorithms. The results will be stored in the cache/*.pickle files.
Those files can then be used by the "partner" script "create_clustering_figure.py" to create the overview plots.
Note that for NEB also TSNE embeddings of the paths between all pairs of nodes are generated, even though this takes a lot of time.

one can call the script with the list of datasets that should be used.
"""


def compute_clusters(X, params, algorithm_name, num_seeds):
    algorithms = list()
    base_seed = params["random_state"]

    for i in range(num_seeds):
        params["random_state"] = base_seed + i * 100
        _, algorithm = corc.our_algorithms.get_clustering_objects(
            params, X, selector=[algorithm_name]
        )[0]
        algorithm.fit(X)

        # Affinity_Propagation: reduce filesize (otherwise huge...)
        if hasattr(algorithm, "affinity_matrix_"):
            algorithm.affinity_matrix_ = None

        algorithms.append(algorithm)
        if algorithm_name in our_algorithms.DETERMINISTIC_ALGORITHMS:
            # we only need a single run for deterministic algorithms
            break
    return algorithms


def main(args):
    corc.utils.create_folder(args.cache_path)
    corc.utils.create_folder(os.path.join(args.cache_path, "cluster_objects"))

    filename = corc.utils.get_filename(
        args.dataset, args.algorithm, args.cache_path, index=args.index
    )
    algorithms = corc.utils.load_algorithms(
        args.dataset, args.algorithm, cache_path=args.cache_path, index=args.index
    )
    if algorithms is None or args.force:
        t0 = time.time()
        X, y, tsne, params = corc.utils.load_dataset(
            args.dataset, index=args.index, cache_path=args.cache_path, return_params=True
        )
        algorithms = compute_clusters(
            X, params, args.algorithm, args.num_seeds
        )
        # print("finished computing. Now starting to evaluate")
        aris = corc.utils.get_ari(algorithms, X, y)
        try:
            purities = corc.utils.get_purity_scores(algorithms, X, y)
        except Exception as e:
            print(f"{filename}: Error computing purity scores: {e}")
            purities = [None]

        msg = f"{args.algorithm} on {args.dataset} ({args.index}): ARI {np.mean(aris):.2f}±{np.std(aris):.2f}"
        if purities[0] is not None:
            msg += f", Purity {np.mean(purities):.2f}±{np.std(purities):.2f}"
        msg += f" ({time.time() - t0:.2f} seconds)"
        print(msg)
        print(f"saving to {filename}")
        with open(filename, "wb") as f:
            pickle.dump(algorithms, f)
    else:
        print(f"{filename} already exists.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        help="Dataset to be used. ",
        default="noisy_moons",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        help="algorithm to be used.",
        default="TMM-NEB",
    )
    parser.add_argument(
        "-c",
        "--cache_path",
        help="Path to the cache directory. (Default: cache)",
        default="cache",
    )
    parser.add_argument(
        "-n",
        "--num_seeds",
        type=int,
        help="Number of seeds to be used for non-deterministic algorithms. (Default: 10)",
        default=10,
    )
    parser.add_argument(
        "-i",
        "--index",
        help="Index to subselect the densired datasets",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation",
    )
    args = parser.parse_args()

    main(args)
