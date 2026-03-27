import os

# make sure that jax does not try allocating the whole GPU
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import pickle
from sklearn.preprocessing import StandardScaler

import corc.utils
import corc.our_algorithms as our_algorithms
import tqdm


def fit_clustering_objects(X, algorithm_name, params):
    clustering_objects = list()
    base_seed = params["random_state"]
    for i in tqdm.trange(10):
        params["random_state"] = base_seed + 100 * i
        _, algorithm = corc.our_algorithms.get_clustering_objects(
            params, X, selector=[algorithm_name]
        )[0]
        algorithm.fit(X)
        clustering_objects.append(algorithm)
        if algorithm_name in our_algorithms.DETERMINISTIC_ALGORITHMS:
            # we only need to train a single one in this case
            break
    return clustering_objects


def main(args):
    target_filename = (
        f"{args.cache_path}/densired10/{args.dataset}{args.dim}_{args.algorithm}.pickle"
    )
    if os.path.exists(target_filename):
        print(f"{target_filename} already exists, exiting.")

    # open the datasets
    filename = f"datasets/densired_{args.dataset}10.pickle"
    with open(filename, "rb") as f:
        data = pickle.load(f)

    # load "params" for the normal densired dataset
    dataset_name = f"densired{'_soft_' if args.dataset=='circles' else ''}{args.dim}"
    _, _, _, params = corc.utils.load_dataset(
        dataset_name, cache_path=args.cache_path, return_params=True
    )

    all_clustering_objects = list()
    for i in range(10):
        X = data[i][args.dim][:, :-1]
        y = data[i][args.dim][:, -1]
        X = StandardScaler().fit_transform(X)
        all_clustering_objects.append(fit_clustering_objects(X, args.algorithm, params))
        print(
            f"({i+1}/10) Finished {len(all_clustering_objects[-1])} runs for {args.algorithm} on {args.dataset}{args.dim}"
        )
    with open(target_filename, "wb") as f:
        pickle.dump(all_clustering_objects, f)
        print(f"stored {target_filename}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["circles", "studt"],
        help="List of datasets to be used.",
    )
    parser.add_argument(
        "--dim",
        choices=[8, 16, 32, 64],
        type=int,
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        help="algorithm to be used",
        default="TMM-NEB",
    )
    parser.add_argument(
        "-c",
        "--cache_path",
        help="Path to the cache directory. (Default: cache)",
        default="cache",
    )
    args = parser.parse_args()

    main(args)
