import os
import pickle
import argparse

from sklearn.preprocessing import StandardScaler
import corc.utils
import corc.our_algorithms as our_algorithms


def fit_clustering_object(X, algorithm_name, params, j):
    # Only one run, set the right random seed
    params = dict(params)  # Copy to avoid sharing state
    params["random_state"] = params["random_state"] + 100 * j
    _, algorithm = corc.our_algorithms.get_clustering_objects(
        params, X, selector=[algorithm_name]
    )[0]
    algorithm.fit(X)
    return algorithm


def main(args):
    output_dir = f"{args.cache_path}/densired10/individuals"
    os.makedirs(output_dir, exist_ok=True)
    out_fn = f"{output_dir}/{args.dataset}{args.dim}_{args.algorithm}_i{args.i}_j{args.j}.pickle"
    if os.path.exists(out_fn):
        print(f"{out_fn} already exists")
        return

    # open the required dataset instance
    filename = f"datasets/densired_{args.dataset}10.pickle"
    with open(filename, "rb") as f:
        data = pickle.load(f)
    X = data[args.i][args.dim][:, :-1]
    y = data[args.i][args.dim][:, -1]
    X = StandardScaler().fit_transform(X)

    # load "params"
    dataset_name = f"densired{'_soft_' if args.dataset=='circles' else ''}{args.dim}"
    _, _, _, params = corc.utils.load_dataset(
        dataset_name, cache_path=args.cache_path, return_params=True
    )

    # Fit model
    algorithm = fit_clustering_object(X, args.algorithm, params, args.j)

    # Store result
    with open(out_fn, "wb") as f:
        pickle.dump(algorithm, f)
        print(f"Stored {out_fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["circles", "studt"], required=True)
    parser.add_argument("--dim", choices=[8, 16, 32, 64], required=True, type=int)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--cache_path", default="cache")
    parser.add_argument("--i", type=int, required=True, help="dataset index (0-9)")
    parser.add_argument("--j", type=int, required=True, help="fit index (0-9)")
    args = parser.parse_args()
    main(args)
