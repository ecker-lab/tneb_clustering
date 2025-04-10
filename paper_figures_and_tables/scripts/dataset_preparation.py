# this file essentially computes TSNE for all datasets and
# saves them to a pickle file (one file per dataset)


import corc.our_datasets
import corc.visualization
import os
import pickle
import time


def main(args):
    """Computes the TSNE embeddings for all datasets and saves them to pickle files."""
    dataset_path = os.path.join(args.cache_path, "datasets")
    corc.utils.create_folder(dataset_path)

    all_datasets = corc.our_datasets.our_datasets(
        dataset_folder=os.path.join(args.cache_path, "../datasets"),
    ).get_datasets()

    for i, dataset in enumerate(all_datasets):
        dataset_name = dataset[1]["name"]
        dataset_filename = os.path.join(dataset_path, f"{dataset_name}.pickle")
        if os.path.exists(dataset_filename):
            print(f"Dataset {dataset_name} already exists. Skipping...")
            continue

        starttime = time.time()
        print(f"Computing TSNE for {dataset_name} ({i+1}/{len(all_datasets)})", end="")
        X, y = dataset[0]
        tsne = corc.visualization.get_TSNE_embedding(X)
        dataset = {
            "dataset": (X, y),
            "X2D": tsne,
            "dataset_name": dataset[1]["name"],
            "dataset_info": dataset[1],
        }
        with open(dataset_filename, "wb") as f:
            pickle.dump(dataset, f)
        print(f" done. {time.time()-starttime:.2f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_path",
        type=str,
        default="cache",
        help="path to cache directory",
    )

    args = parser.parse_args()

    main(args)
