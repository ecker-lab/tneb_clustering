# this file essentially computes TSNE for all datasets and
# saves them to a pickle file (one file per dataset)


import corc.our_datasets
import corc.visualization
import corc.utils
import os
import pickle
import time
import numpy as np
import tqdm


def main(args):
    """Computes the TSNE embeddings for all datasets and saves them to pickle files."""
    dataset_path = os.path.join(args.cache_path, "datasets")
    corc.utils.create_folder(dataset_path)

    all_datasets = corc.our_datasets.our_datasets(
        dataset_folder=os.path.join(args.cache_path, "../datasets"),
    ).get_datasets()

    for i, dataset in enumerate(all_datasets):
        dataset_name = dataset[1]["name"]
        dataset_filename = corc.utils.get_dataset_filename(
            dataset_name, args.cache_path
        )
        if os.path.exists(dataset_filename):
            print(f"Dataset {dataset_name} already exists. Skipping...")
            continue

        starttime = time.time()
        print(f"Computing TSNE for {dataset_name} ({i+1}/{len(all_datasets)})")
        Xs, ys = dataset[0]
        if dataset_name in corc.our_datasets.CORE_HD_DATASETS:
            # if dataset_name.lower().startswith("densired"):
            # densired/mnist → list of 10 sub‑datasets
            results = list()
            for X, y in tqdm.tqdm(zip(Xs, ys)):
                results.append(make_entry(X, y, dataset_name, dataset[1]))
        else:
            # regular single‑dataset case (2D)
            results = make_entry(Xs, ys, dataset_name, dataset[1])

        with open(dataset_filename, "wb") as f:
            pickle.dump(results, f)
        print(f" done. {time.time()-starttime:.2f}s")


def make_entry(X, y, name, info):
    y = np.array(y, dtype=int)
    tsne = corc.visualization.get_TSNE_embedding(X)
    return {
        "dataset": (X, y),
        "X2D": tsne,
        "dataset_name": name,
        "dataset_info": info,
    }


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
