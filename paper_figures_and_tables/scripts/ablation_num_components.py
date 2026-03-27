# import corc
import corc.utils
import corc.graph_metrics.neb
import corc.our_datasets
import os
import pickle
import time
import argparse

cache_path = "cache"

# invoke using:
# for dataset in densired8 densired16 densired32 densired_soft_8 densired_soft_16 densired_soft_32 mnist8 mnist16 mnist32; do python paper_figures_and_tables/scripts/ablation_num_components.py --datasets "$dataset" & done; wait


default_datasets = [
    "densired8",
    "densired16",
    "densired32",
    "densired_soft_8",
    "densired_soft_16",
    "densired_soft_32",
    "mnist8",
    "mnist16",
    "mnist32",
]

# Setup command line argument parser
parser = argparse.ArgumentParser(description="Process datasets with corc.")
parser.add_argument(
    "-d",
    "--datasets",
    type=str,
    nargs="*",
    default=default_datasets,
    help="List of datasets to process",
)
args = parser.parse_args()
datasets = args.datasets

levels = [15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]


for dataset in datasets:
    filename = os.path.join(
        cache_path, "ablations", f"tmms_num_components_{dataset}.pickle"
    )
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            tmm_levels = pickle.load(f)
        if all(level in tmm_levels for level in levels):
            print(f"Not starting {dataset}. All levels already computed.")
            continue
    else:
        tmm_levels = dict()

    print(f"starting {dataset}")
    starttime = time.time()
    X, y, tsne = corc.utils.load_dataset(dataset, cache_path=cache_path)
    for level in levels:
        if level in tmm_levels.keys():
            continue
        tmms = list()
        for i in range(4):
            tmm = corc.graph_metrics.neb.NEB(
                data=X, labels=y, n_components=level, seed=42 + i * 100
            )
            tmm.fit(X)
            tmms.append(tmm)
        tmm_levels[level] = tmms
        with open(filename, "wb") as f:
            pickle.dump(tmm_levels, f)
        print(f"finished level {level} for {dataset}")
    print(f"finished {dataset} in {time.time()-starttime:.2f}s")
