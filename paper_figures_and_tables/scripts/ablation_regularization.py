# import corc
import corc.utils
import corc.graph_metrics
import corc.our_datasets
import os
import pickle
import time

cache_path = "../../cache"

# datasets = corc.our_datasets.CORE_HD_DATASETS + corc.our_datasets.DATASETS2D

datasets = [
    "densired8",
    "densired16",
    "densired32",
    "densired_soft_8",
    "densired_soft_16",
    "densired_soft_32",
    "mnist8",
    "mnist16",
    "mnist32",
] + corc.our_datasets.DATASETS2D

levels = [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

filename = os.path.join(cache_path, "ablations", "tmms_regularization.pickle")

if os.path.exists(filename):
    with open(filename, "rb") as f:
        tmm_levels = pickle.load(f)
    print(f"Loaded {len(tmm_levels)} datasets from {filename}")
else:
    tmm_levels = dict()

for dataset in datasets:
    if dataset in tmm_levels:
        print(f"Skipping {dataset} as it is already present in the pickle file")
        continue

    print(f"starting {dataset}")
    starttime = time.time()
    X, y, tsne = corc.utils.load_dataset(dataset, cache_path=cache_path)
    tmm_levels[dataset] = dict()
    for level in levels:
        tmms = list()
        for _ in range(4):
            tmm = corc.graph_metrics.neb.NEB(
                data=X, labels=y, n_components=25, tmm_regularization=level
            )
            tmm.fit(X)
            tmms.append(tmm)
        tmm_levels[dataset][level] = tmms
    print(f"finished {dataset} in {time.time()-starttime:.2f}s")

    with open(filename, "wb") as f:
        pickle.dump(tmm_levels, f)
