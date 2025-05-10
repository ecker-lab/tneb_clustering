import corc
import corc.utils
import corc.graph_metrics
import corc.our_datasets
import os
import pickle
import time

cache_folder = "cache"
datasets = corc.our_datasets.CORE_HD_DATASETS + corc.our_datasets.DATASETS2D


def get_tNEB(dataset_name, cache_path):
    filename = os.path.join(cache_path, f"{dataset_name}_TMM-NEB.pickle")
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            tmm = pickle.load(f)[0]
    else:
        X, y, tsne = corc.utils.load_dataset(dataset_name, cache_path=cache_folder)
        tmm = corc.graph_metrics.neb.NEB(
            data=X, labels=y, n_components=15, optimization_iterations=500
        )
    return tmm


all_tmms = dict()
for dataset in datasets:
    print(f"starting {dataset}")
    starttime = time.time()
    X, y, tsne = corc.utils.load_dataset(dataset, cache_path=cache_folder)
    tmm = get_tNEB(dataset, cache_folder)
    tmm.n_neighbors = None
    tmm.fit(X, knn=None)
    all_tmms[dataset] = tmm
    print(f"finished {dataset} in {time.time()-starttime:.2f}s")

filename = os.path.join(cache_folder, "ablations", "tmms_fitted_all_pairs.pickle")
with open(filename, "wb") as f:
    pickle.dump(all_tmms, f)
