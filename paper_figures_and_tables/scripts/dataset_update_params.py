import os
import pickle
import corc.our_datasets

cache_path = "cache"

for dataset in corc.our_datasets.DATASET_SELECTOR:
    filename = os.path.join(cache_path, "datasets", f"{dataset}.pickle")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    params = corc.our_datasets.our_datasets(dataset_folder="../datasets").select_datasets([dataset])[0][1]

    if dataset.startswith("mnist") or dataset.startswith("densired"):
        for index in range(10):
            data[index]["dataset_info"] = params
    else: # 2D Datasets
        data["dataset_info"] = params

    with open(filename, "wb") as f:
        pickle.dump(data, f)

