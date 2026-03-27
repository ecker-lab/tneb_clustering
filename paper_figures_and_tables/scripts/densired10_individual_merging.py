import os
import pickle
import corc.our_algorithms


def merge_individuals(dataset, dim, algorithm, cache_path="cache"):
    output_dir = f"{cache_path}/densired10"
    individuals_dir = f"{output_dir}/individuals"
    out_fn = f"{output_dir}/{dataset}{dim}_{algorithm}.pickle"

    # Collect all filenames for this combo
    missing = False
    all_clustering_objects = []
    for i in range(10):
        objects_i = []
        for j in range(10):
            ind_fn = f"{individuals_dir}/{dataset}{dim}_{algorithm}_i{i}_j{j}.pickle"
            if not os.path.exists(ind_fn):
                print(f"  Missing: {ind_fn}")
                missing = True
                break
            with open(ind_fn, "rb") as f:
                obj = pickle.load(f)
            objects_i.append(obj)
        if missing:
            break
        all_clustering_objects.append(objects_i)

    if not missing:
        with open(out_fn, "wb") as f:
            pickle.dump(all_clustering_objects, f)
            print(f"Merged and wrote {out_fn}")
    else:
        print(f"Skipped merging for {dataset}, {dim}, {algorithm}, missing files.")


def main():
    algorithm_names = corc.our_algorithms.CORE_SELECTOR
    dims = [8, 16, 32, 64]
    types = ["circles", "studt"]
    cache_path = "cache"

    for algo in algorithm_names:
        algo_clean = algo.replace("\n", "")
        for dim in dims:
            for dtype in types:
                merged_file = (
                    f"{cache_path}/densired10/{dtype}{dim}_{algo_clean}.pickle"
                )
                if os.path.exists(merged_file):
                    # print(f"{merged_file} already exists, skipping.")
                    continue
                print(f"Checking {dtype}, {dim}, {algo_clean}...")
                merge_individuals(dtype, dim, algo_clean, cache_path=cache_path)


if __name__ == "__main__":
    main()
