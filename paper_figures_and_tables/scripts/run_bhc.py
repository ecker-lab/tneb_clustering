import time
import numpy as np
import sklearn.metrics
import pickle

import networkx as nx
import numpy as np
import os
import argparse

import corc.utils
import corc.our_datasets
import corc.bhc.bhc
import corc.bhc.prior as prior
import corc.purity

# cache_path = "../../../cache"


def get_dataset(dataset, size, cache_path="cache"):
    # dataset loading
    X, y, tsne = corc.utils.load_dataset(dataset, cache_path=cache_path)
    np.random.seed(42)
    if X.shape[0] > size and size > 0:
        _, subsampled_data, _, subsampled_ys = sklearn.model_selection.train_test_split(
            X, y, test_size=size, stratify=y, random_state=42
        )
    else:
        subsampled_data = X
        subsampled_ys = y
    # indices = np.random.choice(X.shape[0], size=size, replace=False)
    # subsampled_data = X[indices]
    # subsampled_ys = y[indices]
    return subsampled_data, subsampled_ys


def get_bhc(subsampled_data, g=20, scale_factor=0.001, alpha=1):
    model = prior.NormalInverseWishart.create(subsampled_data, g, scale_factor)
    start_time = time.time()
    bhc_result = corc.bhc.bhc.BayesianHierarchicalClustering(
        subsampled_data, model, alpha, cut_allowed=False, verbose=True
    ).build()
    end_time = time.time()
    print(f"Time taken for BHC: {end_time - start_time:.2f} seconds")
    return bhc_result


def analyze_result(result, subsampled_data, subsampled_ys):
    arc_list = result.arc_list
    node_ids = result.node_ids

    G = nx.DiGraph()
    for arc in arc_list:
        G.add_edge(arc.source, arc.target)

    # Subgraph excluding last 6 nodes
    num_classes = len(np.unique(subsampled_ys))
    nodes_to_remove = node_ids[-(num_classes - 1) :]
    subgraph = G.subgraph([node for node in G.nodes() if node not in nodes_to_remove])
    subgraph = subgraph.to_undirected()
    # Connected components
    connected_components = list(nx.connected_components(subgraph))

    # Create y_pred labels
    y_pred = np.zeros(len(node_ids), dtype=int)
    for i, component in enumerate(connected_components):
        for node in component:
            y_pred[node] = i

    # Filter y_pred and y_true to only include nodes within subsampled_data range
    valid_indices = np.where(np.array(node_ids) < len(subsampled_data))[0]
    y_pred_filtered = y_pred[valid_indices]
    y_true_filtered = subsampled_ys[valid_indices]

    # print(np.unique(y_pred_filtered, return_counts=True)[1])
    # print(np.unique(y_true_filtered, return_counts=True)[1])

    # ARI calculation
    ari = sklearn.metrics.adjusted_rand_score(y_true_filtered, y_pred_filtered)

    return ari


# SIZE = 1000
# for dataset in corc.our_datasets.CORE_HD_DATASETS:
#     print(f"Processing dataset: {dataset}")
#     filepath = f"{cache_path}/bhc/{dataset}_{SIZE}.pkl"
#     if os.path.exists(filepath):
#         print(f"Result for {dataset} already exists. Skipping...")
#         continue
#     subsampled_data, subsampled_ys = get_dataset(dataset, SIZE)
#     bhc_result = get_bhc(subsampled_data)

#     ari = analyze_result(bhc_result, subsampled_data, subsampled_ys)

#     print(f"ARI of BHC for {dataset}: {ari:.2f}")

#     # store result object
#     with open(f"{cache_path}/bhc/{dataset}_{SIZE}.pkl", "wb") as f:
#         pickle.dump(bhc_result, f)

#     print(" ")
# print("All datasets processed.")


def main():
    parser = argparse.ArgumentParser(description="Process datasets and calculate ARI.")
    parser.add_argument(
        "dataset",
        type=str,
        choices=corc.our_datasets.CORE_HD_DATASETS,
        help="The dataset to process.",
    )
    parser.add_argument(
        "size",
        type=int,
        default=150,
        help="The size of the subsampled dataset.",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="cache",
        help="Path to the cache directory.",
    )
    parser.add_argument(
        "index",
        type=int,
        default=-1,
        help="For densired datasets: which subdataset to run on",
    )

    args = parser.parse_args()

    cache_path = args.cache_path
    # os.makedirs(f"{cache_path}/bhc", exist_ok=True)

    print(f"Processing dataset: {args.dataset}_{args.index} (size: {args.size})")
    filepath = f"{cache_path}/bhc/{args.dataset}_{args.index}_{args.size}.pkl"
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping...")
        return

    subsampled_data, subsampled_ys = get_dataset(
        args.dataset, args.size, cache_path=cache_path
    )
    bhc_result = get_bhc(subsampled_data)

    ari = analyze_result(bhc_result, subsampled_data, subsampled_ys)
    dendrogram = bhc_result.get_dendrogram()
    purity = corc.purity.dendrogram_purity(dendrogram, subsampled_ys)

    print(f"ARI of BHC for {args.dataset}_{args.index}: {ari:.2f}")
    print(f"Purity of BHC for {args.dataset}_{args.index}: {purity:.2f}")

    # store result object
    with open(f"{cache_path}/bhc/{args.dataset}_{args.size}.pkl", "wb") as f:
        pickle.dump(bhc_result, f)

    print(" ")


if __name__ == "__main__":
    main()
