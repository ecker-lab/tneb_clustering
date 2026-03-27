import corc
import corc.utils
import corc.graph_metrics.neb
import corc.graph_metrics.tmm_gmm_neb
import corc.our_datasets

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import jax.numpy as jnp
import tqdm

cache_path = "cache"


def get_tNEB(dataset_name, cache_path):
    filename = os.path.join(cache_path, f"{dataset_name}_TMM-NEB.pickle")
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            tmm = pickle.load(f)[0]
    else:
        X, y, tsne = corc.utils.load_dataset(dataset_name)
        tmm = corc.graph_metrics.neb.NEB(
            data=X, labels=y, n_components=15, optimization_iterations=1
        )
        tmm.fit(X)  # the fitting essentially initializes the mixture model
    return tmm


def compute_iterations_ablation(dataset_name, desired_iterations, cache_path):
    tmm = get_tNEB(dataset_name, cache_path)
    all_adjacencies = dict()
    # all_paths = dict()
    for iterations in desired_iterations:
        all_adjacencies[iterations], _, _ = (
            corc.graph_metrics.tmm_gmm_neb.compute_neb_paths_batch(
                tmm.mixture_model.centers,
                tmm.mixture_model.covs,
                tmm.mixture_model.weights,
                tmm.mixture_model.df,
                iterations=iterations,
                num_NEB_points=tmm.num_NEB_points,
            )
        )
    return all_adjacencies


def compute_pathlength_ablation(dataset_name, desired_pathlength, cache_path):
    tmm = get_tNEB(dataset_name, cache_path)
    all_adjacencies = dict()
    # all_paths = dict()
    for pathlength in desired_pathlength:
        all_adjacencies[pathlength], _, _ = (
            corc.graph_metrics.tmm_gmm_neb.compute_neb_paths_batch(
                tmm.mixture_model.centers,
                tmm.mixture_model.covs,
                tmm.mixture_model.weights,
                tmm.mixture_model.df,
                iterations=500,
                num_NEB_points=pathlength,
            )
        )
    return all_adjacencies


def load_or_compute_adjs_per_dataset(
    datasets, cache_path, ablation_type, desired_values
):
    adjs_per_dataset = dict()
    cached_results_path = os.path.join(
        cache_path, "ablations", f"adjs_per_dataset_{ablation_type}.pkl"
    )
    if os.path.exists(cached_results_path):
        with open(cached_results_path, "rb") as f:
            adjs_per_dataset = pickle.load(f)
    else:
        adjs_per_dataset = dict()

    for dataset in datasets:
        print(f"Starting {dataset}")
        if dataset not in adjs_per_dataset:
            adjs_per_dataset[dataset] = dict()

        # Check if all desired steps have been computed
        missing_values = [
            value for value in desired_values if value not in adjs_per_dataset[dataset]
        ]
        if missing_values:
            print(f"computing missing values for {missing_values}")
            if ablation_type == "iterations":
                all_adjacencies = compute_iterations_ablation(
                    dataset, missing_values, cache_path
                )
            elif ablation_type == "pathlength":
                all_adjacencies = compute_pathlength_ablation(
                    dataset, missing_values, cache_path
                )
            for value in missing_values:
                adjs_per_dataset[dataset][value] = all_adjacencies[value]

            # Save the updated results
            with open(cached_results_path, "wb") as f:
                pickle.dump(adjs_per_dataset, f)

        print(f"Finished {dataset}")

    return adjs_per_dataset


def plot_adjs_per_dataset(adjs_per_dataset, ablation_type, datasets=None):
    # Create a figure and axis
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots()

    if datasets is None:
        datasets = adjs_per_dataset.keys()

    colors = cm.tab20(np.linspace(0, 1, len(datasets)))
    for i, dataset in enumerate(datasets):
        best_adj = adjs_per_dataset[dataset][
            max(list(adjs_per_dataset[dataset].keys()))
        ]  # the one with most iterations

        # Iterate over the adjacency matrices
        values_list = list(adjs_per_dataset[dataset].keys())
        # diff_list = [
        #     np.sum(adjs_per_dataset[dataset][value] - best_adj) for value in values_list
        # ]
        diff_list = [
            np.mean(np.abs(adjs_per_dataset[dataset][value] - best_adj))
            for value in values_list
        ]

        # Normalize the differences
        # normalized_diff_list = [diff / diff_list[0] for diff in diff_list]

        # Plot the normalized differences with markers

        label = corc.our_datasets.dataset_displaynames[dataset].replace("\n", " ")
        ax.plot(values_list, diff_list, label=label, marker="o", color=colors[i])

    # Set the title and labels
    if ablation_type == "iterations":
        ax.set_xlabel("Number of Iterations")
    elif ablation_type == "pathlength":
        ax.set_xlabel("Number of NEB Points")
    ax.set_ylabel("$\Delta$ NEB Difference")
    ax.set_ylim([0, 1])  # Set the y-axis limit to 1.1 to make the plot more readable
    ax.legend()

    # Show the plot
    filename = os.path.join(
        cache_path, "..", "figures", f"ablation_{ablation_type}.pdf"
    )
    plt.savefig(filename)
    print(f"written to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--ablation_type",
        type=str,
        choices=["iterations", "pathlength"],
        default="iterations",
    )
    args = parser.parse_args()

    datasets = corc.our_datasets.DATASETS2D + corc.our_datasets.CORE_HD_DATASETS

    if args.ablation_type == "iterations":
        desired_values = [
            1,
            10,
            20,
            50,
            75,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            450,
            500,
        ]
    elif args.ablation_type == "pathlength":
        desired_values = [10, 20, 50, 75, 100, 150, 200, 300, 400, 500]

    adjs_per_dataset = load_or_compute_adjs_per_dataset(
        datasets, cache_path, args.ablation_type, desired_values
    )
    plot_adjs_per_dataset(
        adjs_per_dataset,
        args.ablation_type,
        datasets=corc.our_datasets.CORE_HD_DATASETS,
    )


if __name__ == "__main__":
    main()
