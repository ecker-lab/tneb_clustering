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

# import argparse
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
            data=X, labels=y, n_components=15, optimization_iterations=500
        )
        tmm.fit(X)  # the fitting essentially initializes the mixture model
    return tmm


def compute_eval_points_ablation(dataset_name, desired_eval_points, cache_path):
    tmm = get_tNEB(dataset_name, cache_path)
    dists = dict()
    paths = np.array(list(tmm.paths_.values()))
    for num_eval_points in tqdm.tqdm(desired_eval_points):
        interpolated_paths = corc.graph_metrics.tmm_gmm_neb.interpolate_paths_batched(
            paths, num_eval_points
        )
        logprobs = corc.graph_metrics.tmm_gmm_neb.tmm_jax_batched(
            interpolated_paths,
            tmm.mixture_model.centers,
            tmm.mixture_model.covs,
            tmm.mixture_model.weights,
            df=tmm.mixture_model.df,
        )
        dists[num_eval_points] = jnp.min(logprobs, axis=1)
    return dists


def load_or_compute_distances_per_dataset(datasets, cache_path, desired_values):
    distances_per_dataset = dict()
    cached_results_path = os.path.join(
        cache_path, "ablations", f"ablation_NEB_eval_points.pkl"
    )
    if os.path.exists(cached_results_path):
        with open(cached_results_path, "rb") as f:
            distances_per_dataset = pickle.load(f)
    else:
        distances_per_dataset = dict()

    for dataset in datasets:
        print(f"Starting {dataset}")
        if dataset not in distances_per_dataset:
            distances_per_dataset[dataset] = dict()

        # Check if all desired steps have been computed
        if not (set(desired_values) == set(distances_per_dataset[dataset].keys())):
            print(f"recomputing values")
            distances = compute_eval_points_ablation(
                dataset, desired_values, cache_path
            )
            distances_per_dataset[dataset] = distances

            # Save the updated results
            with open(cached_results_path, "wb") as f:
                pickle.dump(distances_per_dataset, f)

        print(f"Finished {dataset}")

    return distances_per_dataset


def plot_convergence(distances, datasets=None):
    # Create a figure and axis
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots()
    colors = cm.tab20(np.linspace(0, 1, len(distances)))

    for i, (dataset, eval_steps) in enumerate(distances.items()):
        if datasets is not None and dataset not in datasets:
            continue
        max_eval_steps = max(eval_steps.keys())
        max_distances = eval_steps[max_eval_steps]

        values_list = list(eval_steps.keys())
        diff_list = []

        for eval_step in values_list:
            distances_to_compare = eval_steps[eval_step]
            diff = np.mean(np.abs(distances_to_compare - max_distances))
            diff_list.append(diff)

        # Plot the average differences with markers
        label = corc.our_datasets.dataset_displaynames[dataset].replace("\n", " ")
        ax.plot(values_list, diff_list, label=label, marker="o", color=colors[i])

    # Set the title and labels
    ax.set_xlabel("Number of Evaluation Points")
    ax.set_ylabel("$\Delta$ NEB Distance")
    ax.legend()
    # ax.set_yscale("log")
    ax.set_ylim(0, 1)

    # Show the plot
    filename = os.path.join(cache_path, "..", "figures", f"ablation_eval_points.pdf")
    plt.savefig(filename)
    print(f"written to {filename}")


def main():
    datasets = corc.our_datasets.DATASETS2D + corc.our_datasets.CORE_HD_DATASETS
    # datasets = corc.our_datasets.DATASETS2D

    desired_values = [10, 50, 75, 100, 150, 200, 250, 500, 750, 1000, 1500, 2000, 3000]
    distances = load_or_compute_distances_per_dataset(
        datasets, cache_path, desired_values
    )
    plot_convergence(distances, datasets=corc.our_datasets.CORE_HD_DATASETS)


if __name__ == "__main__":
    main()
