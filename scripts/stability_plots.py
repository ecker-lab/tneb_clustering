import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import argparse
import jax.random as jrandom
import numpy as np
import matplotlib.pyplot as plt
import studenttmixture
import scipy
import tqdm
import sklearn.mixture
import pickle
import itertools
import time

# own code
import corc.datasets2d
import corc.complex_datasets


# import corc.datasets_refactor as datasets2d
from corc.graph_metrics import tmm_gmm_neb
import corc.tmm_plots as tmm_plots
import corc.utils


# we create the plots with 9 different seeds/overclustering values plus GT
NUM_MODELS = 9


def load_datasets():
    # instantiate 2D datasets
    dataset_functions = corc.datasets2d.DATASETS
    datasets_2d = dict()
    for dataset_name, dataset_function in dataset_functions.items():
        datasets_2d[dataset_name] = dataset_function()

    # alternative: 8/16D complex datasets
    complex_datasets = {
        "densired8": corc.complex_datasets.load_densired(
            dim=8, path="datasets/densired.npz"
        ),
        "densired16": corc.complex_datasets.load_densired(
            dim=16, path="datasets/densired.npz"
        ),
        "mnist_nd8": corc.complex_datasets.make_mnist_nd(
            dim=8, path="datasets/mvae_mnist_nd_saved.pkl"
        ),
        "mnist_nd16": corc.complex_datasets.make_mnist_nd(
            dim=16, path="datasets/mvae_mnist_nd_saved.pkl"
        ),
    }

    # datasets = {**datasets_2d, **complex_datasets}
    datasets = complex_datasets

    print(f"there are {len(datasets)} datasets")
    return datasets


def train_multiple_tmm_models_seeds(data_X, data_y, num_seeds=10, neb_iterations=25):
    tmm_models = list()
    for i in range(num_seeds):
        tmm_model = corc.graph_metrics.neb.NEB(
            latent_dim=data_X.shape[1],
            data=data_X,
            labels=data_y,
            optimization_iterations=neb_iterations,
            seed=42 + i,
            n_init=(
                5 if data_X.shape[1] < 10 else 1
            ),  # fitting becomes slow in high dimensions
            n_components=15,
            mixture_model_type="tmm",
        )
        tmm_model.fit(data=data_X)
        tmm_models.append(tmm_model)

    return tmm_models


def train_multiple_tmm_models_overclustering(
    data_X, data_y, num_models=10, neb_iterations=25
):
    num_classes = len(np.unique(data_y))
    tmm_models = list()
    for i in range(num_models):
        tmm_model = corc.graph_metrics.neb.NEB(
            latent_dim=data_X.shape[1],
            data=data_X,
            labels=data_y,
            optimization_iterations=neb_iterations,
            seed=42,
            n_init=(
                5 if data_X.shape[1] < 10 else 1
            ),  # fitting becomes slow in high dimensions
            n_components=num_classes + 2 * i,
            mixture_model_type="tmm",
        )
        tmm_model.fit(data=data_X)
        tmm_models.append(tmm_model)

    return tmm_models


def compute_average_pairwise_ari(tmm_models, data_X, data_y):
    num_classes = len(np.unique(data_y))

    # compute predictions
    predictions = list()
    for tmm_model in tmm_models:
        predictions.append(
            tmm_model.predict_with_target(
                data=data_X, target_number_classes=num_classes
            )
        )

    # pairwise ARI
    pairwise_ari = list()
    for i, j in itertools.combinations(range(len(tmm_models)), 2):
        pairwise_ari.append(
            sklearn.metrics.adjusted_rand_score(predictions[i], predictions[j])
        )
        # ari is not symmetric, so we need to add the other direction as well
        pairwise_ari.append(
            sklearn.metrics.adjusted_rand_score(predictions[j], predictions[i])
        )

    # ari scores against GT data
    ari_scores = list()
    for i in range(len(tmm_models)):
        ari_scores.append(sklearn.metrics.adjusted_rand_score(predictions[i], data_y))

    return np.average(pairwise_ari), np.average(ari_scores)


def load_tsne_from_disk(dataset_filename):
    tsne_filename = f"cache/{dataset_filename}.pickle"
    if os.path.exists(tsne_filename):
        with open(tsne_filename, "rb") as f:
            dataset_info = pickle.load(f)
            tsne = dataset_info["X2D"]
        print(f"loaded tsne for {dataset_filename} from disk")
    else:
        tsne = None
    return tsne


def main(plot_type="seeds"):
    """Computes the stability plots. Two modes are seeds (stability against re-creating the plot
    with different seeds) and overclustering (stability against choosing a different number of
    clusters to start with)"""
    datasets = load_datasets()

    avg_pairwise_aris = dict()
    avg_aris = dict()
    for i, dataset_name in enumerate(datasets.keys()):
        print(f"Working on {dataset_name} ({i+1}/{len(datasets)})")
        dataset_filename = dataset_name.replace(" ", "_")
        data_X, data_y = datasets[dataset_name]

        if data_X.shape[1] > 2:
            tsne = load_tsne_from_disk(dataset_filename)
        else:
            tsne = None

        # check if the plot is already there
        # if os.path.exists(f"figures/stability_{dataset_filename}.pdf"):
        #     continue

        if plot_type == "seeds":
            cache_filename = f"cache/stability_seeds_{dataset_filename}.pkl"
        elif plot_type == "overclustering":
            cache_filename = f"cache/stability_overclustering_{dataset_filename}.pkl"
        else:
            raise "invalid plot type"

        # check if the data is already computed
        tmm_models = None
        if os.path.exists(cache_filename):
            with open(cache_filename, "rb") as f:
                tmm_models = pickle.load(f)
                # recompute when not enough seeds have been computed
                if len(tmm_models) < NUM_MODELS:
                    tmm_models = None

        # compute tmm models
        if tmm_models is None:
            print("computing tmm models...")
            tmm_model_starttime = time.time()
            if plot_type == "seeds":
                tmm_models = train_multiple_tmm_models_seeds(
                    data_X, data_y, num_seeds=NUM_MODELS
                )
            else:
                tmm_models = train_multiple_tmm_models_overclustering(
                    data_X, data_y, num_models=NUM_MODELS
                )
            with open(cache_filename, "wb") as f:
                pickle.dump(tmm_models, f)
            print(
                f"done with model computation. ({time.time() - tmm_model_starttime:.2f}s)"
            )

        # get ari scores
        avg_pairwise_aris[dataset_name], avg_aris[dataset_name] = (
            compute_average_pairwise_ari(
                data_X=data_X, data_y=data_y, tmm_models=tmm_models
            )
        )

        # create the figure
        figure = tmm_plots.plot_tmm_models(
            tmm_models, data_X, data_y, dataset_name, tsne_transform=tsne
        )
        figure.suptitle(
            f"{plot_type.capitalize()} Stability of TMMs on {dataset_name} (avg pairwise ari: {avg_pairwise_aris[dataset_name]:.2f})"
        )
        figure.savefig(f"figures/stability_{plot_type}_{dataset_filename}.pdf")

    # output ari overview
    for dataset_name in datasets.keys():
        print(
            f"Dataset: {dataset_name:20} \t avg pairwise ari: {avg_pairwise_aris[dataset_name]:.2f} \t avg ari: {avg_aris[dataset_name]:.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computing Stability plots for TMM")
    parser.add_argument(
        "-t",
        "--plot_type",
        help="either 'seeds' or 'overclustering'",
        default="seeds",  # Set the default value here
        choices=["seeds", "overclustering"],  # Optional: Validate input
    )
    args = parser.parse_args()

    print(f"Creating stability plot for {args.plot_type} with {NUM_MODELS} plots.")
    main(plot_type=args.plot_type)
