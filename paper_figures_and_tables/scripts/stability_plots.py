import os


os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import jax.random as jrandom
import numpy as np
import matplotlib.pyplot as plt
import studenttmixture
import scipy
import tqdm
import sklearn.preprocessing
import sklearn.mixture
import pickle
import itertools
import time

# own code
import corc.our_datasets
from corc.graph_metrics import tmm_gmm_neb
import corc.tmm_plots
import corc.utils

cache_path = "cache"

# def train_multiple_tmm_models_seeds(
#     data_X, data_y, num_seeds=10, neb_iterations=500, gmm=False, n_components=15
# ):
#     tmm_models = list()
#     for i in range(num_seeds):
#         tmm_model = corc.graph_metrics.neb.NEB(
#             data=data_X,
#             labels=data_y,
#             optimization_iterations=neb_iterations,
#             seed=42 + i,
#             n_init=(
#                 5 if data_X.shape[1] < 10 else 1
#             ),  # fitting becomes slow in high dimensions
#             n_components=n_components,
#             mixture_model_type="gmm" if gmm else "tmm",
#         )
#         tmm_model.fit(data=data_X)
#         tmm_models.append(tmm_model)

#     return tmm_models


def train_multiple_tmm_models_overclustering(
    data_X, data_y, num_models=10, neb_iterations=500, gmm=False
):
    num_classes = len(np.unique(data_y))
    tmm_models = list()
    for i in range(num_models):
        tmm_model = corc.graph_metrics.neb.NEB(
            data=data_X,
            labels=data_y,
            # optimization_iterations=neb_iterations,
            # seed=42,
            # n_init=20,
            n_components=10 + 5 * i,
            mixture_model_type="gmm" if gmm else "tmm",
            batch_size=70,  # small to avoid problems with parallel runs
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


def main(args):
    """Computes the stability plots.

    plot_type: ["seeds","overclustering"] Where seeds shows stability against re-creating the plot
    with different seeds and overclustering shows stability against choosing a different number of
    clusters to start with

    just_ari: if True, no figure is created and only ARI scores are printed to the commandline.
    """

    avg_pairwise_aris = dict()
    avg_aris = dict()
    for i, dataset_name in enumerate(args.dataset_names):
        dataset_displayname = corc.our_datasets.dataset_displaynames[
            dataset_name
        ].replace("\n", " ")
        print(f"Working on {dataset_displayname} ({i+1}/{len(args.dataset_names)})")
        data_X, data_y, tsne = corc.utils.load_dataset(
            dataset_name=dataset_name, cache_path=cache_path
        )

        dataset_filename = dataset_name.replace(" ", "_")
        gmm_string = "_gmm" if args.gmm else ""
        n_components_string = (
            f"_{args.n_components}" if args.plot_type == "seeds" else ""
        )
        cache_filename = f"cache/stability/{args.plot_type}_{dataset_filename}{gmm_string}{n_components_string}.pkl"

        if args.plot_type == "seeds":
            # we only use the precomputed models here.
            algorithm = "GMM-NEB" if args.gmm else "TMM-NEB"
            tmm_models = corc.utils.load_algorithms(
                dataset_name, algorithm=algorithm, cache_path="cache"
            )
            assert tmm_models is not None, "No TMM models found."
        else:  # overclustering
            # those we have to create ourselves

            # check if the data is already computed
            tmm_models = None
            if os.path.exists(cache_filename):
                with open(cache_filename, "rb") as f:
                    tmm_models = pickle.load(f)
                    # recompute when not enough seeds have been computed
                    if len(tmm_models) < args.num_models:
                        tmm_models = None
                    else:
                        print(
                            f"successfully loaded precomputed TMM models for {dataset_displayname} (for {args.plot_type}) from disk"
                        )

            # compute tmm models
            if tmm_models is None:
                print("computing NEB models...")
                tmm_model_starttime = time.time()
                tmm_models = train_multiple_tmm_models_overclustering(
                    data_X, data_y, num_models=args.num_models, gmm=args.gmm
                )
                with open(cache_filename, "wb") as f:
                    pickle.dump(tmm_models, f)
                print(
                    f"done with model computation for {dataset_displayname}. ({time.time() - tmm_model_starttime:.2f}s)"
                )

        # get ari scores
        avg_pairwise_aris[dataset_name], avg_aris[dataset_name] = (
            compute_average_pairwise_ari(
                data_X=data_X, data_y=data_y, tmm_models=tmm_models
            )
        )

        if not args.just_ari:
            # create the figure
            num_tiles = 9 if args.plot_type == "seeds" else 14
            figure = corc.tmm_plots.plot_tmm_models(
                tmm_models[:num_tiles],
                data_X,
                data_y,
                dataset_name,
                tsne_transform=tsne,
                legend=True,
            )
            figure.suptitle(
                f"{args.plot_type.capitalize()} Stability of {'g-NEB' if args.gmm else 't-NEB'} on {dataset_displayname} (avg pairwise ari: {avg_pairwise_aris[dataset_name]:.2f})"
            )
            plt.subplots_adjust(
                left=0.01, right=0.99, top=0.95, bottom=0.05, wspace=0.1, hspace=0.2
            )
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            figure.savefig(
                f"figures/stability/stability_{args.plot_type}_{dataset_filename}{gmm_string}{n_components_string}.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
            print(
                f"Stored figure for {args.plot_type} on {dataset_displayname} in figures/stability"
            )

    # output ari overview
    for dataset_name in args.dataset_names:
        dataset_displayname = corc.our_datasets.dataset_displaynames[
            dataset_name
        ].replace("\n", " ")
        print(
            f"Dataset: {dataset_displayname:30} \t avg pairwise ari: {avg_pairwise_aris[dataset_name]:.2f} \t avg ari: {avg_aris[dataset_name]:.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computing Stability plots for TMM")
    parser.add_argument(
        "-t",
        "--plot_type",
        help="either 'seeds' or 'overclustering'",
        default="overclustering",
        choices=["seeds", "overclustering"],
    )
    parser.add_argument(
        "--just_ari", help="just compute ARI values, no plot", action="store_true"
    )
    parser.add_argument(
        "-d",
        "--dataset_names",
        help="comma separated list of dataset names. Specials: all, 2d, complex, main2, core",
        default=["densired8"],
        nargs="+",
    )
    parser.add_argument(
        "-n",
        "--num_models",
        help="Number of models to compute (default:15) only used for overclustering",
        default=15,
        type=int,
    )
    parser.add_argument(
        "--n_components",
        help="Number of mixture model components. Only used for seed-stability plots",
        default=25,
        type=int,
    )
    parser.add_argument("--gmm", help="use GMM instead of TMM", action="store_true")
    args = parser.parse_args()

    if args.dataset_names[0] == "all":
        args.dataset_names = corc.our_datasets.DATASET_SELECTOR
    elif args.dataset_names[0] == "2d":
        args.dataset_names = corc.our_datasets.DATASETS2D
    elif args.dataset_names[0] == "complex":
        args.dataset_names = corc.our_datasets.COMPLEX_DATASETS
    elif args.dataset_names[0] == "main2" or args.dataset_names[0] == "core":
        args.dataset_names = corc.our_datasets.CORE_HD_DATASETS

    assert os.path.exists("figures/stability"), "figures/stability does not exist"
    assert os.path.exists("cache/stability"), "cache/stability does not exist."

    print(f"Creating stability plot for {args.plot_type} with {args.num_models} plots.")
    main(args)
