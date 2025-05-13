from datetime import datetime
import itertools

import sklearn.metrics
import tqdm
import corc.mixture
import scipy
import sklearn
import sklearn.mixture
import studenttmixture
from pandas.core.common import random_state
import numpy as np
import matplotlib.pyplot as plt
import time

from corc.graph_metrics.graph import Graph
import corc.graph_metrics.tmm_gmm_neb
import corc.utils
import corc
import corc.visualization


class NEB(Graph):
    def __init__(
        self,
        data=None,
        labels=None,
        n_components=25,
        mixture_model_type="tmm",
        n_neighbors=10,  # number of neighbors for the NEB path computation
        dataset_name=None,
        n_clusters=None,  # target number of clusters (filled with GT if labels is given)
        optimization_iterations=200,  # for NEB (huge impact on time consumption)
        num_NEB_points=100,  # number of points in the NEB path
        seed=42,
        latent_dim=2,  # automatically derived from data if provided. One of both is needed.
        path=None,
        tmm_regularization=1e-4,
        n_init=20,  # for fitting TMM/GMM
        thresh=0.01,  # for fitting TMM/GMM
        reduced_tolerance_on_retry=1e-3,  # for fitting TMM/GMM
        max_iter_on_retries=10000,  # for TMM fitting, 10x the default
        max_elongation=None,  # will be set to 500 * dim
        min_cluster_size=10,  # mixture model filtering is only applied to TMM
        batch_size=150,  # for NEB computation (150 <8GB on GPU for d=64, more for lower dimensions)
    ):
        """
        Initialize the NEB (nudged elastic band) based on TMM/GMM.
        """
        # make sure that latent_dim does match the data if provided
        if data is not None:
            latent_dim = data.shape[-1]

        super().__init__(latent_dim, data, labels, path, seed)

        if mixture_model_type == "tmm":
            self.mixture_model = studenttmixture.EMStudentMixture(
                n_components=n_components,
                reg_covar=tmm_regularization,  # this makes the TMM favor ball-like shapes (and avoid extreme elongations)
                n_init=n_init,
                tol=10e-3,  # matching the default of sklearn for GMM
                max_iter=100,  # matching the default of sklearn for GMM
                fixed_df=True,
                df=1.0,  # the minimum value, for df=infty we get gmm
                # fixed_df=False,
                init_type="kmeans",
                random_state=seed,
            )
        elif mixture_model_type == "gmm":
            self.mixture_model = sklearn.mixture.GaussianMixture(
                n_components=n_components,
                n_init=n_init,
                random_state=seed,
                init_params="kmeans",
                covariance_type="full",  # `full` to make it consistent with the TMM covariance
            )

        self.n_components = n_components
        self.n_neighbors = (
            n_neighbors if n_neighbors < n_components else n_components - 1
        )
        self.thresh = thresh
        self.iterations = optimization_iterations
        self.n_init = n_init
        self.max_iter_on_retries = max_iter_on_retries
        self.min_cluster_size = min_cluster_size
        self.max_elongation = (
            max_elongation if max_elongation is not None else 500 * latent_dim
        )
        self.reduced_tolerance_on_retry = reduced_tolerance_on_retry
        self.num_NEB_points = num_NEB_points
        self.dataset_name = dataset_name
        self.n_clusters = n_clusters
        if labels is not None and n_clusters is None:
            self.n_clusters = len(np.unique(labels))
        self.batch_size = batch_size

    def convert_mixture_model(self, old_mixture_model):
        if isinstance(old_mixture_model, studenttmixture.EMStudentMixture):
            mixture_model = corc.mixture.StudentMixture.from_EMStudentMixture(
                mixture_model=self.old_mixture_model
            )
            model_type = "tmm"
        elif isinstance(old_mixture_model, sklearn.mixture.GaussianMixture):
            mixture_model = corc.mixture.GaussianMixtureModel.from_sklearn(
                self.old_mixture_model
            )
            model_type = "gmm"
        return mixture_model, model_type

    def filter_mixture_model(
        self, old_mixture_model, data_X, min_cluster_size, max_elongation
    ):
        """
        Filter a mixture model to select components with at least min_cluster_size elements
        and an elongation of at most max_elongation.
        This function is called by fit() with possibly different values for min_cluster_size and max_elongation.
        Filtering does not happen in-place to be able to redo it.
        """
        mixture_model, model_type = self.convert_mixture_model(old_mixture_model)
        mixture_model.filter_components(
            data_X=data_X,
            min_cluster_size=min_cluster_size,
            max_elongation=max_elongation,
        )

        return mixture_model, model_type

    def fit(self, data, knn=None):
        """
        fit the mixture model (overcluster), compute distances between clusters (based on NEB paths).
        """
        # fit the mixture model (re-use old model if available)
        if hasattr(self, "old_mixture_model") and self.old_mixture_model is not None:
            self.mixture_model = self.old_mixture_model
        else:
            start_mixture = time.time()
            self.mixture_model.fit(data)
            self.time_mixture = time.time() - start_mixture
            print(f"Mixture model fit took {self.time_mixture:.2f} seconds.")

        # make sure that TMM converged (this is sometimes problematic)
        if isinstance(self.mixture_model, studenttmixture.EMStudentMixture):
            # retry fitting if it failed before - happens regularly for high-dim datasets
            for _ in range(self.n_init):
                if self.mixture_model.df_ is None:
                    # we have not converged
                    print("retrying tmm fit with more iterations")
                    self.mixture_model.max_iter = self.max_iter_on_retries
                    self.mixture_model.tol = (
                        self.reduced_tolerance_on_retry
                    )  # default is 1e-5
                    self.mixture_model.fit(data)
        elif isinstance(self.mixture_model, sklearn.mixture.GaussianMixture):
            for _ in range(10):
                if not self.mixture_model.converged_:
                    self.mixture_model.max_iter = self.max_iter_on_retries
                    self.mixture_model.fit(data)

        self.old_mixture_model = self.mixture_model
        original_num_components = self.n_components
        self.mixture_model, _ = self.convert_mixture_model(self.old_mixture_model)
        self.mixture_model.print_elongations_and_counts(data)

        # filter mixture model
        self.mixture_model, model_type = self.filter_mixture_model(
            old_mixture_model=self.old_mixture_model,
            data_X=data,
            min_cluster_size=self.min_cluster_size,
            max_elongation=self.max_elongation,
        )
        if len(self.mixture_model.weights) == 0:
            # no components left, retry with less restrictive filtering.
            # This happens especially for high-dimensional data where components are typically more elongated.
            self.mixture_model, model_type = self.filter_mixture_model(
                old_mixture_model=self.old_mixture_model,
                data_X=data,
                min_cluster_size=3 * self.min_cluster_size,
                max_elongation=50 * self.max_elongation,
            )
        self.centers_ = self.mixture_model.centers

        print(
            f"After filtering {original_num_components} components, we are left with {len(self.mixture_model.weights)} components"
        )
        if original_num_components != len(self.mixture_model.weights):
            self.mixture_model.print_elongations_and_counts(data)
        if knn is None:
            knn = self.n_neighbors

        start_NEB = time.time()
        # compute NEB paths.
        (
            self.adjacency_,
            self.raw_adjacency_,
            self.paths_,
        ) = corc.graph_metrics.tmm_gmm_neb.compute_neb_paths_batch(
            means=self.mixture_model.centers,
            covs=self.mixture_model.covs,
            weights=self.mixture_model.weights,
            df=self.mixture_model.df if (model_type == "tmm") else None,
            gmm=(model_type == "gmm"),
            iterations=self.iterations,
            knn=knn,
            num_NEB_points=self.num_NEB_points,
            batch_size=self.batch_size,
        )
        self.time_NEB = time.time() - start_NEB

    def compute_mst_edges(self):
        """
        Compute the edges of the minimum spanning tree (MST) from the adjacency matrix.
        Those will be used to merge clusters.
        """
        if self.raw_adjacency_ is None:
            raise ValueError(
                "Adjacency matrix not computed, you need to fit NEB first."
            )
        mst = -scipy.sparse.csgraph.minimum_spanning_tree(-self.raw_adjacency_)
        rows, cols = mst.nonzero()
        entries = list(zip(rows, cols))
        return entries

    def predict(self, data_X):
        if self.paths_ is None:  # fitting did not yet take place
            self.fit(data_X)
        return self.mixture_model.predict(data_X)

    def predict_with_target(self, data, target_number_classes):
        predictions = self.predict(data)
        self.target_num_classes = target_number_classes

        # check whether prediction classes need to be merged
        if target_number_classes < self.n_components:
            raw_predictions = predictions

            # extract merging strategy
            threshold_dict, merging_strategy_dict = (
                self.get_thresholds_and_cluster_numbers()
            )
            num_classes = self.get_best_cluster_number(
                threshold_dict, target_number_classes
            )
            merging_strategy = merging_strategy_dict[num_classes]

            # actually merge the predictions
            predictions = merging_strategy[raw_predictions]

        return predictions

    def normalize_adjacency(adjacency):
        """
        normalize adjacency to be between 0 and 1 for the sake of plotting
        """
        if np.isnan(adjacency).any():
            print(f"Adjacency contains {np.isnan(adjacency).sum()} NaN values.")

        # adjacency may contain inf values which we need to actively ignore
        finite_vals = adjacency[np.isfinite(adjacency)]
        norm_adjacency = adjacency - np.nanmin(finite_vals)

        finite_vals = norm_adjacency[np.isfinite(norm_adjacency)]
        norm_adjacency = norm_adjacency / np.nanmax(finite_vals)

        # reset selfloops
        np.fill_diagonal(norm_adjacency, 0)

        return norm_adjacency

    def get_centers(self):
        self.centers_ = self.mixture_model.centers
        return self.centers_

    def get_merged_pairs(self, target_num_classes, only_mst_edges=True):
        thresholds_dict, clustering_dict = self.get_thresholds_and_cluster_numbers()
        num_classes = self.get_best_cluster_number(thresholds_dict, target_num_classes)
        merging_strategy = clustering_dict[num_classes]

        pairs = [
            (i, j)
            for i, j in itertools.combinations(range(len(merging_strategy)), 2)
            if merging_strategy[i] == merging_strategy[j]
        ]

        if only_mst_edges:
            # only those that are also part of the MST
            mst_edges = self.compute_mst_edges()
            pairs = [pair for pair in pairs if pair in mst_edges]

        return pairs

    def create_graph(self, save=True, plot=True, return_graph=False):
        """'
        1. Overcluster data using a GMM/TMM
        2. Construct a weighted undirected graph with the clusters as centers.
        Low weights mean, that the clusters are more disconnected.
        We span elastic bands between all pairs of clusters and then optimize them to stay "high" in probability space
        3. Plots show tsne on samples and gmm/tmm cluster means.
        """

        # apply TSNE to get down to 2D
        # embeddings, cluster_means = self._dim_reduction(self.centers_)

        # edges are expected in the form of a dictionary, so we have to convert our np array
        edges = {
            (i, j): self.adjacency_[i, j]
            for i, j in itertools.combinations(range(self.adjacency_.shape[0]), 2)
        }

        normalized_adjacency = self.normalize_adjacency(self.adjacency_)
        normalized_edges = {
            (i, j): normalized_adjacency[i, j]
            for i, j in itertools.combinations(range(normalized_adjacency.shape[0]), 2)
        }

        self.graph_data = {
            "nodes": self.get_centers(),
            "edges": normalized_edges,
            "raw_edges": edges,
        }

        if plot:
            self.plot_graph()

        if save:
            raise NotImplementedError

        if return_graph:
            return self.graph_data

    def plot_field(
        self,
        data_X,
        levels=20,
        selection=None,  # selection which paths to plot
        save_path=None,
        axis=None,
        plot_points=True,  # whether data_X is plotted
        transformed_points=None,
        grid_resolution=128,
        plot_ids=True,
        bend_paths=False,
        landscape_kwargs={},
    ):
        """Plots the TMM/GMM field and the optimized paths (if available).
        selection: selects which paths are included in the plot, by default, all paths are included.
        other typical options: MST through selection=zip(mst.row,mst.col) and individuals via e.g. [(0,1), (3,4)]

        """
        # extract cluster centers
        locations = corc.utils.mixture_center_locations(self.mixture_model)
        n_components = len(locations)

        # Compute TSNE if necessary
        if data_X.shape[-1] > 2:
            if transformed_points is None:
                transformed_points = corc.visualization.get_TSNE_embedding(data_X)
            locations = corc.visualization.snap_points_to_TSNE(
                locations, data_X, transformed_points
            )
        else:
            transformed_points = data_X

        if axis is None:
            figure, axis = plt.subplots(1, 1)

        # plot the energy landscape if possible
        if data_X.shape[-1] == 2:
            self.mixture_model.plot_energy_landscape(
                data_X=data_X,
                levels=levels,
                grid_resolution=grid_resolution,
                axis=axis,
                kwargs=landscape_kwargs,
            )

        # plot the raw data
        if plot_points:
            axis.scatter(
                transformed_points[:, 0],
                transformed_points[:, 1],
                s=10,
                label="raw data",
            )

        # plot cluster centers and IDs
        axis.scatter(
            locations[:, 0],
            locations[:, 1],
            color="black",
            # marker="X",
            label="mixture centers",
            s=30,
        )
        if plot_ids:
            for i, location in enumerate(locations):
                y_min, y_max = axis.get_ylim()
                scale = y_max - y_min
                axis.annotate(f"{i}", xy=location - 0.05 * scale, color="black")

        # plot paths between centers (by default: all)
        if self.paths_ is not None:
            if selection is None:
                selection = list(itertools.combinations(range(n_components), r=2))
            for i, j in selection:
                if bend_paths:
                    path = self.paths_[(i, j)]
                    axis.plot(path[:, 0], path[:, 1], lw=2, alpha=0.5, color="black")
                else:
                    start = locations[i]
                    end = locations[j]
                    axis.plot(
                        *zip(start, end),
                        color="black",
                        alpha=1,
                    )

        if save_path is not None:
            plt.savefig(save_path)
        # not returning the axis object since it is modified in-place

    def plot_graph(self, X2D=None, pairs=None, target_num_clusters=None, ax=None):
        """
        Note: automatic "pairs" computation only works if self.labels or self.n_clusters is set.
        """
        if ax is None:
            ax = plt.gca()

        if pairs is None:
            if target_num_clusters is None:
                if self.labels is not None:
                    target_num_clusters = len(np.unique(self.labels))
                elif hasattr(self, "n_clusters"):
                    target_num_clusters = self.n_clusters
                else:
                    target_num_clusters = len(
                        self.centers_
                    )  # no clusters are merged, so no edges are drawn

            # by default we do not draw lines for all pairs of nodes that are merged (because some may not have converged)
            # but instead for the shortest edges that form the MST.
            pairs = self.get_merged_pairs(
                target_num_classes=target_num_clusters, only_mst_edges=True
            )

        # drawing the background for NEB in the 2D case
        if self.latent_dim == 2:

            if isinstance(self.mixture_model, corc.mixture.GaussianMixtureModel):
                kwargs = dict(vmax=15)
            else:
                kwargs = dict()

            our_data = self.data if self.data is not None else self.centers_
            self.plot_field(
                data_X=our_data,
                selection=pairs,
                axis=ax,
                plot_points=False,
                plot_ids=False,
                landscape_kwargs=kwargs,
                bend_paths=True,
            )

        else:  # more than 2 dims
            assert self.data is not None, "self.data is needed for tsne matching"
            # get (pseudo) TSNE embedding
            if X2D is None:
                raise Exception(
                    "TSNE transform must be given for high-dimensional datasets"
                )
            # if not hasattr(self, "transformed_centers_"):
            cluster_means = corc.visualization.snap_points_to_TSNE(
                points=self.get_centers(),
                data_X=self.data,
                transformed_X=X2D,
            )
            ax.scatter(
                *cluster_means.T,
                alpha=1.0,
                rasterized=True,
                marker="X",
                s=60,
                c="black",
            )

            # plot paths as straight lines
            if pairs is not None and len(pairs) > 0:
                for pair in pairs:
                    start = cluster_means[pair[0]]
                    end = cluster_means[pair[1]]
                    ax.plot(*zip(start, end), color="black", alpha=0.5, lw=1)

    def get_ari(self, X, y):
        y_pred = self.predict_with_target(X, target_number_classes=len(np.unique(y)))
        ari = sklearn.metrics.adjusted_rand_score(y, y_pred)
        return ari
