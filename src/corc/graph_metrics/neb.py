from datetime import datetime
import itertools

import tqdm
import corc.studentmixture
import scipy
import sklearn
from sklearn.mixture import GaussianMixture
import studenttmixture
from pandas.core.common import random_state
import numpy as np
import matplotlib.pyplot as plt
import time

from corc.graph_metrics.graph import Graph
import corc.graph_metrics.tmm_gmm_neb
import corc.utils
import corc


class NEB(Graph):
    def __init__(
        self,
        latent_dim=2,
        data=None,
        labels=None,
        path=None,
        n_components=15,
        n_neighbors=3,
        thresh=0.01,
        seed=42,
        mixture_model_type="tmm",
        n_init=5,
        optimization_iterations=300,  # for NEB
        max_iter_on_retries=10000,  # for TMM fitting, 10x the default
        dataset_name=None,
        n_clusters=None,
        tmm_regularization=1e-4,
        min_cluster_size=10,  # mixture model filtering is only applied to TMM
        max_elongation=None,  # will be set to 500 * dim**2 as "good" clusters tend to have surprisingly high elongation in high dimensions
    ):
        """
        Initialize the NEB (nudged elastic band) class.
        GMM/TMM

        Args:
            data (np.array): Input data.
            labels (np.array): Labels.
            latent_dim (int): Dimension of the data.
            n_components (int): Number of components for GMM/TMM clustering.
            n_init (int): Number of repetitions during GMM/TMM fitting.
            mixture_model_type (str): GMM or TMM
        """
        # make sure that latent_dim does match the data if provided
        if data is not None:
            latent_dim = data.shape[-1]

        super().__init__(latent_dim, data, labels, path, seed)

        if mixture_model_type == "tmm":
            self.mixture_model = studenttmixture.EMStudentMixture(
                n_components=n_components,
                reg_covar=tmm_regularization,  # this makes the TMM favor ball-like shapes (and avoid extreme elongations)
                n_init=(
                    1 if latent_dim > 10 else n_init
                ),  # convergence is slow in high dimension so we give the model more tries in the fit function if it does not converge immediately.
                fixed_df=True,
                df=1.0,  # the minimum value, for df=infty we get gmm
                init_type="kmeans",
                random_state=seed,
            )
        elif mixture_model_type == "gmm":
            self.mixture_model = GaussianMixture(
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
            max_elongation if max_elongation is not None else 500 * latent_dim**2
        )

    def fit(self, data, knn=5):
        """
        data: data to be fitted on.
        """
        self.mixture_model.fit(data)

        # make sure that TMM converged (this is sometimes problematic)
        if isinstance(self.mixture_model, studenttmixture.EMStudentMixture):
            # retry fitting if it failed before - happens regularly for high-dim datasets
            for _ in range(self.n_init):
                if self.mixture_model.df_ is None:
                    # we have not converged
                    print("retrying tmm fit with more iterations")
                    self.mixture_model.max_iter = self.max_iter_on_retries
                    self.mixture_model.tol = 1e-4  # default is 1e-5
                    self.mixture_model.fit(data)

        self.old_mixture_model = self.mixture_model
        if isinstance(self.old_mixture_model, studenttmixture.EMStudentMixture):
            self.mixture_model = (
                corc.studentmixture.StudentMixture.from_EMStudentMixture(
                    mixture_model=self.old_mixture_model
                )
            )
            model_type = "tmm"
        elif isinstance(self.old_mixture_model, sklearn.mixture.GaussianMixture):
            self.mixture_model = corc.studentmixture.GaussianMixtureModel.from_sklearn(
                self.old_mixture_model
            )
            model_type = "gmm"

        original_num_components = len(self.mixture_model.weights)
        self.mixture_model.print_elongations_and_counts(data)
        self.mixture_model.filter_components(
            data_X=data,
            min_cluster_size=self.min_cluster_size,
            max_elongation=self.max_elongation,
        )
        self.centers_ = self.mixture_model.centers

        print(
            f"After filtering {original_num_components} components, we are left with {len(self.mixture_model.weights)} components"
        )
        ## TODO - remove later or add debug flag
        if original_num_components != len(self.mixture_model.weights):
            self.mixture_model.print_elongations_and_counts(data)

        # compute NEB paths. This is a very time-consuming step
        (
            self.adjacency_,
            self.raw_adjacency_,
            self.paths_,
            self.temps_,
            self.logprobs_,
        ) = corc.graph_metrics.tmm_gmm_neb.compute_neb_paths(
            means=self.mixture_model.centers,
            covs=self.mixture_model.covs,
            weights=self.mixture_model.weights,
            model_type=model_type,
            iterations=self.iterations,
            knn=knn,
        )
        # check quality of generated paths
        corc.graph_metrics.tmm_gmm_neb.evaluate_equidistance(self.paths_)

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
            mst_edges = corc.utils.compute_mst_edges(self.raw_adjacency_)
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
            # "nodes_org_space": self.centers_,
            "edges": normalized_edges,
            "raw_edges": edges,
        }

        if plot:
            self.plot_graph()
        #     self._plt_graph_compare(embeddings, self.labels_, save=f"{filename}.png")

        if save:
            raise NotImplementedError

        if return_graph:
            return self.graph_data

    def plot_graph(self, X2D=None, pairs=None, target_num_clusters=None, axis=None):
        """
        Note: automatic "pairs" computation only works if self.labels or self.n_clusters is set.
        """
        cmap = plt.get_cmap("viridis")  # choose a colormap
        if axis is None:
            axis = plt.gca()

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

            our_data = self.data if self.data is not None else self.centers_
            corc.utils.plot_field(
                data_X=our_data,
                mixture_model=self.mixture_model,
                paths=self.paths_,
                selection=pairs,
                axis=axis,
                plot_points=False,
                plot_ids=False,
            )

        else:  # more than 2 dims

            # get (pseudo) TSNE embedding
            if X2D is None:
                raise Exception(
                    "TSNE transform must be given for high-dimensional datasets"
                )
            # if not hasattr(self, "transformed_centers_"):
            cluster_means = corc.utils.snap_points_to_TSNE(
                points=self.get_centers(),
                data_X=self.data,
                transformed_X=X2D,
            )
            axis.scatter(
                *cluster_means.T,
                alpha=1.0,
                rasterized=True,
                marker="X",
                s=60,
                c="black",
            )

            # plot paths as straight lines
            if pairs is not None and len(pairs) > 0:
                xs, ys = zip(*pairs)
                for pair in pairs:
                    start = cluster_means[pair[0]]
                    end = cluster_means[pair[1]]
                    axis.plot(*zip(start, end), color="black", alpha=0.5, lw=1)
