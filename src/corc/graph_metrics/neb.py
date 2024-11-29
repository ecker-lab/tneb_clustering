from datetime import datetime
import itertools

import tqdm
import scipy
import sklearn
from sklearn.mixture import GaussianMixture
import studenttmixture
from pandas.core.common import random_state
import numpy as np
import matplotlib.pyplot as plt

from corc.graph_metrics.graph import Graph
import corc.graph_metrics.tmm_gmm_neb


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
        optimization_iterations=1000,  # for NEB
        max_iter_on_retries=10000,  # for TMM fitting, 10x the default
        dataset_name=None,
        n_clusters=None,
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
                reg_covar=5e-5,  # this makes the TMM favor ball-like shapes (and avoid extreme elongations)
                n_init=(
                    1 if latent_dim > 10 else n_init
                ),  # convergence is slow in high dimension so we give the model more tries in the fit function if it does not converge immediately.
                fixed_df=True,
                df=1.0,  # the minimum value, for df=infty we get gmm
                init_type="k++",
                random_state=seed,
            )
        elif mixture_model_type == "gmm":
            self.mixture_model = GaussianMixture(
                n_components=n_components,
                n_init=n_init,
                random_state=seed,
                init_params="k-means++",
                covariance_type="spherical",
            )
        self.n_components = n_components
        self.n_neighbors = (
            n_neighbors if n_neighbors < n_components else n_components - 1
        )
        self.thresh = thresh
        self.iterations = optimization_iterations
        self.n_init = n_init
        self.max_iter_on_retries = max_iter_on_retries

    def fit(self, data, max_elongation=100):
        # data: data to be fitted on.
        # max_elongation: for TMM ignore all components that are highly elongated.
        # max_elongation gives an upper bound on biggest_eigenvalue/smallest_eigenvalue
        self.mixture_model.fit(data)

        # extract centers for TMM/GMM
        if isinstance(self.mixture_model, sklearn.mixture.GaussianMixture):
            self.centers_ = self.mixture_model.means_
        elif isinstance(self.mixture_model, studenttmixture.EMStudentMixture):
            # retry fitting if it failed before - happens regularly for high-dim datasets
            for _ in range(self.n_init):
                if self.mixture_model.df_ is None:
                    # we have not converged
                    print("retrying tmm fit with more iterations")
                    self.mixture_model.max_iter = self.max_iter_on_retries
                    self.mixture_model.tol = 1e-4  # default is 1e-5
                    self.mixture_model.fit(data)

            # filter components (remove elongated and very small components)
            original_num_components = self.mixture_model.n_components
            self.mixture_model = self.filter_mixture_model_components(mixture_model=self.mixture_model, data_X=data, max_elongation=1000, factor=15)
            
            # store centers in a central place
            self.centers_ = self.mixture_model.location
            print(
                f"After filtering {original_num_components} components, we are left with {self.mixture_model.n_components} components"
            )

        # compute NEB paths. This is a very time-consuming step
        (
            self.adjacency_,
            self.raw_adjacency_,
            self.paths_,
            self.temps_,
            self.logprobs_,
        ) = corc.graph_metrics.tmm_gmm_neb.compute_neb_paths(
            self.mixture_model, iterations=self.iterations
        )

        # check quality of generated paths
        equidistance_factor, all_factors = (
            corc.graph_metrics.tmm_gmm_neb.evaluate_equidistance(self.paths_)
        )
        if equidistance_factor > 10:
            print(
                f"WARNING: the path is not equidistant! longest segment {equidistance_factor} times too long"
            )
            for edge in all_factors.keys():
                if all_factors[edge] > 10:
                    print(f"{edge} has factor {all_factors[edge]}")

        # normalize adjacency
        norm_adjacency = self.adjacency_ - np.min(self.adjacency_)
        self.adjacency_ = norm_adjacency / np.max(norm_adjacency)

    def predict(self, data):
        if self.paths_ is None:  # fitting did not yet take place
            self.fit(data)
        return self.mixture_model.predict(data)

    # def get_merging_strategy(self, target_number_classes):
    #     if target_number_classes >= self.mixture_model.n_components:
    #         raise "too many classes"

    #     # merging classes if necessary
    #     thresholds_dict, clustering_dict = self.get_thresholds_and_cluster_numbers()

    #     num_classes = self._get_best_cluster_number(
    #         thresholds_dict, target_number_classes
    #     )
    #     threshold = thresholds_dict[num_classes]

    #     # find cluster pairs that need to be merged. We assume a "distance" matrix, so smaller is better
    #     adjacency = self._get_adjacency_matrix()
    #     tmp_adj = np.array(adjacency >= threshold, dtype=int)
    #     _, component_labels = scipy.sparse.csgraph.connected_components(
    #         tmp_adj, directed=False
    #     )

    #     # all pairs of clusters that will be joined
    #     pairs_to_be_merged = [(i, j) for i, j in zip(*np.nonzero(tmp_adj)) if i < j]

    #     return component_labels, pairs_to_be_merged

    # def get_merged_pairs(self, target_num_classes):
    #     merged_classes = list()

    #     thresholds_dict, clustering_dict = self.get_thresholds_and_cluster_numbers()
    #     num_classes = self.get_best_cluster_number(thresholds_dict, target_num_classes)
    #     merging_strategy = clustering_dict[num_classes]

    #     # extract all pairs of merged classes
    #     for id in np.unique(merging_strategy):
    #         # get indices that make a certain class
    #         indices = [i for i, x in enumerate(merging_strategy) if x == id]
    #         # get all pairs of these indices
    #         merged_classes += list(itertools.combinations(indices, 2))

    #     return merged_classes

    def compute_mst_edges(self, raw_adjacency):
        mst = -scipy.sparse.csgraph.minimum_spanning_tree(-raw_adjacency)
        rows, cols = mst.nonzero()
        entries = list(zip(rows, cols))
        return entries

    def get_merged_pairs(self, target_num_classes, only_mst_edges=True):
        thresholds_dict, clustering_dict = self.get_thresholds_and_cluster_numbers()
        num_classes = self.get_best_cluster_number(thresholds_dict, target_num_classes)
        threshold = thresholds_dict[num_classes]
        merging_strategy = clustering_dict[num_classes]

        pairs = [
            (i, j)
            for i, j in itertools.combinations(range(len(merging_strategy)), 2)
            if merging_strategy[i] == merging_strategy[j]
        ]

        if only_mst_edges:
            # only those that are also part of the MST
            mst_edges = self.compute_mst_edges(self.raw_adjacency_)
            pairs = [pair for pair in pairs if pair in mst_edges]

        return pairs

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

        # edges are expected in the form of a dictionary, so we have to convert out np array
        edges = {
            (i, j): self.adjacency_[i, j]
            for i, j in itertools.combinations(range(self.adjacency_.shape[0]), 2)
        }

        # normalized edges
        norm_adjacency = self.adjacency_ - np.min(self.adjacency_)
        norm_adjacency /= np.max(norm_adjacency)

        normalized_edges = {
            (i, j): norm_adjacency[i, j]
            for i, j in itertools.combinations(range(self.adjacency_.shape[0]), 2)
        }

        self.graph_data = {
            "nodes": self.centers_,
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

    def apply_tsne(self, X2D, transform_paths=True, samples_per_path=50):
        # we assume that fitting did take place
        self.transformed_centers_ = X2D.transform(self.centers_)

        if transform_paths:
            self.transformed_paths_ = dict()
            samples_per_path = min(
                samples_per_path, self.paths_[(0, 1)].shape[0]
            )  # do not upsample
            for i, j in tqdm.tqdm(
                itertools.combinations(range(self.n_components), 2),
                total=self.n_components * (self.n_components - 1) // 2,
                desc="converting paths",
            ):
                if i == j:
                    continue
                path = self.paths_[(i, j)]
                path = path[
                    np.linspace(
                        0, len(path) - 1, samples_per_path, dtype=int, endpoint=True
                    )
                ]
                self.transformed_paths_[(i, j)] = X2D.transform(path)
                self.transformed_paths_[(j, i)] = self.transformed_paths_[(i, j)]

    def plot_graph(self, X2D=None, pairs=None, n_clusters=None):
        """
        Note: automatic "pairs" computation only works if self.labels or self.n_clusters is set.
        """
        transformation = X2D
        self.get_graph()  # populates self.graph_data
        # print("got the graph")

        # get means
        if hasattr(self, "transformed_centers_"):
            cluster_means = self.transformed_centers_
        else:
            cluster_means = np.array(self.graph_data["nodes"])
            if transformation is not None and cluster_means.shape[-1] > 2:
                cluster_means = transformation.transform(cluster_means)
                print("transformed means")

        # drawing the background for NEB in the 2D case
        if self.latent_dim == 2:
            image_resolution = 128
            if self.data is not None:
                # if the data is available, we span all datapoints when plotting the contour plot.
                margin = 0.1
                our_data = self.data
            else:
                # as fallback, use the cluster means as a proxy for where data points are (with increased margin)
                margin = 0.5
                our_data = cluster_means

            # creating the grid
            linspace_x = np.linspace(
                our_data[:, 0].min() - margin,
                our_data[:, 0].max() + margin,
                image_resolution,
            )
            linspace_y = np.linspace(
                our_data[:, 1].min() - margin,
                our_data[:, 1].max() + margin,
                image_resolution,
            )
            XY = np.stack(np.meshgrid(linspace_x, linspace_y), -1)

            # scoring the grid points
            tmm_probs = self.mixture_model.score_samples(XY.reshape(-1, 2)).reshape(
                image_resolution, image_resolution
            )

            # and plotting the energy landscape
            plt.contourf(
                linspace_x,
                linspace_y,
                tmm_probs,
                levels=20,
                cmap="coolwarm",
                alpha=0.5,
                zorder=-10,
            )

        # plot cluster means
        plt.scatter(*cluster_means.T, alpha=1.0, rasterized=True, s=30, c="black")

        cmap = plt.get_cmap("viridis")  # choose a colormap

        if pairs is None:
            if n_clusters is not None:
                target_num_clusters = n_clusters
            elif self.labels is not None:
                target_num_clusters = len(np.unique(self.labels))
            elif hasattr(self, "n_clusters"):
                target_num_clusters = self.n_clusters
            else:
                target_num_clusters = len(
                    cluster_means
                )  # no clusters are merged, so no edges are drawn

            # by default we do not draw lines for all pairs of nodes that are merged (because some may not have converged)
            # but instead for the shortest edges that form the MST.
            pairs = self.get_merged_pairs(
                target_num_classes=target_num_clusters, only_mst_edges=True
            )

        if hasattr(self, "transformed_paths_"):
            paths_store = self.transformed_paths_
        else:
            paths_store = self.paths_

        for i, j in pairs:
            if i == j:
                continue
            path = paths_store[(i, j)]
            path = path[np.linspace(0, len(path) - 1, 100, dtype=int, endpoint=True)]
            if path.shape[-1] > 2 and transformation is not None:
                print(f"transforming a path of shape {path.shape}")
                path = transformation.transform(path)

            plt.plot(
                path[:, 0],
                path[:, 1],
                lw=1,
                alpha=0.5,
                color="black",
                # color=cmap(normalized_component_labels[i]),
            )

    @classmethod
    def filter_mixture_model(self, mixture_model, data_X, component_filter):
        """
        This function removes components based on component_filter. 
        Since the mixture model is "unbalanced" after just removing some components, 
        it is re-adjusted by fitting with the reduced components for two rounds.
        
        component_filter: Boolean array containing True for components that should be kept
        """
        # remove components as indicated by component_filter
        mixture_model.location_ = mixture_model.location_[component_filter]
        mixture_model.scale_ = mixture_model.scale_[:, :, component_filter]
        mixture_model.n_components = np.count_nonzero(component_filter)
        intermediate_weights = mixture_model.mix_weights_[component_filter]
        mixture_model.mix_weights_ = intermediate_weights/sum(intermediate_weights)
        mixture_model.df_ = mixture_model.df_[component_filter]
        mixture_model.scale_inv_cholesky_ = mixture_model.scale_inv_cholesky_[
            :, :, component_filter
        ]
        mixture_model.scale_cholesky = mixture_model.scale_cholesky_[
            :, :, component_filter
        ]
        
        '''
        without the mixture_model.fit() call, something is very off (maybe the normalization of mixture component weights, but probably something more) - the energy landscape looks completely different in that case. After "refitting" (essentially doing one E and one M step, or two, just to be sure) the plot looks just as expected (similar to the one before, but without the additional weird components)
        '''
        # perform 2 rounds of model fitting to re-adjust everything
        orig_n_iter = mixture_model.n_iter_
        mixture_model.n_iter_ = 2
        mixture_model.fit(data_X)
        mixture_model.n_iter = orig_n_iter

        return mixture_model

    @classmethod
    def filter_mixture_model_elongation(self, mixture_model, data_X, max_elongation=1000):
        # removes very elongated components one-by-one

        while True:
            # compute cluster elongations
            elongations = []
            for i in range(mixture_model.scale_.shape[2]):
                eigenvalues = np.linalg.eigvalsh(mixture_model.scale_[:, :, i])
                elongation = max(eigenvalues) / min(eigenvalues)
                elongations.append(elongation)
            elongation_filter = np.array(elongations) < (max(elongations)-1)
        
            # remove worst component
            if max(elongations) > max_elongation:
                mixture_model = self.filter_mixture_model(mixture_model=mixture_model, data_X= data_X, component_filter=elongation_filter)
            else:
                break
        
        return mixture_model

    @classmethod
    def filter_mixture_model_small_components(self, mixture_model, data_X, factor=15):
        
        # compute min size of a component to keep
        n_items = len(data_X)
        n_components = len(mixture_model.location_)
        min_n_items = n_items / n_components / factor

        # check sizes of predicted cluster classes
        y_pred = mixture_model.predict(data_X)
        _, counts = np.unique(y_pred, return_counts=True)
        component_filter = counts > min_n_items

        self.filter_mixture_model(mixture_model=mixture_model, data_X=data_X, component_filter=component_filter)

        return mixture_model

    @classmethod
    def filter_mixture_model_components(self, mixture_model, data_X, max_elongation, factor):
        """Performs both types of filtering until the model stabilizes"""
        previous_num_components = len(mixture_model.location_)
        while True:
            # perform both types of filtering
            mixture_model = self.filter_mixture_model_elongation(mixture_model=mixture_model, data_X=data_X, max_elongation=max_elongation)
            mixture_model = self.filter_mixture_model_small_components(mixture_model=mixture_model,data_X=data_X,factor=factor)
            if previous_num_components != len(mixture_model.location_):
                previous_num_components = len(mixture_model.location_)
            else:
                # nothing changed, so we are done.
                break

        return mixture_model
