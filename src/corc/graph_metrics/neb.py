from datetime import datetime
import itertools

import tqdm
import scipy
import sklearn
import studenttmixture
from pandas.core.common import random_state
import numpy as np
import matplotlib.pyplot as plt

from corc.graph_metrics.graph import Graph
import corc.graph_metrics.tmm_gmm_neb


class NEB(Graph):
    def __init__(
        self,
        latent_dim,
        data=None,
        labels=None,
        path=None,
        n_components=15,
        n_neighbors=3,
        thresh=0.01,
        seed=42,
        mixture_model_type="tmm",
        n_init=5,
        optimization_iterations=1000,
        dataset_name=None,
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
        super().__init__(latent_dim, data, labels, path, seed)

        if mixture_model_type == "tmm":
            self.mixture_model = studenttmixture.EMStudentMixture(
                n_components=n_components,
                n_init=n_init,
                fixed_df=False,
                # df=1.0,  # the minimum value, for df=infty we get gmm
                init_type="k++",
                random_state=seed,
            )
        elif mixture_model_type == "gmm":
            self.mixture_model = sklearn.mixture.GaussianMixture(
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

    def fit(self, data):
        self.mixture_model.fit(data)

        # extract centers for TMM/GMM
        if isinstance(self.mixture_model, sklearn.mixture.GaussianMixture):
            self.centers_ = self.mixture_model.means_
        elif isinstance(self.mixture_model, studenttmixture.EMStudentMixture):
            for _ in range(20):
                if self.mixture_model.df_ is None:
                    # we have not converged
                    print("retrying tmm fit with more iterations")
                    self.mixture_model.max_iter = 10000
                    self.mixture_model.fit(data)
            self.centers_ = self.mixture_model.location

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

        # normalize adjacency
        norm_adjacency = self.adjacency_ - np.min(self.adjacency_)
        self.adjacency_ = norm_adjacency / np.max(norm_adjacency)

    def predict(self, data):
        if self.paths_ is None:  # fitting did not yet take place
            self.fit(data)
        return self.mixture_model.predict(data)

    def predict_with_target(self, data, target_number_classes):
        predictions = self.predict(data)

        # merging classes if necessary
        if target_number_classes < self.n_components:
            raw_predictions = predictions
            thresholds, cluster_numbers, clusterings = (
                self.get_thresholds_and_cluster_numbers()
            )

            # check whether it is possible to reach target_number_clusters and choose next-smaller value if necessary
            if target_number_classes not in cluster_numbers:
                print(f"{target_number_classes} clusters is not achievable.")
                target_number_classes = max(
                    num for num in cluster_numbers if num < target_number_classes
                )
                print(f"Working with {target_number_classes} clusters instead.")

            # find matching threshold
            threshold = thresholds[
                np.where(cluster_numbers == target_number_classes)[0][0]
            ]

            # find cluster pairs that need to be merged. We assume a "distance" matrix, so smaller is better
            adjacency = self._get_adjacency_matrix()
            tmp_adj = np.array(adjacency >= threshold, dtype=int)
            _, component_labels = scipy.sparse.csgraph.connected_components(
                tmp_adj, directed=False
            )
            # actually merge the predictions
            predictions = component_labels[raw_predictions]

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
            for i, j in itertools.combinations(range(self.n_components), 2)
        }

        # normalized edges
        norm_adjacency = self.adjacency_ - np.min(self.adjacency_)
        norm_adjacency /= np.max(norm_adjacency)

        normalized_edges = {
            (i, j): norm_adjacency[i, j]
            for i, j in itertools.combinations(range(self.n_components), 2)
        }

        self.graph_data = {
            "nodes": self.centers_,
            # "nodes_org_space": self.centers_,
            "edges": normalized_edges,
            "raw_edges": edges,
        }

        if plot:
            raise NotImplementedError
        #     self._plt_graph_compare(embeddings, self.labels_, save=f"{filename}.png")

        if save:
            raise NotImplementedError

        if return_graph:
            return self.graph_data

    def compute_mst_edges(self, raw_adjacency):
        mst = -scipy.sparse.csgraph.minimum_spanning_tree(-raw_adjacency)
        rows, cols = mst.nonzero()
        entries = list(zip(rows, cols))
        return entries

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

    def plot_graph(self, X2D=None):
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

        # plot cluster means
        plt.scatter(*cluster_means.T, alpha=1.0, rasterized=True, s=30, c="black")

        cmap = plt.get_cmap("viridis")  # choose a colormap

        if hasattr(self, "pairs_"):
            pairs = self.pairs_
        else:
            pairs = self.compute_mst_edges(self.raw_adjacency_)

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
