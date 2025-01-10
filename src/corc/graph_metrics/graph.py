from pathlib import Path
import os
import scipy
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph


class Graph:
    def __init__(self, latent_dim, data=None, labels=None, path=None, seed=42):
        """
        Initialize the Graph class.

        Args:
            data (np.array): Input data.
            labels (np.array): Labels.
            latent_dim (int): Dimension of the data.
        """
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.seed = seed
        self.data = data
        self.labels = labels
        self.latent_dim = latent_dim
        self.seed = seed

        self.path = Path("figures") if path is None else path

        self.graph_data = {"nodes": None, "edges": None, "nodes_org_space": None}

    def fit(self, data):
        """
        Abstract method that fits the model. To be implemented by subclasses.
        """
        pass

    def predict(self, data, target_number_classes=0):
        """
        Abstract method that predicts the model. To be implemented by subclasses.

        target_number_classes contains number of classes that may be predicted
            (i.e. possibly merging existing classes to get to the right number)
        """
        pass

    def create_graph(self, save=True, plot=True, return_graph=False, *args, **kwargs):
        """
        Abstract method to create a graph.

        This method should be implemented by subclasses.
        """
        pass

    def apply_tsne(self, X2D, transform_paths=True, samples_per_path=50):
        # we assume that fitting did take place
        if self.graph_data["nodes"] is not None:
            self.transformed_centers_ = X2D.transform(self.graph_data["nodes"])

    def save_graph(self, file_name):
        """
        Save the graph to a file.

        Args:
            file_name (str): The file name where the graph should be saved.
        """
        import pickle

        if self.graph_data["nodes"] is None:
            raise ValueError(
                "Graph data is not created yet. Please create the graph before saving."
            )

        with open(self.path / f"graph{file_name}", "wb") as f:
            pickle.dump(self.graph_data, f)

        print(f"Graph saved to {self.path / f'graph{file_name}'}.")

    def _dim_reduction(self, cluster_centers):
        from openTSNE import TSNE

        if self.latent_dim > 2:
            tsne = TSNE(
                perplexity=len(self.data) / 100,
                metric="euclidean",
                n_jobs=8,
                random_state=self.seed,
                verbose=False,
            )
            embeddings = tsne.fit(self.data)
            cluster_means = embeddings.transform(cluster_centers)
        else:
            embeddings = self.data
            cluster_means = cluster_centers

        return embeddings, cluster_means

    def get_graph(self):
        if self.graph_data is None or self.graph_data["nodes"] is None:
            self.create_graph(save=False, plot=False, return_graph=False)
        return self.graph_data

    def plot_graph(self, transformation=None, target_num_clusters=None):
        """
        from openTSNE import TSNE
        tsne = TSNE(
            perplexity=perplexity,
            metric='euclidean',
            n_jobs=8,
            random_state=42,
            verbose=False,
        )
        transformation = tsne.fit(self.data)
        """

        self.get_graph()  # populates self.graph_data

        cluster_means = np.array(self.graph_data["nodes"])

        # if transformation is not None:
        #     cluster_means = transformation.transform(cluster_means)

        plt.scatter(*cluster_means.T, alpha=1.0, rasterized=True, s=15, c="black")

        for (cm, neigh), dip in self.graph_data["edges"].items():
            plt.plot(
                [cluster_means[cm][0], cluster_means[neigh][0]],
                [cluster_means[cm][1], cluster_means[neigh][1]],
                alpha=dip,
                c="black",
            )

    def _get_adjacency_matrix(self):
        if hasattr(self, "adjacency_") and self.adjacency_ is not None:
            return self.adjacency_
        else:
            # create adjacency from "edges"
            adjacency_list = self.graph_data["edges"]
            adjacency_matrix = np.zeros(shape=(self.n_components, self.n_components))

            for (node_u, node_v), value in adjacency_list.items():
                adjacency_matrix[(node_u, node_v)] = value

            self.adjacency_ = adjacency_matrix
            return adjacency_matrix

    def get_thresholds_and_cluster_numbers(self):
        adjacency = self._get_adjacency_matrix()
        thresholds, counts = np.unique(adjacency, return_counts=True)
        thresholds = sorted(thresholds.tolist())

        threshold_dict = dict()
        clustering_dict = dict()
        # record the number of clusters achievable with each threshold
        for threshold in thresholds:
            tmp_adj = np.array(adjacency >= threshold, dtype=int)
            n_components, clusters = scipy.sparse.csgraph.connected_components(
                tmp_adj, directed=False
            )
            # by only updating the threshold (and merging strategy) when necessary, we choose the smallest threshold that leads to this number of clusters.
            if n_components not in threshold_dict.keys():
                threshold_dict[n_components] = threshold
                # also store the merging strategy for that
                clustering_dict[n_components] = clusters

        return threshold_dict, clustering_dict

    def get_best_cluster_number(self, threshold_dict, target_number_classes):
        if target_number_classes in threshold_dict.keys():
            return target_number_classes
        else:
            print(f"{int(target_number_classes)} clusters is not achievable.")
            # try smaller n_clusters instead (and if there is none, use the smallest n_clusters)
            smallest_n_clusters = min(set(threshold_dict.keys()))
            num_classes = max(
                [key for key in threshold_dict.keys() if key < target_number_classes]
                + [
                    smallest_n_clusters
                ]  # this makes sure that we always return something
            )
            print(f"Working with {num_classes} clusters instead.")
            return num_classes


class GWGGraph(Graph):
    def __init__(
        self,
        latent_dim,
        data=None,
        labels=None,
        path=None,
        n_clusters=None,
        n_components=10,
        n_neighbors=3,
        covariance="full",
        clustering_method="gmm",
        filter_edges=True,
        seed=42,
    ):
        """
        Initialize the GWG class.
        GMM + weighted graph.

        Args:
            data (np.array): Input data.
            labels (np.array): Labels.
            latent_dim (int): Dimension of the data.
            n_components (int): Number of components for GMM clustering.
        """

        super().__init__(latent_dim, data, labels, path, seed)

        self.n_clusters = n_clusters
        self.n_components = n_components
        self.n_neighbors = (
            n_neighbors if n_neighbors < n_components else n_components - 1
        )
        self.covariance = covariance
        self.clustering_method = clustering_method
        self.filter_edges = filter_edges

    def _cluster(self):
        if self.clustering_method == "gmm":
            from sklearn.mixture import GaussianMixture

            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance,
                init_params="k-means++",
                random_state=self.seed,
            )
            gmm.fit(self.data)
            pred_labels = gmm.predict(self.data)
            return gmm.means_, pred_labels
        elif self.clustering_method == "tmm":
            import studenttmixture

            tmm = studenttmixture.EMStudentMixture(
                n_components=self.n_components,
                n_init=1,
                fixed_df=False,  # True,
                # df=1.0,
                init_type="k++",
                random_state=self.seed,
            )
            tmm.fit(self.data)
            pred_labels = tmm.predict(self.data)
            return tmm.location, pred_labels
        else:
            print("[ERROR] Clustering method not yet implemented.")
            exit()

    def _dim_reduction(self, gmm_means):
        from openTSNE import TSNE

        if self.latent_dim > 2:
            tsne = TSNE(
                perplexity=len(self.data) / 100,
                metric="euclidean",
                n_jobs=8,
                random_state=self.seed,
                verbose=False,
            )
            embeddings = tsne.fit(self.data)
            cluster_means = embeddings.transform(gmm_means)
        else:
            embeddings = self.data
            cluster_means = gmm_means

        return embeddings, cluster_means

    def _compute_projection(self, cluster1, cluster2, means, predictions):
        c = means[cluster1] - means[cluster2]
        unit_vector = c / np.linalg.norm(c)

        points1 = self.data[predictions == cluster1]
        points2 = self.data[predictions == cluster2]
        cluster1_proj = np.dot(points1, unit_vector)
        cluster2_proj = np.dot(points2, unit_vector)

        mean = (np.mean(cluster1_proj) + np.mean(cluster2_proj)) / 2

        cluster1_proj -= mean
        cluster2_proj -= mean

        return cluster1_proj, cluster2_proj

    def _get_recoloring(self, level, clusterings, pred_labels):
        _, threshold, component_labels = clusterings[level]
        return component_labels[pred_labels], threshold

    def _get_recoloring(self, pred_labels):
        adjacency = self._get_adjacency_matrix()
        n_components, component_labels = scipy.sparse.csgraph.connected_components(
            adjacency, directed=False
        )
        return component_labels[pred_labels]

    def _get_adjacency_matrix(self):
        adjacency_matrix = np.zeros(shape=(self.n_components, self.n_components))

        for (cm, neigh), weight in self.graph_data["edges"].items():
            adjacency_matrix[(cm, neigh)] = weight
            adjacency_matrix[(neigh, cm)] = weight

        return adjacency_matrix

    def plot_thresholds(self):
        threshold_dict, _ = self.get_thresholds_and_cluster_numbers

        # clusters vs thresholds
        plt.plot(list(threshold_dict.keys()), list(threshold_dict.values()), marker="o")
        plt.xlabel("Number of clusters")
        plt.ylabel("Threshold")
        plt.title("Clusters")
        plt.grid()

    def _get_edges_dict_initial(self, knn_dict, weights_dict):
        edges = {}
        for (cm, neighs), (_, weights) in zip(knn_dict.items(), weights_dict.items()):
            for n, weight in zip(list(neighs), list(weights)):
                edges[(cm, n)] = weight
        return edges

    def _get_edges_dict(self, thresh):
        edges = {}
        for (cm, neigh), weight in self.graph_data["edges"].items():
            if weight >= thresh:
                edges[(cm, neigh)] = weight
        return edges

    def _get_knn_dict(self, means, k=3, thresh=np.inf):
        knn = kneighbors_graph(means, k, mode="distance", include_self=False).toarray()
        knn[knn > thresh] = 0

        knn_dict = {}
        for i in range(len(means)):
            neighbors = np.where(knn[i])[0]
            knn_dict[i] = set(neighbors)
        return knn_dict

    def predict(self, data, target_number_clusters=None):
        if target_number_clusters is not None:
            self.n_clusters = target_number_clusters
        threshold = self._get_threshold()
        edges = self._get_edges_dict(thresh=threshold)
        self.graph_data["edges"] = edges
        self.labels_ = self._get_recoloring(pred_labels=self.pred_labels)
        return self.labels_

    def _get_threshold(self):
        threshold_dict, cluster_dict = self.get_thresholds_and_cluster_numbers()
        target_number_classes = self.n_clusters

        # check whether the target number is achievable and select next-best alternative instead
        best_num_classes = self.get_best_cluster_number(
            threshold_dict, target_number_classes
        )
        return threshold_dict[best_num_classes]

    def _get_threshold_filter_edges(self, means, k):
        knn_all_to_all = kneighbors_graph(
            means, k, mode="distance", include_self=False
        ).toarray()
        knn_all_to_all_sorted = np.sort(knn_all_to_all)
        """
        Why 3: I looked at the average distances between 1st, 2nd, etc neighbors and there was a jump in the change of distances between 2nd and 3rd neighbor (second point in the plot), so I set max distance threshold to be the average distance of the 3rd neighbor.
        """
        return knn_all_to_all_sorted.mean(axis=0)[3]

    def fit(self, data):
        """'
        1. Overcluster data using a GMM
        2. Construct a weighted undirected graph with the clusters as centers. Low weights mean, that the clusters are more disconnected.
        We projected the samples of two clusters onto the line connecting the two cluster means and apply the diptest on that.
        We use the pvalue of the diptest as edge weights (line thickness) for the graph.
        3. Plots show tsne on samples and gmm cluster means.
        """
        self.data = data

        center_points, pred_labels = self._cluster()

        # threshold to remove spurious edges
        threshold = (
            self._get_threshold_filter_edges(center_points, k=self.n_components - 1)
            if self.filter_edges
            else np.inf
        )
        knn_dict = self._get_knn_dict(
            center_points, k=self.n_neighbors, thresh=threshold
        )
        weight_dict = self._get_weight_dict(center_points, pred_labels, knn_dict)
        edges = self._get_edges_dict_initial(knn_dict, weight_dict)

        self.graph_data = {
            "nodes": center_points,
            "edges": edges,
            "nodes_org_space": center_points,
            "norm": [
                np.array(list(edges.values())).min(),
                np.array(list(edges.values())).max(),
            ],
        }
        self.pred_labels = pred_labels
        if self.n_clusters is None:
            self.labels_ = self._get_recoloring(pred_labels=pred_labels)
