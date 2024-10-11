from pathlib import Path
import os
import scipy
import random
import numpy as np
import matplotlib.pyplot as plt


class Graph():
    def __init__(self, latent_dim, data=None, labels=None, path=None, seed=42):
        """
        Initialize the Graph class.

        Args:
            data (np.array): Input data.
            labels (np.array): Labels.
            latent_dim (int): Dimension of the data.
        """

        os.environ['PYTHONHASHSEED']=str(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        self.seed = seed

        self.data = data
        self.labels = labels
        self.latent_dim = latent_dim
        self.seed = seed

        self.path = Path('figures') if path is None else path

        self.graph_data = {'nodes':None, 'edges':None, 'nodes_org_space':None}


    def create_graph(self, save=True, plot=True, return_graph=False, *args, **kwargs):
        """
        Abstract method to create a graph.

        This method should be implemented by subclasses.
        """
        pass


    def save_graph(self, file_name):
        """
        Save the graph to a file.

        Args:
            file_name (str): The file name where the graph should be saved.
        """
        import pickle

        if self.graph_data["nodes"] is None:
            raise ValueError("Graph data is not created yet. Please create the graph before saving.")
        
        with open(self.path / f'graph{file_name}', 'wb') as f:
            pickle.dump(self.graph_data, f)

        print(f"Graph saved to {self.path / f'graph{file_name}'}.")


class GWGGraph(Graph):
    def __init__(
        self,
        latent_dim,
        data=None,
        labels=None,
        path=None,
        n_components=10,
        n_neighbors=3,
        covariance='full',
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

        self.n_components = n_components
        self.n_neighbors = (
            n_neighbors if n_neighbors < n_components else n_components - 1
        )
        self.covariance = covariance


    def get_graph(self):
        if self.graph_data["nodes"] is None:
            self.create_graph(save=False, plot=False, return_graph=False)
        return self.graph_data


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


    def _get_edges_dict(self, knn_dict, pvalue_dict):
        edges = {}

        for (cm, neighs), (_, dips) in zip(knn_dict.items(), pvalue_dict.items()):
            for n, dip in zip(list(neighs), list(dips)):
                edges[(cm, n)] = dip
        return edges


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
    

    def get_thresholds_and_cluster_numbers(self):
        adjacency = self._get_adjacency_matrix()
        thresholds, counts = np.unique(adjacency, return_counts=True)
        thresholds = sorted(thresholds.tolist())

        cluster_numbers = list()
        clusterings = list()
        for threshold in thresholds:
            tmp_adj = np.array(adjacency >= threshold, dtype=int)
            n_components, clusters = scipy.sparse.csgraph.connected_components(
                tmp_adj, directed=False
            )
            cluster_numbers.append((threshold, n_components))
            clusterings.append((n_components, threshold, clusters))

        cluster_numbers = np.array(cluster_numbers)

        return thresholds, cluster_numbers, clusterings

    def _get_recoloring(self, level, clusterings, pred_labels):
        _, threshold, clustering = clusterings[level]
        O2R = dict(zip(range(len(clustering)), clustering))
        return np.array([O2R[yp] for yp in pred_labels]), threshold


    def _get_recoloring(self, pred_labels):
        adjacency = self._get_adjacency_matrix()
        n_components, clustering = scipy.sparse.csgraph.connected_components(
            adjacency, directed=False
        )
        O2R = dict(zip(range(len(clustering)), clustering))
        return np.array([O2R[yp] for yp in pred_labels])

    def _get_adjacency_matrix(self):
        adjacency_list = self.graph_data["edges"]
        adjacency_matrix = np.zeros(shape=(self.n_components, self.n_components))

        for i in range(self.n_components):
            for j in range(self.n_components):
                adjacency_matrix[i, j] = adjacency_matrix[j, i] = (
                    adjacency_list[(i, j)] if (i, j) in adjacency_list.keys() else 0.0
                )

        return adjacency_matrix

    def plot_thresholds(self, cluster_numbers):
        # clusters vs thresholds
        plt.plot(cluster_numbers[:, 1], cluster_numbers[:, 0], marker="o")
        plt.xlabel("Number of clusters")
        plt.ylabel("Threshold")
        plt.title("Clusters")
        plt.grid()
