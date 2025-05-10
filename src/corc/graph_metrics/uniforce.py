import matplotlib.pyplot as plt
import numpy as np

from Uniforce import Uniforce
from data.enums.Algorithm_Over_Clustering import Algorithm_Over_Clustering
from data.options.SpanningTreeOptions import SpanningTreeOptions
from data.options.UniforceOptions import UniforceOptions

Algorithm_Over_Clustering.KmeansPp  # This is just so that auto import clean up won't remove the import of the Algorithms when we save the file and don't use the Algorithms class


class Uniforce_Wrapper:
    def __init__(self, alpha, num_clusters=None):
        self.alpha = alpha
        self.target_num_clusters = num_clusters
        self.uniforce = Uniforce(
            UniforceOptions(
                spanning_tree_options=SpanningTreeOptions(
                    alpha=self.alpha,
                    specific_number_of_clusters=self.target_num_clusters,
                ),
                algorithm_over_clustering=Algorithm_Over_Clustering.GlobalKMeansPpParallel,
            )
        )
        self.result = None
        self.graph_data = {"nodes": None, "edges": None, "nodes_org_space": None}

    def fit(self, data):
        self.result = self.uniforce.fit(data)

    def predict(self, data):
        return self.result.labels

    def _adjacency_matrix_to_dict(self, matrix):
        adjacency_dict = {}

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] != 0:  # Only store non-zero edges
                    adjacency_dict[(i, j)] = matrix[i][j]  # Store edge with weight

        return adjacency_dict

    def _get_edges_dict(self, edges, thresh=0):
        edges = {}
        for (cm, neigh), weight in edges.items():
            if weight >= thresh:
                edges[(cm, neigh)] = weight
        return edges

    def create_graph(self, save=True, plot=True, return_graph=False, *args, **kwargs):
        cluster_means = self.result.sub_cluster_centers[self.result.is_active == True]
        edges = self._adjacency_matrix_to_dict(self.result.adjacency)
        edges = self._get_edges_dict(edges)

        self.graph_data = {
            "nodes": cluster_means,
            "edges": edges,
            # "nodes_org_space": center_points,
        }

        filename = f"_uniforce"

        if plot:
            self._plt_graph_compare(embeddings, pred_labels, save=f"{filename}.png")
        if save:
            self.save_graph(f"{filename}.pkl")
        if return_graph:
            return self.graph_data

    def get_graph(self):
        if self.graph_data is None or self.graph_data["nodes"] is None:
            self.create_graph(save=False, plot=False, return_graph=False)
        return self.graph_data

    def plot_graph(self, X2D=None, target_num_clusters=None, ax=None):
        """
        from openTSNE import TSNE
        tsne = TSNE(
            perplexity=perplexity,
            metric='euclidean',
            n_jobs=8,
            random_state=42,
            verbose=False,
        )
        X2D = tsne.fit(self.data)
        """
        if ax is None:
            ax = plt.gca()

        self.get_graph()  # populates self.graph_data

        cluster_means = np.array(self.graph_data["nodes"])

        if X2D is not None:
            cluster_means = corc.utils.snap_points_to_TSNE(
                points=cluster_means, data_X=self.data, transformed_X=X2D
            )
            self.graph_data["nodes"] = cluster_means

        ax.scatter(*cluster_means.T, alpha=1.0, rasterized=True, s=15, c="black")

        for (cm, neigh), weight in self.graph_data["edges"].items():
            ax.plot(
                [cluster_means[cm][0], cluster_means[neigh][0]],
                [cluster_means[cm][1], cluster_means[neigh][1]],
                # alpha=weight,
                c="black",
            )
