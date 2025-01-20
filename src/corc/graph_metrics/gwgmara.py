from datetime import datetime
import colorcet as cc
import diptest
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns


from corc.graph_metrics.graph import GWGGraph
import corc.utils


class GWGMara(GWGGraph):
    def __init__(
        self,
        latent_dim,
        data=None,
        labels=None,
        path=None,
        n_clusters=None,
        n_components=10,
        n_neighbors=3,
        covariance="diag",
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
        super().__init__(
            latent_dim,
            data,
            labels,
            path,
            n_clusters,
            n_components,
            n_neighbors,
            covariance,
            clustering_method,
            filter_edges,
            seed,
        )

    def create_graph(self, save=True, plot=True, return_graph=False):
        """'
        1. Overcluster data using a GMM
        2. Construct a weighted undirected graph with the clusters as centers. Low weights mean, that the clusters are more disconnected.
        We projected the samples of two clusters onto the line connecting the two cluster means and apply the diptest on that.
        We use the dip statistic as edge weights (line thickness) for the graph.
        3. Plots show tsne on samples and gmm cluster means.
        """
        center_points, pred_labels = self._cluster()
        embeddings, cluster_means = self._dim_reduction(center_points)

        knn_dict = self._get_knn_dict(
            center_points,
            k=self.n_neighbors,
            thresh=self._get_threshold_filter_edges(
                center_points, k=self.n_components - 1
            ),
        )
        dip_dict = self._get_weight_dict(center_points, pred_labels, knn_dict)
        edges = self._get_edges_dict_initial(knn_dict, dip_dict, thresh=np.inf)

        self.graph_data = {
            "nodes": cluster_means,
            "edges": edges,
            "nodes_org_space": center_points,
            "norm": [
                np.array(list(edges.values())).min(),
                np.array(list(edges.values())).max(),
            ],
        }

        filename = f'_{self.n_components}_vae_latent_dim_{self.latent_dim}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'

        if plot:
            self._plt_graph_compare(embeddings, pred_labels, save=f"{filename}.png")
        if save:
            self.save_graph(f"{filename}.pkl")
        if return_graph:
            return self.graph_data

    def _get_weight_dict(self, means, labels, knn_dict):
        dip_dict = {}
        for cm, neighs in knn_dict.items():
            dip_list = []
            for n in list(neighs):
                cluster1_proj, cluster2_proj = self._compute_projection(
                    cm, n, means, labels
                )

                dip = (
                    1  # take the inverse of the dip statistic to have low values if the clusters are bimodal and high values if unimodal
                    - diptest.dipstat(np.concatenate([cluster1_proj, cluster2_proj]))
                    * 2  # the *2 in here scales the output such that the maximum value is 1
                )
                dip_list.append(dip)
            dip_dict[cm] = dip_list
        return dip_dict

    def _plt_graph_compare(self, embeddings, pred_labels, save=None, to_norm=True):
        super()._plt_graph_compare(embeddings, pred_labels, save=save, to_norm=to_norm)
