from datetime import datetime
import colorcet as cc
import diptest
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph


from corc.graph_metrics.graph import GWGGraph


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
        covariance='diag',
        clustering_method='gmm',
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
        super().__init__(latent_dim, data, labels, path, n_clusters, n_components, n_neighbors, covariance, clustering_method, seed)


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
            thresh=self._get_threshold(center_points, k=self.n_components - 1),
        )
        dip_dict = self._get_dip_dict(center_points, pred_labels, knn_dict)
        edges = self._get_edges_dict(knn_dict, dip_dict)

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

    def _get_threshold(self, means, k):
        knn_all_to_all = kneighbors_graph(
            means, k, mode="distance", include_self=False
        ).toarray()
        knn_all_to_all_sorted = np.sort(knn_all_to_all)
        """
        Why 3: I looked at the average distances between 1st, 2nd, etc neighbors and there was a jump in the change of distances between 2nd and 3rd neighbor (second point in the plot), so I set max distance threshold to be the average distance of the 3rd neighbor.
        """
        return knn_all_to_all_sorted.mean(axis=0)[3]

    def _get_knn_dict(self, means, k=3, thresh=np.inf):
        knn = kneighbors_graph(means, k, mode="distance", include_self=False).toarray()
        knn[knn > thresh] = 0

        knn_dict = {}
        for i in range(len(means)):
            neighbors = np.where(knn[i])[0]
            knn_dict[i] = set(neighbors)
        return knn_dict

    def _get_dip_dict(self, means, labels, knn_dict):
        dip_dict = {}
        for cm, neighs in knn_dict.items():
            dip_list = []
            for n in list(neighs):
                cluster1_proj, cluster2_proj = self._compute_projection(
                    cm, n, means, labels
                )

                dip = (
                    diptest.dipstat(np.concatenate([cluster1_proj, cluster2_proj]))
                    * 2
                    # the *2 in here scales the output such that the maximum value is 1
                )
                dip_list.append(dip)
            dip_dict[cm] = dip_list
        return dip_dict


    def _get_edges_dict(self, knn_dict, pvalue_dict):
        edges = {}

        for (cm, neighs), (_, dips) in zip(knn_dict.items(), pvalue_dict.items()):
            for n, dip in zip(list(neighs), list(dips)):
                edges[(cm, n)] = dip
        return edges


    def _plt_graph_compare(self, embeddings, pred_labels, save=None):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        palette = (
            sns.color_palette(cc.glasbey, n_colors=self.n_components)
            if self.n_components > 20
            else sns.color_palette("tab20")
        )

        # colored by labels
        axs[0].set_title("Embeddings colored by GT label")
        for c in range(len(np.unique(self.labels))):
            axs[0].scatter(
                *embeddings[self.labels == c].T,
                s=5,
                color=palette[c],
                alpha=1.0,
                rasterized=True,
                label=c,
            )
        axs[0].legend(bbox_to_anchor=(1, 1), markerscale=3)

        # colored by gmm clustering prediction
        axs[1].set_title("Clustering and graph")
        for c in range(self.n_components):
            axs[1].scatter(
                *embeddings[pred_labels == c].T,
                s=5,
                color=palette[c],
                alpha=1.0,
                rasterized=True,
                label=c,
            )

        cluster_means = self.graph_data["nodes"]
        axs[1].scatter(*cluster_means.T, alpha=1.0, rasterized=True, s=15, c="black")

        norm = mpl.colors.Normalize(
            vmin=self.graph_data["norm"][0], vmax=self.graph_data["norm"][1]
        )

        for (cm, neigh), dip in self.graph_data["edges"].items():
            axs[1].plot(
                [cluster_means[cm][0], cluster_means[neigh][0]],
                [cluster_means[cm][1], cluster_means[neigh][1]],
                alpha=(1 - norm(dip)),
                # linewidth=(norm(dip) - 1) * -3 + 0.1,
                c="black",
            )

            axs[1].text(
                (cluster_means[cm][0] + cluster_means[neigh][0]) / 2,
                (cluster_means[cm][1] + cluster_means[neigh][1]) / 2,
                f"{dip:.3f}",
                fontsize=8,
                alpha=1.0,
            )

        for ax in axs:
            ax.axis("off")

        if save:
            self.path.mkdir(parents=True, exist_ok=True)
            path = self.path / f"graph_compare_gwgmara{save}"
            plt.savefig(path, bbox_inches="tight")
            plt.show()
            plt.close()
            print(f"WARNING: saving figure to file {path}")
        else:
            plt.show()

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

        threshold = self._get_threshold(center_points, k=self.n_components - 1)
        knn_dict = self._get_knn_dict(
            center_points, k=self.n_neighbors, thresh=threshold
        )
        dip_dict = self._get_dip_dict(center_points, pred_labels, knn_dict)
        edges = self._get_edges_dict(knn_dict, dip_dict)

        self.graph_data = {
            "nodes": center_points,
            "edges": edges,
            "nodes_org_space": center_points,
            "norm": [
                np.array(list(edges.values())).min(),
                np.array(list(edges.values())).max(),
            ],
        }
        self.labels_ = self._get_recoloring(pred_labels=pred_labels)

    def plot_graph(self, X2D=None):
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
        cluster_means = self.graph_data["nodes"]

        if X2D is not None:
            cluster_means = X2D.transform(cluster_means)
        self.graph_data["nodes"] = cluster_means

        plt.scatter(*cluster_means.T, alpha=1.0, rasterized=True, s=15, c="black")

        norm = mpl.colors.Normalize(
            vmin=self.graph_data["norm"][0], vmax=self.graph_data["norm"][1]
        )
        for (cm, neigh), weight in self.graph_data["edges"].items():
            plt.plot(
                [cluster_means[cm][0], cluster_means[neigh][0]],
                [cluster_means[cm][1], cluster_means[neigh][1]],
                # linewidth=(1 - norm(weight)),
                c="black",
            )
