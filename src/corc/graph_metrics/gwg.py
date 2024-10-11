from datetime import datetime
import colorcet as cc
import diptest
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph


from corc.graph_metrics.graph import GWGGraph


class GWG(GWGGraph):
    def __init__(
        self,
        latent_dim,
        data=None,
        labels=None,
        path=None,
        n_components=10,
        n_neighbors=3,
        thresh=0.01,
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

        super().__init__(latent_dim, data, labels, path, n_components, n_neighbors, covariance, seed)

        self.thresh = thresh

    def create_graph(self, save=True, plot=True, return_graph=False):
        """'
        1. Overcluster data using a GMM
        2. Construct a weighted undirected graph with the clusters as centers. Low weights mean, that the clusters are more disconnected.
        We projected the samples of two clusters onto the line connecting the two cluster means and apply the diptest on that.
        We use the pvalue of the diptest as edge weights (line thickness) for the graph.
        3. Plots show tsne on samples and gmm cluster means.
        """
        gmm_fit = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance,
            init_params="k-means++",
            random_state=self.seed,
        )
        gmm_fit = gmm_fit.fit(self.data)
        pred_labels = gmm_fit.predict(self.data)

        embeddings, cluster_means = self._dim_reduction(gmm_fit.means_)

        knn_dict = self._get_knn_dict(gmm_fit.means_, k=self.n_neighbors)
        pvalue_dict = self._get_pvalue_dict(gmm_fit.means_, pred_labels, knn_dict)
        edges = self._get_edges_dict(knn_dict, pvalue_dict)

        self.graph_data = {
            "nodes": cluster_means,
            "edges": edges,
            "nodes_org_space": gmm_fit.means_,
        }

        filename = f'_{self.n_components}_vae_latent_dim_{self.latent_dim}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'

        if plot:
            self._plt_graph_compare(embeddings, pred_labels, save=f"{filename}.png")

        if save:
            self.save_graph(f"{filename}.pkl")

        if return_graph:
            return self.graph_data


    def _get_knn_dict(self, means, k=3):
        knn = kneighbors_graph(means, k, mode="distance", include_self=False).toarray()
        knn[knn < self.thresh] = 0

        knn_dict = {}
        for i in range(len(means)):
            neighbors = np.where(knn[i])[0]
            knn_dict[i] = neighbors
        return knn_dict

    def _get_pvalue_dict(self, means, labels, knn_dict):
        pvalue_dict = {}
        for cm, neighs in knn_dict.items():
            pvalue_list = []
            for n in list(neighs):

                cluster1_proj, cluster2_proj = self._compute_projection(
                    cm, n, means, labels
                )
                dip, pvalue = diptest.diptest(
                    np.concatenate([cluster1_proj, cluster2_proj])
                )
                pvalue_list.append(pvalue)
            pvalue_dict[cm] = pvalue_list
        return pvalue_dict


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

        for (cm, neigh), dip in self.graph_data["edges"].items():
            axs[1].plot(
                [cluster_means[cm][0], cluster_means[neigh][0]],
                [cluster_means[cm][1], cluster_means[neigh][1]],
                alpha=dip,
                c="black",
            )

            axs[1].text(
                (cluster_means[cm][0] + cluster_means[neigh][0]) / 2,
                (cluster_means[cm][1] + cluster_means[neigh][1]) / 2,
                f"{dip:.3f}",
                fontsize=8,
                alpha=dip,
            )

        for ax in axs:
            ax.axis("off")

        if save:
            self.path.mkdir(parents=True, exist_ok=True)
            path = self.path / f"graph_compare{save}"
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

        gmm_fit = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance,
            init_params="k-means++",
        )
        gmm_fit = gmm_fit.fit(self.data)
        pred_labels = gmm_fit.predict(self.data)

        knn_dict = self._get_knn_dict(gmm_fit.means_, k=self.n_neighbors)

        pvalue_dict = self._get_pvalue_dict(gmm_fit.means_, pred_labels, knn_dict)
        edges = self._get_edges_dict(knn_dict, pvalue_dict)

        self.graph_data = {
            "nodes": gmm_fit.means_,
            "edges": edges,
            "nodes_org_space": gmm_fit.means_,
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

        for (cm, neigh), dip in self.graph_data["edges"].items():
            plt.plot(
                [cluster_means[cm][0], cluster_means[neigh][0]],
                [cluster_means[cm][1], cluster_means[neigh][1]],
                alpha=dip,
                c="black",
            )
