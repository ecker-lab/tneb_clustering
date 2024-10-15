import numpy as np
import pandas as pd
import seaborn as sns
import colorcet as cc
import scanpy as sc
import anndata as ad
import matplotlib.pylab as plt
from scipy.sparse import csr_matrix

from corc.graph_metrics.graph import Graph


class PAGA(Graph):
    def __init__(
        self,
        latent_dim,
        data=None,
        labels=None,
        path=None,
        resolution=0.1,
        clustering_method="leiden",
        seed=42,
    ):
        """
        Initialize the PAGA class.

        Args:
            data (np.array): Input data.
            labels (np.array): Labels.
            latent_dim (int): Dimension of the data.
            resolution (float):
            clustering_method (str): Set clustering method. Options are 'leiden' and 'louvain'.
        """
        super().__init__(latent_dim, data, labels, path, seed)

        self.resolution = resolution
        self.clustering_method = clustering_method

    def create_graph(self, save=True):

        counts = csr_matrix(self.data, dtype=np.float32)
        adata = ad.AnnData(counts)
        adata.obs["category"] = self.labels
        self.adata = adata

        sc.pp.neighbors(adata, random_state=self.seed)
        sc.tl.umap(adata, random_state=self.seed)

        if self.clustering_method == "leiden":
            sc.tl.leiden(
                adata,
                flavor="igraph",
                n_iterations=2,
                resolution=self.resolution,
                random_state=self.seed,
            )
        elif self.clustering_method == "louvain":
            sc.tl.louvain(
                adata,
                flavor="igraph",
                resolution=self.resolution,
                random_state=self.seed,
            )
        else:
            print('Wrong clustering method. Choose "louvain" or "leiden".')
            exit()

        # sc.pl.umap(self.adata, color=[self.clustering_method, "category"], legend_loc="on data", color_map='tab20', save=f'_{self.clustering_method}_{self.resolution}_vae_latent_dim_{self.latent_dim}.png')

        sc.tl.paga(self.adata, groups=self.clustering_method)
        # sc.pl.paga_compare(
        #     self.adata,
        #     threshold=0.05,
        #     title="",
        #     right_margin=0.2,
        #     size=10,
        #     edge_width_scale=0.5,
        #     legend_fontsize=12,
        #     fontsize=12,
        #     frameon=False,
        #     edges=True,
        #     save=f'{filename}.png',
        # )

        nodes_org_space = (
            pd.DataFrame(self.data, index=self.adata.obs_names)
            .groupby(self.adata.obs[self.clustering_method], observed=True)
            .median()
            .sort_index()
        ).to_numpy()
        self.labels_ = np.array(self.adata.obs[self.clustering_method], dtype="int")
        self.n_components = len(np.unique(self.labels_))

        embeddings, cluster_means = self._dim_reduction(nodes_org_space)
        edges = self._adj_matrix_to_edges_list(
            self.adata.uns["paga"]["connectivities"].toarray()
        )

        self.graph_data = {
            "nodes": cluster_means,
            "edges": edges,
            "nodes_org_space": nodes_org_space,
        }

        filename = f"_{self.clustering_method}_{self.resolution}_vae_latent_dim_{self.latent_dim}"
        self._plt_graph_compare(embeddings, self.labels_, save=f"{filename}.png")

        if save:
            self.save_graph(f"{filename}.pkl")

    def _adj_matrix_to_edges_list(self, matrix, thresh=0.05):
        edges = {}
        num_rows, num_cols = matrix.shape

        for i in range(num_rows):
            for j in range(num_cols):
                if matrix[i, j] > thresh:
                    edges[(i, j)] = matrix[i, j]
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

        for (cm, neigh), dip in self.graph_data["edges"].items():
            axs[1].plot(
                [cluster_means[cm][0], cluster_means[neigh][0]],
                [cluster_means[cm][1], cluster_means[neigh][1]],
                alpha=1.0,
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
            path = self.path / f"graph_compare{save}"
            plt.savefig(path, bbox_inches="tight")
            plt.show()
            plt.close()
            print(f"WARNING: saving figure to file {path}")
        else:
            plt.show()

    def _dim_reduction(self, means):
        if self.latent_dim > 2:
            cluster_means = (
                pd.DataFrame(self.adata.obsm["X_umap"], index=self.adata.obs_names)
                .groupby(self.adata.obs[self.clustering_method], observed=True)
                .median()
                .sort_index()
            ).to_numpy()
            embeddings = self.adata.obsm["X_umap"]
        else:
            embeddings = self.data
            cluster_means = means

        return embeddings, cluster_means

    def fit(self, data):
        self.data = data

        counts = csr_matrix(self.data, dtype=np.float32)
        adata = ad.AnnData(counts)
        adata.obs["category"] = self.labels

        self.adata = adata

        sc.pp.neighbors(self.adata)
        sc.tl.umap(adata)

        if self.clustering_method == "leiden":
            sc.tl.leiden(
                self.adata,
                flavor="igraph",
                n_iterations=2,
                resolution=self.resolution,
                random_state=self.seed,
            )
        elif self.clustering_method == "louvain":
            sc.tl.louvain(
                self.adata,
                flavor="igraph",
                resolution=self.resolution,
                random_state=self.seed,
            )
        else:
            print('Wrong clustering method. Choose "louvain" or "leiden".')
            exit()

        self.labels_ = self.adata.obs[self.clustering_method]
        pos = (
            pd.DataFrame(data, index=self.adata.obs_names)
            .groupby(self.labels_, observed=True)
            .median()
            .sort_index()
        ).to_numpy()

        sc.tl.paga(self.adata, groups=self.clustering_method)

        edges = self._adj_matrix_to_edges_list(
            self.adata.uns["paga"]["connectivities"].toarray()
        )
        self.graph_data = {"nodes": pos, "edges": edges, "nodes_org_space": pos}

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

        for (cm, neigh), weight in self.graph_data["edges"].items():
            plt.plot(
                [cluster_means[cm][0], cluster_means[neigh][0]],
                [cluster_means[cm][1], cluster_means[neigh][1]],
                alpha=1.0,
                c="black",
            )
