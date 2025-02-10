import numpy as np
import pandas as pd
import seaborn as sns
import colorcet as cc
import scanpy as sc
import anndata as ad
import matplotlib.pylab as plt
from scipy.sparse import csr_matrix

from corc.graph_metrics.graph import Graph
import corc.utils


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
        use_rep="X",
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
        super().__init__(
            latent_dim=latent_dim, data=data, labels=labels, path=path, seed=seed
        )

        self.resolution = resolution
        self.clustering_method = clustering_method
        self.use_rep = use_rep

    def create_graph(self, save=True):
        self.fit(self.data)
        embeddings, cluster_means = self._dim_reduction(
            self.graph_data["nodes_org_space"]
        )
        self.graph_data["nodes"] = cluster_means
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

        sc.pp.neighbors(self.adata, random_state=self.seed, use_rep=self.use_rep)
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
            raise NotImplementedError(
                'Wrong clustering method. Choose "louvain" or "leiden".'
            )

        self.labels_ = self.adata.obs[self.clustering_method].astype(int)
        self.n_components = len(np.unique(self.labels_))
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
        self.labels_ = self._get_recoloring(pred_labels=list(self.labels_))

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
        cluster_means = self.graph_data["nodes"]

        if ax is None:
            ax = plt.gca()

        # get TSNE(centers) if needed
        if self.latent_dim > 2:
            if hasattr(self, "transformed_centers_"):
                cluster_means = self.transformed_centers_
            else:
                if X2D is not None:
                    # cluster_means = X2D.transform(cluster_means)
                    cluster_means = corc.utils.snap_points_to_TSNE(
                        points=cluster_means, data_X=self.data, transformed_X=X2D
                    )
                    self.graph_data["nodes"] = cluster_means
                    self.transformed_centers_ = cluster_means
                else:
                    print("transformation missing!")

        ax.scatter(*cluster_means.T, alpha=1.0, rasterized=True, s=30, c="black")

        for (cm, neigh), weight in self.graph_data["edges"].items():
            ax.plot(
                [cluster_means[cm][0], cluster_means[neigh][0]],
                [cluster_means[cm][1], cluster_means[neigh][1]],
                alpha=1.0,
                c="black",
            )
