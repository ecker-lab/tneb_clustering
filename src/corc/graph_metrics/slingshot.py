import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from pyslingshot import Slingshot as PySlingshot
from scipy.sparse import csr_matrix
import anndata as ad

from corc.graph_metrics.graph import Graph


class Slingshot(Graph):
    def __init__(
        self,
        latent_dim,
        data=None,
        labels=None,
        path=None,
        n_components=10,
        covariance_type='diag',
        seed=4,
    ):

        """
        code: https://github.com/mossjacob/pyslingshot

        Args:
            data (np.array): Input data.
            labels (np.array): Labels.
            latent_dim (int): Dimension of the data.
        """

        super().__init__(latent_dim, data, labels, path, seed)
        self.n_components = n_components
        self.covariance_type = covariance_type

    
    def _get_labels(self):
        '''
        NOTE: this clustering can be replaced by any other method
        Slingshot does not specify any.
        '''
        from sklearn.mixture import GaussianMixture

        gmm_fit = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            init_params="k-means++",
            random_state=self.seed,
        )
        gmm_fit = gmm_fit.fit(self.data)
        pred_labels = gmm_fit.predict(self.data)
        return pred_labels


    def _dim_reduction(self):
        '''
        NOTE: this dimension reduction algorithm can be replaced by any other method
        Slingshot does not specify any.
        '''
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
        else:
            embeddings = self.data

        return embeddings


    def fit(self, data):
        self.data = data

        counts = csr_matrix(self.data, dtype=np.float32)
        adata = ad.AnnData(counts)
        adata.obsm["X_umap"] = self._dim_reduction()

        self.labels_ = self._get_labels()
        adata.obs["category"] = self.labels_
        self.adata = adata

        slingshot = PySlingshot(self.adata, celltype_key="category", obsm_key="X_umap", start_node=0, debug_level='verbose')
        slingshot.fit(num_epochs=1, debug_axes=None)
        self.curves = slingshot.curves


    def plot_graph(self, X2D=None):
        if X2D is not None:
            self.adata.obsm["X_umap"] = X2D
            slingshot = PySlingshot(self.adata, celltype_key="category", obsm_key="X_umap", start_node=0, debug_level='verbose')
            slingshot.fit(num_epochs=1, debug_axes=None)
            self.curves = slingshot.curves

        for l_idx, curve in enumerate(self.curves):
            s_interp, p_interp, order = curve.unpack_params()
            plt.plot(
                p_interp[order, 0],
                p_interp[order, 1],
                alpha=1,
                c='black')