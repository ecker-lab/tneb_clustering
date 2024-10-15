from sklearn import cluster, mixture
import studenttmixture
from src.corc.graph_metrics import paga, gwg, gwgmara, neb
from scipy.sparse import csr_matrix
import scanpy
import anndata
import numpy as np


class Leiden:
    def __init__(self, resolution=1.0, seed=42):
        self.resolution = resolution
        self.seed = seed

    def fit(self, data):
        self.data = data

        counts = csr_matrix(self.data, dtype=np.float32)
        adata = anndata.AnnData(counts)

        self.adata = adata

        scanpy.pp.neighbors(self.adata)
        scanpy.tl.leiden(
            self.adata,
            flavor="igraph",
            n_iterations=2,
            resolution=self.resolution,
            random_state=self.seed,
        )

        self.labels_ = self.adata.obs["leiden"]


def get_clustering_objects(
    params,  # default parameters are in our_datasets.default_base
    bandwidth,
    connectivity,
):
    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(
        n_clusters=params["n_clusters"],
        random_state=params["random_state"],
    )
    ward = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
    )
    spectral = cluster.SpectralClustering(
        n_clusters=params["n_clusters"],
        eigen_solver="arpack",
        affinity="nearest_neighbors",
        random_state=params["random_state"],
    )
    dbscan = cluster.DBSCAN(eps=params["eps"])
    hdbscan = cluster.HDBSCAN(
        min_samples=params["hdbscan_min_samples"],
        min_cluster_size=params["hdbscan_min_cluster_size"],
        allow_single_cluster=params["allow_single_cluster"],
    )
    optics = cluster.OPTICS(
        min_samples=params["min_samples"],
        xi=params["xi"],
        min_cluster_size=params["min_cluster_size"],
    )
    affinity_propagation = cluster.AffinityPropagation(
        damping=params["damping"],
        preference=params["preference"],
        random_state=params["random_state"],
    )
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        metric="cityblock",
        n_clusters=params["n_clusters"],
        connectivity=connectivity,
    )
    birch = cluster.Birch(n_clusters=params["n_clusters"])
    gmm = mixture.GaussianMixture(
        n_components=params["n_clusters"],
        covariance_type="full",
        random_state=params["random_state"],
    )
    tmm = studenttmixture.EMStudentMixture(
        n_components=params["n_clusters"],
        n_init=5,
        fixed_df=False,  # True,
        # df=1.0,
        init_type="k++",
        random_state=params["random_state"],
    )
    leiden = Leiden(resolution=params["resolution_leiden"], seed=params["random_state"])
    mgwg = gwg.GWG(
        latent_dim=params["dim"],
        n_components=params["n_components"],
        n_neighbors=params["n_neighbors"],
        seed=params["random_state"],
    )
    mgwgmara = gwgmara.GWGMara(
        latent_dim=params["dim"],
        n_components=params["n_components"],
        n_neighbors=params["n_neighbors"],
        seed=params["random_state"],
    )
    mpaga = paga.PAGA(
        latent_dim=params["dim"],
        resolution=params["resolution"],
        seed=params["random_state"],
    )
    tmm_neb = neb.NEB(
        latent_dim=params["dim"],
        n_components=params["n_components"],
        seed=params["random_state"],
        mixture_model_type="tmm",
        n_init=1,
        optimization_iterations=50,
    )
    gmm_neb = neb.NEB(
        latent_dim=params["dim"],
        n_components=params["n_components"],
        seed=params["random_state"],
        mixture_model_type="gmm",
        n_init=5,
        optimization_iterations=50,
    )

    clustering_algorithms = (
        ("MiniBatch\nKMeans", two_means),
        ("Agglomerative\nClustering", average_linkage),
        ("HDBSCAN", hdbscan),
        ("Gaussian\nMixture", gmm),
        ("t-Student\nMixture", tmm),
        ("Leiden", leiden),
        ("PAGA", mpaga),
        ("GWG-dip", mgwgmara),
        ("GWG-pvalue", mgwg),
        ("Affinity\nPropagation", affinity_propagation),
        ("MeanShift", ms),
        ("Spectral\nClustering", spectral),
        ("Ward", ward),
        ("DBSCAN", dbscan),
        ("OPTICS", optics),
        ("BIRCH", birch),
        ("TMM-NEB", tmm_neb),
        ("GMM-NEB", gmm_neb),
    )
    return clustering_algorithms
