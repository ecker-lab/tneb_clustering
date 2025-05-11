from sklearn import cluster, mixture
import studenttmixture
from corc.graph_metrics import paga, gwgmara, neb, uniforce, smmp
from scipy.sparse import csr_matrix
import scanpy
import anndata
import numpy as np
from sklearn.neighbors import kneighbors_graph


ALGORITHM_SELECTOR = [
    "MiniBatch\nKMeans",
    "Agglomerative\nClustering",
    "HDBSCAN",
    "Gaussian\nMixture",
    "t-Student\nMixture",
    # "DBSCAN",
    # "BIRCH",
    # "OPTICS",
    "Spectral\nClustering",
    "Affinity\nPropagation",
    "MeanShift",
    "Leiden",
    "PAGA",
    "UniForCE",
    "SMMP",
    "GWG-dip",
    "GMM-NEB",
    "TMM-NEB",
]

CORE_SELECTOR = [
    "Agglomerative\nClustering",
    "HDBSCAN",
    "Gaussian\nMixture",
    # "t-Student\nMixture",
    "Leiden",
    "GWG-dip",
    "UniForCE",
    "SMMP",
    "GMM-NEB",
    "TMM-NEB",
]

ALG_DISPLAYNAMES = {
    "TMM-NEB": "t-NEB (ours)",
    "GMM-NEB": "g-NEB (ours)",
    "t-Student\nMixture": "Student-t\nMixture",
}

DETERMINISTIC_ALGORITHMS = [
    "Agglomerative\nClustering",
    "HDBSCAN",
    "Spectral\nClustering",
    "Leiden",
]


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
    params,  # default parameters are returned together with the datasets
    X,
    selector=None,
):
    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params["n_neighbors"], include_self=False
    )
    # make connectivity symmetric  (factor of 0.5 since we are averaging)
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    # default params can be found in our_datasets.py
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
    # birch = cluster.Birch(n_clusters=params["n_clusters"])
    gmm = mixture.GaussianMixture(
        n_components=params["n_clusters"],
        covariance_type="full",
        random_state=params["random_state"],
    )
    tmm = studenttmixture.EMStudentMixture(
        n_components=params["n_clusters"],
        n_init=10,
        fixed_df=False,  # True,
        # df=1.0,
        init_type="k++",
        random_state=params["random_state"],
        reg_covar=1e-4,
        tol=1e-3,
        max_iter=5000,
    )
    leiden = Leiden(resolution=params["resolution_leiden"], seed=params["random_state"])
    mgwgmara = gwgmara.GWGMara(
        latent_dim=params["dim"],
        n_components=params["gwg_n_components"],
        n_neighbors=params["gwg_n_neighbors"],
        covariance=params["gwg_covariance"],
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
        n_clusters=params["n_clusters"],
        seed=params["random_state"],
        mixture_model_type="tmm",
        n_init=20,
        optimization_iterations=200,
        tmm_regularization=params["tmm_regularization"],
    )
    gmm_neb = neb.NEB(
        latent_dim=params["dim"],
        n_components=params["n_components"],
        n_clusters=params["n_clusters"],
        seed=params["random_state"],
        mixture_model_type="gmm",
        n_init=20,
        optimization_iterations=200,
    )
    uniforce_algo = uniforce.Uniforce_Wrapper(
        alpha=0.0, num_clusters=params["n_clusters"]
    )
    smmp_algo = smmp.SMMP(
        n_clusters=params["n_clusters"],
    )

    clustering_algorithms = [
        ("MiniBatch\nKMeans", two_means),
        ("Agglomerative\nClustering", ward),
        # ("BIRCH", birch), # very old method that people don't really use
        ("HDBSCAN", hdbscan),
        # ("DBSCAN", dbscan), # HDBSCAN is always better than DBSCAN
        ("OPTICS", optics),
        ("Gaussian\nMixture", gmm),
        ("t-Student\nMixture", tmm),
        ("Affinity\nPropagation", affinity_propagation),
        ("Spectral\nClustering", spectral),
        ("MeanShift", ms),
        ("Leiden", leiden),
        ("PAGA", mpaga),
        ("GWG-dip", mgwgmara),
        ("UniForCE", uniforce_algo),
        ("SMMP", smmp_algo),
        ("GMM-NEB", gmm_neb),
        ("TMM-NEB", tmm_neb),
    ]

    if selector is None:
        selector = ALGORITHM_SELECTOR

    selected_algorithms = [
        (name, algo)  # return the full set of parameters
        for name, algo in clustering_algorithms  # global variable
        if name in selector
    ]

    return selected_algorithms
