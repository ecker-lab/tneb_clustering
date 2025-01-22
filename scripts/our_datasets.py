import numpy as np
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from corc import complex_datasets, datasets2d


DENSIRED_PATH = "datasets/densired.npz"
DENSIRED_SOFT_PATH = "datasets/densired_soft.npz"
MNIST_PATH = "datasets/mvae_mnist_nd_saved.pkl"

# ============
# select which datasets to return
# ============
DATASET_SELECTOR = [
    "noisy_circles",
    "noisy_moons",
    "varied",
    "aniso",
    "blobs",
    # "worms", # missing completely
    # "bowtie", # missing gwg-parameters
    # "zigzag", # missing gwg-parameters
    # "zigzig", # missing gwg-parameters
    # "uniform_circle", # missing gwg-parameters
    "clusterlab10",  # missing gwg-parameters
    ###########################
    ##### fig 2 datasets ######
    ###########################
    "blobs1_8",
    "blobs1_16",
    "blobs1_32",
    "blobs1_64",
    "blobs2_8",
    "blobs2_16",
    "blobs2_32",
    "blobs2_64",
    "densired8",
    "densired16",
    "densired32",
    "densired64",
    "densired_soft_8",
    "densired_soft_16",
    "densired_soft_32",
    "densired_soft_64",
    "mnist8",
    "mnist16",
    "mnist32",
    "mnist64",
]

DATASETS2D = [
    # "clusterlab1",
    # "clusterlab2",
    # "clusterlab3",
    # "clusterlab4",
    # "clusterlab5",
    # "clusterlab6",
    # "clusterlab7",
    # "clusterlab8",
    # "clusterlab9",
    "noisy_circles",
    "noisy_moons",
    "blobs",
    "varied",
    "aniso",
    "clusterlab10",
]

COMPLEX_DATASETS = [
    "blobs1_8",
    "blobs1_16",
    "blobs1_32",
    "blobs1_64",
    "blobs2_8",
    "blobs2_16",
    "blobs2_32",
    "blobs2_64",
    "densired8",
    "densired16",
    "densired32",
    "densired64",
    "densired_soft_8",
    "densired_soft_16",
    "densired_soft_32",
    "densired_soft_64",
    "mnist8",
    "mnist16",
    "mnist32",
    "mnist64",
]

CORE_HD_DATASETS = [
    "densired8",
    "densired16",
    "densired32",
    "densired64",
    "densired_soft_8",
    "densired_soft_16",
    "densired_soft_32",
    "densired_soft_64",
    "mnist8",
    "mnist16",
    "mnist32",
    "mnist64",
]


class our_datasets:

    def __init__(
        self,
        n_samples=1000,
        seed=30,
    ) -> None:
        # ============
        # Set up cluster parameters
        # ============
        self.default_base = {
            "name": "unknown",
            "dim": 2,
            "quantile": 0.3,
            "eps": 0.3,
            "damping": 0.9,
            "preference": -200,
            "n_neighbors": 3,
            "n_clusters": 3,
            "n_components": 25,
            "min_samples": 7,
            "xi": 0.05,
            "min_cluster_size": 0.1,
            "allow_single_cluster": True,
            "hdbscan_min_cluster_size": 15,
            "hdbscan_min_samples": 3,
            "resolution": 1.0,
            "resolution_leiden": 1.0,
            "random_state": 42,
            "use_rep": "X",
            "gwg_n_components": 15,
            "gwg_n_neighbors": 3,
            "gwg_covariance": "diag",
        }

        # ============
        # Generate 2D datasets.
        # We choose the size big enough to see the scalability
        # of the algorithms, but not too big to avoid too long running times
        # ============
        noisy_circles = datasets.make_circles(
            n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
        )
        noisy_moons = datasets.make_moons(
            n_samples=n_samples, noise=0.05, random_state=seed
        )
        blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
        rng = np.random.RandomState(seed)
        no_structure = rng.rand(n_samples, 2), None

        # Anisotropicly distributed data
        random_state = 170
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        aniso = (X_aniso, y)

        # blobs with varied variances
        varied = datasets.make_blobs(
            n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
        )

        bowtie = datasets2d.bowtie_dataset(n=n_samples)
        zigzag = datasets2d.zigzag_dataset(n=n_samples)
        zigzig = datasets2d.zigzig_dataset(n=n_samples)
        uniform_circle = datasets2d.uniform_circle(n=n_samples)

        clusterlab10 = datasets2d.clusterlab_dataset10()

        # ============
        # Generate high-dimensional datasets.
        # We choose the size big enough to see the scalability
        # of the algorithms, but not too big to avoid too long running times
        # ============
        dims = [8, 16, 32, 64]
        std = 0.075
        # Gaussians with 6 clusters in 8D, 16D, 32D, 64D
        blobs1_0 = complex_datasets.make_gaussians(
            dim=dims[0], std=std * np.sqrt(dims[0]), n_samples=n_samples
        )
        blobs1_1 = complex_datasets.make_gaussians(
            dim=dims[1], std=std * np.sqrt(dims[1]), n_samples=n_samples
        )
        blobs1_2 = complex_datasets.make_gaussians(
            dim=dims[2], std=std * np.sqrt(dims[2]), n_samples=n_samples
        )
        blobs1_3 = complex_datasets.make_gaussians(
            dim=dims[3], std=std * np.sqrt(dims[3]), n_samples=n_samples
        )

        # Gaussians with 6 clusters with varying frequency in 8D, 16D, 32D, 64D
        blobs2_0 = complex_datasets.make_gaussians(
            dim=dims[0],
            std=std * np.sqrt(dims[0]),
            n_samples=n_samples,
            equal_sized_clusters=False,
        )
        blobs2_1 = complex_datasets.make_gaussians(
            dim=dims[1],
            std=std * np.sqrt(dims[1]),
            n_samples=n_samples,
            equal_sized_clusters=False,
        )
        blobs2_2 = complex_datasets.make_gaussians(
            dim=dims[2],
            std=std * np.sqrt(dims[2]),
            n_samples=n_samples,
            equal_sized_clusters=False,
        )
        blobs2_3 = complex_datasets.make_gaussians(
            dim=dims[3],
            std=std * np.sqrt(dims[3]),
            n_samples=n_samples,
            equal_sized_clusters=False,
        )

        # funky shapes with 6 clusters in 8D, 16D, 32D, 64D
        densired0 = complex_datasets.load_densired(dim=dims[0], path=DENSIRED_PATH)
        densired1 = complex_datasets.load_densired(dim=dims[1], path=DENSIRED_PATH)
        densired2 = complex_datasets.load_densired(dim=dims[2], path=DENSIRED_PATH)
        densired3 = complex_datasets.load_densired(dim=dims[3], path=DENSIRED_PATH)

        densired_soft_0 = complex_datasets.load_densired(
            dim=dims[0], path=DENSIRED_SOFT_PATH
        )
        densired_soft_1 = complex_datasets.load_densired(
            dim=dims[1], path=DENSIRED_SOFT_PATH
        )
        densired_soft_2 = complex_datasets.load_densired(
            dim=dims[2], path=DENSIRED_SOFT_PATH
        )
        densired_soft_3 = complex_datasets.load_densired(
            dim=dims[3], path=DENSIRED_SOFT_PATH
        )

        # MNIST-Nd
        mnist0 = complex_datasets.make_mnist_nd(dim=dims[0], path=MNIST_PATH)
        mnist1 = complex_datasets.make_mnist_nd(dim=dims[1], path=MNIST_PATH)
        mnist2 = complex_datasets.make_mnist_nd(dim=dims[2], path=MNIST_PATH)
        mnist3 = complex_datasets.make_mnist_nd(dim=dims[3], path=MNIST_PATH)

        ############
        # store dataset together with default parameters
        ############
        self.dataset_store = [
            (
                noisy_circles,
                {
                    "name": "noisy_circles",
                    "damping": 0.77,
                    "preference": -240,
                    "quantile": 0.2,
                    "n_clusters": 2,
                    "min_samples": 7,
                    "xi": 0.08,
                    "resolution": 1.0,
                    "resolution_leiden": 0.01,
                    "gwg_n_components": 15,
                    "gwg_n_neighbors": 3,
                    "gwg_covariance": "diag",
                    "n_components": 15,
                },
            ),
            (
                noisy_moons,
                {
                    "name": "noisy_moons",
                    "damping": 0.75,
                    "preference": -220,
                    "n_clusters": 2,
                    "min_samples": 7,
                    "xi": 0.1,
                    "resolution": 0.5,
                    "resolution_leiden": 0.01,
                    "gwg_n_components": 10,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "diag",
                    "n_components": 15,
                },
            ),
            (
                varied,
                {
                    "name": "varied",
                    "eps": 0.18,
                    "n_neighbors": 2,
                    "min_samples": 7,
                    "xi": 0.01,
                    "min_cluster_size": 0.2,
                    "resolution": 0.5,
                    "resolution_leiden": 0.1,
                    "gwg_n_components": 25,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "full",
                    "n_components": 15,
                },
            ),
            (
                aniso,
                {
                    "name": "aniso",
                    "eps": 0.15,
                    "n_neighbors": 2,
                    "min_samples": 7,
                    "xi": 0.1,
                    "min_cluster_size": 0.2,
                    "resolution": 0.1,
                    "resolution_leiden": 0.1,
                    "gwg_n_components": 5,
                    "gwg_n_neighbors": 5,
                    "gwg_covariance": "full",
                    "n_components": 15,
                },
            ),
            (
                blobs,
                {
                    "name": "blobs",
                    "min_samples": 7,
                    "xi": 0.1,
                    "min_cluster_size": 0.2,
                    "resolution": 0.1,
                    "resolution_leiden": 0.1,
                    "gwg_n_components": 5,
                    "gwg_n_neighbors": 3,
                    "gwg_covariance": "spherical",
                    "n_components": 15,
                },
            ),
            (
                no_structure,
                {
                    "name": "no_structure",
                    "n_clusters": 1,
                    "resolution": 0.1,
                    "resolution_leiden": 0.01,
                    "gwg_n_components": 5,
                    "gwg_n_neighbors": 3,
                    "gwg_covariance": "spherical",
                },
            ),
            # (
            #     worms,
            #     {
            #         "n_clusters": 4,
            #         "resolution": 0.01,
            #         "resolution_leiden": 0.01,
            #     },
            # ),
            (
                bowtie,
                {
                    "name": "bowtie",
                    "n_clusters": 2,
                    "resolution": 0.1,
                    "resolution_leiden": 0.1,
                },
            ),
            (
                zigzag,
                {
                    "name": "zigzag",
                    "resolution": 0.01,
                    "resolution_leiden": 0.01,
                },
            ),
            (
                zigzig,
                {
                    "name": "zigzig",
                    "n_clusters": 5,
                    "resolution": 0.01,
                    "resolution_leiden": 0.01,
                },
            ),
            (
                uniform_circle,
                {
                    "name": "uniform_circle",
                    "n_clusters": 1,
                    "resolution": 0.1,
                    "resolution_leiden": 0.1,
                },
            ),
            (
                clusterlab10,
                {
                    "name": "clusterlab10",
                    "n_clusters": 6,
                    "resolution": 1.0,
                    "resolution_leiden": 0.01,
                    "n_components": 15,
                },
            ),
            (
                blobs1_0,
                {
                    "name": "blobs1_8",
                    "dim": dims[0],
                    "n_clusters": 6,
                    "gwg_n_components": 25,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "diag",
                    "resolution": 1.0,
                    "resolution_leiden": 1.0,
                    "quantile": 0.04,
                    "preference": -200,
                },
            ),
            (
                blobs1_1,
                {
                    "name": "blobs1_16",
                    "dim": dims[1],
                    "n_clusters": 6,
                    "gwg_n_components": 10,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "spherical",
                    "resolution": 1.0,
                    "resolution_leiden": 1.0,
                    "quantile": 0.045,
                    "preference": -500,
                },
            ),
            (
                blobs1_2,
                {
                    "name": "blobs1_32",
                    "dim": dims[2],
                    "n_clusters": 6,
                    "gwg_n_components": 25,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "diag",
                    "resolution": 1.0,
                    "resolution_leiden": 1.0,
                    "quantile": 0.033,
                    "preference": -700,
                },
            ),
            (
                blobs1_3,
                {
                    "name": "blobs1_64",
                    "dim": dims[3],
                    "n_clusters": 6,
                    "gwg_n_components": 10,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "diag",
                    "resolution": 1.0,
                    "resolution_leiden": 1.0,
                    "quantile": 0.031,
                    "preference": -1300,
                },
            ),
            (
                blobs2_0,
                {
                    "name": "blobs2_8",
                    "dim": dims[0],
                    "n_clusters": 6,
                    "gwg_n_components": 25,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "diag",
                    "resolution": 0.6,
                    "resolution_leiden": 0.6,
                    "quantile": 0.07,
                    "preference": -200,
                },
            ),
            (
                blobs2_1,
                {
                    "name": "blobs2_16",
                    "dim": dims[1],
                    "n_clusters": 6,
                    "gwg_n_components": 10,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "spherical",
                    "resolution": 0.7,
                    "resolution_leiden": 0.7,
                    "quantile": 0.052,  # I did not managed to get the exact same number of clusters, its the closes one above
                    "preference": -500,
                },
            ),
            (
                blobs2_2,
                {
                    "name": "blobs2_32",
                    "dim": dims[2],
                    "n_clusters": 6,
                    "gwg_n_components": 50,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "diag",
                    "resolution": 0.85,
                    "resolution_leiden": 0.85,
                    "quantile": 0.06,
                    "preference": -900,
                },
            ),
            (
                blobs2_3,
                {
                    "name": "blobs2_64",
                    "dim": dims[3],
                    "n_clusters": 6,
                    "gwg_n_components": 50,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "spherical",
                    "resolution": 0.93,
                    "resolution_leiden": 0.93,
                    "quantile": 0.045,
                    "preference": -900,
                },
            ),
            (
                densired0,
                {
                    "name": "densired8",
                    "dim": dims[0],
                    "n_clusters": 6,
                    "gwg_n_components": 50,
                    "gwg_n_neighbors": 5,
                    "gwg_covariance": "spherical",
                    "resolution": 0.2,
                    "resolution_leiden": 0.2,
                    "quantile": 0.03,
                    "preference": -2000,
                },
            ),
            (
                densired1,
                {
                    "name": "densired16",
                    "dim": dims[1],
                    "n_clusters": 6,
                    "gwg_n_components": 50,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "spherical",
                    "resolution": 0.2,
                    "resolution_leiden": 0.2,
                    "quantile": 0.04,
                    "preference": -7000,
                },
            ),
            (
                densired2,
                {
                    "name": "densired32",
                    "dim": dims[2],
                    "n_clusters": 6,
                    "gwg_n_components": 50,
                    "gwg_n_neighbors": 3,
                    "gwg_covariance": "spherical",
                    "resolution": 0.1,
                    "resolution_leiden": 0.1,
                    "quantile": 0.035,
                    "preference": -7000,
                },
            ),
            (
                densired3,
                {
                    "name": "densired64",
                    "dim": dims[3],
                    "n_clusters": 6,
                    "gwg_n_components": 25,
                    "gwg_n_neighbors": 5,
                    "gwg_covariance": "full",
                    "resolution": 0.1,
                    "resolution_leiden": 0.1,
                    "quantile": 0.052,
                    "preference": -14_000,
                },
            ),
            (
                densired_soft_0,
                {
                    "name": "densired_soft_8",
                    "dim": dims[0],
                    "n_clusters": 6,
                    "gwg_n_components": 50,
                    "gwg_n_neighbors": 5,
                    "gwg_covariance": "spherical",
                    "resolution": 0.2,
                    "resolution_leiden": 0.2,
                    "quantile": 0.03,
                    "preference": -2000,
                },
            ),
            (
                densired_soft_1,
                {
                    "name": "densired_soft_16",
                    "dim": dims[1],
                    "n_clusters": 6,
                    "gwg_n_components": 50,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "spherical",
                    "resolution": 0.2,
                    "resolution_leiden": 0.2,
                    "quantile": 0.04,
                    "preference": -7000,
                },
            ),
            (
                densired_soft_2,
                {
                    "name": "densired_soft_32",
                    "dim": dims[2],
                    "n_clusters": 6,
                    "gwg_n_components": 50,
                    "gwg_n_neighbors": 3,
                    "gwg_covariance": "spherical",
                    "resolution": 0.1,
                    "resolution_leiden": 0.1,
                    "quantile": 0.035,
                    "preference": -7000,
                },
            ),
            (
                densired_soft_3,
                {
                    "name": "densired_soft_64",
                    "dim": dims[3],
                    "n_clusters": 6,
                    "gwg_n_components": 25,
                    "gwg_n_neighbors": 5,
                    "gwg_covariance": "full",
                    "resolution": 0.1,
                    "resolution_leiden": 0.1,
                    "quantile": 0.052,
                    "preference": -14_000,
                },
            ),
            (
                mnist0,
                {
                    "name": "mnist8",
                    "dim": dims[0],
                    "n_clusters": 10,
                    "n_components": 20,
                    "gwg_n_components": 10,
                    "gwg_n_neighbors": 5,
                    "gwg_covariance": "full",
                    "resolution": 0.6,
                    "resolution_leiden": 0.6,
                    "quantile": 0.023,
                    "preference": -1500,  # -2000 would work as well,
                },
            ),
            (
                mnist1,
                {
                    "name": "mnist16",
                    "dim": dims[1],
                    "n_clusters": 10,
                    "n_components": 20,
                    "gwg_n_components": 10,
                    "gwg_n_neighbors": 3,
                    "gwg_covariance": "full",
                    "resolution": 0.7,
                    "resolution_leiden": 0.7,
                    "quantile": 0.02,
                    "preference": -3500,
                },
            ),
            (
                mnist2,
                {
                    "name": "mnist32",
                    "dim": dims[2],
                    "n_clusters": 10,
                    "n_components": 20,
                    "gwg_n_components": 20,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "full",
                    "resolution": 0.4,
                    "resolution_leiden": 0.4,
                    "quantile": 0.041,
                    "preference": -5300,
                },
            ),
            (
                mnist3,
                {
                    "name": "mnist64",
                    "dim": dims[3],
                    "n_clusters": 10,
                    "n_components": 20,
                    "gwg_n_components": 15,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "full",
                    "resolution": 0.49,
                    "resolution_leiden": 0.49,
                    "quantile": 0.13,
                    "preference": -6_000,
                },
            ),
        ]

    def get_datasets(self):
        return self.select_datasets(DATASET_SELECTOR)

    def select_datasets(self, dataset_selector):
        datasets = [
            (
                dataset,
                {**self.default_base, **params},
            )  # return the full set of parameters
            for dataset, params in self.dataset_store  # global variable
            if params["name"] in dataset_selector
        ]
        return datasets
