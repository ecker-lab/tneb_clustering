import numpy as np
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from corc import complex_datasets

class our_datasets:

    def __init__(self, n_samples=1000, seed=30, ) -> None:
          # ============
        # select which datasets to return
        # ============
        self.dataset_selector = [
            "noisy_circles",
            "noisy_moons",
            "varied",
            "aniso",
            "blobs",
            "worms",
            "bowtie",
            "zigzag",
            "zigzig",
            "uniform_circle",
            "clusterlab10",
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
            "mnist8",
            "mnist16",
            "mnist32",
            "mnist64",
            # "paul15",
        ]

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
            "n_components": 15,
            "min_samples": 7,
            "xi": 0.05,
            "min_cluster_size": 0.1,
            "allow_single_cluster": True,
            "hdbscan_min_cluster_size": 15,
            "hdbscan_min_samples": 3,
            "resolution": 1.0,
            "resolution_leiden": 1.0,
            "random_state": 42,
        }

        # ============
        # Generate 2D datasets.
        # We choose the size big enough to see the scalability
        # of the algorithms, but not too big to avoid too long running times
        # ============
        noisy_circles = datasets.make_circles(
            n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
        )
        noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
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

        # worms
        # worms = complex_datasets.make_2d_worms(6)

        # bowtie = datasets2d.bowtie_dataset(n=n_samples)
        # zigzag = datasets2d.zigzag_dataset(n=n_samples)
        # zigzig = datasets2d.zigzig_dataset(n=n_samples)
        # uniform_circle = datasets2d.uniform_circle(n=n_samples)

        # clusterlab10 = datasets2d.clusterlab_dataset10()


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

        # Worms with 6 clusters in 8D, 16D, 32D, 64D

        # funky shapes with 6 clusters in 8D, 16D, 32D, 64D
        funky_shapes_path = "funky_shapes.npz"
        densired0 = complex_datasets.load_densired(dim=dims[0], path=funky_shapes_path)
        densired1 = complex_datasets.load_densired(dim=dims[1], path=funky_shapes_path)
        densired2 = complex_datasets.load_densired(dim=dims[2], path=funky_shapes_path)
        densired3 = complex_datasets.load_densired(dim=dims[3], path=funky_shapes_path)
        # densired0 = complex_datasets.make_densired(dim=dims[0], n_samples=n_samples, std=std*np.sqrt(dims[0]))
        # densired1 = complex_datasets.make_densired(dim=dims[1], n_samples=n_samples, std=std*np.sqrt(dims[1]))
        # densired2 = complex_datasets.make_densired(dim=dims[2], n_samples=n_samples, std=std*np.sqrt(dims[2]))
        # densired3 = complex_datasets.make_densired(dim=dims[3], n_samples=n_samples, std=std*np.sqrt(dims[3]))

        # MNIST-Nd
        mnist_path = "mvae_mnist_nd_saved.pkl"
        mnist0 = complex_datasets.make_mnist_nd(dim=dims[0], path=mnist_path)
        mnist1 = complex_datasets.make_mnist_nd(dim=dims[1], path=mnist_path)
        mnist2 = complex_datasets.make_mnist_nd(dim=dims[2], path=mnist_path)
        mnist3 = complex_datasets.make_mnist_nd(dim=dims[3], path=mnist_path)

        # transcriptomic dataset (one mentioned in PAGA paper)
        # paul15 = make_Paul15()

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
                    "gwg_covariance": "diag"
                    
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
                    "gwg_covariance": "diag"
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
                    "gwg_covariance": "full"
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
                    "gwg_covariance": "full"
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
                    "gwg_covariance": "spherical"
                },
            ),
            (
                no_structure,
                {
                    "name": "no_structure",
                    "resolution": 0.1,
                    "resolution_leiden": 0.01,
                },
            ),
            # (
            #     worms,
            #     {
            #         "resolution": 0.01,
            #         "resolution_leiden": 0.01,
            #     },
            # ),
            # (
            #     bowtie,
            #     {
            #         "resolution":0.1,
            #         "resolution_leiden":0.1,
            #     }
            # ),
            # (
            #     zigzag,
            #     {
            #         "resolution":0.01,
            #         "resolution_leiden":0.01,
            #     }
            # ),
            # (
            #     zigzig,
            #     {
            #         "resolution":0.01,
            #         "resolution_leiden":0.01,
            #     }
            # ),
            # (
            #     uniform_circle,
            #     {
            #         "resolution":0.1,
            #         "resolution_leiden":0.1,
            #     }
            # ),
            # (
            #     clusterlab10,
            #     {
            #         "resolution":1.0,
            #         "resolution_leiden":0.01,
            #     }
            # ),
            (
                blobs1_0,
                {
                    "name": "blobs1_8",
                    "dim": dims[0],
                    "n_clusters": 6,
                    "resolution": 1.0,
                    "resolution_leiden": 1.0,
                    "gwg_n_components": 25,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "diag"
                },
            ),
            (
                blobs1_1,
                {
                    "name": "blobs1_16",
                    "dim": dims[1],
                    "n_clusters": 6,
                    "resolution": 1.0,
                    "resolution_leiden": 1.0,
                    "gwg_n_components": 10,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "spherical"
                },
            ),
            (
                blobs1_2,
                {
                    "name": "blobs1_32",
                    "dim": dims[2],
                    "n_clusters": 6,
                    "resolution": 1.0,
                    "resolution_leiden": 1.0,
                    "gwg_n_components": 25,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "diag"
                },
            ),
            (
                blobs1_3,
                {
                    "name": "blobs1_64",
                    "dim": dims[3],
                    "n_clusters": 6,
                    "resolution": 1.0,
                    "resolution_leiden": 1.0,
                    "gwg_n_components": 10,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "diag"
                },
            ),
            (
                blobs2_0,
                {
                    "name": "blobs2_8",
                    "dim": dims[0],
                    "n_clusters": 6,
                    "resolution": 1.0,
                    "resolution_leiden": 1.0,
                    "gwg_n_components": 25,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "diag"
                },
            ),
            (
                blobs2_1,
                {
                    "name": "blobs2_16",
                    "dim": dims[1],
                    "n_clusters": 6,
                    "resolution": 1.0,
                    "resolution_leiden": 1.0,
                    "gwg_n_components": 10,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "spherical"
                },
            ),
            (
                blobs2_2,
                {
                    "name": "blobs2_32",
                    "dim": dims[2],
                    "n_clusters": 6,
                    "resolution": 1.0,
                    "resolution_leiden": 1.0,
                    "gwg_n_components": 50,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "diag"
                },
            ),
            (
                blobs2_3,
                {
                    "name": "blobs2_64",
                    "dim": dims[3],
                    "n_clusters": 6,
                    "resolution": 1.0,
                    "resolution_leiden": 1.0,
                    "gwg_n_components": 50,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "spherical"
                },
            ),
            (
                densired0,
                {
                    "name": "densired8",
                    "dim": dims[0],
                    "n_clusters": 6,
                    "resolution": 1.0,
                    "resolution_leiden": 1.0,
                    "gwg_n_components": 50,
                    "gwg_n_neighbors": 5,
                    "gwg_covariance": "spherical"
                },
            ),
            (
                densired1,
                {
                    "name": "densired16",
                    "dim": dims[1],
                    "n_clusters": 6,
                    "resolution": 1.0,
                    "resolution_leiden": 1.0,
                    "gwg_n_components": 50,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "spherical"
                },
            ),
            (
                densired2,
                {
                    "name": "densired32",
                    "dim": dims[2],
                    "n_clusters": 6,
                    "resolution": 1.0,
                    "resolution_leiden": 1.0,
                    "gwg_n_components": 50,
                    "gwg_n_neighbors": 3,
                    "gwg_covariance": "spherical"
                },
            ),
            (
                densired3,
                {
                    "name": "densired64",
                    "dim": dims[3],
                    "n_clusters": 6,
                    "resolution": 1.0,
                    "resolution_leiden": 1.0,
                    "gwg_n_components": 25,
                    "gwg_n_neighbors": 5,
                    "gwg_covariance": "full"
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
                    "gwg_covariance": "full"
                }
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
                    "gwg_covariance": "full"
                }
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
                    "gwg_covariance": "full"
                }),
            (
                mnist3, 
                {
                    "name": "mnist64",
                    "dim": dims[3],
                    "n_clusters": 10,
                    "n_components": 20,
                    "gwg_n_components": 15,
                    "gwg_n_neighbors": 2,
                    "gwg_covariance": "full"
                }
            ),
            # (paul15, {"dim": 1000, "n_clusters": 12, "n_components": 20}),
        ]




    def get_datasets(self):
        return self.select_datasets(self.dataset_selector)

    def select_datasets(self, dataset_selector):
        datasets = [
            (dataset, {**self.default_base, **params}) # return the full set of parameters
            for dataset, params in self.dataset_store  # global variable
            if params["name"] in dataset_selector
        ]
        return datasets