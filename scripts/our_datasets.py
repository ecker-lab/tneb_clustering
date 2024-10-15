import numpy as np
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from corc import complex_datasets


# ============
# select which datasets to return
# ============
dataset_selector = [
    # "noisy_circles",
    # "noisy_moons",
    # "varied",
    # "aniso",
    # "blobs",
    # "worms",
    # "bowtie",
    # "zigzag",
    # "zigzig",
    # "uniform_circle",
    # "clusterlab10",
    ###########################
    ##### fig 2 datasets ######
    ###########################
    "blobs1_0",
    "blobs1_1",
    "blobs1_2",
    "blobs1_3",
    "blobs2_0",
    "blobs2_1",
    "blobs2_2",
    "blobs2_3",
    # "densired0",
    # "densired1",
    # "densired2",
    # "densired3",
    "mnist0",
    "mnist1",
    "mnist2",
    "mnist3",
    # "paul15",
]


# ============
# Generate 2D datasets.
# We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1000
seed = 30
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
# densired0 = complex_datasets.make_densired(dim=dims[0], n_samples=n_samples, std=std*np.sqrt(dims[0]))
# densired1 = complex_datasets.make_densired(dim=dims[1], n_samples=n_samples, std=std*np.sqrt(dims[1]))
# densired2 = complex_datasets.make_densired(dim=dims[2], n_samples=n_samples, std=std*np.sqrt(dims[2]))
# densired3 = complex_datasets.make_densired(dim=dims[3], n_samples=n_samples, std=std*np.sqrt(dims[3]))

# MNIST-Nd
mnist0 = complex_datasets.make_mnist_nd(dim=dims[0])
mnist1 = complex_datasets.make_mnist_nd(dim=dims[1])
mnist2 = complex_datasets.make_mnist_nd(dim=dims[2])
mnist3 = complex_datasets.make_mnist_nd(dim=dims[3])

# transcriptomic dataset (one mentioned in PAGA paper)
# paul15 = make_Paul15()


# ============
# Set up cluster parameters
# ============
default_base = {
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

dataset_store = [
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
            "name": "blobs1_0",
            "dim": dims[0],
            "n_clusters": 6,
            "resolution": 1.0,
            "resolution_leiden": 1.0,
        },
    ),
    (
        blobs1_1,
        {
            "name": "blobs1_1",
            "dim": dims[1],
            "n_clusters": 6,
            "resolution": 1.0,
            "resolution_leiden": 1.0,
        },
    ),
    (
        blobs1_2,
        {
            "name": "blobs1_2",
            "dim": dims[2],
            "n_clusters": 6,
            "resolution": 1.0,
            "resolution_leiden": 1.0,
        },
    ),
    (
        blobs1_3,
        {
            "name": "blobs1_3",
            "dim": dims[3],
            "n_clusters": 6,
            "resolution": 1.0,
            "resolution_leiden": 1.0,
        },
    ),
    (
        blobs2_0,
        {
            "name": "blobs2_0",
            "dim": dims[0],
            "n_clusters": 6,
            "resolution": 1.0,
            "resolution_leiden": 1.0,
        },
    ),
    (
        blobs2_1,
        {
            "name": "blobs2_1",
            "dim": dims[1],
            "n_clusters": 6,
            "resolution": 1.0,
            "resolution_leiden": 1.0,
        },
    ),
    (
        blobs2_2,
        {
            "name": "blobs2_2",
            "dim": dims[2],
            "n_clusters": 6,
            "resolution": 1.0,
            "resolution_leiden": 1.0,
        },
    ),
    (
        blobs2_3,
        {
            "name": "blobs2_3",
            "dim": dims[3],
            "n_clusters": 6,
            "resolution": 1.0,
            "resolution_leiden": 1.0,
        },
    ),
    # (densired0, {
    #     "dim":dims[0],
    #     "n_clusters": 6,
    #     "resolution":1.0,
    #     "resolution_leiden":1.0,
    # }),
    # (densired1, {
    #     "dim":dims[1],
    #     "n_clusters": 6,
    #     "resolution":1.0,
    #     "resolution_leiden":1.0,
    # }),
    # (densired2, {
    #     "dim":dims[2],
    #     "n_clusters": 6,
    #     "resolution":1.0,
    #     "resolution_leiden":1.0,
    # }),
    # (densired3, {
    #     "dim":dims[3],
    #     "n_clusters": 6,
    #     "resolution":1.0,
    #     "resolution_leiden":1.0,
    # }),
    (mnist0, {"name": "mnist0", "dim": dims[0], "n_clusters": 10, "n_components": 20}),
    (mnist1, {"name": "mnist1", "dim": dims[1], "n_clusters": 10, "n_components": 20}),
    (mnist2, {"name": "mnist2", "dim": dims[2], "n_clusters": 10, "n_components": 20}),
    (mnist3, {"name": "mnist3", "dim": dims[3], "n_clusters": 10, "n_components": 20}),
    # (paul15, {"dim": 1000, "n_clusters": 12, "n_components": 20}),
]

datasets = [
    (dataset, params)
    for dataset, params in dataset_store
    if params["name"] in dataset_selector
]
