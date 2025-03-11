import numpy as np
from scipy.spatial import distance

# import densired
import pandas as pd
from sklearn.preprocessing import StandardScaler
from corc.utils import set_seed



def make_gaussians(
    dim, std, n_samples, equal_sized_clusters=True, n_centers=6, random_state=42
):
    """
    dim : int
        dimensionality of latent embedding
    std : float/list
        standard deviations of clusters. If float, this will be converted to a list of same std for each cluster.
        If list, individual std for each cluster, should have same length as n_centers.
    n_samples : int
        number of data points
    equal_sized_clusters : bool
        flag if clusters should contain same amount of samples or different
    n_centers : int
        number of cluster centers
    """
    set_seed(random_state)

    # define number of samples per cluster
    if equal_sized_clusters:
        n_samples_around_c = np.array([n_samples // n_centers] * n_centers)
    else:
        n_samples_around_c = np.random.randint(1,n_samples//n_centers, size=n_centers-1)
        n_samples_around_c = np.concatenate((n_samples_around_c, [n_samples-n_samples_around_c.sum()]))

    # save cluster label for every point
    labels = []
    for ci, samples in enumerate(n_samples_around_c):
        labels = np.concatenate((labels, np.ones(samples)*ci), axis=None)

    # define std per cluster
    assert std is not None, "Std should not be None"
    if not isinstance(std, np.ndarray):
        assert isinstance(std, list) or isinstance(std, float), f'Std needs to be float or list or np.ndarray'
        std = np.array(std) if isinstance(std, list) else np.array([std] * n_centers)
    assert len(std) == n_centers, f'Number of stds needs to be same as number of cluster centers.'

    # define cluster centers
    cluster_centers = np.random.uniform(0,1,size=(n_centers, dim))

    # sample data points around cluster centers
    latent_emb = []
    for (cluster_center, samples, std_) in zip(cluster_centers, n_samples_around_c, std):
        samples = np.random.normal(cluster_center, std_, size=(samples, dim))
        latent_emb = np.concatenate((latent_emb, samples), axis=None)
    latent_emb = np.array(latent_emb).reshape(n_samples_around_c.sum(), -1)

    # normalize data
    X = StandardScaler().fit_transform(latent_emb)
    return X, labels


# def make_densired(dim, n_samples, std=1.0, random_state=42):
#     skeleton = densired.datagen.densityDataGen(
#         dim=dim,
#         radius=5,
#         clunum=6,
#         core_num=200,
#         min_dist=0.7,
#         step_spread=0.3,
#         # ratio_noise=0.1,
#         ratio_con=0.01,
#         dens_factors=True,  # list of factors, where factor = round((np.random.rand() * 1.5) + 0.5, 2)
#         # dens_factors=False, # list of [1]s
#         # dens_factors=[std]*6, # set std per cluster
#         seed=random_state,
#     )
#     data = skeleton.generate_data(data_num=n_samples)
#     X = data[:, :-1]
#     y = data[:, -1]
#     X = StandardScaler().fit_transform(X)
#     return X, y


def load_densired(dim, path="../datasets/densired.npz"):
    with open(path, "rb") as f:
        data = np.load(f)
        # "files" within a npz-file cannot be named with numbers only, thus the f-string
        X = data[f"d{dim}"][:, :-1]
        y = data[f"d{dim}"][:, -1]
        X = StandardScaler().fit_transform(X)
        return X, y


def make_mnist_nd(dim, path="../datasets/mvae_mnist_nd_saved.pkl"):
    df = pd.read_pickle(path)
    X = df["data"][dim]
    y = df["labels"][dim]
    X = StandardScaler().fit_transform(X)
    return X, y
