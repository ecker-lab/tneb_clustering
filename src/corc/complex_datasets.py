import numpy as np
from scipy.spatial import distance

# import densired
import pandas as pd
from sklearn.preprocessing import StandardScaler

from corc import generation


def make_gaussians(
    dim, std, n_samples, equal_sized_clusters=True, n_centers=6, random_state=42
):
    gen = generation.GenerationModel(
        center_structure="uniform",
        n_centers=n_centers,
        n_samples=n_samples,
        dim=dim,
        std=std,
        equal_sized_clusters=equal_sized_clusters,
        save_file=False,
        outdir=".",
        distance=None,
        random_state=random_state,
    )
    X = gen.sample_embedding()
    y = gen.labels
    X = StandardScaler().fit_transform(X)
    return X, y


def make_densired(dim, n_samples, std=1.0, random_state=42):
    skeleton = densired.datagen.densityDataGen(
        dim=dim,
        radius=5,
        clunum=6,
        core_num=200,
        min_dist=0.7,
        step_spread=0.3,
        # ratio_noise=0.1,
        ratio_con=0.01,
        dens_factors=True,  # list of factors, where factor = round((np.random.rand() * 1.5) + 0.5, 2)
        # dens_factors=False, # list of [1]s
        # dens_factors=[std]*6, # set std per cluster
        seed=random_state,
    )
    data = skeleton.generate_data(data_num=n_samples)
    X = data[:, :-1]
    y = data[:, -1]
    X = StandardScaler().fit_transform(X)
    return X, y


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
