import numpy as np
from scipy.spatial import distance

# import densired
import pandas as pd

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

    return gen.sample_embedding(), gen.labels


def make_2d_worms(max_clusters, noise=False, random_state=42):
    # Initialize parameters
    numsteps = 200
    labels = []
    i_clu = 1
    X = []
    c_trail = []
    num_noisep = 8
    nump = 2

    np.random.seed(random_state)

    for i_shape in range(max_clusters):
        steepness = 1 + np.random.rand() * 6
        numsteps = np.random.randint(100, 301)
        var_range = [10, 80]
        c = np.random.rand(2) * 2000 - 1000
        X, c_trail, labels, i_clu = _gen_one_worm(
            X,
            c,
            var_range,
            numsteps,
            steepness,
            nump,
            i_clu,
            num_noisep,
            c_trail,
            labels,
            noise=noise,
        )

    X = np.array(X)
    minX = X.min(axis=0)
    X = X - minX

    return X, np.array([i[0] for i in labels])


def _gen_one_worm(
    X,
    c,
    var_range,
    numsteps,
    steepness,
    nump,
    i_clu,
    num_noisep,
    c_trail,
    labels,
    noise=False,
):
    stepl = 5
    dims = 2
    new_cs = []
    num_rdirs = 3
    Xnew = []
    labels_new = []

    rdir = [np.random.rand(dims) - 0.5 for _ in range(num_rdirs)]
    rdir = [x / np.linalg.norm(x) for x in rdir]

    v = rdir[0]
    i = 0

    for i_step in range(1, numsteps + 1):
        p = i_step / numsteps
        v2 = np.array([-v[1], v[0]])  # Rotate 90 degrees
        v = v + v2 * (steepness / numsteps)
        v = v / np.linalg.norm(v)
        varr = var_range[0] * (1 - p) + var_range[1] * p
        c = c + v * stepl

        if len(c_trail) > 0:
            d = np.min(distance.cdist([c], c_trail))
            if d < 50:
                break

        new_cs.append(c)

        for _ in range(nump):
            b = np.random.normal(c, varr, dims)
            Xnew.append(b)
            labels_new.append([i_clu, 1])
            i += 1
        if noise:
            for _ in range(num_noisep):
                b = np.random.normal(c, varr * 8, dims)
                Xnew.append(b)
                labels_new.append([i_clu, 2])
                i += 1

    if i_step > numsteps / 4:
        X.extend(Xnew)
        c_trail.extend(new_cs)
        labels.extend(labels_new)
        i_clu += 1

    return X, c_trail, labels, i_clu


def make_Paul15(path="../paul15_dataset.pkl"):
    df = pd.read_pickle(path)
    data = np.array(df.iloc[:, :-2], dtype="float64")
    labels_num = np.array(df["paul15_clusters_num"])
    return data, labels_num


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
    return data[:, :-1], data[:, -1]


def load_densired(dim, path="../funky_shapes.npz"):
    with open(path, "rb") as f:
        data = np.load(f)
        # "files" within a npz-file cannot be named with numbers only, thus the f-string
        return data[f"d{dim}"][:, :-1], data[f"d{dim}"][:, -1]


def make_mnist_nd(dim, path="../mvae_mnist_nd_saved.pkl"):
    df = pd.read_pickle(path)
    return df["data"][dim], df["labels"][dim]
