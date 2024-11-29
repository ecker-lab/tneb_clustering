from __future__ import annotations

import numpy as np
import sklearn.datasets
import functools


def vec2d_from_angle(angle: float | np.ndarray) -> np.ndarray:
    """
    Convert an angle to a 2D vector
    """
    return np.stack([np.cos(angle), np.sin(angle)], -1)


def clusterlab_dataset1() -> tuple[np.ndarray, np.ndarray]:
    """
    A single cluster, example dataset 1 from the R package clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """
    # generate the normally distributed data
    X = np.random.normal(size=(100, 2))

    # the cluster label is always zero, because there is only one cluster
    y = np.zeros(100)
    return X, y


def clusterlab_dataset2() -> tuple[np.ndarray, np.ndarray]:
    """
    Four clusters with equal variances, example dataset 2 from the R package
    clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """
    # the vectors of centers just point into the four cardinal directions
    # create the angles of the four cardinal directions
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)

    # get the vectors from the angles
    centers = vec2d_from_angle(angles)

    # create the data points and the corresponding cluster labels
    Xs = []
    ys = []

    for i, c in enumerate(centers):
        # generate the data points
        X = np.random.normal(size=(50, 2))

        # do the scaling like the clusterlab package
        X = X * 2.5

        # shift the data points to the center (also scaled)
        X = X + c * 8
        Xs.append(X)

        # the cluster label is always the same for the same cluster
        y = np.zeros(50) + i
        ys.append(y)

    # combine the data points and the cluster labels into single arrays
    Xs = np.vstack(Xs)
    ys = np.concatenate(ys)
    return Xs, ys


def clusterlab_dataset3() -> tuple[np.ndarray, np.ndarray]:
    """
    Four clusters with different variances, example dataset 3 from the R package
    clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """
    # see the 'clusterlab_dataset2' for comments explaining the code
    cs = vec2d_from_angle(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    xs = []
    ys = []

    for i, (c, s) in enumerate(zip(cs, [1, 1, 2.5, 2.5])):
        xs.append(np.random.normal(size=(50, 2)) * s + c * 8)
        ys.append(np.zeros(50) + i)

    xs = np.vstack(xs)
    ys = np.concatenate(ys)
    return xs, ys


def clusterlab_dataset4() -> tuple[np.ndarray, np.ndarray]:
    """
    Simulating four clusters with one cluster pushed to the outside, example
    dataset 4 from the R package clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """
    # see the 'clusterlab_dataset2' for comments explaining the code
    cs = vec2d_from_angle(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    xs = []
    ys = []

    for i, (c, a) in enumerate(zip(cs, [1, 2, 1, 1])):
        xs.append(np.random.normal(size=(50, 2)) * 2.5 + c * 8 * a)
        ys.append(np.zeros(50) + i)

    xs = np.vstack(xs)
    ys = np.concatenate(ys)
    return xs, ys


def clusterlab_dataset5() -> tuple[np.ndarray, np.ndarray]:
    """
    Simulating four clusters with one smaller cluster, example dataset 5 from
    the R package clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """
    # see the 'clusterlab_dataset2' for comments explaining the code
    cs = vec2d_from_angle(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    xs = []
    ys = []

    for i, (c, n) in enumerate(zip(cs, [15, 50, 50, 50])):
        xs.append(np.random.normal(size=(n, 2)) * 2.5 + c * 8)
        ys.append(np.zeros(n) + i)

    xs = np.vstack(xs)
    ys = np.concatenate(ys)
    return xs, ys


def clusterlab_dataset6() -> tuple[np.ndarray, np.ndarray]:
    """
    Simulating five clusters with one central cluster, example dataset 6 from
    the R package clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """
    # see the 'clusterlab_dataset2' for comments explaining the code
    cs = vec2d_from_angle(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    cs = np.concatenate([cs, np.zeros((1, 2))])
    xs = []
    ys = []

    for i, c in enumerate(cs):
        xs.append(np.random.normal(size=(50, 2)) * 2.5 + c * 2 * 8)
        ys.append(np.zeros(50) + i)

    xs = np.vstack(xs)
    ys = np.concatenate(ys)
    return xs, ys


def clusterlab_dataset7() -> tuple[np.ndarray, np.ndarray]:
    """
    Simulating five clusters with ten outliers, example dataset 7 from the R
    package clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """
    # see the 'clusterlab_dataset2' for comments explaining the code
    cs = vec2d_from_angle(np.linspace(0, 2 * np.pi, 5, endpoint=False))
    xs = []
    ys = []

    for i, c in enumerate(cs):
        xs.append(np.zeros((50, 2)) + c * 7 * 2)
        ys.append(np.zeros(50) + i)

    xs = np.vstack(xs)
    ys = np.concatenate(ys)

    zs = np.random.normal(size=xs.shape)
    ms = np.sum(zs**2, axis=-1).argsort()[-10:]

    rs = np.random.normal(size=(10, 2))
    rs = rs / np.linalg.norm(rs, axis=-1, keepdims=True)

    xs += zs * 2
    xs[ms] += rs * 20
    return xs, ys


def clusterlab_dataset8() -> tuple[np.ndarray, np.ndarray]:
    """
    Simulating six clusters with different variances, example dataset 8 from
    the R package clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """
    # see the 'clusterlab_dataset2' for comments explaining the code
    cs = vec2d_from_angle(np.linspace(0, 2 * np.pi, 6, endpoint=False))
    xs = []
    ys = []

    for i, (c, s) in enumerate(zip(cs, [0.5, 1, 1.5, 1.75, 1.85, 1.95, 2.05])):
        xs.append(np.random.normal(size=(50, 2)) * s + c * 7 * 2)
        ys.append(np.zeros(50) + i)

    xs = np.vstack(xs)
    ys = np.concatenate(ys)
    return xs, ys


def clusterlab_dataset9() -> tuple[np.ndarray, np.ndarray]:
    """
    Simulating six clusters with different push apart degrees, example dataset
    9 from the R package clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """
    # see the 'clusterlab_dataset2' for comments explaining the code
    cs = vec2d_from_angle(np.linspace(0, 2 * np.pi, 6, endpoint=False))
    xs = []
    ys = []

    for i, (c, a) in enumerate(zip(cs, [0.5, 1, 1.5, 1.75, 1.85, 1.95, 2.05])):
        xs.append(np.random.normal(size=(50, 2)) + c * 9 * a)
        ys.append(np.zeros(50) + i)

    xs = np.vstack(xs)
    ys = np.concatenate(ys)
    return xs, ys


def clusterlab_dataset10() -> tuple[np.ndarray, np.ndarray]:
    """
    Simulating six clusters with different push apart degrees and variances,
    example dataset 10 from the R package clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """
    # see the 'clusterlab_dataset2' for comments explaining the code
    cs = vec2d_from_angle(np.linspace(0, 2 * np.pi, 6, endpoint=False))
    xs = []
    ys = []

    for i, (c, s, a) in enumerate(
        zip(
            cs,
            [0.5, 1.0, 1.5, 1.75, 2.0, 2.25, 2.25],
            [0.5, 1.0, 1.5, 1.75, 1.85, 1.95, 2.05],
        )
    ):
        xs.append(np.random.normal(size=(50, 2)) * s + c * 9 * a)
        ys.append(np.zeros(50) + i)

    xs = np.vstack(xs)
    ys = np.concatenate(ys)
    return xs, ys


def _random_projection(xs: np.ndarray, dim: int) -> np.ndarray:
    """
    Random projection of the data to a higher dimensional space, used in the
    clusterlab R package and in the paper "Sparse clusterability: testing for
    cluster structure in high dimensions" by Laborde et al. (2023).

    This is a simple way to simulate high-dimensional data. Although this does
    not add any "interesting new structure" to the data, because the dataset
    can be recovered by doing PCA. PCA would also show, that only the first two
    eigenvalues are non-zero.
    """
    assert xs.ndim == 2
    assert xs.shape[1] <= dim

    ev = np.random.normal(size=(dim, xs.shape[1])) * 0.1
    return xs @ ev.T


def bowtie_dataset(n=256):
    """
    A 2D dataset with two clusters in the shape of a bowtie, that thins out
    towards the ends.

    This is done by creating two 2D correlated abs normal distributions, that
    point in opposite directions. The data points are then rotated by 45
    degrees, to create the bowtie shape.
    """
    X = np.random.normal(size=(n, 2))

    # make correlated
    X = X @ np.array([[1, 0.3], [0.3, 1]], dtype="float32")

    # abs normal with offset
    X = np.abs(X) + 0.1

    # random {-1, 1} labels
    y = np.random.randint(0, 2, size=X.shape[0])

    r = y * 2 - 1
    X = X * r[:, None]

    # rotate 45 degrees
    s = np.sin(np.pi / 4)
    c = np.cos(np.pi / 4)
    R = np.array([[c, -s], [s, c]], dtype="float32")

    X = X @ R
    return X, y


def _correlated_2d_normal(n, rho):
    x = np.random.normal(size=(n, 2))
    x = x @ np.array([[1, rho], [rho, 1]], dtype="float32")
    return x


def zigzag_dataset(n=64, rho=0.8):
    """
    A 2D dataset with six clusters in the shape of a zigzag.

    This is done by creating six 2D correlated normal distributions, that point
    in the same direction. Every second cluster is mirrored along the x-axis
    to create the zigzag shape.
    """
    Xs = []
    ys = []

    for i in range(5):
        X = _correlated_2d_normal(n, rho)

        if i % 2 == 0:
            X[:, 0] = -X[:, 0]

        X[:, 0] += i * 4
        Xs.append(X)

        # all 'clusters' have the same label
        y = np.zeros(n)
        ys.append(y)

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    return X, y


def zigzig_dataset(n=64, rho=0.9):
    """
    A 2D dataset with seven correlated normal clusters right next to each other.
    """
    Xs = []
    ys = []

    for i in range(5):
        X = _correlated_2d_normal(n, rho)

        X[:, 0] += i * 1.5
        Xs.append(X)

        y = np.zeros(n) + i
        ys.append(y)

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    return X, y


def uniform_circle(n=256, r=1):
    """
    A 2D dataset with a uniform distribution on a circle.
    """
    t = np.random.uniform(0, 2 * np.pi, n)

    X = np.stack([r * np.cos(t), r * np.sin(t)], -1)

    # add noise
    z = np.random.normal(size=(n, 2)) * 0.1  # type: ignore
    X += z

    y = np.zeros(n)
    return X, y


def uniform_square(n=256, r=1):
    """
    A 2D dataset with a uniform distribution in a square.
    """
    X = np.random.uniform(-r, r, size=(n, 2))
    y = np.zeros(n)
    return X, y


def aniso_blobs(n=256):
    """
    A 2D dataset with a uniform distribution in a square.
    """
    X, y = sklearn.datasets.make_blobs(  # type: ignore
        n_samples=n,
        random_state=170,
    )

    X = X @ np.array([[0.6, -0.6], [-0.4, 0.8]], dtype="float32")
    return X, y


DATASETS = {
    # "Clusterlab1": clusterlab_dataset1,
    "Clusterlab2": clusterlab_dataset2,
    "Clusterlab3": clusterlab_dataset3,
    "Clusterlab4": clusterlab_dataset4,
    "Clusterlab5": clusterlab_dataset5,
    "Clusterlab6": clusterlab_dataset6,
    "Clusterlab7": clusterlab_dataset7,
    "Clusterlab8": clusterlab_dataset8,
    "Clusterlab9": clusterlab_dataset9,
    "Clusterlab10": clusterlab_dataset10,
    "Bowtie": bowtie_dataset,
    "ZigZag": zigzag_dataset,
    "ZigZig": zigzig_dataset,
    "Uniform Circle": uniform_circle,
    "Uniform Square": uniform_square,
    "Aniso Blobs": aniso_blobs,
    "Moons": functools.partial(
        sklearn.datasets.make_moons,
        n_samples=256,
        noise=0.05,
    ),
    "Circles": functools.partial(
        sklearn.datasets.make_circles,
        n_samples=256,
        factor=0.5,
        noise=0.05,
    ),
    "Blobs": functools.partial(
        sklearn.datasets.make_blobs,
        n_samples=256,
        cluster_std=[1.0, 2.5, 0.5],
        random_state=170,
    ),
}
