import abc

import flarejax as fj
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, Int, PRNGKeyArray


@staticmethod
def _vec2d_from_angle(angle: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Convert an angle to a 2D vector
    """
    return jnp.stack([jnp.cos(angle), jnp.sin(angle)], -1)


class Dataset(abc.ABC):
    @abc.abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        n: int,
    ) -> tuple[Float[Array, "{n} ..."], Int[Array, "{n}"]]: ...

    @property
    def dim(self) -> int:
        return jax.eval_shape(self.sample, jrandom.PRNGKey(0), 1)[1]


class GMM(fj.Module, Dataset, register=False):
    """
    Gaussian Mixture Model to sample from.

    Parameters
    ---
    means: Float[Array, "components dim"]
        The means of the components

    covariances: Float[Array, "components dim dim"]
        The covariances of the components

    frequencies: Float[Array, "components"]
        The relative frequencies of the components
    """

    means: Float[Array, "components dim"]
    covariances: Float[Array, "components dim dim"]
    frequencies: Float[Array, "components"]

    def __post_init__(self) -> None:
        assert abs(sum(self.frequencies) - 1) < 1e-6

    def components(self) -> int:
        return self.means.shape[0]

    def dim(self) -> int:
        return self.means.shape[1]

    def sample(
        self,
        key: PRNGKeyArray,
        n: int,
    ) -> tuple[Float[Array, "{n} ..."], Int[Array, "{n}"]]:
        keys = jrandom.split(key, n)
        keys = jnp.stack(keys, axis=0)

        return jax.vmap(self._sample)(keys)

    def _sample(
        self,
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "dim"], Int[Array, ""]]:
        key_index, key_sample = jrandom.split(key)
        index = jrandom.choice(
            key_index,
            a=jnp.arange(self.components),
            p=self.frequencies,
        )

        mean = self.means[index]
        covariance = self.covariances[index]

        return jrandom.multivariate_normal(key_sample, mean, covariance), index

    @staticmethod
    def _isotropic_covariances(
        components: int,
        dim: int,
    ) -> Float[Array, "{components} {dim} {dim}"]:
        covariances = jnp.eye(dim)
        covariances = jnp.stack([covariances] * components, axis=0)
        return covariances

    @classmethod
    def from_uniform_centers(
        cls,
        key: PRNGKeyArray,
        components: int,
        dim: int,
        stddev: float,
    ) -> "GMM":
        means = jrandom.uniform(key, (components, dim))

        covariances = cls._isotropic_covariances(components, dim) * stddev**2
        frequencies = jnp.ones(components) / components

        return cls(
            means,
            covariances,
            frequencies,
        )

    @classmethod
    def equidistant_triangle(
        cls,
        stddev: float,
    ) -> "GMM":
        angles = jnp.linspace(0, 2 * jnp.pi, 3, endpoint=False)
        means = _vec2d_from_angle(angles)

        covariances = cls._isotropic_covariances(3, 2) * stddev**2
        frequencies = jnp.ones(3) / 3

        return cls(
            means,
            covariances,
            frequencies,
        )


class Clusterlab1(Dataset, fj.Module, register=False):
    """
    A single cluster, example dataset 1 from the R package clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """

    def sample(self, key, n):
        # generate the normally distributed data
        X = jrandom.normal(key, (n, 2))

        # the cluster label is always zero, because there is only one cluster
        y = jnp.zeros(n)
        return X, y


class Clusterlab2(Dataset, fj.Module, register=False):
    """
    Four clusters with equal variances, example dataset 2 from the R package
    clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """

    std_dev: float = 2.5
    radius_data: float = 8

    def sample(self, key, n):
        # the vectors of centers just point into the four cardinal directions
        # create the angles of the four cardinal directions
        n_directions = 4
        angles = jnp.linspace(0, 2 * jnp.pi, n_directions, endpoint=False)

        # get the vectors from the angles
        centers = _vec2d_from_angle(angles)

        # create the data points and the corresponding cluster labels
        Xs = []
        ys = []

        for i, c in enumerate(centers):
            key, subkey = jrandom.split(key)

            ni = n // len(centers)

            if i == 0:
                ni += n % len(centers)

            # generate the data points
            X = jrandom.normal(subkey, (ni, 2))

            # do the scaling like the clusterlab package
            X = X * self.std_dev

            # shift the data points to the center (also scaled)
            X = X + c * self.radius_data
            Xs.append(X)

            # the cluster label is always the same for the same cluster
            y = jnp.zeros(ni) + i
            ys.append(y)

        # combine the data points and the cluster labels into single arrays
        Xs = jnp.vstack(Xs)
        ys = jnp.concatenate(ys)
        return Xs, ys


class Clusterlab3(Dataset, fj.Module, register=False):
    """
    Four clusters with different variances, example dataset 3 from the R package
    clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """

    std_devs: tuple[float, ...] = (1, 1, 2.5, 2.5)
    radius_data: float = 8

    def sample(self, key, n):
        # see the 'clusterlab_dataset2' for comments explaining the code
        cs = _vec2d_from_angle(jnp.linspace(0, 2 * jnp.pi, 4, endpoint=False))
        xs = []
        ys = []

        for i, (c, s) in enumerate(zip(cs, self.std_devs)):
            key, subkey = jrandom.split(key)

            ni = n // len(cs)
            if i == 0:
                ni += n % len(cs)

            xs.append(jrandom.normal(subkey, (ni, 2)) * s + c * self.radius_data)
            ys.append(jnp.zeros(ni) + i)

        xs = jnp.vstack(xs)
        ys = jnp.concatenate(ys)
        return xs, ys


class Clusterlab4(Dataset, fj.Module, register=False):
    """
    Simulating four clusters with one cluster pushed to the outside, example
    dataset 4 from the R package clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """

    std_dev: float = 2.5
    radius_data: float = 8
    dists: tuple[float, ...] = (1, 2, 1, 1)

    def sample(self, key, n):

        # see the 'clusterlab_dataset2' for comments explaining the code
        cs = _vec2d_from_angle(jnp.linspace(0, 2 * jnp.pi, 4, endpoint=False))
        xs = []
        ys = []

        for i, (c, a) in enumerate(zip(cs, self.dists)):
            key, subkey = jrandom.split(key)

            ni = n // len(cs)
            if i == 0:
                ni += n % len(cs)

            z = jrandom.normal(subkey, (ni, 2)) * self.std_dev
            xs.append(z + c * self.radius_data * a)
            ys.append(jnp.zeros(ni) + i)

        xs = jnp.vstack(xs)
        ys = jnp.concatenate(ys)
        return xs, ys


class Clusterlab6(Dataset, fj.Module, register=False):
    """
    Simulating five clusters with one central cluster, example dataset 6 from
    the R package clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """

    std_dev: float = 2.5
    radius_data: float = 16

    def sample(self, key, n):
        # see the 'clusterlab_dataset2' for comments explaining the code
        cs = _vec2d_from_angle(jnp.linspace(0, 2 * jnp.pi, 4, endpoint=False))
        cs = jnp.concatenate([cs, jnp.zeros((1, 2))])
        xs = []
        ys = []

        for i, c in enumerate(cs):
            key, subkey = jrandom.split(key)

            ni = n // len(cs)
            if i == 0:
                ni += n % len(cs)

            z = jrandom.normal(subkey, (ni, 2)) * self.std_dev
            xs.append(z + c * self.radius_data)
            ys.append(jnp.zeros(ni) + i)

        xs = jnp.vstack(xs)
        ys = jnp.concatenate(ys)
        return xs, ys


class Clusterlab8(Dataset, fj.Module, register=False):
    """
    Simulating six clusters with different variances, example dataset 8 from
    the R package clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """

    std_devs: tuple[float, ...] = (0.5, 1, 1.5, 1.75, 1.85, 1.95, 2.05)
    radius_data: float = 14

    def sample(self, key, n):
        # see the 'clusterlab_dataset2' for comments explaining the code
        n_directions = 6
        cs = _vec2d_from_angle(jnp.linspace(0, 2 * jnp.pi, n_directions, endpoint=False))
        xs = []
        ys = []

        for i, (c, s) in enumerate(zip(cs, self.std_devs)):
            key, subkey = jrandom.split(key)

            ni = n // len(cs)
            if i == 0:
                ni += n % len(cs)

            xs.append(jrandom.normal(subkey, (ni, 2)) * s + c * self.radius_data)
            ys.append(jnp.zeros(ni) + i)

        xs = jnp.vstack(xs)
        ys = jnp.concatenate(ys)
        return xs, ys


class Clusterlab9(Dataset, fj.Module, register=False):
    """
    Simulating six clusters with different push apart degrees, example dataset
    9 from the R package clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """

    radii: tuple[float, ...] = (0.5, 1, 1.5, 1.75, 1.85, 1.95, 2.05)
    radius_data: float = 9

    def sample(self, key, n):
        # see the 'clusterlab_dataset2' for comments explaining the code
        n_directions = 6
        cs = _vec2d_from_angle(jnp.linspace(0, 2 * jnp.pi, n_directions, endpoint=False))
        xs = []
        ys = []

        for i, (c, a) in enumerate(zip(cs, self.radii)):
            key, subkey = jrandom.split(key)

            ni = n // len(cs)
            if i == 0:
                ni += n % len(cs)

            xs.append(jrandom.normal(subkey, (ni, 2)) + c * self.radius_data * a)
            ys.append(jnp.zeros(ni) + i)

        xs = jnp.vstack(xs)
        ys = jnp.concatenate(ys)
        return xs, ys


class Clusterlab10(Dataset, fj.Module, register=False):
    """
    Simulating six clusters with different push apart degrees and variances,
    example dataset 10 from the R package clusterlab.
    Used in "Sparse clusterability: testing for cluster structure in high
    dimensions" by Laborde et al. (2023).
    """

    std_devs: tuple[float, ...] = (0.5, 1, 1.5, 1.75, 2, 2.25, 2.25)
    radii: tuple[float, ...] = (4.5, 9, 13.5, 15.75, 16.65, 17.55, 18.45

    def sample(self, key, n):
        # see the 'clusterlab_dataset2' for comments explaining the code
        n_directions = 6
        cs = _vec2d_from_angle(jnp.linspace(0, 2 * jnp.pi, n_directions, endpoint=False))
        xs = []
        ys = []

        for i, (c, s, a) in enumerate(zip(cs, self.std_devs, self.radii)):
            key, subkey = jrandom.split(key)

            ni = n // len(cs)
            if i == 0:
                ni += n % len(cs)

            xs.append(jrandom.normal(subkey, (ni, 2)) * s + c * a)
            ys.append(jnp.zeros(ni) + i)

        xs = jnp.vstack(xs)
        ys = jnp.concatenate(ys)
        return xs, ys


class BowTie(Dataset, fj.Module, register=False):
    """
    A 2D dataset with two clusters in the shape of a bowtie, that thins out
    towards the ends.

    This is done by creating two 2D correlated abs normal distributions, that
    point in opposite directions. The data points are then rotated by 45
    degrees, to create the bowtie shape.
    """

    rho: float = 0.3

    def sample(self, key, n):
        key_x, key_y = jrandom.split(key)
        X = jrandom.normal(key_x, (n, 2))

        # make correlated
        X = X @ jnp.array([[1, self.rho], [self.rho, 1]], dtype="float32")

        # abs normal with offset
        offset = 0.1
        X = jnp.abs(X) + offset

        # random {-1, 1} labels
        y = jrandom.randint(key_y, (X.shape[0],), 0, 2)

        r = y * 2 - 1
        X = X * r[:, None]

        # rotate 45 degrees
        s = jnp.sin(jnp.pi / 4)
        c = jnp.cos(jnp.pi / 4)
        R = jnp.array([[c, -s], [s, c]], dtype="float32")

        X = X @ R
        return X, y


def _correlated_2d_normal(key, n, rho: float):
    x = jrandom.normal(key, (n, 2))
    x = x @ jnp.array([[1, rho], [rho, 1]], dtype="float32")
    return x


class ZigZag(Dataset, fj.Module, register=False):
    """
    A 2D dataset with seven correlated normal clusters right next to each other.
    """

    rho: float = 0.9
    nclusters: int = fj.field(default=5, static=True)
    scale: float = 1.5

    def sample(self, key, n):
        Xs = []
        ys = []

        for i in range(self.nclusters):
            key, subkey = jrandom.split(key)

            ni = n // self.nclusters
            if i == 0:
                ni += n % self.nclusters

            X = _correlated_2d_normal(subkey, ni, self.rho)

            X[:, 0] += i * self.scale
            Xs.append(X)

            y = jnp.zeros(n) + i
            ys.append(y)

        X = jnp.concatenate(Xs)
        y = jnp.concatenate(ys)
        return X, y


class UniformCircle(Dataset, fj.Module, register=False):
    """
    A 2D dataset with two clusters in the shape of a circle.

    The data points are sampled from a uniform distribution in the unit circle.
    """

    r: float = 1
    std_dev: float = 0.1

    def sample(self, key, n):
        key_t, key_r = jrandom.split(key)

        t = jrandom.uniform(key_t, (n,), minval=0, maxval=2 * jnp.pi)

        X = jnp.stack([jnp.cos(t), jnp.sin(t)], -1)
        X = X * self.r

        # add noise
        z = jrandom.normal(key_r, (n, 2)) * self.std_dev
        X += z

        y = jnp.zeros(n)
        return X, y


class UniformSquare(Dataset, fj.Module, register=False):
    """
    A 2D dataset with a uniform distribution in a square.
    """

    r: float = 1

    def sample(self, key, n):
        X = jrandom.uniform(key, (n, 2), minval=-self.r, maxval=self.r)
        y = jnp.zeros(n)
        return X, y
