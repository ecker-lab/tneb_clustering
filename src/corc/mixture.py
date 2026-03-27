import corc.graph_metrics.tmm_gmm_neb
import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class MixtureModel(ABC):
    def __init__(self, centers, covs, weights):
        self.weights = jnp.array(weights)
        self.centers = jnp.array(centers)
        self.covs = jnp.array(covs)

    @abstractmethod
    def predict(self, data_X, return_probs=False): ...

    @abstractmethod
    def score_samples(self, data_X): ...

    def filter_components(self, data_X, min_cluster_size=10, max_elongation=1000):
        # compute elongations and cluster sizes
        counts = self.get_counts(data_X)
        elongations = self.get_elongations()

        # construct filter (True will be kept)
        component_filter = np.logical_and(
            counts > min_cluster_size, np.array(elongations) < max_elongation
        )

        # filter the components
        self.centers = self.centers[component_filter]
        self.covs = self.covs[component_filter]
        self.weights = self.weights[component_filter]
        if hasattr(self, "df"):
            self.df = self.df[component_filter]

    def get_elongations(self):
        """
        elongation of a component is defined by the quotient
        between largest and smallest eigenvalue
        """
        elongations = []
        for cov in self.covs:
            eigenvalues = np.linalg.eigvalsh(cov)
            elongation = max(eigenvalues) / min(eigenvalues)
            elongations.append(elongation)
        return elongations

    def get_counts(self, X):
        y_pred = self.predict(data_X=X)
        unique_values, counts = np.unique(y_pred, return_counts=True)
        padded_counts = jnp.zeros(len(self.weights), dtype=int)
        padded_counts = padded_counts.at[unique_values].set(counts)
        return padded_counts

    def print_elongations_and_counts(self, X):
        counts = self.get_counts(X)
        elongations = self.get_elongations()

        print(np.array(list(zip(counts, elongations)), dtype=int))
        return counts, elongations

    def plot_energy_landscape(
        self,
        data_X,  # for boundary computation
        levels=20,
        grid_resolution=128,
        axis=None,
        kwargs={},
    ):
        if axis == None:
            axis = plt.gca()

        # grid coordinates
        margin = 0.5
        x_grid = np.linspace(
            data_X[:, 0].min() - margin, data_X[:, 0].max() + margin, grid_resolution
        )
        y_grid = np.linspace(
            data_X[:, 1].min() - margin, data_X[:, 1].max() + margin, grid_resolution
        )
        XY = np.stack(np.meshgrid(x_grid, y_grid), -1)

        # get scores for the grid values
        mm_probs = self.score_samples(XY.reshape(-1, 2)).reshape(
            grid_resolution, grid_resolution
        )
        mm_probs = np.clip(mm_probs, None, 0)
        # plotting the energy landscape
        axis.contourf(
            x_grid,
            y_grid,
            mm_probs,
            levels=levels,
            cmap="coolwarm",
            alpha=0.5,
            zorder=-10,
            **kwargs,
        )


class GaussianMixtureModel(MixtureModel):
    def __init__(self, centers, covs, weights):
        super().__init__(centers, covs, weights)

    @classmethod
    def from_sklearn(cls, mixture_model):
        centers = mixture_model.means_
        covs = mixture_model.covariances_
        weights = mixture_model.weights_
        return cls(centers, covs, weights)

    def predict(self, data_X, return_probs=False):
        predictions, probs = corc.graph_metrics.tmm_gmm_neb.predict_gmm_jax(
            X=data_X,
            means=self.centers,
            covs=self.covs,
            weights=self.weights,
        )
        if return_probs:
            return predictions, probs
        else:
            return predictions

    def score_samples(self, data_X):
        return corc.graph_metrics.tmm_gmm_neb.gmm_jax(
            X=data_X, means=self.centers, covs=self.covs, weights=self.weights
        )


class StudentMixture(MixtureModel):

    def __init__(self, centers, covs, weights, df=1):
        super().__init__(centers, covs, weights)
        self.df = jnp.array(df)

    @classmethod
    def from_EMStudentMixture(cls, mixture_model):
        centers = mixture_model.location
        covs = np.transpose(mixture_model.scale, axes=(2, 0, 1))
        weights = mixture_model.mix_weights
        return cls(centers, covs, weights, df=mixture_model.df_)

    def predict(self, data_X, return_probs=False):
        predictions, probs = corc.graph_metrics.tmm_gmm_neb.predict_tmm_jax(
            X=data_X,
            means=self.centers,
            covs=self.covs,
            weights=self.weights,
            df=self.df,
        )
        if return_probs:
            return predictions, probs
        else:
            return predictions

    def score_samples(self, data_X):
        return corc.graph_metrics.tmm_gmm_neb.tmm_jax(
            X=data_X,
            means=self.centers,
            covs=self.covs,
            weights=self.weights,
            df=self.df,
        )
