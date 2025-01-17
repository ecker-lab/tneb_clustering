import corc.graph_metrics.tmm_gmm_neb
import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod


class MixtureModel(ABC):
    def __init__(self, centers, covs, weights):
        self.weights = weights
        self.centers = centers
        self.covs = covs

    @abstractmethod
    def predict(self, data_X, return_probs=False): ...

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


class StudentMixture(MixtureModel):

    def __init__(self, centers, covs, weights):
        super().__init__(centers, covs, weights)

    @classmethod
    def from_EMStudentMixture(cls, mixture_model):
        centers = mixture_model.location
        covs = np.transpose(mixture_model.scale, axes=(2, 0, 1))
        weights = mixture_model.mix_weights
        return cls(centers, covs, weights)

    def predict(self, data_X, return_probs=False):
        predictions, probs = corc.graph_metrics.tmm_gmm_neb.predict_tmm_jax(
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
        return corc.graph_metrics.tmm_gmm_neb.tmm_jax(
            X=data_X, means=self.centers, covs=self.covs, weights=self.weights
        )
