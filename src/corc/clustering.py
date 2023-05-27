import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


class Clustering():
    def __init__(self, type, n_runs, n_clusters) -> None:
        """clustering algorithm

        Parameters
        ----------
        type : str
            type of clustering, i.e. Gaussian Mixture Model (GMM)
        n_runs : int
            number of runs of cluster fitting
        n_clusters : int
            number of clusters
        """
        self.clustering_type = type
        self.n_runs = n_runs
        self.n_clusters = n_clusters


    def run(self, train_latents, test_latents):
        """ fit clustering algorithm and predict test set

        Parameters
        ----------
        train_latents : ndarray
            training set of latent embeddings
        test_latents : ndarray
            test set of latent embeddings

        Returns
        -------
        ndarray, ndarray
            prediction and scores of clustering algorithm
        """
        if self.clustering_type == 'gmm':
            return self.run_gmm_clustering(train_latents, test_latents)
        if self.clustering_type == 'kmeans':
            return self.run_kmeans_clustering(train_latents, test_latents)


    def run_gmm_clustering(self, train_latents, test_latents):
        """ fit GMM and predict test set

        Parameters
        ----------
        train_latents : ndarray
            training set of latent embeddings
        test_latents : ndarray
            test set of latent embeddings

        Returns
        -------
        ndarray, ndarray
            prediction and scores of GMM
        """
        predictions = np.zeros((self.n_runs, len(test_latents)))
        scores = np.zeros((self.n_runs))

        for i in range(self.n_runs):
            # fit GMM & predict test set
            # The type is covariance_type="diag", which means that the size of the cluster along each dimension can be set independently, with the resulting ellipse constrained to align with the axes.
            gmm = GaussianMixture(n_components=self.n_clusters, covariance_type='diag').fit(train_latents)
            predictions[i] = gmm.predict(test_latents)
            scores[i] = gmm.score(test_latents)
        return predictions, scores


    def run_kmeans_clustering(self, train_latents, test_latents):
        """ fit kmeans and predict test set

        Parameters
        ----------
        train_latents : ndarray
            training set of latent embeddings
        test_latents : ndarray
            test set of latent embeddings

        Returns
        -------
        ndarray, ndarray
            prediction and scores of kmeans
        """
        predictions = np.zeros((self.n_runs, len(test_latents)))
        scores = np.zeros((self.n_runs))

        for i in range(self.n_runs):
            # fit kmeans & predict test set
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto').fit(train_latents)
            predictions[i] = kmeans.predict(test_latents)
            scores[i] = kmeans.score(test_latents)
        return predictions, scores
