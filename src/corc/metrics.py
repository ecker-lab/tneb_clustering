import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from corc.utils import save


class Metric():
    def __init__(self, type, n_best, save_file, K, stds, outdir) -> None:
        """metric to assess clustering tendency

        Parameters
        ----------
        type : str
            type of metric, i.e. adjusted rand index (ARI)
        n_best : int
            number of how many clustering runs to consider for metric calculated
        save_file : bool
            if metric summary should be saved to pkl
        K : int
            execute clustering and metric calculation for K times
        stds : list of float
            standard deviations of sampled data
        outdir : str
            folder name where to save data
        k_results : ndarry, K x n_stds
            summary array of metric
        """
        self.metric_type = type
        self.n_best = n_best
        self.save_file = save_file
        self.K = K
        self.stds = stds
        self.outdir = outdir

        self.k_results = np.zeros((self.K, len(self.stds)))


    def calculate(self, train_latents, test_latents, predictions, scores, k, s):
        """ calculate metric between best clusterings

        Parameters
        ----------
        predictions : ndarray
            of clustering algorithm
        scores : ndarray
            of clustering algorithm
        k : int
            k-th iteration of K times
        s : int
            s-th standard deviation in #stds

        Returns
        -------
        ndarray, n_best x n_best
            mean metric between n_best runs of clustering
        """
        if self.metric_type == 'ARI':
            results = self.calculate_ARI(predictions, scores)
            self.k_results[k, s] = results[np.triu_indices(self.n_best, k=1)].mean()
        if self.metric_type == 'silhouette':
            results = self.calculate_silhouette_score(test_latents, predictions, scores)
            self.k_results[k, s] = results.mean()
        if self.metric_type == 'hopkins':
            results = self.calc_hopkins_statistics(train_latents)
            self.k_results[k, s] = results.mean()


    def summarize(self):
        """take mean metric per std over K and print/save
        """
        mean_metrics = self.k_results.mean(axis=0)
        if self.save_file:
            data = pd.DataFrame(data=np.stack((self.stds, mean_metrics)).T, columns=['std', self.metric_type])
            save(data, self.metric_type, self.outdir)
        else:
            print(mean_metrics)


    def calculate_ARI(self, predictions, scores):
        """ calculate adjusted rand index (ARI) between best clusterings

        Parameters
        ----------
        predictions : ndarray
            of clustering algorithm
        scores : ndarray
            of clustering algorithm

        Returns
        -------
        ndarray, n_best x n_best
            mean ARI between n_best runs of clustering
        """
        sorted_scores = np.argsort(scores)[::-1]
        sorted_preds = predictions[sorted_scores[:self.n_best]]

        aris = np.zeros((self.n_best, self.n_best))

        for i in range(self.n_best):
            for j in range(i, self.n_best):
                aris[i, j] = adjusted_rand_score(sorted_preds[i], sorted_preds[j])
        return aris


    def calculate_silhouette_score(self, test_latents, predictions, scores):
        """ calculate Silhouette score for best clusterings

        Parameters
        ----------
        test_latents : ndarray
            test split of data points
        predictions : ndarray
            of clustering algorithm
        scores : ndarray
            of clustering algorithm

        Returns
        -------
        ndarray, n_best x 1
            Silhouette score of n_best runs of clustering
        """
        sorted_scores = np.argsort(scores)[::-1]
        sorted_preds = predictions[sorted_scores[:self.n_best]]

        silhouette = np.zeros((self.n_best, 1))

        for i in range(self.n_best):
            silhouette[i] = silhouette_score(test_latents, sorted_preds[i])
        return silhouette


    def calc_hopkins_statistics(self, train_latents):
        """ calculate Hopkins statistics for n_best runs

        Parameters
        ----------
        train_latents : ndarray
            train split of data points

        Returns
        -------
        ndarray, n_best x 1
            Hopkins statistics of n_best runs
        """
        hopkins = np.zeros((self.n_best, 1))

        for i in range(self.n_best):
            sample_size = int(train_latents.shape[0] * 0.2) # 0.05 (5%) based on paper by Lawson and Jures

            # uniform random sample in the original data space
            X_uniform_random_sample = np.random.uniform(train_latents.min(axis=0), train_latents.max(axis=0), (sample_size, train_latents.shape[1]))

            # random sample of size sample_size from the original data X
            random_indices = np.random.randint(0, train_latents.shape[0], (sample_size,))
            X_sample = train_latents[random_indices]

            # initialize unsupervised learner for implementing neighbor searches
            neigh = NearestNeighbors(n_neighbors=2)
            nbrs = neigh.fit(train_latents)

            # u_distances = nearest neighbour distances from uniform random sample
            u_distances, _ = nbrs.kneighbors(X_uniform_random_sample, n_neighbors=2)
            u_distances = u_distances[:, 0] # distance to the first (nearest) neighbour

            # w_distances = nearest neighbour distances from a sample of points from original data X
            w_distances , _ = nbrs.kneighbors(X_sample, n_neighbors=2)
            # distance to the second nearest neighbour (as the first neighbour will be the point itself, with distance = 0)
            w_distances = w_distances[:, 1]

            u_sum = np.sum(u_distances)
            w_sum = np.sum(w_distances)

            # compute and return Hopkins' statistic
            hopkins[i] = u_sum / (u_sum + w_sum)
        return hopkins