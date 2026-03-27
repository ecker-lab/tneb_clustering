import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.neighbors import NearestNeighbors
import pickle
import os


class Metric:
    def __init__(self, type, n_best, save_file, K, stds, outdir) -> None:
        """metric to assess clustering tendency

        Parameters
        ----------
        type : list
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
        k_results : dict -> str: ndarray, K x n_stds
            summary dict of metrics
        """
        self.metric_type = type
        self.n_best = n_best
        self.save_file = save_file
        self.K = K
        self.stds = stds
        self.outdir = outdir

        self.k_results = {}
        for t in self.metric_type:
            self.k_results[t] = np.zeros((self.K, len(self.stds)))

    def calculate(self, train_latents, test_latents, predictions, scores, k, s):
        """calculate metric between best clusterings

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
        dict -> str: ndarray, metric: n_best x n_best
            dict over metrics of mean metric between n_best runs of clustering
        """
        if "ARI" in self.metric_type:
            results = self.calculate_ARI(predictions, scores)
            self.k_results["ARI"][k, s] = results[
                np.triu_indices(self.n_best, k=1)
            ].mean()
        if "silhouette" in self.metric_type:
            results = self.calculate_silhouette_score(test_latents, predictions, scores)
            self.k_results["silhouette"][k, s] = results.mean()
        if "CH" in self.metric_type:
            results = self.calculate_calinski_harabasz_score(
                test_latents, predictions, scores
            )
            self.k_results["CH"][k, s] = results.mean()
        if "DB" in self.metric_type:
            results = self.calculate_davies_bouldin_score(
                test_latents, predictions, scores
            )
            self.k_results["DB"][k, s] = results.mean()
        if "hopkins" in self.metric_type:
            results = self.calc_hopkins_statistics(train_latents)
            self.k_results["hopkins"][k, s] = results.mean()
        else:
            raise NotImplementedError("[ERROR] Metric not yet implemented.")

    def summarize(self):
        """take mean metric per std over K and print/save"""
        if self.save_file:
            for t in self.metric_type:
                mean_metrics = self.k_results[t].mean(axis=0)
                data = pd.DataFrame(
                    data=np.stack((self.stds, mean_metrics)).T, columns=["std", t]
                )
                with open(os.path.join(self.outdir, f"{t}.pkl"), "wb") as f:
                    pickle.dump(data, f)
        else:
            for t in self.metric_type:
                mean_metrics = self.k_results[t].mean(axis=0)
                print(t, mean_metrics)

    def calculate_ARI(self, predictions, scores):
        """calculate adjusted rand index (ARI) between best clusterings

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
        sorted_preds = predictions[sorted_scores[: self.n_best]]

        aris = np.zeros((self.n_best, self.n_best))

        for i in range(self.n_best):
            for j in range(i, self.n_best):
                aris[i, j] = adjusted_rand_score(sorted_preds[i], sorted_preds[j])
        return aris

    def calculate_silhouette_score(self, test_latents, predictions, scores):
        """calculate Silhouette score for best clusterings

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
        sorted_preds = predictions[sorted_scores[: self.n_best]]

        silhouette = np.zeros((self.n_best, 1))

        for i in range(self.n_best):
            silhouette[i] = silhouette_score(test_latents, sorted_preds[i])
        return silhouette

    def calculate_calinski_harabasz_score(self, test_latents, predictions, scores):
        """calculate Calinski Harabasz score for best clusterings

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
            Calinski Harabasz score of n_best runs of clustering
        """
        sorted_scores = np.argsort(scores)[::-1]
        sorted_preds = predictions[sorted_scores[: self.n_best]]

        score = np.zeros((self.n_best, 1))

        for i in range(self.n_best):
            score[i] = calinski_harabasz_score(test_latents, sorted_preds[i])
        return score

    def calculate_davies_bouldin_score(self, test_latents, predictions, scores):
        """calculate Davies Bouldin score for best clusterings

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
            Davies Bouldin score of n_best runs of clustering
        """
        sorted_scores = np.argsort(scores)[::-1]
        sorted_preds = predictions[sorted_scores[: self.n_best]]

        score = np.zeros((self.n_best, 1))

        for i in range(self.n_best):
            score[i] = davies_bouldin_score(test_latents, sorted_preds[i])
        return score

    def calc_hopkins_statistics(self, train_latents):
        """calculate Hopkins statistics for n_best runs

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
            sample_size = int(
                train_latents.shape[0] * 0.2
            )  # 0.05 (5%) based on paper by Lawson and Jures

            # uniform random sample in the original data space
            X_uniform_random_sample = np.random.uniform(
                train_latents.min(axis=0),
                train_latents.max(axis=0),
                (sample_size, train_latents.shape[1]),
            )

            # random sample of size sample_size from the original data X
            random_indices = np.random.randint(
                0, train_latents.shape[0], (sample_size,)
            )
            X_sample = train_latents[random_indices]

            # initialize unsupervised learner for implementing neighbor searches
            neigh = NearestNeighbors(n_neighbors=2)
            nbrs = neigh.fit(train_latents)

            # u_distances = nearest neighbour distances from uniform random sample
            u_distances, _ = nbrs.kneighbors(X_uniform_random_sample, n_neighbors=2)
            u_distances = u_distances[:, 0]  # distance to the first (nearest) neighbour

            # w_distances = nearest neighbour distances from a sample of points from original data X
            w_distances, _ = nbrs.kneighbors(X_sample, n_neighbors=2)
            # distance to the second nearest neighbour (as the first neighbour will be the point itself, with distance = 0)
            w_distances = w_distances[:, 1]

            u_sum = np.sum(u_distances)
            w_sum = np.sum(w_distances)

            # compute and return Hopkins' statistic
            hopkins[i] = u_sum / (u_sum + w_sum)
        return hopkins


def variation_of_information(labels1, labels2, return_split_merge=False):
    # taken from https://github.com/cfpark00/pyvoi/blob/master/pyvoi.py
    assert len(labels2) == len(labels1)
    size = len(labels2)

    mutual_labels = (labels1.astype(np.uint64) << 32) + labels2.astype(np.uint64)

    sm_unique, sm_inverse, sm_counts = np.unique(
        labels2, return_inverse=True, return_counts=True
    )
    fm_unique, fm_inverse, fm_counts = np.unique(
        labels1, return_inverse=True, return_counts=True
    )
    _, mutual_inverse, mutual_counts = np.unique(
        mutual_labels, return_inverse=True, return_counts=True
    )

    terms_mutual = -np.log(mutual_counts / size) * mutual_counts / size
    terms_mutual_per_count = (
        terms_mutual[mutual_inverse] / mutual_counts[mutual_inverse]
    )
    terms_sm = -np.log(sm_counts / size) * sm_counts / size
    terms_fm = -np.log(fm_counts / size) * fm_counts / size
    if not return_split_merge:
        terms_mutual_sum = np.sum(terms_mutual_per_count)
        vi_split = terms_mutual_sum - terms_sm.sum()
        vi_merge = terms_mutual_sum - terms_fm.sum()
        vi = vi_split + vi_merge
        return vi, vi_split, vi_merge

    vi_split_each = np.zeros(len(sm_unique))
    np.add.at(vi_split_each, sm_inverse, terms_mutual_per_count)
    vi_split_each -= terms_sm
    vi_merge_each = np.zeros(len(fm_unique))
    np.add.at(vi_merge_each, fm_inverse, terms_mutual_per_count)
    vi_merge_each -= terms_fm

    vi_split = np.sum(vi_split_each)
    vi_merge = np.sum(vi_merge_each)
    vi = vi_split + vi_merge

    i_splitters = np.argsort(vi_split_each)[::-1]
    i_mergers = np.argsort(vi_merge_each)[::-1]

    vi_split_sorted = vi_split_each[i_splitters]
    vi_merge_sorted = vi_merge_each[i_mergers]

    splitters = np.stack([vi_split_sorted, sm_unique[i_splitters]], axis=1)
    mergers = np.stack([vi_merge_sorted, fm_unique[i_mergers]], axis=1)
    return vi, vi_split, vi_merge, splitters, mergers
