import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

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

    def calculate(self, predictions, scores, k, s):
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


    def summarize(self):
        """take mean metric per std over K and print/save
        """
        mean_aris = self.k_results.mean(axis=0)
        if self.save_file:
            data = pd.DataFrame(data=np.stack((self.stds, mean_aris)).T, columns=['std', self.metric_type])
            save(data, self.metric_type, self.outdir)
        else:
            print(mean_aris)


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
