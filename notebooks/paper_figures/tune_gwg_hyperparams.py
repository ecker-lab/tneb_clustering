import numpy as np
from sklearn import datasets
from pathlib import Path

from corc.graph_metrics import gwgmara

import sys

sys.path.append('../../scripts/')
import our_datasets

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score


output_path = './results'

# Hyperparameters to search over.
n_components = [2, 5, 10, 15, 20, 25, 30, 50]
covs = ['spherical', 'diag', 'tied', 'full']
n_neighbors = [2, 3, 5]


def main():
    my_datasets = our_datasets.our_datasets()
    default_base = my_datasets.default_base
    dataset_selector = my_datasets.dataset_selector
    datasets = my_datasets.select_datasets(dataset_selector)

    params_per_dataset = pd.DataFrame()
    for i_dataset, (dataset, params) in enumerate(datasets):

        # Load and normalize data.
        X, y = dataset
        y = [0] * len(X) if y is None else np.array(y, dtype="int")

        # Normalize dataset for easier parameter selection.
        X = StandardScaler().fit_transform(X)

        # Load dataset parameters.
        name = params['name']
        dim = params['dim']
        n_clusters = params['n_clusters']

        results = pd.DataFrame()
        for n_c, cov, n_n in (
            (n_c, cov, n_n)
            for n_c in n_components
            for cov in covs
            for n_n in n_neighbors
        ):
            # Init Model.
            algorithm = gwgmara.GWGMara(
                latent_dim=dim,
                n_clusters=n_clusters,
                n_components=n_c,
                n_neighbors=n_n,
                covariance=cov,
                filter_edges=False,
                seed=42,
            )

            # Fit Model.
            algorithm.fit(X)
            y_pred = algorithm.predict(X)

            # Compute ARI to ground truth.
            ari = adjusted_rand_score(y, y_pred)

            item = {
                'gmm_n_components': [n_c],
                'n_neighbors': [n_n],
                'gmm_covariance': [cov],
                'ari': [ari],
                'dataset': [name],
            }
            results = pd.concat([results, pd.DataFrame.from_dict(item)])

            # Not testing all combinations, stopping as soon as a config has been found
            # that leads to ARI = 1.
            if ari == 1:
                break

        # Select best result.
        row = results.sort_values('ari', ascending=False).iloc[0:1]
        params_per_dataset = pd.concat([params_per_dataset, row], axis=0)

        # Save results.
        results.to_pickle(Path(output_path, f'results_{name}.pkl'))
    params_per_dataset.to_pickle(Path(output_path, f'results_params_per_dataset.pkl'))


if __name__ == '__main__':
    main()
