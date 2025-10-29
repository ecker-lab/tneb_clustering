# -*- coding: utf-8 -*-

# License: GPL 3.0

import numpy as np
from scipy.special import gammaln
import sklearn.metrics
import networkx as nx

import time
import tqdm

import corc.bhc.api as api
import corc.purity


class BayesianHierarchicalClustering(api.AbstractBayesianBasedHierarchicalClustering):
    """
    Reference: HELLER, Katherine A.; GHAHRAMANI, Zoubin.
               Bayesian hierarchical clustering.
               In: Proceedings of the 22nd international conference on
                   Machine learning. 2005. p. 297-304.
               http://mlg.eng.cam.ac.uk/zoubin/papers/icml05heller.pdf
    """

    def __init__(self, data, model, alpha, cut_allowed, verbose=False):
        super().__init__(data, model, alpha, cut_allowed)
        self.verbose = verbose

    def _print(self, string):
        if self.verbose:
            print(string)

    def fit(self, data):
        self.data = data
        self.build()

    def build(self):
        print("starting computation")
        n_objects = self.data.shape[0]

        weights = []

        # active nodes
        active_nodes = np.arange(n_objects)
        # assignments - starting each point in its own cluster
        assignments = np.arange(n_objects)
        # stores information from temporary merges
        tmp_merge = None
        hierarchy_cut = False

        # for every single data point
        log_p = np.zeros(n_objects)
        log_d = np.zeros(n_objects)
        counts = np.ones(n_objects, dtype=int)
        for i in range(n_objects):
            # compute log(d_k)
            log_d[i] = BayesianHierarchicalClustering.__calc_log_d(
                self.alpha, counts[i], None
            )
            # compute log(p_i)
            log_p[i] = self.model.calc_log_mlh(self.data[i])

        ij = n_objects - 1

        starttime = time.time()
        pair_count = n_objects * (n_objects - 1) // 2
        tmp_merge = np.empty((pair_count, 5), dtype=float)
        row = 0
        # for every pair of data points
        for i in range(n_objects):
            log_p_k_row = self.model.row_of_log_likelihood_for_pairs(self.data, i)
            for j in range(i + 1, n_objects):
                # compute log(d_k)
                n_ch = counts[i] + counts[j]
                log_d_ch = log_d[i] + log_d[j]
                log_dk = BayesianHierarchicalClustering.__calc_log_d(
                    self.alpha, n_ch, log_d_ch
                )
                # compute log(pi_k)
                log_pik = np.log(self.alpha) + gammaln(n_ch) - log_dk
                # compute log(p_k)
                log_p_k = log_p_k_row[j - i - 1]  # since j starts at i + 1
                # compute log(r_k)
                log_p_ch = log_p[i] + log_p[j]
                r1 = log_pik + log_p_k
                r2 = log_d_ch - log_dk + log_p_ch
                log_r = r1 - r2
                # store results
                tmp_merge[row] = [i, j, log_r, r1, r2]
                row += 1
        self._print(
            f"Time taken for initial pairwise calculations: {time.time() - starttime:.2f} seconds"
        )

        starttime = time.time()
        new_comparison_time = 0
        # find clusters to merge
        arc_list = np.empty(0, dtype=api.Arc)

        data_per_cluster = [np.array([self.data[i]]) for i in range(n_objects)]

        with tqdm.tqdm(
            total=active_nodes.size - 1,
            desc="Merging clusters",
            disable=not self.verbose,
        ) as pbar:
            while active_nodes.size > 1:
                # find i, j with the highest probability of the merged hypothesis
                position = np.argmax(tmp_merge[:, 2])
                # max_log_rk = np.max(tmp_merge[:, 2])
                # ids_matched = np.argwhere(tmp_merge[:, 2] == max_log_rk)
                # position = np.min(ids_matched)
                i, j, log_r, r1, r2 = tmp_merge[position]
                i = int(i)
                j = int(j)
                weights.append(log_r)

                # cut if required and stop
                # if self.cut_allowed and log_r < 0:
                #     hierarchy_cut = True
                #     break

                # new node ij
                ij = counts.size
                n_ch = counts[i] + counts[j]
                counts = np.append(counts, n_ch)
                # compute log(d_ij)
                log_d_ch = log_d[i] + log_d[j]
                log_d_ij = BayesianHierarchicalClustering.__calc_log_d(
                    self.alpha, counts[ij], log_d_ch
                )
                log_d = np.append(log_d, log_d_ij)
                # update assignments
                data_per_cluster.append(
                    np.vstack((data_per_cluster[i], data_per_cluster[j]))
                )
                data_per_cluster[i] = None
                data_per_cluster[j] = None
                assignments[np.argwhere(assignments == i)] = ij
                assignments[np.argwhere(assignments == j)] = ij

                # create arcs from ij to i,j
                arc_i = api.Arc(ij, i)
                arc_j = api.Arc(ij, j)
                arc_list = np.append(arc_list, [arc_i, arc_j])

                # delete i,j from active list and add ij
                i_idx = np.argwhere(active_nodes == i).flatten()
                j_idx = np.argwhere(active_nodes == j).flatten()
                active_nodes = np.delete(active_nodes, [i_idx, j_idx])
                active_nodes = np.append(active_nodes, ij)

                # turn nodes i,j off
                # keep rows where neither column 0 nor column 1 equals i or j
                mask = ~np.isin(tmp_merge[:, :2], [i, j]).any(axis=1)
                tmp_merge = tmp_merge[mask]

                # compute log(p_ij)
                t1 = np.maximum(r1, r2)
                t2 = np.minimum(r1, r2)
                log_p_ij = t1 + np.log(1 + np.exp(t2 - t1))
                log_p = np.append(log_p, log_p_ij)

                comparison_time = time.time()
                # for every pair ij x active
                collected_merge_info = np.empty((len(active_nodes) - 1, 5), dtype=float)
                for k in range(active_nodes.size - 1):
                    # compute log(d_k)
                    n_ch = counts[k] + counts[ij]
                    log_d_ch = log_d[k] + log_d[ij]
                    log_dij = BayesianHierarchicalClustering.__calc_log_d(
                        self.alpha, n_ch, log_d_ch
                    )
                    # compute log(pi_k)
                    log_pik = np.log(self.alpha) + gammaln(n_ch) - log_dij
                    # compute log(p_k)
                    assert (
                        data_per_cluster[active_nodes[k]] is not None
                    ), f"data_per_cluster[{active_nodes[k]}] is None! {active_nodes}"
                    data_merged = np.vstack(
                        (data_per_cluster[ij], data_per_cluster[active_nodes[k]])
                    )
                    log_p_ij = self.model.calc_log_mlh(data_merged)
                    # compute log(r_k)
                    log_p_ch = log_p[ij] + log_p[active_nodes[k]]
                    r1 = log_pik + log_p_ij
                    r2 = log_d_ch - log_dij + log_p_ch
                    log_r = r1 - r2
                    # store results
                    collected_merge_info[k] = [ij, active_nodes[k], log_r, r1, r2]

                pbar.update(1)
                pbar.set_postfix_str(f"clusters: {active_nodes.size}, log_r: {log_r:.2f}")
                # append
                tmp_merge = np.vstack((tmp_merge, collected_merge_info))
                new_comparison_time += time.time() - comparison_time
        self._print(f"Time taken for merging: {time.time() - starttime:.2f} seconds")
        self._print(
            f"Time taken for new comparisons: {new_comparison_time:.2f} seconds"
        )


        self.result = api.Result(
            arc_list=arc_list,
            node_ids=np.arange(0, ij + 1),
            last_log_p=log_p[-1],
            weights=np.array(weights),
            hierarchy_cut=hierarchy_cut,
            n_clusters=len(np.unique(assignments)),
        )
        return self.result

    def predict_with_target(self, X, target_number_clusters):
        """
        Returns the cluster assignment based on target_number_clusters (i.e. cuts the
        tree accordingly). X is ignored.
        """
        if self.result is None:
            raise ValueError("The model has not been fitted yet.")
        assert X is None or len(X) == len(
            self.data
        ), "Only works on training data! X is ignored."

        G = nx.DiGraph()
        for arc in self.result.arc_list:
            G.add_edge(arc.source, arc.target)

        remaining_nodes = self.result.node_ids[: (-(target_number_clusters - 1))]
        subgraph = G.subgraph(remaining_nodes)
        subgraph = subgraph.to_undirected()
        components = list(nx.connected_components(subgraph))

        y_pred = np.zeros(len(self.data), dtype=int)
        for i, component in enumerate(components):
            for node in component:
                if node < len(self.data):  # only original nodes, no intermediate ones
                    y_pred[node] = i

        return y_pred

    def get_purity(self, y, X=None):
        if not hasattr(self, "purity"):
            dendrogram = self.result.get_dendrogram()
            self.purity = corc.purity.dendrogram_purity(dendrogram, y)
        return self.purity

    def get_ari(self, y):
        if not hasattr(self, "ari"):
            self.ari = sklearn.metrics.adjusted_rand_score(
                y, self.predict_with_target(self.data, len(np.unique(y)))
            )
        return self.ari

    @staticmethod
    def __calc_log_d(alpha, nk, log_d_ch):
        if nk == 1 and log_d_ch is None:
            return np.log(alpha)
        else:
            dk_t1 = np.log(alpha) + gammaln(nk)
            dk_t2 = log_d_ch
            a = np.maximum(dk_t1, dk_t2)
            b = np.minimum(dk_t1, dk_t2)
            return a + np.log(1 + np.exp(b - a))
