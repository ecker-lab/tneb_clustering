# Copyright 2022 Xin Han
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from tqdm import trange


def dendrogram_purity(dendrogram: np.ndarray, y: np.array, y_pred_raw=None):
    """
    A :math:`(n-1)` by 4 matrix ``Z`` is returned. At the
    :math:`i`-th iteration, clusters with indices ``Z[i, 0]`` and
    ``Z[i, 1]`` are combined to form cluster :math:`n + i`. A
    cluster with an index less than :math:`n` corresponds to one of
    the :math:`n` original observations. The distance between
    clusters ``Z[i, 0]`` and ``Z[i, 1]`` is given by ``Z[i, 2]``. The
    fourth value ``Z[i, 3]`` represents the number of original
    observations in the newly formed cluster.
    """
    if y_pred_raw is None:
        y_pred_raw = list(range(len(y)))
    base_clusters, base_counts = np.unique(y_pred_raw, return_counts=True)
    n_instance = len(base_clusters)

    parent_matrix = _get_parent(dendrogram=dendrogram, n_instance=n_instance)
    node_purity = _get_node_purity_mixed(
        parent_matrix=parent_matrix, y=y, y_pred_raw=y_pred_raw
    )
    y_label = np.unique(y)
    purity = 0
    pairs_counter = 0
    for true_cluster_index in range(len(y_label)):
        current_instances = np.argwhere(y == y_label[true_cluster_index]).flatten()
        purity_cache = dict()
        # purity scores for all pairs
        for i in trange(len(current_instances)):
            for j in range(len(current_instances))[i + 1 :]:
                cluster_a = int(y_pred_raw[current_instances[i]])
                cluster_b = int(y_pred_raw[current_instances[j]])
                if not (cluster_a, cluster_b) in purity_cache.keys():
                    purity_cache[(cluster_a, cluster_b)] = _purity_score(
                        y_pred_raw[current_instances[i]],
                        y_pred_raw[current_instances[j]],
                        true_cluster_index,
                        parent_matrix,
                        node_purity,
                        n_instance,
                    )
                purity += purity_cache[(cluster_a, cluster_b)]
                pairs_counter += 1
    purity = purity / pairs_counter

    return purity


def _purity_score(
    i: int,
    j: int,
    ci: int,
    parent_matrix: np.ndarray,
    node_purity: np.ndarray,
    n_instances: int,
):
    if i == j:
        score = node_purity[ci, i]
    else:
        lca = np.argwhere(parent_matrix[i, :] * parent_matrix[j, :] == 1).flatten()[0]
        score = node_purity[ci, lca + n_instances]
    return score


def _get_parent(dendrogram: np.ndarray, n_instance: int):
    """
    A :math:`(n-1)` by 4 matrix ``Z`` is returned. At the
    :math:`i`-th iteration, clusters with indices ``Z[i, 0]`` and
    ``Z[i, 1]`` are combined to form cluster :math:`n + i`. A
    cluster with an index less than :math:`n` corresponds to one of
    the :math:`n` original observations. The distance between
    clusters ``Z[i, 0]`` and ``Z[i, 1]`` is given by ``Z[i, 2]``. The
    fourth value ``Z[i, 3]`` represents the number of original
    observations in the newly formed cluster.
    """
    parent = [[i + n_instance] for i in range(n_instance - 1)]
    dendrogram = np.append(dendrogram, parent, axis=1)
    parent_matrix = np.zeros(shape=[n_instance, 2 * n_instance - 1], dtype=np.int8)

    for i in range(n_instance - 1):
        current_ind = []

        if dendrogram[i, 0] >= n_instance:
            ind = np.argwhere(parent_matrix[:, int(dendrogram[i, 0])] == 1).flatten()
            for item in ind:
                current_ind.append(item)
        else:
            current_ind.append(int(dendrogram[i, 0]))

        if dendrogram[i, 1] >= n_instance:
            ind = np.argwhere(parent_matrix[:, int(dendrogram[i, 1])] == 1).flatten()
            for item in ind:
                current_ind.append(item)
        else:
            current_ind.append(int(dendrogram[i, 1]))

        parent_matrix[current_ind, int(dendrogram[i, 4])] = 1

    parent_matrix = np.delete(parent_matrix, np.s_[:n_instance], axis=1)

    return parent_matrix


def _get_node_purity_mixed(parent_matrix: np.ndarray, y: np.array, y_pred_raw):
    true_clusters = np.unique(y)
    raw_clusters, leaf_sizes = np.unique(y_pred_raw, return_counts=True)
    n_leaves = len(np.unique(y_pred_raw))

    node_purity = np.zeros(shape=(len(true_clusters), 2 * len(y) - 1))

    # leaf purity per true cluster
    for true_cluster_index, true_cluster in enumerate(true_clusters):
        true_cluster_instances = np.argwhere(y == true_cluster).flatten()
        for leaf_index, raw_cluster in enumerate(raw_clusters):
            raw_cluster_instances = np.argwhere(y_pred_raw == raw_cluster).flatten()
            intersection = np.intersect1d(true_cluster_instances, raw_cluster_instances)
            # compute purity
            node_purity[true_cluster_index, leaf_index] = (
                len(intersection) / leaf_sizes[leaf_index]
            )

    # propagate purity values in the tree
    # Internal nodes in the dendrogram/parent_matrix are indexed from n_leaf … 2*n_leaves‑2
    for internal_node_idx in range(2 * n_leaves - 1)[n_leaves:]:
        # children of the current internal node (columns in parent_matrix are offset by n_leaf)
        child_indices = np.argwhere(
            parent_matrix[:, internal_node_idx - n_leaves] == 1
        ).flatten()

        for true_class_idx, true_cluster in enumerate(true_clusters):
            child_purities = node_purity[true_class_idx, child_indices]
            child_weights = leaf_sizes[child_indices]

            weighted_sum = (child_purities * child_weights).sum()
            total_weight = child_weights.sum()

            # Store the purity of the internal node for this true class
            node_purity[true_class_idx, internal_node_idx] = weighted_sum / total_weight

    return node_purity
