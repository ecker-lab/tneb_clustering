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
import tqdm
import itertools


def dendrogram_purity(dendrogram: np.ndarray, y: np.array, y_pred_raw=None):
    """
    dendrogram definition: A :math:`(n-1)` by 4 matrix ``Z`` is returned. At the
    :math:`i`-th iteration, clusters with indices ``Z[i, 0]`` and
    ``Z[i, 1]`` are combined to form cluster :math:`n + i`. A
    cluster with an index less than :math:`n` corresponds to one of
    the :math:`n` original observations. The distance between
    clusters ``Z[i, 0]`` and ``Z[i, 1]`` is given by ``Z[i, 2]``. The
    fourth value ``Z[i, 3]`` represents the number of original
    observations in the newly formed cluster.
    """
    if y_pred_raw is None:
        y_pred_raw = np.array(list(range(len(y))))

    num_true_classes = len(np.unique(y))
    # y_pred_raw = np.array(list(range(len(y))))
    base_clusters = np.unique(y_pred_raw)
    num_base_clusters = len(base_clusters)
    parent_matrix = _get_parent_matrix(
        dendrogram=dendrogram, n_instance=num_base_clusters
    )

    # counts_per_class contains for every true class label the number of datapoints
    # with this label for each base class. For a full hierarchy this is either 0 or 1.
    # For an overclustering, larger values are possible.
    counts_per_class = np.zeros((num_true_classes, num_base_clusters), dtype=int)
    for true_label_index in range(num_true_classes):
        filter = y == (np.unique(y)[true_label_index])
        values, counts = np.unique(y_pred_raw[filter], return_counts=True)
        # store the counts in the big array
        for i in range(len(values)):
            base_cluster_index = np.where(base_clusters == values[i])
            counts_per_class[true_label_index, base_cluster_index] = counts[i]

    # parent_matrix[i,:] contains a 1 at position j if j is a descendant of i.
    # the matrix multiplication thus sums up how many original nodes per class are at each
    # node in the tree.
    node_counts = counts_per_class @ parent_matrix.T

    purity = 0.0
    total_pairs = 0
    # within-leaf purities (always 0 for complete hierarchies)
    for true_label_index in range(num_true_classes):
        for base_cluster_index in range(num_base_clusters):
            class_count = counts_per_class[true_label_index, base_cluster_index]
            leaf_purity = class_count / np.sum(counts_per_class[:, base_cluster_index])
            weight = class_count * (class_count - 1) / 2
            purity += leaf_purity * weight
            total_pairs += weight

    # purity where inner nodes of the dendrogram serve as LCA
    for true_label_index in range(num_true_classes):
        for index, (child1, child2, _, _) in enumerate(dendrogram):
            counts_child1 = node_counts[true_label_index, int(child1)]
            counts_child2 = node_counts[true_label_index, int(child2)]
            weight = counts_child1 * counts_child2

            class_count = node_counts[true_label_index, index + num_base_clusters]
            node_purity = class_count / np.sum(
                node_counts[:, index + num_base_clusters]
            )

            purity += node_purity * weight
            total_pairs += weight

    purity /= total_pairs
    return purity


def _get_parent_matrix(dendrogram, n_instance):
    "The parent matrix stores for each cluster which of the basic elements/clusters form it."
    # each line of the dendrogram corresponds to a cluster, plus we need the initial clusters
    parent_matrix = np.zeros(
        (dendrogram.shape[0] + n_instance, n_instance), dtype=np.int8
    )
    # each initial cluster consists just of itself
    for i in range(n_instance):
        parent_matrix[i, i] = 1

    for i, (child1, child2, _, _) in enumerate(dendrogram):
        parent_matrix[i + n_instance] = (
            parent_matrix[int(child1)] + parent_matrix[int(child2)]
        )

    assert np.max(parent_matrix) == 1, "input dendrogram does not correspond to a tree"

    return parent_matrix
