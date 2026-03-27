import numpy as np
from scipy.spatial import KDTree, distance
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt

import corc

"""
Python implementation of https://github.com/Guanjunyi/SMMP-A-Stable-Membership-based-Auto-tuning-Multi-Peak-Clustering-Algorithm.git
"""


class SMMP:
    def __init__(self, eta=0.1, k=None, n_clusters=None, isshowresult=False):
        self.eta = eta
        self.isshowresult = isshowresult
        self.k = k
        self.n_clusters = n_clusters
        self.labels = None

    def predict(self, data):
        return self.labels

    def fit(self, data):
        self.graph_data = {"nodes": None, "edges": None, "nodes_org_space": None}
        self.data = data

        n, d = data.shape  # Get number of samples (n) and dimensions (d)
        # fast search of KNN matrix based on kd-tree (when dimension is not larger than 10)
        max_k, knn, knn_dist = self.compute_knn_matrix(data)
        # adaptive tuning of parameter k
        if self.k is None:
            self.k = self.adaptive_tuning_k(knn, knn_dist, max_k)
        if self.isshowresult:
            print(f"Number of inital subclusters: {self.k=}")

        # setting of parameter k_b for our border link detection
        k_b = int(min(round(self.k / 2), 2 * np.floor(np.log(n))))
        # density estimation
        rho = 1 / (
            self.k * np.sum(knn_dist[:, 1 : self.k], axis=1)
        )  # within-surrounding-similarity-based density w.r.t 'k'
        # identify density peaks and calculate center-representativeness
        npn, theta, descendant, pk, n_pk, OrdRho = self.compute_density_peaks(rho, knn)
        if self.isshowresult:
            print(f"Number of density peaks: {n_pk=}")

        # generate sub-clsuters
        sl, edge = self.assign_sub_labels(n, pk, npn, OrdRho, descendant)
        # obtain cross-cluster border pairs
        borderpair = self.obtain_borderpairs(sl, k_b, knn, knn_dist)
        # obtain border links
        blink = self.obtain_borderlinks(borderpair)

        if self.isshowresult:
            self.resultshow(data, sl, edges=blink, title="subclusters")
        self.graph_data["edges"] = blink

        if blink.size == 0:  # if there is no border link, output sub-clustering result
            CL = sl.T
            self.n_clusters = n_pk
            if self.isshowresult:
                self.resultshow(data, CL, title="clusters (no blink)")
        else:
            # calculate representativeness of border links for the similarity estimation between subclusters
            simimesgs = np.empty(
                (n_pk, n_pk), dtype=object
            )  # Initialize an empty object array to hold the sets of similarity messages

            for ii, jj in blink:
                pk1 = sl[ii]
                pk2 = sl[jj]
                smesgs = smesgs = (
                    simimesgs[pk1, pk2] if simimesgs[pk1, pk2] is not None else []
                )  # Get the current similarity messages set
                smesgs.append(
                    (theta[ii] + theta[jj]) / 2
                )  # Append the average of theta[ii] and theta[jj] to the set
                simimesgs[pk1, pk2] = (
                    smesgs  # Update the similarity messages set for pk1, pk2
                )
                simimesgs[pk2, pk1] = (
                    smesgs  # Update the set for pk2, pk1 (since similarity is bidirectional)
                )
            simimesgs = np.array(simimesgs)

            # similarity estimation between subclusters
            sim = np.ones((n_pk, n_pk))
            np.fill_diagonal(sim, 0)

            for pk1 in range(n_pk - 1):
                for pk2 in range(pk1 + 1, n_pk):
                    smesgs = simimesgs[pk1, pk2]  # Retrieve similarity messages
                    if smesgs != None:
                        smesgs = smesgs[
                            0
                        ]  # Unwrap the list (similar to smesgs{:} in MATLAB)
                        max_smesg = np.max(
                            smesgs
                        )  # Find the maximum similarity message
                        min_n_smesg = int(
                            np.ceil(min(edge[pk1], edge[pk2]) * self.eta)
                        )  # min_n_smesg: the minimum standard number of similarity message samples
                        smesgs = np.sort(
                            np.concatenate([[smesgs], np.zeros(min_n_smesg)]), axis=0
                        )[
                            ::-1
                        ]  # Sort in descending order
                        smesgs = smesgs[
                            :min_n_smesg
                        ]  # Take the top min_n_smesg elements

                        if max_smesg > 0:
                            Gamma = np.mean(np.abs(smesgs - max_smesg)) / max_smesg
                            sim[pk1, pk2] = max_smesg * (1 - Gamma)
                            sim[pk2, pk1] = max_smesg * (1 - Gamma)

            # Single-linkage clustering of sub-clusters according to SIM
            SingleLink = linkage(
                1 - np.array(sim), method="single"
            )  # Perform hierarchical clustering using 'single' linkage
            if self.isshowresult:
                self.dendrogramshow(SingleLink)

            # Check if NC (number of clusters) is provided
            if self.n_clusters is None:
                bata = np.concatenate(
                    ([0], SingleLink[:, -1])
                )  # Last column of linkage
                bata[bata < 0] = 0  # Ensure no negative values in bata
                bataratio = np.column_stack(
                    [np.arange(n_pk + 1, 0, -1)[: n_pk - 1], np.diff(bata)]
                )  # Compute bataratio
                bataratio = bataratio[bataratio[:, 1].argsort()][
                    ::-1
                ]  # Sort by the second column in descending order
                self.n_clusters = int(
                    bataratio[0, 0]
                )  # Stable number of clusters with maximum bata-interval

            # Assign clusters using fcluster
            CL_pk = fcluster(SingleLink, t=self.n_clusters, criterion="maxclust")

            # Assign final cluster labels
            CL = np.zeros(
                len(sl), dtype=int
            )  # Initialize the CL array to store cluster labels
            for i in range(n_pk):
                CL[sl == i] = CL_pk[
                    i
                ]  # Assign the cluster label based on the index in sl

            if self.isshowresult:
                self.resultshow(data, CL, title="clusters")

            self.labels = CL

    def obtain_borderlinks(self, borderpair):
        """
        obtain border links
        """
        if len(borderpair) == 0:
            return np.array([])  # Return an empty array if no border pairs exist

        borderpair[:, :2] = np.sort(
            borderpair[:, :2], axis=1
        )  # Sort to identify similar rows
        _, unique_indices = np.unique(borderpair[:, 2], return_index=True)
        borderpair = borderpair[unique_indices]  # Get unique rows
        borderpair = borderpair[
            np.argsort(borderpair[:, 2])
        ]  # Sort by the third column (distance)

        n_pairs = borderpair.shape[0]
        blink = []  # Border link storage

        for i in range(n_pairs):
            bp = borderpair[i, :2]

            # Check if bp shares elements with blink
            if not any(np.intersect1d(bp, np.array(blink).flatten())):
                blink.append(bp.tolist())

        return np.array(blink, dtype="int")

    def obtain_borderpairs(self, sl, k_b, knn, knn_dist):
        """
        obtain cross-cluster border pairs
        """
        borderpair = []
        n = len(sl)

        for i in range(n):
            label_i = sl[i]
            for j in range(1, k_b):
                i_nei = knn[i, j]
                dist_i_nei = knn_dist[i, j]
                label_nei = sl[i_nei]

                if label_i != label_nei and i in knn[i_nei, 1:k_b]:  # Check mutual KNN
                    borderpair.append([i, i_nei, dist_i_nei])
                    break

        return np.array(borderpair)

    def assign_sub_labels(self, n, pk, npn, OrdRho, descendant):
        """
        generate sub clusters
        """
        sl = -1 * np.ones(n, dtype=int)  # Sub-labels of points (initialized to -1)
        n_pk = len(pk)
        for i, pki in enumerate(pk):  # Assign unique sub-labels to density peaks
            sl[pki] = i

        # Inherit sub-labels from nearest parent node (NPN)
        for i in range(n):
            if sl[OrdRho[i]] == -1:
                sl[OrdRho[i]] = sl[npn[OrdRho[i]]]

        # Compute the number of edges in each sub-cluster
        edge = np.zeros(n_pk, dtype=int)
        for i in range(0, n_pk):
            child_sub = descendant[sl == i]
            edge[i] = np.sum(
                child_sub == 0
            )  # Count leaf nodes (no descendants) edge(i): the edge number of sub-cluster 'i'

        return sl, edge

    def compute_density_peaks(self, rho, knn):
        """
        identify density peaks and calculate center-representativeness
        """
        n = len(rho)
        theta = np.ones(
            n
        )  # theta(i): the center-representativeness of point 'i' (initialization)
        descendant = np.zeros(
            n, dtype=int
        )  # descendant(i): the descendant node number of point 'i' (initialization)
        npn = np.zeros(n, dtype=int)  # Neighbor-based parent node (initialization)
        OrdRho = np.argsort(-rho)  # Sort indices by density (descending order)

        for i in range(n):
            for j in range(
                1, self.k
            ):  # MATLAB index starts at 2, so Python starts at 1
                neigh = knn[OrdRho[i], j]
                if rho[OrdRho[i]] < rho[neigh]:
                    npn[OrdRho[i]] = (
                        neigh  # NPN:neigbor-based parent node, i.e., nearest higher density point within the KNN area.
                    )
                    theta[OrdRho[i]] = theta[neigh] * (rho[OrdRho[i]] / rho[neigh])
                    descendant[neigh] += 1
                    break

        pk = np.where(theta == 1)[0]  # find density peaks (i.e., sub-cluster centers)
        n_pk = len(pk)  # number of density peaks

        return npn, theta, descendant, pk, n_pk, OrdRho

    def compute_knn_matrix(self, data):
        """
        fast search of KNN matrix based on kd-tree (when dimension is not larger than 10)
        """
        n, d = data.shape  # Get number of samples (n) and dimensions (d)

        # Define max_k based on n
        if n > 200:
            max_k = int(np.ceil(np.sqrt(n)))
        else:
            max_k = max(15, round(n / 10))

        if d <= 11:
            # Use KDTree for fast nearest neighbor search
            tree = KDTree(data)
            knn_dist, knn = tree.query(data, k=max_k * 2)
        else:
            # Compute full pairwise distance matrix
            dist_matrix = distance.cdist(data, data, metric="euclidean")
            knn = np.argsort(dist_matrix, axis=1)  # Sorted indices
            knn_dist = np.sort(dist_matrix, axis=1)  # Sorted distances

        return max_k, knn, knn_dist

    def adaptive_tuning_k(self, knn, knn_dist, max_k):
        """
        adaptive tuning of parameter k
        """
        n = knn.shape[0]
        n_k = np.zeros(
            n
        )  # n_k: the number of different 'k' that satisfy the number of desnity peaks 'n_pk'
        k_sum = np.zeros(
            n
        )  # k_sum : the sum of different 'k' that satisfy the number of desnity peaks 'n_pk'

        for cur_k in range(
            2, max_k + 1, max(1, max_k // 20)
        ):  # ceil(kmax/20): run about 20 times (to reduce computation time)
            cur_rho = 1 / (
                cur_k * np.sum(knn_dist[:, 1:cur_k], axis=1)
            )  # within-surrounding-similarity-based density w.r.t 'cur_k'

            ispk = np.ones(n, dtype=int)  # density peak label
            for i in range(n):
                for j in range(1, cur_k):  # check neighborhood
                    if cur_rho[i] < cur_rho[knn[i, j]]:
                        ispk[i] = 0  # not a density peak
                        break

            n_pk = np.sum(ispk)  # n_pk: the number of denisty peak w.r.t 'cur_k'
            n_k[n_pk] += 1
            k_sum[n_pk] += cur_k

        # Find the stable number of density peaks with the maximum k-interval
        stb_n_pk = np.argmax(n_k)  # Index of the most stable density peak count

        # obtain our parameter $k$
        k = round(k_sum[stb_n_pk] / n_k[stb_n_pk]) if n_k[stb_n_pk] > 0 else 2
        return k

    def resultshow(self, data, labels, edges=None, title=""):
        scatter = plt.scatter(
            data[:, 0], data[:, 1], c=labels, cmap="viridis", marker="o"
        )
        if edges is not None:
            for i, i_nei in edges:
                plt.plot(
                    [data[i][0], data[i_nei][0]],
                    [data[i][1], data[i_nei][1]],
                    alpha=1.0,
                    c="red",
                )
        plt.colorbar(scatter, label="Cluster Label")
        plt.title(title)
        plt.show()

    def dendrogramshow(self, Z):
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z)
        plt.show()

    def plot_graph(self, X2D=None, target_num_clusters=None, ax=None):
        """
        from openTSNE import TSNE
        tsne = TSNE(
            perplexity=perplexity,
            metric='euclidean',
            n_jobs=8,
            random_state=42,
            verbose=False,
        )
        X2D = tsne.fit(self.data)
        """
        if ax is None:
            ax = plt.gca()

        if X2D is not None:
            cluster_means = corc.utils.snap_points_to_TSNE(
                points=cluster_means, data_X=self.data, transformed_X=X2D
            )
            self.graph_data["nodes"] = cluster_means

        for cm, neigh in self.graph_data["edges"]:
            ax.plot(
                [self.data[cm][0], self.data[neigh][0]],
                [self.data[cm][1], self.data[neigh][1]],
                alpha=1.0,
                c="red",
            )
