import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from typing import Optional
import pyVIA.core as pyvia
from pyVIA.utils_via import *

from corc.graph_metrics.graph import Graph


class Stavia(Graph):
    def __init__(
        self,
        latent_dim,
        data=None,
        labels=None,
        path=None,
        n_neighbors=20,
        resolution=0.15,
        seed=4,
    ):
        """
        code: https://github.com/ShobiStassen/VIA

        Args:
            data (np.array): Input data.
            labels (np.array): Labels.
            latent_dim (int): Dimension of the data.
        """

        super().__init__(latent_dim, data, labels, path, seed)
        self.n_neighbors = n_neighbors
        self.edgepruning_clustering_resolution = resolution

    def fit(self, data):
        self.data = data

        self.via = pyvia.VIA(
            self.data,
            self.labels,
            edgepruning_clustering_resolution_local=1.0,
            resolution_parameter=1.0,
            edgepruning_clustering_resolution=self.edgepruning_clustering_resolution,
            knn=self.n_neighbors,
            too_big_factor=0.3,
            root_user=[0],
            dataset="",
            random_seed=self.seed,
            memory=2,
            preserve_disconnected=True,
        )
        try:
            self.via.run_VIA()
        except UnboundLocalError:
            self.labels_ = np.array([0] * len(self.data))
            print("[WARNING] caught exception")
        self.labels_ = self._get_recoloring(
            np.array(self.via.labels, dtype="int"),
            np.array(self.via.connected_comp_labels, dtype="int"),
        )

    def _get_recoloring(
        self, pred_labels, clustering
    ):  # MR: this should already be available from the graph class
        O2R = dict(zip(range(len(clustering)), clustering))
        return np.array([O2R[yp] for yp in pred_labels])

    def plot_graph(
        self,
        X2D=None,
        n_clusters=None,  # not used here
        idx: Optional[list] = None,
        draw_all_curves: bool = True,
        arrow_width_scale_factor: float = 15.0,
        linewidth: float = 1.5,
        highlight_terminal_states: bool = True,
        use_maxout_edgelist: bool = False,
    ):
        """

        projects the graph based coarse trajectory onto a umap/tsne embedding

        :param X2D: (earlier: embedding) 2d array [n_samples x 2] with x and y coordinates of all n_samples. Umap, tsne, pca OR use the via computed embedding self.via.embedding
        :param idx: default: None. Or List. if you had previously computed a umap/tsne (embedding) only on a subset of the total n_samples (subsampled as per idx), then the via objects and results will be indexed according to idx too
        :param draw_all_curves: if the clustergraph has too many edges to project in a visually interpretable way, set this to False to get a simplified view of the graph pathways
        :param arrow_width_scale_factor:
        :param linewidth:
        :param highlight_terminal_states: whether or not to highlight/distinguish the clusters which are detected as the terminal states by via
        :param use_maxout_edgelist:
        """
        embedding = X2D
        if embedding is None:
            embedding = self.data
            if embedding is None:
                print(
                    f"{datetime.now()}\t ERROR please provide an embedding or compute using via_mds() or via_umap()"
                )

        if idx is None:
            idx = np.arange(0, self.via.nsamples)
        cluster_labels = list(np.asarray(self.via.labels)[idx])
        super_cluster_labels = list(np.asarray(self.via.labels)[idx])
        super_edgelist = self.via.edgelist
        if use_maxout_edgelist == True:
            super_edgelist = self.via.edgelist_maxout
        final_super_terminal = self.via.terminal_clusters

        sub_terminal_clusters = self.via.terminal_clusters
        sc_pt_markov = list(np.asarray(self.via.single_cell_pt_markov)[idx])
        super_root = self.via.root[0]

        sc_supercluster_nn = sc_loc_ofsuperCluster_PCAspace(
            self.via, np.arange(0, len(cluster_labels))
        )
        # draw_all_curves. True draws all the curves in the piegraph, False simplifies the number of edges
        # arrow_width_scale_factor: size of the arrow head
        X_dimred = embedding  # * 1. / np.max(embedding, axis=0)
        x = X_dimred[:, 0]
        y = X_dimred[:, 1]
        max_x = np.percentile(x, 90)
        noise0 = max_x / 1000

        df = pd.DataFrame(
            {
                "x": x,
                "y": y,
                "cluster": cluster_labels,
                "super_cluster": super_cluster_labels,
                "projected_sc_pt": sc_pt_markov,
            },
            columns=["x", "y", "cluster", "super_cluster", "projected_sc_pt"],
        )
        df_mean = df.groupby("cluster", as_index=False).mean()
        sub_cluster_isin_supercluster = df_mean[["cluster", "super_cluster"]]

        sub_cluster_isin_supercluster = sub_cluster_isin_supercluster.sort_values(
            by="cluster"
        )
        sub_cluster_isin_supercluster["int_supercluster"] = (
            sub_cluster_isin_supercluster["super_cluster"].round(0).astype(int)
        )

        df_super_mean = df.groupby("super_cluster", as_index=False).mean()
        pt = df_super_mean["projected_sc_pt"].values
        num_cluster = len(set(super_cluster_labels))

        G_orange = ig.Graph(n=num_cluster, edges=super_edgelist)
        ll_ = []  # this can be activated if you intend to simplify the curves
        for fst_i in final_super_terminal:

            path_orange = G_orange.get_shortest_paths(super_root, to=fst_i)[0]
            len_path_orange = len(path_orange)
            for enum_edge, edge_fst in enumerate(path_orange):
                if enum_edge < (len_path_orange - 1):
                    ll_.append((edge_fst, path_orange[enum_edge + 1]))

        edges_to_draw = super_edgelist if draw_all_curves else list(set(ll_))
        for e_i, (start, end) in enumerate(edges_to_draw):
            if pt[start] >= pt[end]:
                start, end = end, start

            x_i_start = df[df["super_cluster"] == start]["x"].values
            y_i_start = df[df["super_cluster"] == start]["y"].values
            x_i_end = df[df["super_cluster"] == end]["x"].values
            y_i_end = df[df["super_cluster"] == end]["y"].values

            super_start_x = X_dimred[sc_supercluster_nn[start], 0]
            super_end_x = X_dimred[sc_supercluster_nn[end], 0]
            super_start_y = X_dimred[sc_supercluster_nn[start], 1]
            super_end_y = X_dimred[sc_supercluster_nn[end], 1]
            direction_arrow = -1 if super_start_x > super_end_x else 1

            minx = min(super_start_x, super_end_x)
            maxx = max(super_start_x, super_end_x)

            miny = min(super_start_y, super_end_y)
            maxy = max(super_start_y, super_end_y)

            x_val = np.concatenate([x_i_start, x_i_end])
            y_val = np.concatenate([y_i_start, y_i_end])

            idx_keep = np.where((x_val <= maxx) & (x_val >= minx))[0]
            idy_keep = np.where((y_val <= maxy) & (y_val >= miny))[0]

            idx_keep = np.intersect1d(idy_keep, idx_keep)

            x_val = x_val[idx_keep]
            y_val = y_val[idx_keep]

            super_mid_x = (super_start_x + super_end_x) / 2
            super_mid_y = (super_start_y + super_end_y) / 2
            from scipy.spatial import distance

            very_straight = False
            straight_level = 3
            noise = noise0
            x_super = np.array(
                [
                    super_start_x,
                    super_end_x,
                    super_start_x,
                    super_end_x,
                    super_start_x,
                    super_end_x,
                    super_start_x,
                    super_end_x,
                    super_start_x + noise,
                    super_end_x + noise,
                    super_start_x - noise,
                    super_end_x - noise,
                ]
            )
            y_super = np.array(
                [
                    super_start_y,
                    super_end_y,
                    super_start_y,
                    super_end_y,
                    super_start_y,
                    super_end_y,
                    super_start_y,
                    super_end_y,
                    super_start_y + noise,
                    super_end_y + noise,
                    super_start_y - noise,
                    super_end_y - noise,
                ]
            )

            if abs(minx - maxx) <= 1:
                very_straight = True
                straight_level = 10
                x_super = np.append(x_super, super_mid_x)
                y_super = np.append(y_super, super_mid_y)

            for i in range(straight_level):  # DO THE SAME FOR A MIDPOINT TOO
                y_super = np.concatenate([y_super, y_super])
                x_super = np.concatenate([x_super, x_super])

            list_selected_clus = list(zip(x_val, y_val))

            if len(list_selected_clus) >= 1 & very_straight:
                dist = distance.cdist(
                    [(super_mid_x, super_mid_y)], list_selected_clus, "euclidean"
                )
                k = min(2, len(list_selected_clus))
                midpoint_loc = dist[0].argsort()[:k]

                midpoint_xy = []
                for i in range(k):
                    midpoint_xy.append(list_selected_clus[midpoint_loc[i]])

                noise = noise0 * 2

                if k == 1:
                    mid_x = np.array(
                        [
                            midpoint_xy[0][0],
                            midpoint_xy[0][0] + noise,
                            midpoint_xy[0][0] - noise,
                        ]
                    )
                    mid_y = np.array(
                        [
                            midpoint_xy[0][1],
                            midpoint_xy[0][1] + noise,
                            midpoint_xy[0][1] - noise,
                        ]
                    )
                if k == 2:
                    mid_x = np.array(
                        [
                            midpoint_xy[0][0],
                            midpoint_xy[0][0] + noise,
                            midpoint_xy[0][0] - noise,
                            midpoint_xy[1][0],
                            midpoint_xy[1][0] + noise,
                            midpoint_xy[1][0] - noise,
                        ]
                    )
                    mid_y = np.array(
                        [
                            midpoint_xy[0][1],
                            midpoint_xy[0][1] + noise,
                            midpoint_xy[0][1] - noise,
                            midpoint_xy[1][1],
                            midpoint_xy[1][1] + noise,
                            midpoint_xy[1][1] - noise,
                        ]
                    )
                for i in range(3):
                    mid_x = np.concatenate([mid_x, mid_x])
                    mid_y = np.concatenate([mid_y, mid_y])

                x_super = np.concatenate([x_super, mid_x])
                y_super = np.concatenate([y_super, mid_y])
            x_val = np.concatenate([x_val, x_super])
            y_val = np.concatenate([y_val, y_super])

            x_val = x_val.reshape((len(x_val), -1))
            y_val = y_val.reshape((len(y_val), -1))
            xp = np.linspace(minx, maxx, 500)

            gam50 = pg.LinearGAM(n_splines=4, spline_order=3, lam=10).gridsearch(
                x_val, y_val
            )
            XX = gam50.generate_X_grid(term=0, n=500)
            preds = gam50.predict(XX)

            idx_keep = np.where((xp <= (maxx)) & (xp >= (minx)))[0]
            plt.plot(XX, preds, linewidth=linewidth, c="#323538")  # 3.5#1.5

            mean_temp = np.mean(xp[idx_keep])
            closest_val = xp[idx_keep][0]
            closest_loc = idx_keep[0]

            for i, xp_val in enumerate(xp[idx_keep]):
                if abs(xp_val - mean_temp) < abs(closest_val - mean_temp):
                    closest_val = xp_val
                    closest_loc = idx_keep[i]
            step = 1

            head_width = (
                noise * arrow_width_scale_factor
            )  # arrow_width needs to be adjusted sometimes # 40#30  ##0.2 #0.05 for mESC #0.00001 (#for 2MORGAN and others) # 0.5#1
            if direction_arrow == 1:
                plt.arrow(
                    xp[closest_loc],
                    preds[closest_loc],
                    xp[closest_loc + step] - xp[closest_loc],
                    preds[closest_loc + step] - preds[closest_loc],
                    shape="full",
                    lw=0,
                    length_includes_head=False,
                    head_width=head_width,
                    color="#323538",
                )

            else:
                plt.arrow(
                    xp[closest_loc],
                    preds[closest_loc],
                    xp[closest_loc - step] - xp[closest_loc],
                    preds[closest_loc - step] - preds[closest_loc],
                    shape="full",
                    lw=0,
                    length_includes_head=False,
                    head_width=head_width,
                    color="#323538",
                )

        c_edge = []
        width_edge = []
        pen_color = []
        super_cluster_label = []
        terminal_count_ = 0
        dot_size = []

        for i in sc_supercluster_nn:
            if i in final_super_terminal:
                print(
                    f"{datetime.now()}\tSuper cluster {i} is a super terminal with sub_terminal cluster",
                    sub_terminal_clusters[terminal_count_],
                )
                c_edge.append("yellow")  # ('yellow')
                if highlight_terminal_states == True:
                    width_edge.append(2)
                    super_cluster_label.append(
                        "TS" + str(sub_terminal_clusters[terminal_count_])
                    )
                else:
                    width_edge.append(0)
                    super_cluster_label.append("")
                pen_color.append("black")
                dot_size.append(20)  # 60
                terminal_count_ = terminal_count_ + 1
            else:
                width_edge.append(0)
                c_edge.append("black")
                pen_color.append("red")
                super_cluster_label.append(str(" "))  # i or ' '
                dot_size.append(00)  # 20

        count_ = 0
        loci = [sc_supercluster_nn[key] for key in sc_supercluster_nn]
        for i, c, w, pc, dsz, lab in zip(
            loci, c_edge, width_edge, pen_color, dot_size, super_cluster_label
        ):
            plt.scatter(
                X_dimred[i, 0], X_dimred[i, 1], c="black", s=dsz, linewidth=w
            )  # edgecolors=c)
            count_ = count_ + 1
