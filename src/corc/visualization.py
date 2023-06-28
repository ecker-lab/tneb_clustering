import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from openTSNE import TSNE
import scipy


class Visualization():
    def __init__(self, plot_data=False, plot_histplot=False, plot_gaussplot=False,  plot_tsne=False,  plot_ivat=False, perplexity=30, savefig=False, outdir='logs'):
        """ visualize toy data

        Parameters
        ----------
        plot_data : bool
            flag to create scatterplot of data; only possible in 2d
        plot_histplot : bool
            flag to create histplot
        plot_gaussplot : bool
            flag to create gaussplot
        plot_tsne : bool
            flag to create tsne plot
        plot_ivat : bool
            flag to create iVAT plot
        perplexity : int
            perplexity parameter for tsne
        savefig : bool
            if figures should be saved to png and pdf
        outdir : str
            folder name where to save data
        """
        self.plot_data = plot_data
        self.plot_histplot = plot_histplot
        self.plot_gaussplot = plot_gaussplot
        self.plot_tsne = plot_tsne
        self.plot_ivat = plot_ivat
        self.perplexity = perplexity
        self.savefig = savefig
        self.outdir = outdir


    def visualize(self, dists, std, latent_emb, labels):
        """visualize different plots

        Parameters
        ----------
        dists : ndarray, n_centers x 1
            min dists per cluster center to neighboring cluster centers
        std : float
            standard deviation of sampled data
        latent_emb : ndarray, (n_centers*n_samples_around_c) x dim
            sampled data points
        labels : ndarray, n_samples x 1
            labels of gt clusters; each cluster center has a label which is shared with points sampled Normal(center, std) around it
        """
        if self.plot_data and latent_emb.shape[1] == 2:
            self.dataplot(latent_emb, labels, std)
        if self.plot_histplot:
            self.histplot(dists, std)
        if self.plot_gaussplot:
            self.gaussplot(dists, std)
        if self.plot_tsne:
            self.tsne(latent_emb, labels, std)
        if self.plot_ivat:
            self.ivat(latent_emb, std)


    def save_plot(self, fig, std, plotname):
        """ save figure of plot in folder under given name

        Parameters
        ----------
        fig : plt.figure()
            figure of plot
        std : float
            standard deviation of sampled data
        plotname : str
            name under which plot is saved
        outdir : str
            folder name where to save plot
        """
        filename = f'{plotname}_{std:.2f}'
        fig.savefig(join(self.outdir, 'pdf', f'{filename}.pdf'))
        fig.savefig(join(self.outdir, 'png', f'{filename}.png'))
        fig.clear(True)


    def dataplot(self, data, labels, std):
        fig = plt.figure()
        df = pd.DataFrame()
        df['x'], df['y'], df['label'] = data[:,0], data[:,1], labels
        sns.scatterplot(data=df, x='x', y='y', hue='label', legend=False)
        if self.savefig:
            self.save_plot(fig, std, plotname='data')
        plt.close()


    def histplot(self, dists, std):
        """ plot minimum distances between cluster centers divided by standard deviation as histogram and save 

        Parameters
        ----------
        dists : ndarray, n_centers x 1
            min dists per cluster center to neighboring cluster centers
        std : float
            standard deviation of sampled data
        """
        dists_std_ratio = dists / std
        fig = plt.figure()
        sns.histplot(data=pd.DataFrame(dists_std_ratio), legend=False, multiple='stack')
        if self.savefig:
            self.save_plot(fig, std, plotname='hist')
        plt.close()


    def gaussplot(self, dists, std):
        """ plot Gaussian(0, std) and Gaussian(mean distance to standard deviation ratio, std) to visualize closeness and overlap of sampled data
        and save

        Parameters
        ----------
        dists : ndarray, n_centers x 1
            min dists per cluster center to neighboring cluster centers
        std : float
            standard deviation of sampled data
        outdir : str
            folder name where to save plot
        """
        dists_std_ratio = dists #/ std
        dist = dists_std_ratio.mean()
        x = np.linspace(0-3*std, 0+3*std, 1000)
        y = scipy.stats.norm.pdf(x, 0, std)
        fig = plt.figure()
        plt.plot(x, y, color='r')
        x2 = np.linspace(dist-3*std, dist+3*std, 1000)
        y2 = scipy.stats.norm.pdf(x2, dist, std)
        plt.plot(x2, y2, color='r')
        if self.savefig:
            self.save_plot(fig, std, plotname='gauss')
        plt.close()


    def tsne(self, latent_emb, labels, std):
        """ t-SNE with colors according to gt clusters
        and save

        Parameters
        ----------
        latent_emb : ndarray, (n_centers*n_samples_around_c) x dim
            sampled data
        labels : ndarray, n_samples x 1
            labels of gt clusters; each cluster center has a label which is shared with points sampled Normal(center, std) around it
        std : float
            standard deviation of sampled data
        perplexity : int
            value for perplexity parameter of t-SNE
        outdir : str
            folder name where to save plot
        """
        fig = plt.figure()
        tsne = TSNE(n_components=2, perplexity=self.perplexity)
        emb = tsne.fit(latent_emb).transform(latent_emb)
        check = pd.DataFrame(emb, columns=['x', 'y'])
        check['labels'] = labels
        check['sizes'] = np.ones(len(latent_emb))
        cp = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
        sns.scatterplot(data=check, x='x', y='y', hue='labels', legend=False, palette=cp, size='sizes')
        if self.savefig:
            self.save_plot(fig, std, plotname='tsne')
        plt.close()


        # https://pyclustertend.readthedocs.io/en/latest/pyclustertend.html#pyclustertend.visual_assessment_of_tendency.ivat
    def ivat(self, latent_emb, std):
        """iVat return a visualisation based on the Vat but more reliable and easier to
        interpret.

        Parameters
        ----------
        latent_emb : ndarray, (n_centers*n_samples_around_c) x dim
            sampled data
        """

        def _compute_ordered_dissimilarity_matrix(X):
            """The ordered dissimilarity matrix is used by visual assesement of tendency. It is a just a a reordering
            of the dissimilarity matrix.

            Parameters
            ----------
            X : matrix
                numpy array

            Return
            -------
            ODM : matrix
                the ordered dissimalarity matrix .
            """

            # Step 1 :
            from sklearn.metrics import pairwise_distances

            observation_path = []

            matrix_of_pairwise_distance = pairwise_distances(X)
            list_of_int = np.zeros(matrix_of_pairwise_distance.shape[0], dtype="int")

            index_of_maximum_value = np.argmax(matrix_of_pairwise_distance)

            column_index_of_maximum_value = index_of_maximum_value // matrix_of_pairwise_distance.shape[1]

            list_of_int[0] = column_index_of_maximum_value
            observation_path.append(column_index_of_maximum_value)

            K = np.linspace(0, matrix_of_pairwise_distance.shape[0] - 1, matrix_of_pairwise_distance.shape[0], dtype="int")
            J = np.delete(K, column_index_of_maximum_value)

            # Step 2 :
            for r in range(1, matrix_of_pairwise_distance.shape[0]):

                p, q = (-1, -1)

                mini = np.max(matrix_of_pairwise_distance)

                for candidate_p in observation_path:
                    for candidate_j in J:
                        if matrix_of_pairwise_distance[candidate_p, candidate_j] < mini:
                            p = candidate_p
                            q = candidate_j
                            mini = matrix_of_pairwise_distance[p, q]

                list_of_int[r] = q
                observation_path.append(q)

                ind_q = np.where(np.array(J) == q)[0][0]
                J = np.delete(J, ind_q)

            # Step 3
            ordered_matrix = np.zeros(matrix_of_pairwise_distance.shape)

            for column_index_of_maximum_value in range(ordered_matrix.shape[0]):
                for j in range(ordered_matrix.shape[1]):
                    ordered_matrix[column_index_of_maximum_value, j] = matrix_of_pairwise_distance[
                        list_of_int[column_index_of_maximum_value], list_of_int[j]]

            return ordered_matrix

        def _compute_ivat_ordered_dissimilarity_matrix(X):
            """The ordered dissimilarity matrix is used by ivat. It is a just a a reordering
            of the dissimilarity matrix.

            Parameters
            ----------
            X : matrix
                numpy array

            Return
            -------
            D_prim : matrix
                the ordered dissimalarity matrix .
            """

            ordered_matrix = _compute_ordered_dissimilarity_matrix(X)
            re_ordered_matrix = np.zeros((ordered_matrix.shape[0], ordered_matrix.shape[0]))

            for r in range(1, ordered_matrix.shape[0]):
                # Step 1 : find j for which D[r,j] is minimum and j in [1:r-1]
                j = np.argmin(ordered_matrix[r, 0:r])

                # Step 2 :
                re_ordered_matrix[r, j] = ordered_matrix[r, j]

                # Step 3 : pour c : 1,r-1 avec c !=j
                c_tab = np.array(range(0, r))
                c_tab = c_tab[c_tab != j]

                for c in c_tab:
                    re_ordered_matrix[r, c] = max(ordered_matrix[r, j], re_ordered_matrix[j, c])
                    re_ordered_matrix[c, r] = re_ordered_matrix[r, c]

            return re_ordered_matrix


        ordered_matrix = _compute_ivat_ordered_dissimilarity_matrix(latent_emb)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(ordered_matrix, cmap='gray', vmin=0, vmax=np.max(ordered_matrix))

        if self.savefig:
            self.save_plot(fig, std, plotname='iVAT')
        plt.close()
