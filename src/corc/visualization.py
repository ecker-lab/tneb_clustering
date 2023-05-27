import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from openTSNE import TSNE
import scipy


class Visualization():
    def __init__(self, plot_histplot, plot_gaussplot,  plot_tsne,  perplexity, savefig, outdir):
        """ visualize toy data

        Parameters
        ----------
        plot_histplot : bool
            flag to create histplot or not
        plot_gaussplot : bool
            flag to create gaussplot or not
        plot_tsne : bool
            flag to create tsne plot or not
        perplexity : int
            perplexity parameter for tsne
        savefig : bool
            if figures should be saved to png and pdf
        outdir : str
            folder name where to save data
        """
        self.plot_histplot = plot_histplot
        self.plot_gaussplot = plot_gaussplot
        self.plot_tsne = plot_tsne
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
        if self.plot_histplot:
            self.histplot(dists, std, self.outdir)
        if self.plot_gaussplot:
            self.gaussplot(dists, std, self.outdir)
        if self.plot_tsne:
            self.tsne(latent_emb, labels, std, self.perplexity, self.outdir)


    def save_plot(self, fig, std, plotname, outdir):
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
        filename = f'{plotname}_{std}'
        fig.savefig(join(outdir, 'pdf', f'{filename}.pdf'))
        fig.savefig(join(outdir, 'png', f'{filename}.png'))
        fig.clear(True)


    def histplot(self, dists, std, outdir, savefig=True):
        """ plot minimum distances between cluster centers divided by standard deviation as histogram and save 

        Parameters
        ----------
        dists : ndarray, n_centers x 1
            min dists per cluster center to neighboring cluster centers
        std : float
            standard deviation of sampled data
        outdir : str
            folder name where to save plot
        """
        dists_std_ratio = dists / std
        fig = plt.figure()
        sns.histplot(data=pd.DataFrame(dists_std_ratio), legend=False, multiple='stack')
        if self.savefig:
            self.save_plot(fig, std, plotname='hist', outdir=outdir)


    def gaussplot(self, dists, std, outdir, savefig=True):
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
        dists_std_ratio = dists / std
        dist = dists_std_ratio.mean()
        x = np.linspace(0-3*std, 0+3*std, 1000)
        y = scipy.stats.norm.pdf(x, 0, std)
        fig = plt.figure()
        plt.plot(x, y, color='r')
        x2 = np.linspace(dist-3*std, dist+3*std, 1000)
        y2 = scipy.stats.norm.pdf(x2, dist, std)
        plt.plot(x2, y2, color='r')
        if self.savefig:
            self.save_plot(fig, std, plotname='gauss', outdir=outdir)


    def tsne(self, latent_emb, labels, std, perplexity, outdir, savefig=True):
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
        tsne = TSNE(n_components=2, perplexity=perplexity, metric='cosine')
        emb = tsne.fit(latent_emb).transform(latent_emb)
        check = pd.DataFrame(emb, columns=['x', 'y'])
        check['labels'] = labels
        check['sizes'] = np.ones(len(latent_emb))
        cp = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
        sns.scatterplot(data=check, x='x', y='y', hue='labels', legend=False, palette=cp, size='sizes')
        if self.savefig:
            self.save_plot(fig, std, plotname='tsne', outdir=outdir)