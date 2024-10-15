import numpy as np
from scipy.spatial import distance
import os
import random

from corc.utils import save


class GenerationModel():
    def __init__(self, center_structure, n_centers, n_samples, dim, std=None, equal_sized_clusters=True, save_file=False, outdir='.', distance=None, random_state=42) -> None:
        """ generation of toy dataset

        Parameters
        ----------
        center_structure : str
            i.e. uniform or grid
        n_centers : int
            number of cluster centers
        n_samples : int
            number of data points
        dim : int
            dimensionality of latent embedding
        std : float/list
            standard deviations of clusters. If float, this will be converted to a list of same std for each cluster.
            If list, individual std for each cluster, should have same length as n_centers.
        equal_sized_clusters : bool
            flag if clusters should contain same amount of samples or different
        save_file : bool
            if embeddings should be saved to pkl
        outdir : str
            folder name where to save data
        distance : int
            distance between cluster centers in equidistant structure

        labels : ndarray, n_samples x 1
            labels of gt clusters; each cluster center has a label which is shared with points sampled Normal(center, std) around it
        cluster_centers : ndarray, n_centers x 1
            min dists per cluster center to neighboring cluster centers
        dists :  ndarray, n_centers x 1
            min dists per cluster center to neighboring cluster centers
        """
        os.environ['PYTHONHASHSEED']=str(random_state)
        random.seed(random_state)
        np.random.seed(random_state)

        self.center_structure =  center_structure
        self.distance = distance
        self.n_centers = n_centers
        self.equal_sized_clusters = equal_sized_clusters
        self.dim = dim
        self.save_file = save_file
        self.outdir = outdir

        self.stds = std
        self._check_std(self.stds)

        self.n_samples_around_c = self._get_sample_points(n_samples)
        self.n_samples = self.n_samples_around_c.sum()

        self.labels = self._get_labels()
        self.cluster_centers = self._get_cluster_centers()
        self.dists = self._calc_min_dists()


    def _check_std(self, stds):
        if not isinstance(stds, np.ndarray) and stds is not None:
            assert isinstance(stds, list) or isinstance(stds, float), f'Std needs to be float or list.'
            self.stds = np.array(stds) if isinstance(stds, list) else np.array([stds] * self.n_centers)
            assert len(self.stds) == self.n_centers, f'Number of stds needs to be same as number of cluster centers.'


    def _get_sample_points(self, n_samples):
        if self.equal_sized_clusters:
            n_samples_around_c = np.array([n_samples // self.n_centers] * self.n_centers)
        else:
            n_samples_around_c = np.random.randint(1,n_samples//self.n_centers, size=self.n_centers-1)
            n_samples_around_c = np.concatenate((n_samples_around_c, [n_samples-n_samples_around_c.sum()]))
        return n_samples_around_c


    def _calc_min_dists(self):
        """ calculate the minimum distances of the cluster centers to each other

        Parameters
        ----------
        cluster_centers : ndarray, n_centers x dim
            points sampled uniformly in [0,1]^dim

        Returns
        -------
        ndarray, n_centers x 1
            min dists per cluster center to neighboring cluster centers
        """
        dists = distance.cdist(self.cluster_centers, self.cluster_centers) + np.eye(len(self.cluster_centers))*1000
        return np.min(dists, axis=0)


    def _calculate_coordinates_triangle(self):
        # Define the coordinates of the first point as (0, 0)
        x1, y1 = 0, 0
        # Calculate the coordinates of the second point
        x2, y2 = self.distance, 0
        # Calculate the distance between the second and third points using the Law of Cosines
        cos_theta = self.distance ** 2 / (2 * self.distance * self.distance)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        # Calculate the coordinates of the third point using the calculated angles and distances
        x3 = self.distance * cos_theta
        y3 = self.distance * sin_theta

        return np.array([[x1, y1], [x2, y2], [x3, y3]])
    

    def _get_cluster_centers(self):
        """ place cluster centers

        Returns
        -------
        ndarray, n_centers x dim
            N cluster centers of given dimensionality
        """
        if self.center_structure == 'uniform':
            cluster_centers = np.random.uniform(0,1,size=(self.n_centers, self.dim))
        elif self.center_structure == 'equidistant_triangle':
            assert self.n_centers == 3, f'Number of cluster centers needs to be 3 for equidistant triangle.'
            assert self.dim == 2, f'Dimension has to be 2 for equidistant triangle.'
            cluster_centers = self._calculate_coordinates_triangle()
        else:
            raise NotImplementedError('[ERROR] Type of placing cluster centers not valid.')
        return cluster_centers


    def _get_labels(self):
        """ create labels for latent embeddings according to cluster samples

        Returns
        -------
        ndarray, n_samples x 1
            labels of gt clusters; each cluster center has a label which is shared with points sampled Normal(center, std) around it
        """
        labels = []
        for ci, samples in enumerate(self.n_samples_around_c):
            labels = np.concatenate((labels, np.ones(samples)*ci), axis=None)
        return labels


    def sample_embedding(self, std=None):
        """ sample points normal distributed with std around cluster centers in given dimension
        and save

        Parameters
        ----------
        std : float
            standard deviation of sampled data

        Returns
        -------
        ndarray, (n_centers*n_samples_around_c) x dim
            sampled data points
        """
        if std is not None:
            print(f'Warning: self.stds was overwritten!')
        self._check_std(std)
        assert self.stds is not None, f'Stds need to be set either in init() or sample_embedding()!'

        latent_emb = []
        for (cluster_center, samples, std) in zip(self.cluster_centers, self.n_samples_around_c, self.stds):
            samples = np.random.normal(cluster_center, std, size=(samples, self.dim))
            latent_emb = np.concatenate((latent_emb, samples), axis=None)
        latent_emb = np.array(latent_emb).reshape(self.n_samples_around_c.sum(), -1)
        if self.save_file:
            save(latent_emb, filename='latent_emb', outdir=self.outdir)
        return latent_emb


    def split_data(self, latent_emb):
        """ split latent embeddings into training and test set

        Parameters
        ----------
        latent_emb : ndarray, (n_centers*n_samples_around_c) x dim
            sampled data

        Returns
        -------
        ndarray, ndarray
            train set and test set
        """
        n_train = int(len(latent_emb) * 0.9)
        idcs = np.arange(len(latent_emb))
        np.random.shuffle(idcs)
        train_latents = latent_emb[idcs[:n_train]]
        test_latents = latent_emb[idcs[n_train:]]
        return train_latents, test_latents