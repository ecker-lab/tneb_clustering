import numpy as np
from scipy.spatial import distance

from corc.utils import save


class GenerationModel():
    def __init__(self, center_structure, n_centers, n_samples, dim, save_file, outdir, distance=None) -> None:
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
        n_samples_around_c : int
            number of points sampled around cluster center
        """
        self.center_structure =  center_structure
        self.distance = distance
        self.n_centers = n_centers
        self.n_samples = n_samples
        self.dim = dim
        self.save_file = save_file
        self.outdir = outdir
        self.labels = []
        self.cluster_centers = []
        self.dists = []

        self.n_samples_around_c = self.n_samples // self.n_centers


    def generate(self):
        """ generate the toy dataset

        Returns
        -------
        ndarrys
            labels, cluster centers and mean distances between cluster neighbors
        """
        self.get_labels()
        self.get_cluster_centers()
        self.calc_min_dists()


    def calc_min_dists(self):
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
        self.dists = np.min(dists, axis=0)


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
    

    def get_cluster_centers(self):
        """ place cluster centers

        Returns
        -------
        ndarray, n_centers x dim
            N cluster centers of given dimensionality
        """
        if self.center_structure == 'uniform':
            self.cluster_centers = np.random.uniform(0,1,size=(self.n_centers, self.dim))
        elif self.center_structure == 'equidistant_triangle':
            assert self.n_centers == 3, f'Number of cluster centers needs to be 3 for equidistant triangle.'
            assert self.dim == 2, f'Dimension has to be 2 for equidistant triangle.'
            self.cluster_centers = self._calculate_coordinates_triangle()
        else:
            raise NotImplementedError('[ERROR] Type of placing cluster centers not valid.')


    def get_labels(self):
        """ create labels for latent embeddings according to cluster samples

        Returns
        -------
        ndarray, n_samples x 1
            labels of gt clusters; each cluster center has a label which is shared with points sampled Normal(center, std) around it
        """
        labels = []
        for ci in range(self.n_centers):
            labels.append(np.ones(self.n_samples_around_c)*ci)
        self.labels = np.array(labels).reshape(self.n_centers*self.n_samples_around_c, -1)


    def sample_embedding(self, std):
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
        latent_emb = []
        for c in self.cluster_centers:
            samples = np.random.normal(c, std, size=(self.n_samples_around_c, self.dim))
            latent_emb.append(samples)
        latent_emb = np.array(latent_emb).reshape(self.n_centers*self.n_samples_around_c, -1)
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