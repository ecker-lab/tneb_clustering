from pathlib import Path
import matplotlib.pyplot as plt


class Graph():
    def __init__(self, latent_dim, data=None, labels=None, path=None, seed=42):
        """
        Initialize the Graph class.

        Args:
            data (np.array): Input data.
            labels (np.array): Labels.
            latent_dim (int): Dimension of the data.
        """
        self.data = data
        self.labels = labels
        self.latent_dim = latent_dim
        self.seed = seed

        self.path = Path('figures') if path is None else path

        self.graph_data = {'nodes':None, 'edges':None, 'nodes_org_space':None}


    def create_graph(self, save=True, *args, **kwargs):
        """
        Abstract method to create a graph.

        This method should be implemented by subclasses.
        """
        pass


    def save_graph(self, file_name):
        """
        Save the graph to a file.

        Args:
            file_name (str): The file name where the graph should be saved.
        """
        import pickle

        if self.graph_data["nodes"] is None:
            raise ValueError("Graph data is not created yet. Please create the graph before saving.")
        
        with open(self.path / f'graph{file_name}', 'wb') as f:
            pickle.dump(self.graph_data, f)

        print(f"Graph saved to {self.path / f'graph{file_name}'}.")


    def plot_graph(self, X2D=None):
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
        cluster_means = self.graph_data['nodes']

        if X2D is not None:
            cluster_means = X2D.transform(cluster_means)
        self.graph_data['nodes'] = cluster_means

        plt.scatter(*cluster_means.T, alpha=1.0, rasterized=True, s=15, c='black')

        for (cm, neigh), weight in self.graph_data['edges'].items():
            plt.plot(
                [cluster_means[cm][0], cluster_means[neigh][0]],
                [cluster_means[cm][1], cluster_means[neigh][1]],
                alpha=weight,
                c="black",
            )
