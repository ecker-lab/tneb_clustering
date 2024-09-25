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


    def create_graph(self, save=True, plot=True, return_graph=False, *args, **kwargs):
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
