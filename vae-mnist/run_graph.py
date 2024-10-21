import configargparse
import pandas as pd
import numpy as np

from corc.graph_metrics import paga, gwg, gwgmara

'''
Prerequisites:
this class requires to have 
- a trained vae checkpoint at checkpoint_path (run train.py)
'''

def main():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config', required=False, is_config_file=True, 
                help='Path to config file.')
    # general
    p.add_argument('--latent_dim', type=int, default=2,
                help='Dimension of the latent embedding.')
    p.add_argument('--checkpoint_path', type=str, default='figures',
                help='Checkpoint path of VAE model.')
    p.add_argument('--graph_metric', type=str, default='gwg',
                help='Options are "gwg" and "paga".')
    # hyperparameters
    p.add_argument('--n_components', type=int, default=10,
                help='GWG hyperparameter.')
    p.add_argument('--resolution', type=float, default=0.1,
                help='PAGA hyperparameter.')
    p.add_argument('--clustering_method', type=str, default='leiden',
                help='PAGA hyperparameter.')
    

    opt = p.parse_args()

    df = pd.read_pickle(f'{opt.checkpoint_path}/vae_embeddings_latent_dim_{opt.latent_dim}.pkl')

    latents = np.stack(df['latent'].values).astype(float)
    colors = np.stack(df['label'].values)
    
    if opt.graph_metric == 'paga':
        graph = paga.PAGA(latents, colors, latent_dim=opt.latent_dim, resolution=opt.resolution, clustering_method=opt.clustering_method)
    elif opt.graph_metric == 'gwg':
        graph = gwg.GWG(latents, colors, latent_dim=opt.latent_dim, n_components=opt.n_components)
    elif opt.graph_metric == 'gwgmara':
        graph = gwgmara.GWGMara(latents, colors, latent_dim=opt.latent_dim, n_components=opt.n_components)
    else:
        raise ValueError("Graph metric is not implemented yet. Please choose between 'paga', 'gwg' and 'gwgmara'.")
    graph.create_graph()
    


if __name__=="__main__": 
   main()