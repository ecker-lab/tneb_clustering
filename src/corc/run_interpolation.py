import torch
from datetime import datetime
import configargparse
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import torch
import pickle
from pathlib import Path
import os

from vae import VAE

hidden_dim = 512
n = 15
path = Path('figures')

'''
Prerequisites:
this class requires to have 
- a trained vae checkpoint at checkpoint_path (run mnist-nd/pytorch/main.py)
- a graph created at graph_path (run run_graph.py)

Example call:
python run_interpolation.py --latent_dim 2 --checkpoint_path /usr/users/pede1/morphology/cluster_vs_continuum/vae-mnist/runs/mnist/vae_20240729-134322 --graph_path /usr/users/pede1/morphology/cluster_vs_continuum/vae-mnist/figures/graph_10_vae_latent_dim_2_20240729-161818.pkl --s_node 9 --e_node 7
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
    p.add_argument('--graph_path', type=str, default='figures',
                help='Path where graph is saved.')
    p.add_argument('--s_node', type=int, default=0,
                help='Start node of edge between s_node and e_node where to interpolate between.')
    p.add_argument('--e_node', type=int, default=1,
                help='End node of edge between s_node and e_node where to interpolate between.')
    opt = p.parse_args()

    print(f'[INFO] Latent dimension is set to {opt.latent_dim}.')

    model = load_model(opt.checkpoint_path, opt.latent_dim)
    _, _, nodes_org_space = load_graph(opt.graph_path)

    gn = os.path.splitext((opt.graph_path).split('/')[-1])[0].replace('.', '')
    filename = f'_{gn}'
    interpolate(model, nodes_org_space, (opt.s_node, opt.e_node), opt.latent_dim, save=filename)



def load_model(checkpoint_path, latent_dim):
    model = VAE(input_dim=784, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(f'{checkpoint_path}/model_final_epoch.pth'))
    return model


def load_graph(graph_path):
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    return graph['nodes'], graph['edges'], graph['nodes_org_space']


def interpolate(model, nodes, edge, latent_dim, save=None):
    c1 = torch.Tensor(nodes[edge[0]])
    c2 = torch.Tensor(nodes[edge[1]])
    z = torch.zeros((n, latent_dim))
    for i, step in enumerate(torch.linspace(0,1,n)):
        z[i] = c1*(1-step)+c2*step
    samples = model.decode(z)
    samples = torch.sigmoid(samples)

    # Plot the generated images
    fig, ax = plt.subplots(1, n, figsize=(n, 1))
    for i in range(n):
        ax[i].imshow(samples[i].view(28, 28).cpu().detach().numpy(), cmap='gray')
        ax[i].axis('off')
    
    if save:
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / f'interpolation_between_{edge[0]}_and_{edge[1]}{save}', bbox_inches='tight')
    plt.close()

    

if __name__=="__main__": 
   main()