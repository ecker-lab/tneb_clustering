import yaml
from os.path import join
import numpy as np
import pandas as pd
from tqdm import tqdm

from corc import utils
from corc.visualization import Visualization
from corc.generation import GenerationModel
from corc.metrics import Metric
from corc.clustering import Clustering


def main():
    with open('configs/config.yaml') as f:
        opt = yaml.safe_load(f)

    experiment_name = opt['experiment_name']
    outdir = join('logs', experiment_name)

    utils.cond_mkdir(outdir)
    utils.cond_mkdir(join(outdir,'png'))
    utils.cond_mkdir(join(outdir,'pdf'))

    with open(join(outdir,'config.yaml'), 'w') as f:
        yaml.dump(opt, f)

    vis = Visualization(**opt['visualization'], outdir=outdir)
    gen = GenerationModel(**opt['generation'], outdir=outdir)
    metric = Metric(**opt['metric'], K=opt['K'], outdir=outdir)
    clustering = Clustering(**opt['clustering'])

    # execute clustering and metric calculation for K times
    for k in tqdm(range(opt['K'])):
        # cluster centers are the same for each std
        for s, std in enumerate(opt['metric']['stds']):
            # create data
            gen.stds = std
            latent_emb = gen.sample_embedding()
            train_latents, test_latents = gen.split_data(latent_emb)

            # plot once
            if k==0:
                vis.visualize(gen.dists, std, latent_emb, gen.labels)

            # run clustering on train/test split
            predictions, scores = clustering.run(train_latents, test_latents)

            # calculate metric
            metric.calculate(train_latents, test_latents, predictions, scores, k, s)

    # save metrics
    metric.summarize()

    # generate overview plots
    utils.generate_overview_lineplot(outdir)
    utils.generate_overview_visplot(outdir)


if __name__ == '__main__':
    main()