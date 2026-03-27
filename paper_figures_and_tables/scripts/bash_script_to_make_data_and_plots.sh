#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.01

python paper_figures_and_tables/scripts/dataset_preparation.py

# to be run on a GPU machine
echo """
noisy_circles
noisy_moons
varied
aniso
blobs
clusterlab10
blobs1_8
blobs1_16
blobs1_32
blobs1_64
blobs2_8
blobs2_16
blobs2_32
blobs2_64
densired8
densired16
densired32
densired64
densired_soft_8
densired_soft_16
densired_soft_32
densired_soft_64
mnist8
mnist16
mnist32
mnist64
""" | xargs -P 6 -I {} nice -6 python paper_figures_and_tables/scripts/compute_clustering_pickles.py --algorithms ours --datasets {}

# computing all remaining algorithms (this may happen on a CPU machine)
echo """
noisy_circles
noisy_moons
varied
aniso
blobs
clusterlab10
blobs1_8
blobs1_16
blobs1_32
blobs1_64
blobs2_8
blobs2_16
blobs2_32
blobs2_64
densired8
densired16
densired32
densired64
densired_soft_8
densired_soft_16
densired_soft_32
densired_soft_64
mnist8
mnist16
mnist32
mnist64
""" | xargs -P 6 -I {} nice -6 python paper_figures_and_tables/scripts/compute_clustering_pickles.py --datasets {} 


python paper_figures_and_tables/scripts/04_05_create_clustering_figure.py -d main1

python paper_figures_and_tables/scripts/04_05_create_clustering_figure.py -d main2

python paper_figures_and_tables/scripts/04_05_create_clustering_figure.py -d fig1

python paper_figures_and_tables/scripts/04_05_create_clustering_figure.py -d fig2


# arranged the 64d datasets to be in front because they take longest (by far)
echo """
densired_soft_64
mnist64
densired64
blobs1_64
blobs2_64
noisy_circles
noisy_moons
varied
aniso
blobs
clusterlab10
blobs1_8
blobs1_16
blobs1_32
blobs2_8
blobs2_16
blobs2_32
densired8
densired16
densired32
densired_soft_8
densired_soft_16
densired_soft_32
mnist8
mnist16
mnist32
""" | xargs -P 16 -I {} nice -6 python paper_figures_and_tables/scripts/stability_plots.py -t overclustering -d {}

python paper_figures_and_tables/scripts/stability_plots.py -t seeds -d core

echo """
noisy_circles
noisy_moons
varied
aniso
blobs
clusterlab10
blobs1_8
blobs1_16
blobs1_32
blobs1_64
blobs2_8
blobs2_16
blobs2_32
blobs2_64
densired8
densired16
densired32
densired64
densired_soft_8
densired_soft_16
densired_soft_32
densired_soft_64
mnist8
mnist16
mnist32
mnist64
""" | xargs -P 6 -I {} nice -6 python paper_figures_and_tables/scripts/stability_plots.py -t overclustering -d {} --gmm


echo """
varied
aniso
densired8
densired16
densired_soft_8
densired_soft_16
mnist8
mnist16
""" | xargs -P 6 -I {} nice -6 python paper_figures_and_tables/scripts/plot_join_strategies.py -d {}

# num_components (calling all 9 datasets in parallel)
for dataset in densired8 densired16 densired32 densired_soft_8 densired_soft_16 densired_soft_32 mnist8 mnist16 mnist32; do python paper_figures_and_tables/scripts/ablation_num_components.py --datasets "$dataset" & done; wait
