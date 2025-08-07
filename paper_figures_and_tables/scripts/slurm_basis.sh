#!/bin/bash
#SBATCH --partition=standard96s:shared
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=FAIL
#SBATCH --time=8:00:00

pwd
echo $@
source ~/.bashrc
export DISABLE_TQDM=True

micromamba activate tneb
python paper_figures_and_tables/scripts/$@
# python paper_figures_and_tables/scripts/densired10_individual_pickles.py $@
# python paper_figures_and_tables/scripts/densired10_clustering_pickles.py $@