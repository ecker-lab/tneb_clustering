# Tneb clustering
This is the repo with code for `Hierarchical clustering with maximum density paths and mixture models` 2025 paper.
To replicate the images please go through the setup and then go to `paper_figures_and_tables/scripts` and run the bash script `bash_script_to_make_data_and_plots.sh`. After this you could also go to the notebooks and replicate the rest of the figures there.

## Setup
Clone the repo, initialize the environment and install the package locally.
The first line of the yml file sets the new environment's name `corc_env`.
```
git clone https://github.com/ecker-lab/tneb_clustering
cd tneb_clustering
conda env create -f environment.yml 
conda activate corc_env # The first line of the yml file sets the new environment's name `corc_env`.
pip install -e .
```
To pull the submodules run `git submodule update --init --recursive`.
Please note that if you want to generate densire or densired soft datasets yourself, this requires a different environment.


## Submoduled uninstallation
To remove a submodule later
```
git submodule deinit -f path/to/submodule
git rm -f path/to/submodule
rm -rf .git/modules/path/to/submodule
```
## tsnecude installation
If tsne-cuda installation causes problem, it is likely that your cuda drivers have different version. For more details and installation, see [here](https://github.com/CannyLab/tsne-cuda/blob/main/INSTALL.md).
