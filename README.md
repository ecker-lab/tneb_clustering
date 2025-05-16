# Tneb clustering
This is the repo with code for "Hierarchical clustering with maximum density paths and mixture models" 2025 paper.  
To replicate the images please go through the setup and then go to `paper_figures_and_tables/scripts` and run the bash script `bash_script_to_make_data_and_plots.sh`. After this you could also go to the notebooks (both `notebooks` and `paper_figures_and_tables/notebooks`) to replicate the rest of the figures.

## Setup
Clone the repo, initialize the environment and install the package locally.
The first line of the yml file sets the new environment's name `tneb`. This can be overwritten through `-n`.
```
git clone --recurse-submodules https://github.com/ecker-lab/tneb_clustering
cd tneb_clustering
conda env create -f environment.yml 
conda activate tneb 
```

## tsne-cuda installation
If tsne-cuda installation causes problem, it is likely that your cuda drivers have different version. For more details and installation, see [here](https://github.com/CannyLab/tsne-cuda/blob/main/INSTALL.md).

## UniForCE

To install external method UniForCE, go to the folder
```
cd external/UniForCE
```
and install repo with
```
find . -type f -name "*.py" -print0 | while IFS= read -r -d '' file; do   sed -i 's/from code\./from /g; s/import code\./import /g' "$file";  done
pip install -e code
```
If the folder external/UniForCE does not exist (for example when clonding without the `--recurse-submodules` option), try calling `git submodule init` and `git submodule update --checkout`.


## Recreating the densired datasets

Since densired relies on modern packages while some baselines necessitate `numpy<2`, an additional environment is needed for running the code in the `densired.ipynb` notebook. It can be installed using `conda env create -f environment_densired.yml` which creates the environment `densired`. This step is _not_ necessary to reproduce our results as we provide an precompiled `.npz` version of the densired datasets.
