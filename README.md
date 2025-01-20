# cluster vs continuum
Finding methods for assessing clustering tendency.

## Setup

Go into folder
```
cd cluster_vs_continuum
```
To pull the submodules run
```
git submodule update --init --recursive
```
Create the environment from the environment.yaml file:
```
conda env create -f environment.yml
```

The first line of the yml file sets the new environment's name `corc_env`.

Activate the new environment:
```
conda activate corc_env
```

Verify that the new environment was installed correctly:
```
conda env list
```

Install repo with
```
pip install -e .
```

## Notebooks

There is a folder with example notebooks for different clustering tendency assessment algorithms.

## Automatic Notebook Clean-up

Add this to `.git/config`:
```bash
[filter "clean-notebook-output"]
    clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
```
and add a file `.gitattributes` with the following content:
```
**/*.ipynb filter=clean-notebook-output
```

## Submoduled uninstallation
To remove a submodule later
```
git submodule deinit -f path/to/submodule
git rm -f path/to/submodule
rm -rf .git/modules/path/to/submodule
```
## tsnecude installation
If tsne-cuda installation causes problem, it is likely that your cuda drivers have different version. For more details and installation, see [here](https://github.com/CannyLab/tsne-cuda/blob/main/INSTALL.md).
