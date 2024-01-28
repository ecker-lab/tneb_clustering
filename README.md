# cluster vs continuum
Finding methods for assessing clustering tendency.

## Setup

Go into folder
```
cd cluster_vs_continuum
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

## Run experiments

Experiments include the generation of toy data, visualization of data distribution, clustering and calculating a metric to assess clustering tendency.

Start program with
```
python main.py
```
which loads parameters set in `configs/config.yaml`.

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