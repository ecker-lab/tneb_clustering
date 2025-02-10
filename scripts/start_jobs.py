import os
import subprocess
import corc.our_datasets
import corc.our_algorithms
import argparse


def start_main_jobs():
    # check for missing files before starting the computation
    missing_files = []
    for dataset_name in corc.our_datasets.DATASET_SELECTOR:
        dataset_filename = f"{cache_path}/{dataset_name}.pickle"

        for algorithm_name in corc.our_algorithms.ALGORITHM_SELECTOR:
            algorithm_name = algorithm_name.replace("\\n", "\n").replace("\n", "")
            alg_filename = f"{cache_path}/{dataset_name}_{algorithm_name}.pickle"
            if not os.path.exists(alg_filename):
                missing_files.append(dataset_name)
                break

    # starting jobs
    for dataset_name in missing_files:
        subprocess.run(
            f"sbatch --job-name='NEB {dataset_name}' scripts/cluster_pickles.sh {dataset_name}",
            shell=True,
        )
        print(f"Started job for {dataset_name}")


def start_seeds_jobs(n_components, gmm):
    gmm_string = "_gmm" if gmm else ""

    for dataset_name in corc.our_datasets.DATASET_SELECTOR:
        dataset_filename = dataset_name.replace(" ", "_")
        filename = f"{cache_path}/stability/seeds_{dataset_filename}{gmm_string}_{n_components}.pkl"
        if not os.path.exists(filename):
            if not gmm:
                subprocess.run(
                    f"sbatch --job-name='seeds {dataset_name} tmm' scripts/cluster_stability.sh -t seeds --n_components {n_components}  -d {dataset_name}",
                    shell=True,
                )
            else:
                subprocess.run(
                    f"sbatch --job-name='seeds {dataset_name} gmm' scripts/cluster_stability.sh -t seeds --n_components {n_components}  -d {dataset_name} --gmm",
                    shell=True,
                )
            print(f"Started job for {dataset_name}")


def start_overclustering_jobs(gmm):
    gmm_string = "_gmm" if gmm else ""

    for dataset_name in corc.our_datasets.COMPLEX_DATASETS:
        dataset_filename = dataset_name.replace(" ", "_")
        filename = (
            f"{cache_path}/stability/overclustering_{dataset_filename}{gmm_string}.pkl"
        )
        if not os.path.exists(filename):
            if not gmm:
                subprocess.run(
                    f"sbatch --job-name='overclustering {dataset_name} tmm' scripts/cluster_stability.sh -t overclustering -d {dataset_name}",
                    shell=True,
                )
            else:
                subprocess.run(
                    f"sbatch --job-name='overclustering {dataset_name} gmm' scripts/cluster_stability.sh -t overclustering -d {dataset_name} --gmm",
                    shell=True,
                )
            print(f"Started job for {dataset_name}")


parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--type",
    choices=["seeds", "overclustering", "main"],
    default="main",
    help="which jobs to start",
)
parser.add_argument(
    "-c",
    "--n_components",
    type=int,
    default=25,
    help="number of components for seeds",
)
args = parser.parse_args()


cache_path = "cache"

if args.type == "main":
    start_main_jobs()
elif args.type == "seeds":
    start_seeds_jobs(n_components=args.n_components, gmm=False)
    start_seeds_jobs(n_components=args.n_components, gmm=True)
elif args.type == "overclustering":
    start_overclustering_jobs(gmm=False)
    start_overclustering_jobs(gmm=True)
