import subprocess
import corc.our_algorithms
import corc.our_datasets
import corc.utils
import os.path
import argparse

# TIMEOUT = "60m"
TIMEOUT = "48h"

# ALGORITHMS_TO_RUN = corc.our_algorithms.CORE_SELECTOR # just some
# ALGORITHMS_TO_RUN = corc.our_algorithms.ALGORITHM_SELECTOR # all
# ALGORITHMS_TO_RUN = ["BHC"] 
# ALGORITHMS_TO_RUN = ["UniForCE"] 
# ALGORITHMS_TO_RUN = ["TMM-NEB", "GMM-NEB"] 

# DATASETS = corc.our_datasets.DATASETS2D
DATASETS = corc.our_datasets.DATASET_SELECTOR

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dry-run",
    "--dry",
    action="store_true",
    help="Only display the generated jobs without executing them",
)
parser.add_argument(
    "--algorithm",
    "-a",
    nargs="+",
    help="Specify algorithms to run (default: all)",
    default=corc.our_algorithms.ALGORITHM_SELECTOR,
)
parser.add_argument(
    "--n_parallel",
    "-n",
    type=int,
    help="Number of parallel jobs to run (default: 32)",
    default=32,
)
args = parser.parse_args()


jobs = list()
for dataset in DATASETS:
    num_datasets = 10 if dataset in corc.our_datasets.CORE_HD_DATASETS else 1
    for index in range(num_datasets):
        for algorithm in args.algorithm:
            algorithm = algorithm.replace("\n", "")
            filename = corc.utils.get_filename(dataset, algorithm, "cache", index=index)
            if not os.path.exists(filename):
                jobs.append(
                    f"--algorithm {algorithm} --dataset {dataset} --index {index}"
                )
                # print(jobs[-1])


print("\n".join(jobs))
print(f"\nTotal jobs: {len(jobs)}\n\n")

if not args.dry_run:
    proc = subprocess.run(
        [
            "xargs",  # program name
            "-t",
            "-L",
            "1",  # one line = one command
            "-P",
            str(args.n_parallel),  # max parallel jobs
            "timeout",
            TIMEOUT,
            "python",
            "paper_figures_and_tables/scripts/compute_clustering.py",
        ],
        input="\n".join(jobs).encode(),  # {{ convert list → bytes }}
        check=False,
    )
