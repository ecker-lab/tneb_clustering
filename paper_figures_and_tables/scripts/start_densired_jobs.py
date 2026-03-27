import os
import subprocess
import corc.our_algorithms

# import argparse

# check missing files

# parser = argparse.ArgumentParser()
dry_run = False
# dry_run = True
if dry_run:
    print("dry run!")
algorithm_names = corc.our_algorithms.CORE_SELECTOR

dims = [8, 16, 32, 64]

types = ["circles", "studt"]

for algo in algorithm_names:
    algo = algo.replace("\n", "")
    for dim in dims:
        for type in types:
            call_string = (
                f"sbatch --job-name 'densired_{type}_{dim}_{algo}' paper_figures_and_tables/scripts/slurm_basis.sh --dataset {type} --dim {dim} --algorithm {algo}",
            )
            target_file = f"cache/densired10/{type}{dim}_{algo}.pickle"
            if not os.path.exists(target_file):
                if dry_run:
                    print(call_string)
                else:
                    subprocess.run(call_string, shell=True)
            else:
                print(f"file {target_file} already exists, skipping.")
