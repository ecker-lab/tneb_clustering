import os
import subprocess
import corc.our_algorithms

dry_run = False  # set to True to preview only

algorithm_names = ["TMM-NEB", "GMM-NEB"]
dims = [32, 64]
types = ["circles", "studt"]

for algo in algorithm_names:
    algo_clean = algo.replace("\n", "")  # avoid linebreaks
    for dim in dims:
        for dtype in types:
            # File containing all results for this setting
            aggregate_file = f"cache/densired10/{dtype}{dim}_{algo_clean}.pickle"
            # If clustered file exists, do not run ANY jobs for this combo
            if os.path.exists(aggregate_file):
                print(f"{aggregate_file} already exists, skipping all individuals.")
                continue
            for i in range(10):
                for j in range(10):
                    target_file = f"cache/densired10/individuals/{dtype}{dim}_{algo_clean}_i{i}_j{j}.pickle"
                    if not os.path.exists(target_file):
                        jobname = f"ind_{dtype}_{dim}_{algo_clean}_i{i}_j{j}"
                        call_string = (
                            f"sbatch --job-name '{jobname}' "
                            "paper_figures_and_tables/scripts/slurm_basis.sh "
                            f"--dataset {dtype} --dim {dim} --algorithm {algo_clean} --i {i} --j {j}"
                        )
                        if dry_run:
                            print(call_string)
                        else:
                            subprocess.run(call_string, shell=True)
                    else:
                        print(f"{target_file} already exists, skipping.")
