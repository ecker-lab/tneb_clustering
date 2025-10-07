import os
import subprocess
import corc.our_datasets


dry_run = False
# dry_run = True

if dry_run:
    print("dry run!")
datasets = corc.our_datasets.CORE_HD_DATASETS

size = 200

for dataset in datasets:
    # algo = algo.replace("\n", "")
    call_string = (
        f"sbatch --job-name 'bhc_{dataset}' paper_figures_and_tables/scripts/slurm_basis.sh run_bhc.py  {dataset} {size}",
    )
    target_file = f"cache/bhc/{dataset}_{size}.pkl"
    if not os.path.exists(target_file):
        if dry_run:
            print(call_string)
        else:
            subprocess.run(call_string, shell=True)
    else:
        print(f"file {target_file} already exists, skipping.")
