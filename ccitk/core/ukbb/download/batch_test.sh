#!/bin/bash

# Example of running python script in a batch mode
#SBATCH -c 1                        # Number of CPU Cores
#SBATCH -p roclong                  # Partition (queue)
#SBATCH --mem 12288                 # memory pool for all cores
#SBATCH --nodelist roc01            # SLURM node
#SBATCH --output=slurm.%N.%j.log    # Standard output and error log

# Source virtual environment (pip)
source /homes/sli9/venv/bin/activate

# Run python script
python job.py --csv-file /vol/biodata/data/biobank/40616/csv/ukbb_40616_new_eids.csv --key-path /vol/biodata/data/biobank/40616/key/ukbb.key --ukbfetch-path /vol/biodata/data/biobank/40616/utils/ukbfetch --output-dir /vol/biodata/data/biobank/40616/output --n-thread 0