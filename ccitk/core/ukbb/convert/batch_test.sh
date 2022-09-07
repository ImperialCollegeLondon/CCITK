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
python job.py --input-dir /vol/biodata/data/biobank/40616/output/images/zip --csv-file /vol/biodata/data/biobank/40616/csv/ukbb_40616_new_eids.csv --output-dir /vol/biodata/data/biobank/40616/output/images --n-thread 0