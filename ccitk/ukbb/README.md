```
ssh sli9@vm-biomedia/slurm.doc.ic.ac.uk

# Create a virtual environment 

# Activate the virtual environment
source /vol/biomedic2/sli9/venv/bin/activate

# Pip install dependencies 
cd /vol/biomedic2/sli9/projects/CCITK/
python setup.py develop

export CUDA_HOME=/vol/cuda/11.3.1-cudnn8.2.1/
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113/

# MIRTK
export PATH="$PATH:/vol/biomedic2/wbai/git/MIRTK_bin/bin"


# download job.py has to be run when in ukbb/download dir.

ccitk-ukbb-download --n-partition 10 --n-threads 0 --fields ao
--csv-file /vol/biodata/data/biobank/40616/csv/ukbb_40616_new_eids.csv
--key-path /vol/biodata/data/biobank/40616/key/ukbb.key
--ukbfetch-path /vol/biodata/data/biobank/40616/utils/ukbfetch
--output-dir /vol/biodata/data/biobank/40616/output/

ccitk-ukbb-convert --n-partition 10 --n-threads 0 --fields ao
--input-dir /vol/biodata/data/biobank/40616/output/images/zip/
--csv-file /vol/biodata/data/biobank/40616/csv/ukbb_40616_new_eids.csv
--output-dir /vol/biodata/data/biobank/40616/output/images/

ccitk-ukbb-segment --data-dir --process-seq --save-seg
--sa-model-path /vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_sa 
--la-2ch-model-path /vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_la_2ch
--la-4ch-model-path /vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_la_4ch
--la-4ch-seg4-model-path /vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_la_4ch_seg4
--output-dir /vol/biodata/data/biobank/40616/output/segs


ccitk-ukbb-analyze --sa --la --data-dir /vol/biodata/data/biobank/40616/output/images/nii 
--outout-dir /vol/biodata/data/biobank/40616/output/

# SLURM job management

cat slurm.roc01.41941.log
tail -f slurm.roc01.41941.log
scancel 41941

```