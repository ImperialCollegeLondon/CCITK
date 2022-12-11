```
ssh sli9@vm-biomedia/slurm.doc.ic.ac.uk

# Create a virtual environment 
virtualenv /vol/biomedic2/sli9/ven

# Activate the virtual environment
source /vol/biomedic2/sli9/venv/bin/activate

# Pip install dependencies 
cd /vol/biomedic2/sli9/projects/CCITK/
python setup.py develop

export CUDA_HOME=/vol/cuda/11.3.1-cudnn8.2.1/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vol/cuda/11.3.1-cudnn8.2.1/targets/x86_64-linux/lib
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113/

pip install nibabel scipy  # ccitk pyhocon

pip install pandas numpy  # download

pip install opencv-python pydicom SimpleITK numpy  # convert

TMPDIR=/vol/biomedic2/sli9/temp/ pip install tensorflow numpy tqdm  # segment, don't install tensorflow if using batch

# MIRTK
export PATH="$PATH:/vol/biomedic2/wbai/git/MIRTK_bin/bin"

# average_3d_ffd
export PATH=$PATH:/vol/biomedic2/wbai/git/ukbb_cardiac/third_party/ubuntu_16.04_bin/


# download job.py has to be run when in ukbb/download dir. 10k about 6-8 hours for sa, la and ao
ccitk-ukbb-batch-download 
# batch args
--n-partition 10 --n-threads 0 
--csv-file /vol/biodata/data/biobank/40616/csv/ukbb_40616_new_eids.csv
--venv-activate-path /homes/sli9/venv/bin/activate
--partition-csv-dir /vol/biodata/data/biobank/40616/output/temp/csv
# job args
--fields ao
--key-path /vol/biodata/data/biobank/40616/key/ukbb.key
--ukbfetch-path /vol/biodata/data/biobank/40616/utils/ukbfetch
--output-dir /vol/biodata/data/biobank/40616/output/images/

ccitk-ukbb-batch-convert 
# batch args
--n-partition 10 --n-threads 0 
--csv-file /vol/biodata/data/biobank/40616/csv/ukbb_40616_new_eids.csv
--venv-activate-path /homes/sli9/venv/bin/activate
--partition-csv-dir /vol/biodata/data/biobank/40616/output/temp/csv
# job args
--fields ao
--input-dir /vol/biodata/data/biobank/40616/output/images/zip/
--output-dir /vol/biodata/data/biobank/40616/output/images/

ccitk-ukbb-segment 
--process-seq --save-seg
--data-dir /vol/biodata/data/biobank/40616/output/images/nii/
--csv-file /vol/biodata/data/biobank/40616/csv/ukbb_40616_new_eids.csv
# --output-dir /vol/biodata/data/biobank/40616/output/segs
--model sa  # choose from sa, la_2ch, la_4ch, la_4ch_seg4, ao
--model-path  /vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_sa 

# Complete list of model paths
sa-model-path /vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_sa 
la-2ch-model-path /vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_la_2ch
la-4ch-model-path /vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_la_4ch
la-4ch-seg4-model-path /vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_la_4ch_seg4_modified
ao-model-path /vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/UNet-LSTM_ao_modified


export PYTHONPATH=$PYTHONPATH:/vol/biomedic2/wbai/git/
ccitk-ukbb-batch-segment 
# batch args
--n-partition 10
--csv-file /vol/biodata/data/biobank/40616/csv/ukbb_40616_new_eids.csv
--venv-activate-path /homes/sli9/venv/bin/activate
--partition-csv-dir /vol/biodata/data/biobank/40616/output/temp/csv
# job args
--process-seq --save-seg
--data-dir /vol/biodata/data/biobank/40616/output/images/nii/
# --output-dir /vol/biodata/data/biobank/40616/output/segs
--model sa  # choose from sa, la_2ch, la_4ch, la_4ch_seg4, ao

ccitk-ukbb-analyze --sa --la --data-dir /vol/biodata/data/biobank/40616/output/images/nii 
--outout-dir /vol/biodata/data/biobank/40616/output/

ccitk-ukbb-batch-analyze 
# batch args
--n-partition 10
--csv-file /vol/biodata/data/biobank/40616/csv/ukbb_40616_new_eids.csv
--venv-activate-path /homes/sli9/venv/bin/activate
--partition-csv-dir /vol/biodata/data/biobank/40616/output/temp/csv
# job args
--sa --la --ao --data-dir /vol/biodata/data/biobank/40616/output/images/nii 
--output-dir /vol/biodata/data/biobank/40616/output/
# if --ao
--pressure-csv /vol/biodata/data/biobank/40616/csv/ukb52223_image_subset.csv

# SLURM job management
sinfo -lNe
cat slurm.roc01.41941.log
tail -f slurm.roc01.41941.log
scancel 41941

```