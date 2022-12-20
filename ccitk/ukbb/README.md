# Introduction
This application consists of four modules: `download`, `convert`, `segment`, and `analyze`. 

`download` is to fetch bulk DICOM image data from UKBB; 
`convert` is to unzip and convert them to NIFTY images;
`segment` is to segment long-axis (LA), short-axis (SA) and aortic (AO) images using 2D networks;
dnd `analyze` is to analyze segmentations and extract informations such as 
volumes of the atrial and ventricle, strains, wall thickness abd aortic area.

# Dependencies
```
pip install pandas tqdm numpy scipy nibabel opencv-python pydicom SimpleVTK scikit-image vtk
```
It requires tensorflow for segmentation. The code was written for tensorflow 1, but it also now 
back support tensorflow 2. 

It also requires `MIRTK` and `average_3d_ffd` in the `analyze` module. Add their paths to your `PATH` variable, 
for example
```
# MIRTK
export PATH="$PATH:/vol/biomedic2/wbai/git/MIRTK_bin/bin"

# average_3d_ffd
export PATH=$PATH:/vol/biomedic2/wbai/git/ukbb_cardiac/third_party/ubuntu_16.04_bin/
```

# Modules
## download
Command line command
```
ccitk-ukbb-download 
--csv-file /vol/biodata/data/biobank/40616/csv/ukbb_40616_new_eids.csv
--fields sa la ao
--key-path /vol/biodata/data/biobank/40616/key/ukbb.key
--ukbfetch-path /vol/biodata/data/biobank/40616/utils/ukbfetch
--output-dir /vol/biodata/data/biobank/40616/output/images/
```

* `--csv-file` specifiles the location of a csv file that contains a list of eids to be processed in a column.
* `--fields` specifies what images to download. You can choose one or more from three options: [sa, la, ao], where
sa stands for short-axis images, la is long-axis images, and ao is aortic images. 
* `--key-path` specifies the ukbb key path
* `--ukbfetch-path` specifies the location of the ukbfetch command
* `--output-dir` specifies where to output

Important: This command has to run when you are inside the [ccitk/ukbb/download](download) directory
## convert
Command line command
```
ccitk-ukbb-convert 
--csv-file /vol/biodata/data/biobank/40616/csv/ukbb_40616_new_eids.csv
--fields sa la ao
--input-dir /vol/biodata/data/biobank/40616/output/images/zip/
--output-dir /vol/biodata/data/biobank/40616/output/images/
```

* `--csv-file` specifiles the location of a csv file that contains a list of eids to be processed in a column.
* `--fields` specifies what images to convert. You can choose one or more from three options: sa la ao, where
sa stands for short-axis images, la is long-axis images, and ao is aortic images. 
* `--input-dir` specifies the location of zip files
* `--output-dir` specifies where to output, which are a folder of dicom images, and a folder of nii images. 

## segment
To use prebuilt tensorflow 1
```
export PYTHONPATH=$PYTHONPATH:/vol/biomedic2/wbai/git/
```
Command line command
```
ccitk-ukbb-segment 
--csv-file /vol/biodata/data/biobank/40616/csv/ukbb_40616_new_eids.csv
--process-seq --save-seg
--data-dir /vol/biodata/data/biobank/40616/output/images/nii/
--model sa  # choose from sa, la_2ch, la_4ch, la_4ch_seg4, ao
```
* `--csv-file` specifiles the location of a csv file that contains a list of eids to be processed in a column.
* `--process-seq` whether to process the whole cine sequence or just ED ES. This should always be supplied, 
as we always need the whole cine segmentations for analysis
* `--save-seg` whether to save segmentations. Always should be supplied. 
* `--model` specifies what model to use. You can choose 1 from 5 options: ["sa", "ao", "la_2ch", "la_4ch", "la_4ch_seg4"], where
sa stands for short-axis model, la is long-axis model, and ao is aortic model. 

The complete list of model paths are listed in below for reference. They have been hard coded so we do not need to supple
anything else, other than the name of the model ["sa", "ao", "la_2ch", "la_4ch", "la_4ch_seg4"]:
```
sa-model-path /vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_sa 
la-2ch-model-path /vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_la_2ch
la-4ch-model-path /vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_la_4ch
la-4ch-seg4-model-path /vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_la_4ch_seg4_modified
ao-model-path /vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/UNet-LSTM_ao_modified
```
You can only run one model at a time. To finish the segmentation process, make sure to run all 5 models. 

## analyze
Command line command
```
ccitk-ukbb-analyze 
--csv-file /vol/biodata/data/biobank/40616/csv/ukbb_40616_new_eids.csv
--sa --la --ao --data-dir /vol/biodata/data/biobank/40616/output/images/nii 
--output-dir /vol/biodata/data/biobank/40616/output/
# if --ao
--pressure-csv /vol/biodata/data/biobank/40616/csv/ukb52223_image_subset.csv
```
* `--csv-file` specifiles the location of a csv file that contains a list of eids to be processed in a column.
* `--data-dir` specifiles the location of nii images
* `--sa`: whether to run short-axis analysis
* `--la`, whether to run long-axis analysis
* `--ao` whether to run aortic analysis
* `--output-dir` location to output csv files
* `--pressure-csv`: if --ao, then need to supple the location of a csv file that contains the blood pressure of each subject


# SLURM - CPU/GPU clusters
To run above commands for tens of thousands of subjects take an enormous amount of time, months maybe. So we must utilise
HPC clusters to parallel these computations. 

To submit a SLURM job, it is very easy, 

```
ccitk-slurm-submit --batch-conf /path-to-batch.conf
```
See more information in [ccitk/slurm](../../ccitk/slurm) package

In the batch.conf, you need to fill out the following fields
```
batch {
  job_type = "cpu"  # or "gpu"
  job_script_path = "/path-to-python-script"  # or command line command
  job_args = "--key value --other-options" # all but --csv-file
  csv_file = "/path-to-the-main-eid-csv"
  partition_csv_dir = "/directory-to-store-the-partitioned-eid-csvs"
  venv_activate_path = "/homes/sli9/venv/bin/activate"
  jobs = [
    # cpu job
    {
      n_cpu = 16
      mem = 12288
      nodelist = "roc01"
    },
    # or gpu job
    {
      n_cpu = 4
      mem = 12288
      nodelist = "lory"
      n_gpu = 1
    },
  ]
```
Batch confs that was used to process newly added 10k UKBB images are listed here:
* [batch_download.conf](batch_download.conf)
    * remember to CD into [ccitk/ukbb/download](download) directory before running ccitk-slurm-submit
    * Maximum you can run is 10 parallel jobs at the same time for downloading as it is limited on the UKBB server side. 
* [batch_convert.conf](batch_convert.conf)
* [batch_segment_sa.conf](batch_segment_sa.conf)
* [batch_segment_la_2ch.conf](batch_segment_la_2ch.conf)
* [batch_segment_la_4ch.conf](batch_segment_la_4ch.conf)
* [batch_segment_la_4ch_seg4.conf](batch_segment_la_4ch_seg4.conf)
* [batch_segment_ao.conf](batch_segment_ao.conf)
* [batch_analyze.conf](batch_analyze.conf)


# Miscellaneous
```
ssh sli9@vm-biomedia-slurm.doc.ic.ac.uk

# Create a virtual environment 
virtualenv /vol/biomedic2/sli9/ven

# Activate the virtual environment
source /vol/biomedic2/sli9/venv/bin/activate

# Pip install dependencies 
cd /vol/biomedic2/sli9/projects/CCITK/
python setup.py develop

export CUDA_HOME=/vol/cuda/11.3.1-cudnn8.2.1/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vol/cuda/11.3.1-cudnn8.2.1/targets/x86_64-linux/lib

# MIRTK
export PATH="$PATH:/vol/biomedic2/wbai/git/MIRTK_bin/bin"

# average_3d_ffd
export PATH=$PATH:/vol/biomedic2/wbai/git/ukbb_cardiac/third_party/ubuntu_16.04_bin/

# SLURM job management
sinfo -lNe
cat slurm.roc01.41941.log
tail -f slurm.roc01.41941.log
scancel 41941
```
