# Introduction
This application submits any script to SLURM managed clusters in BioMedia given a user specified 
`batch.conf` file.

Dependencies:
```
pyhocon
numpy
pandas
```

# batch.conf format
A template of CPU and GPU `batch.conf` is provided in [batch.conf](batch.conf)

For CPU jobs
```hocon
batch {
  job_type = "cpu"
  job_script_path = "/path-to-.py"
  job_args = "--input-dir /path-to-input --key value"  # all but --csv-file
  csv_file = "/path-to-csv-file-to-be-partitioned"
  partition_csv_dir = "/path-on-shared-drive-to-stored-partitions"
  venv_activate_path = "/path-to-venv-activate"
  jobs = [
    {
      n_cpu = 32
      mem = 12288
      nodelist = "roc01"  # 01 to 16
    },
    {
      n_cpu = 32
      mem = 12288
      nodelist = "roc01"  # 01 to 16
    },
    {
      n_cpu = 32
      mem = 12288
      nodelist = "roc01"  # 01 to 16
    }
  ]
}

```

For GPU jobs
```hocon
batch {
  job_type = "gpu"
  job_script_path = "/path-to-.py"
  job_args = "--input-dir /path-to-input --key value"  # all but --csv-file
  csv_file = "/path-to-csv-file-to-be-partitioned"
  partition_csv_dir = "/path-on-shared-drive-to-stored-partitions"
  venv_activate_path = "/path-to-venv-activate"
  jobs = [
    {
      n_cpu = 32
      mem = 12288
      nodelist = "lory"  # lory 16 x Tesla T4 (16GB), monal03 4 x Tesla P100 (16GB), monal04 4 x Tesla P40 (24GB), monal 05 and 06 10 x Geforce Titan XP (12GB)
      n_gpu = 1
    },
    {
      n_cpu = 32
      mem = 12288
      nodelist = "lory"  # lory 16 x Tesla T4 (16GB), monal03 4 x Tesla P100 (16GB), monal04 4 x Tesla P40 (24GB), monal 05 and 06 10 x Geforce Titan XP (12GB)
      n_gpu = 1
    },
    {
      n_cpu = 32
      mem = 12288
      nodelist = "lory"  # lory 16 x Tesla T4 (16GB), monal03 4 x Tesla P100 (16GB), monal04 4 x Tesla P40 (24GB), monal 05 and 06 10 x Geforce Titan XP (12GB)
      n_gpu = 1
    }
  ]
}

```

# Available machines

These internal BioMedia pages contains details about machine availables: 

https://biomedia.doc.ic.ac.uk/internal/computer-resources/the-slurm-resource-manager/

https://biomedia.doc.ic.ac.uk/internal/computer-resources/gpu-user-guide/

For CPU jobs, `roc01` to `roc16`, each has 32 CPUs.

```
NODELIST   NODES   PARTITION       STATE CPUS    S:C:T MEMORY TMP_DISK WEIGHT AVAIL_FE REASON              
  roc01          1     roclong        idle 32      2:8:2 257000        0      1   (null) none                
  roc02          1     roclong        idle 32      2:8:2 257000        0      1   (null) none                
  roc03          1     roclong        idle 32      2:8:2 257000        0      1   (null) none                
  roc04          1     roclong        idle 32      2:8:2 257000        0      1   (null) none                
  roc05          1     roclong        idle 32      2:8:2 257000        0      1   (null) none                
  roc06          1     roclong        idle 32      2:8:2 257000        0      1   (null) none                
  roc07          1     roclong        idle 32      2:8:2 257000        0      1   (null) none                
  roc08          1     roclong        idle 32      2:8:2 257000        0      1   (null) none                
  roc09          1     roclong        idle 32      2:8:2 257000        0      1   (null) none                
  roc10          1     roclong        idle 32      2:8:2 257000        0      1   (null) none                
  roc11          1     roclong        idle 32      2:8:2 257000        0      1   (null) none                
  roc12          1     roclong        idle 32      2:8:2 257000        0      1   (null) none                
  roc13          1     roclong        idle 32      2:8:2 257000        0      1   (null) none                
  roc14          1     roclong        idle 32      2:8:2 257000        0      1   (null) none                
  roc15          1     roclong        idle 32      2:8:2 257000        0      1   (null) none                
  roc16          1     roclong        idle 32      2:8:2 257000        0      1   (null) none  
```

For GPU jobs, 

```
NODELIST   NODES   PARTITION       STATE CPUS    S:C:T MEMORY TMP_DISK WEIGHT AVAIL_FE REASON              
  lory           1       gpus*       mixed 80     2:20:2 380000        0      1   (null) none                
  monal03        1       gpus*        idle 56     2:14:2 250000        0      1   (null) none                
  monal04        1       gpus*       mixed 56     2:14:2 250000        0      1   (null) none                
  monal05        1       gpus*        idle 56     2:14:2 220000        0      1   (null) none                
  monal06        1       gpus*        idle 56     2:14:2 250000        0      1   (null) none                
```

### lory (Tesla T4)

lory features 16 Tesla T4 cards, each with 16 GB. It also has 20T of fast local SSD storage. Use this as scratch storage for copies of input data and intermedate storage. Do not rely on this and always back up important results to /vol/ directories.

```
CPU: Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz
RAM: 12 x Samsung M393A4K40CB2-CTD (384GB)
GPU: 16 x Tesla T4 (16GB)
```

### monal03 (Tesla P100)
monal03 features 4 Tesla P100 cards, each with 16 GB, connected via NVLINK, ideal for multi-GPU computing (i.e. where there is a lot of communication between the GPUs).

```
CPU: Intel(R) Xeon(R) CPU E5-2660 v4 @ 2.00GHz
RAM: 8 x Samsung M393A4K40BB1-CRC (256GB)
GPU: 4 x Tesla P100 (16GB)
```

### monal04 (Tesla P40)
```
CPU: Intel(R) Xeon(R) CPU E5-2660 v4 @ 2.00GHz
RAM: 8 x Samsung M393A4K40BB1-CRC (256GB)
GPU: 4 x Tesla P40 (24GB)
```

### monal05 and monal06 (Titan Xp)

```
CPU: Intel(R) Xeon(R) CPU E5-2660 v4 @ 2.00GHz
RAM: 8 x Micron 36ASF4G72PZ-2G3A1 (256GB)
GPU: 10 x Geforce Titan XP (12GB)
```

# Run

To submit a batch job, after creating a `batch.conf` and specifying the job accordingly, do

```
ccitk-slurm-submit --batch-conf /path-to-user-batch-conf
```
