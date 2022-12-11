import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
from argparse import ArgumentParser
from pyhocon import ConfigTree, ConfigFactory

from ccitk.slurm import SlurmBatchJobManager
from ccitk.slurm.config import CPUBatchJobConfig, GPUBatchJobConfig


def get_conf(conf: ConfigTree, key: str = "", default=None):
    key = ".".join(["batch", key])
    return conf.get(key, default)


def partition_csv(csv_path: Path, n_partition: int, output_dir: Path) -> List[Path]:
    output_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(str(csv_path))
    dfs = np.array_split(df, n_partition)
    csvs = []
    for i, df in enumerate(dfs):
        df.to_csv(str(output_dir.joinpath(f"partition_csv_{i}.csv")), index=False)
        csvs.append(output_dir.joinpath(f"partition_csv_{i}.csv"))
    return csvs


def main():
    parser = ArgumentParser()
    parser.add_argument("--batch-conf", dest="batch_conf", type=str)
    args = parser.parse_args()
    batch_conf_path = Path(args.batch_conf)
    batch_conf = ConfigFactory.parse_file(str(batch_conf_path))

    job_script_path = Path(get_conf(batch_conf, key="job_script_path"))
    job_args = get_conf(batch_conf, key="job_args")  # all but --csv-file
    csv_file = Path(get_conf(batch_conf, key="csv_file"))
    partition_csv_dir = Path(get_conf(batch_conf, key="partition_csv_dir"))
    venv_activate_path = Path(get_conf(batch_conf, key="venv_activate_path"))
    job_type = get_conf(batch_conf, key="job_type")

    jobs = get_conf(batch_conf, key="jobs")

    n_partition = len(jobs)

    csv_files = partition_csv(csv_path=csv_file, n_partition=n_partition, output_dir=partition_csv_dir)
    configs = []
    for job, csv_file in zip(jobs, csv_files):
        python_command = SlurmBatchJobManager.make_python_command(job_script_path, job_args, csv_file)
        if job_type == "gpu":
            config = GPUBatchJobConfig(
                venv_activate_path=venv_activate_path,
                python_command=python_command,
                n_cpu=job.get("n_cpu"),
                mem=job.get("mem"),
                nodelist=job.get("nodelist"),
                n_gpu=job.get("n_gpu")
            )
        else:
            config = CPUBatchJobConfig(
                venv_activate_path=venv_activate_path,
                python_command=python_command,
                n_cpu=job.get("n_cpu"),
                mem=job.get("mem"),
                nodelist=job.get("nodelist")
            )
        configs.append(config)
    manager = SlurmBatchJobManager()
    manager.submit(
        configs=configs
    )


if __name__ == '__main__':
    main()
