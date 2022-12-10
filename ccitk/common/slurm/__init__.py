from pathlib import Path
from ccitk.common.slurm.config import CPUBatchJobConfig, GPUBatchJobConfig, BatchJobConfig
from typing import List
import os
import tempfile
from argparse import ArgumentParser, Namespace
import pandas as pd
import numpy as np


def partition_csv(csv_path: Path, n_partition: int, output_dir: Path) -> List[Path]:
    output_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(str(csv_path))
    dfs = np.array_split(df, n_partition)
    csvs = []
    for i, df in enumerate(dfs):
        df.to_csv(str(output_dir.joinpath(f"partition_csv_{i}.csv")), index=False)
        csvs.append(output_dir.joinpath(f"partition_csv_{i}.csv"))
    return csvs


class SlurmBatchJobManager:
    config_csl = BatchJobConfig

    def __init__(self, temp_dir: Path = None):
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp())
        self.temp_dir = temp_dir

    @classmethod
    def submit_cli(cls, job_parser: ArgumentParser, job_script_path: Path):
        # job_parser.add_argument("--master-csv-file", dest="master_csv_file", type=str, required=True,
        #                     help="List of EIDs to download, column name eid")
        job_parser.add_argument("--n-partition", dest="n_partition", type=int)
        job_parser.add_argument("--partition-csv-dir", dest="partition_csv_dir", type=str)
        job_parser.add_argument("--venv-activate-path", dest="venv_activate_path", type=str,
                            default="/homes/sli9/venv/bin/activate")
        args = job_parser.parse_args()
        csv_file = Path(args.csv_file)
        n_partition = args.n_partition
        venv_activate_path = Path(args.venv_activate_path)
        partition_csv_dir = Path(args.partition_csv_dir)
        manager = cls()
        delattr(args, 'csv_file')
        delattr(args, 'n_partition')
        delattr(args, 'partition_csv_dir')
        delattr(args, 'venv_activate_path')

        manager.submit(
            job_script_path=job_script_path,
            job_namespace=args,
            master_csv_file=csv_file,
            n_partition=n_partition,
            partition_csv_dir=partition_csv_dir,
            venv_activate_path=venv_activate_path,
        )

    @staticmethod
    def make_python_command(job_script_path: Path, **kwargs):
        python_command = "python {job_script_path}".format(job_script_path=str(job_script_path))
        python_command += " --csv-file {csv_file}"
        for key, value in kwargs.items():
            key = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    value = ""
                else:
                    continue
            if isinstance(value, list):
                value = " ".join(value)
            if isinstance(value, Path) or isinstance(value, int) or isinstance(value, float):
                value = str(value)
            python_command += f" {key} {value}"
        return python_command

    def create_batch_files(self, csv_paths: List[Path], python_command: str, venv_activate_path: Path):
        batch_files = []
        for idx, csv_file in enumerate(csv_paths):
            batch_file = self.temp_dir.joinpath(f"job_{idx}.sh")
            batch_job_config = self.config_csl(
                node_num=idx + 1,  # random?
                python_command=python_command.format(csv_file=csv_file),
                venv_activate_path=venv_activate_path,
            )
            with open(str(batch_file), "w") as file:
                file.write(batch_job_config.sbatch)
            batch_files.append(batch_file)
        return batch_files

    def submit(self, job_script_path: Path, job_namespace: Namespace, master_csv_file: Path, n_partition: int,
               partition_csv_dir: Path, venv_activate_path: Path):
        python_command = self.make_python_command(
            job_script_path,
            **vars(job_namespace)
        )
        csv_files = partition_csv(csv_path=master_csv_file, n_partition=n_partition, output_dir=partition_csv_dir)
        batch_files = self.create_batch_files(
            csv_paths=csv_files,
            python_command=python_command,
            venv_activate_path=venv_activate_path,
        )
        for batch_file in batch_files:
            os.system(f"sbatch {str(batch_file)}")


class SlurmCPUBatchJobManager(SlurmBatchJobManager):
    config_csl = CPUBatchJobConfig


class SlurmGPUBatchJobManager(SlurmBatchJobManager):
    config_csl = GPUBatchJobConfig
