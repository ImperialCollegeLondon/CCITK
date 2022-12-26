import os
import tempfile
from typing import List
from pathlib import Path

from ccitk.slurm.config import CPUBatchJobConfig, GPUBatchJobConfig, BatchJobConfig


class SlurmBatchJobManager:
    config_csl = BatchJobConfig

    def __init__(self, temp_dir: Path = None):
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp())
        self.temp_dir = temp_dir

    @staticmethod
    def make_python_command(job_script_path: Path, job_args: str, csv_file: Path):
        python_command = "python {job_script_path}".format(job_script_path=str(job_script_path))
        python_command += " --csv-file {csv_file} ".format(csv_file=str(csv_file))
        python_command += job_args
        return python_command

    def create_batch_files(self, configs: List[BatchJobConfig]):
        batch_files = []
        for idx, config in enumerate(configs):
            batch_file = self.temp_dir.joinpath(f"job_{idx}.sh")
            with open(str(batch_file), "w") as file:
                file.write(config.to_sbatch())
            batch_files.append(batch_file)
        return batch_files

    def submit(self, configs: List[BatchJobConfig]):
        batch_files = self.create_batch_files(configs)
        for batch_file in batch_files:
            os.system(f"sbatch {str(batch_file)}")
