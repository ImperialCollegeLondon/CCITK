import dataclasses
from dataclasses import field
from pathlib import Path


@dataclasses.dataclass
class BatchJobConfig:
    nodelist: str
    python_command: str
    venv_activate_path: Path  # Path("/homes/sli9/venv/bin/activate")
    template_path: Path = field(init=False)
    partition: str = field(init=False)
    n_cpu: int = field(default=32)
    mem: int = field(default=12288)
    log_output: str = field(default="slurm.%N.%j.log")

    def to_sbatch(self) -> str:
        with open(str(self.template_path), "r") as file:
            sbatch = file.read()
        sbatch = sbatch.format(
            n_cpu=self.n_cpu,
            mem=self.mem,
            log_output=self.log_output,
            partition=self.partition,
            venv_path=str(self.venv_activate_path),
            python_command=self.python_command,
            nodelist=self.nodelist,
        )
        return sbatch


@dataclasses.dataclass
class CPUBatchJobConfig(BatchJobConfig):
    partition: str = field(default="roclong")
    template_path: Path = field(default=Path(__file__).parent.joinpath("batch_template_cpu.txt"))


@dataclasses.dataclass
class GPUBatchJobConfig(BatchJobConfig):
    partition: str = field(default="gpus")
    n_gpu: int = field(default=1)
    template_path: Path = field(default=Path(__file__).parent.joinpath("batch_template_gpu.txt"))

    def to_sbatch(self) -> str:
        # nodelist = GPUNodeList(self.node_num).name
        # self.nodelist = nodelist
        with open(str(self.template_path), "r") as file:
            sbatch = file.read()
        sbatch = sbatch.format(
            n_cpu=self.n_cpu,
            mem=self.mem,
            log_output=self.log_output,
            partition=self.partition,
            venv_path=str(self.venv_activate_path),
            python_command=self.python_command,
            nodelist=self.nodelist,
            n_gpu=self.n_gpu,
        )
        return sbatch
