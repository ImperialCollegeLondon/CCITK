"""
Master script to partition image csv, and submit one job per partition
"""
from pathlib import Path

from ccitk.common.slurm import SlurmCPUBatchJobManager
from ccitk.ukbb.download.cli import make_parser

CWD = Path(__file__).parent


def main():
    parser = make_parser()
    job_script_path = CWD.joinpath("cli.py")

    SlurmCPUBatchJobManager.submit_cli(
        job_parser=parser,
        job_script_path=job_script_path,
    )


if __name__ == '__main__':
    main()
