
from pathlib import Path

from ccitk.common.slurm import SlurmGPUBatchJobManager
from ccitk.ukbb.segment.cli import make_parser

CWD = Path(__file__).parent


def main():
    parser = make_parser()
    SlurmGPUBatchJobManager.submit_cli(
        job_parser=parser,
        job_script_path=CWD.joinpath("cli.py")
    )


if __name__ == '__main__':
    main()
