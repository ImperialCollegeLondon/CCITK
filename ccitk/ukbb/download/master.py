"""
Master script to partition image csv, and submit one job per partition
"""

import os
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List


CWD = Path(__file__).parent


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--csv-file", dest="csv_file", type=str, required=True,
                        help="List of EIDs to download, column name eid")
    parser.add_argument("--key-path", dest="key_path", type=str, required=True)
    parser.add_argument("--ukbfetch-path", dest="ukbfetch_path", type=str, required=True)
    parser.add_argument("--n-partition", dest="n_partition", type=int)
    parser.add_argument("--output-dir", dest="output_dir", type=str)
    parser.add_argument("--n-thread", dest="n_thread", type=int, default=0)
    parser.add_argument("--fields", nargs="+", choices=["la", "sa", "ao"], type=str, default=["la", "sa", "ao"])
    return parser.parse_args()


def partition_csv(csv_path: Path, n_partition: int, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(str(csv_path))
    dfs = np.array_split(df, n_partition)
    csvs = []
    for i, df in enumerate(dfs):
        df.to_csv(str(output_dir.joinpath(f"partition_csv_{i}.csv")), index=False)
        csvs.append(output_dir.joinpath(f"partition_csv_{i}.csv"))
    return csvs


def create_batch_files(key_path: Path, ukbfetch_path: Path, part_csv_files: List[Path], output_dir: Path, n_thread, fields: List[str]) -> List[Path]:
    job_script_path = CWD.joinpath("job.py")
    python_command = f"python {str(job_script_path)}"
    with open(str(CWD.joinpath("batch_template.txt")), "r") as file:
        sbatch = file.read()
    temp_batch_file_dir = output_dir.joinpath("temp")
    temp_batch_file_dir.mkdir(exist_ok=True, parents=True)
    batch_files = []
    fields = " ".join(fields)
    for idx, csv_file in enumerate(part_csv_files):
        command = python_command + f" --key-path {str(key_path)} --ukbfetch-path {str(ukbfetch_path)} " \
                                   f"--csv-file {str(csv_file)} --output-dir {str(output_dir)}  --n-thread {n_thread} --fields {fields}"
        batch = sbatch.format(idx + 1) + f"{command}\n"
        batch_file = temp_batch_file_dir.joinpath(f"job_{idx}.sh")
        with open(str(batch_file), "w") as file:
            file.write(batch)
        batch_files.append(batch_file)
    return batch_files


def submit_batch_files(batch_files: List[Path]):
    for batch_file in batch_files:
        os.system(f"sbatch {str(batch_file)}")


def main():
    args = parse_args()
    csv_file = Path(args.csv_file)
    key_path = Path(args.key_path)
    ukbfetch_path = Path(args.ukbfetch_path)

    output_dir = Path(args.output_dir)
    n_partition = args.n_partition

    part_csv_files = partition_csv(csv_file, n_partition, output_dir.joinpath("temp", "csv"))
    batch_files = create_batch_files(key_path, ukbfetch_path, part_csv_files, output_dir, args.n_thread, args.fields)
    submit_batch_files(batch_files)


if __name__ == '__main__':
    main()
