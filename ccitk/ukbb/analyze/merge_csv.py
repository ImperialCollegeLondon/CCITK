import glob
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--csv-dir", dest="csv_dir", type=str)
    parser.add_argument("--prefix", dest="prefix", type=str)
    parser.add_argument("--output-dir", dest="output_dir", type=str)
    return parser.parse_args()


def main():
    # find all files with prefix_partition_*.csv
    args = parse_args()
    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    file_prefix = args.prefix
    file_path = csv_dir.joinpath(file_prefix)
    csv_files = []
    for name in glob.glob(f'{str(file_path)}_partition_csv_[0-9]*.csv'):
        file_path = Path(name)
        csv_files.append(file_path)
    dfs = [pd.read_csv(str(f), index_col=[0], parse_dates=[0]) for f in csv_files]

    finaldf = pd.concat(dfs, axis=0, join='inner').drop_duplicates().sort_index()
    finaldf.to_csv(str(output_dir.joinpath(file_prefix)) + ".csv")
    pass


if __name__ == '__main__':
    main()
