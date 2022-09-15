from pathlib import Path
from argparse import ArgumentParser
from CMRSegment.preprocessor import DataPreprocessor


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data-dir", dest="data_dir", type=str, required=True)
    parser.add_argument("-o", "--output-dir", dest="output_dir", required=True, type=str)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    preprocessor = DataPreprocessor()
    subjects = list(preprocessor.run(data_dir=Path(args.data_dir), output_dir=args.output_dir, overwrite=args.overwrite))


if __name__ == '__main__':
    main()
