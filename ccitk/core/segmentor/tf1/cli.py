from pathlib import Path
from argparse import ArgumentParser
from CMRSegment.common.resource import PhaseImage
from CMRSegment.segmentor.tf1.HR import TF13DSegmentor


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-path", dest="model_path", type=str, default=None)

    parser.add_argument("--input-path", dest="input_path", type=str, required=True, help="Input path of a nii.gz file.")
    parser.add_argument("--phase", dest="phase", default="phase", type=str)

    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path)
    with TF13DSegmentor(model_path=model_path, overwrite=args.overwrite) as segmentor:
        segmentation = segmentor.apply(
            image=PhaseImage(phase=args.phase, path=Path(args.input_path)),
            output_path=output_dir.joinpath("seg.nii.gz")
        )


if __name__ == '__main__':
    main()
