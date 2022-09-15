from pathlib import Path
from CMRSegment.pipeline import CMRPipeline
from CMRSegment.pipeline.config import PipelineConfig


def parse_args():
    parser = PipelineConfig.argument_parser()
    parser.add_argument("--data-dir", dest="data_dir", required=True, type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = PipelineConfig.from_args(args)
    pipeline = CMRPipeline(config=config)
    pipeline.run(data_dir=Path(args.data_dir))


if __name__ == '__main__':
    main()
