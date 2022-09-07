from argparse import ArgumentParser
from .inference import segment_sa_la, segment_ao
from pathlib import Path


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--data-dir", required=True,
        type=str, help='Path to the data set directory, under which images '
                       'are organised in subdirectories for each subject.'
    )

    parser.add_argument("--output-dir", default=None,type=str)
    parser.add_argument(
        "--sa-model-path", default=None,
        type=str, help='Path to the saved trained model.',
    )
    parser.add_argument(
        "--la-2ch-model-path", default=None,
        type=str, help='Path to the saved trained model.',
    )
    parser.add_argument(
        "--la-4ch-model-path", default=None,
        type=str, help='Path to the saved trained model.',
    )
    parser.add_argument(
        "--la-4ch-seg4-model-path", default=None,
        type=str, help='Path to the saved trained model.',
    )
    parser.add_argument(
        "--ao-model-path", default=None,
        type=str, help='Path to the saved trained model.',
    )

    parser.add_argument("--process-seg", action="store_true", help="Process a time sequence of images.")
    parser.add_argument("--save-seg", action="store_true", help="Save segmentation")
    parser.add_argument(
        "--seg4", action="store_true",
        help='Segment all the 4 chambers in long-axis 4 chamber view. '
             'This seg4 network is trained using 200 subjects from Application 18545.'
             'By default, for all the other tasks (ventricular segmentation'
             'on short-axis images and atrial segmentation on long-axis images,'
             'the networks are trained using 3,975 subjects from Application 2964.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).absolute()
    else:
        output_dir = None

    # short-axis segmentations
    if args.sa_model_path is not None:
        segment_sa_la(
            data_dir=Path(args.data_dir),
            seq_name="sa",
            model_path=Path(args.sa_model_path),
            process_seq=args.process_seg,
            save_seg=args.save_seg,
            seg4=False,
            output_dir=output_dir,
        )

    # long-axis segmentations
    if args.la_2ch_model_path is not None:
        segment_sa_la(
            data_dir=Path(args.data_dir),
            seq_name="la_2ch",
            model_path=Path(args.la_2ch_model_path),
            process_seq=args.process_seg,
            save_seg=args.save_seg,
            seg4=False,
            output_dir=output_dir,
        )
    if args.la_4ch_model_path is not None:
        segment_sa_la(
            data_dir=Path(args.data_dir),
            seq_name="la_4ch",
            model_path=Path(args.la_4ch_model_path),
            process_seq=args.process_seg,
            save_seg=args.save_seg,
            seg4=False,
            output_dir=output_dir,
        )
    if args.la_4ch_seg4_model_path is not None and args.seg4:
        segment_sa_la(
            data_dir=Path(args.data_dir),
            seq_name="la_4ch",
            model_path=Path(args.la_4ch_seg4_model_path),
            process_seq=args.process_seg,
            save_seg=args.save_seg,
            seg4=args.seg4,
            output_dir=output_dir,
        )

    if args.ao_model_path is not None:
        segment_ao(
            data_dir=str(Path(args.data_dir)),
            model="UNet",
            seq_name="ao",
            model_path=str(Path(args.la_4ch_seg4_model_path)),
            process_seq=args.process_seg,
            save_seg=args.save_seg,
            z_score=True,
            weight_R=5,
            weight_r=0.1,
            time_step=1,
        )


if __name__ == '__main__':
    main()
