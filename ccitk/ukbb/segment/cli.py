import inspect
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path

from ccitk.ukbb.segment.inference import segment_sa_la, segment_ao


DEFAULT_MODEL_DICT = {
    "sa": "/vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_sa",
    "la_2ch": "/vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_la_2ch",
    "la_4ch": "/vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_la_4ch",
    "la_4ch_seg4": "/vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/FCN_la_4ch_seg4_modified",
    "ao": "/vol/biomedic2/wbai/git/ukbb_cardiac/trained_model/UNet-LSTM_ao_modified",
}


def make_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "--data-dir", required=True,
        type=str, help='Path to the data set directory, under which images '
                       'are organised in subdirectories for each subject.'
    )
    parser.add_argument("--csv-file", dest="csv_file", type=str, required=True,
                        help="List of EIDs to download, column name eid")

    parser.add_argument("--model", choices=["sa", "ao", "la_2ch", "la_4ch", "la_4ch_seg4"], type=str, required=True)

    parser.add_argument("--process-seq", action="store_true", help="Process a time sequence of images.")
    parser.add_argument("--save-seg", action="store_true", help="Save segmentation")
    return parser


def parse_args():
    parser = make_parser()
    return parser.parse_args()


def main():
    args = parse_args()
    # if args.output_dir is not None:
    #     output_dir = Path(args.output_dir).absolute()
    # else:
    #     output_dir = None
    data_dir = Path(args.data_dir)
    csv_file = Path(args.csv_file)
    df = pd.read_csv(str(csv_file))
    eids = df["eid"]
    data_list = [data_dir.joinpath(str(eid)) for eid in eids]
    model = args.model
    # TODO: if seg exists, skip; if data not exists, skip and record.
    # TODO: Can only run on at a time
    # short-axis segmentations
    # if args.model_path is None:
    #     model_path = Path(DEFAULT_MODEL_DICT[model])
    # else:
    #     model_path = Path(args.model_path)
    model_path = Path(DEFAULT_MODEL_DICT[model])

    print(model_path)
    if model == "sa":
        print(inspect.getfullargspec(segment_sa_la))
        segment_sa_la(
            data_list=data_list,
            seq_name="sa",
            model_path=model_path,
            process_seq=args.process_seq,
            save_seg=args.save_seg,
            seg4=False,
        )
    elif model == "la_2ch":
        # long-axis segmentations
        segment_sa_la(
            data_list=data_list,
            seq_name="la_2ch",
            model_path=model_path,
            process_seq=args.process_seq,
            save_seg=args.save_seg,
            seg4=False,
        )
    elif model == "la_4ch":
        segment_sa_la(
            data_list=data_list,
            seq_name="la_4ch",
            model_path=model_path,
            process_seq=args.process_seq,
            save_seg=args.save_seg,
            seg4=False,
        )
    elif model == "la_4ch_seg4":
        segment_sa_la(
            data_list=data_list,
            seq_name="la_4ch",
            model_path=model_path,
            process_seq=args.process_seq,
            save_seg=args.save_seg,
            seg4=True,
        )
    elif model == "ao":
        segment_ao(
            data_list=data_list,
            model="UNet-LSTM",
            seq_name="ao",
            model_path=str(model_path),
            process_seq=args.process_seq,
            save_seg=args.save_seg,
            z_score=True,
            weight_R=5,
            weight_r=0.1,
            time_step=1,
        )


if __name__ == '__main__':
    main()
