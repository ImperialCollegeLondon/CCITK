from pathlib import Path
from argparse import ArgumentParser
from ccitk.ukbb.analyze.short_axis import eval_ventricular_volume, eval_wall_thickness, eval_strain_sax
from ccitk.ukbb.analyze.long_axis import eval_atrial_volume, eval_strain_lax
from ccitk.ukbb.analyze.aortic import eval_aortic_area
import pandas as pd


def make_parser():
    parser = ArgumentParser()
    parser.add_argument("--sa", action="store_true")
    parser.add_argument("--la", action="store_true")
    parser.add_argument("--ao", action="store_true")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--csv-file", dest="csv_file", type=str, required=True, help="List of EIDs to download, column name eid")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--pressure-csv", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--distributed", action="store_true", help="Running in distributed mode.")
    return parser


def parse_args():
    parser = make_parser()
    return parser.parse_args()


def main():
    # TODO: if in distributed mode, output csv file has suffix _{input_csv_name}, to distinguish between different nodes.
    #  Then after all finished, merge csv files
    # TODO: output csv currently overwrite if exist, change it to merging csv, or different csv names
    args = parse_args()
    data_dir = Path(args.data_dir).absolute()
    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    par_dir = Path(__file__).parent.joinpath("par").absolute()
    csv_file = Path(args.csv_file)
    df = pd.read_csv(str(csv_file))
    distributed = args.distributed
    eids = df["eid"]
    data_list = [data_dir.joinpath(str(eid)) for eid in eids]
    overwrite = args.overwrite
    print("Here")
    if args.sa:
        if distributed:
            if not output_dir.joinpath(f"sa_ventricular_volume_{csv_file.stem}.csv").exists() or overwrite:
                eval_ventricular_volume(
                    data_list=data_list,
                    output_csv=str(output_dir.joinpath(f"sa_ventricular_volume_{csv_file.stem}.csv")),
                )
            if not output_dir.joinpath(f"sa_wall_thickness_{csv_file.stem}.csv").exists() or overwrite:
                eval_wall_thickness(
                    data_list=data_list,
                    output_csv=str(output_dir.joinpath(f"sa_wall_thickness_{csv_file.stem}.csv")),
                )
            if not output_dir.joinpath(f"sa_strain_{csv_file.stem}.csv").exists() or overwrite:
                eval_strain_sax(
                    data_list=data_list,
                    output_csv=str(output_dir.joinpath(f"sa_strain_{csv_file.stem}.csv")),
                    par_dir=str(par_dir),
                )
        else:
            if not output_dir.joinpath("sa_ventricular_volume.csv").exists() or overwrite:
                eval_ventricular_volume(
                    data_list=data_list,
                    output_csv=str(output_dir.joinpath("sa_ventricular_volume.csv")),
                )
            if not output_dir.joinpath("sa_wall_thickness.csv").exists() or overwrite:
                eval_wall_thickness(
                    data_list=data_list,
                    output_csv=str(output_dir.joinpath("sa_wall_thickness.csv")),
                )
            if not output_dir.joinpath("sa_strain.csv").exists() or overwrite:
                eval_strain_sax(
                    data_list=data_list,
                    output_csv=str(output_dir.joinpath("sa_strain.csv")),
                    par_dir=str(par_dir),
                )

    if args.la:
        if distributed:
            if not output_dir.joinpath(f"la_atrial_volume_{csv_file.stem}.csv").exists() or overwrite:
                eval_atrial_volume(
                    data_list=data_list,
                    output_csv=str(output_dir.joinpath(f"la_atrial_volume_{csv_file.stem}.csv")),
                )
            if not output_dir.joinpath(f"la_strain_{csv_file.stem}.csv").exists() or overwrite:
                eval_strain_lax(
                    data_list=data_list,
                    output_csv=str(output_dir.joinpath(f"la_strain_{csv_file.stem}.csv")),
                    par_dir=str(par_dir),
                )
        else:
            if not output_dir.joinpath("la_atrial_volume.csv").exists() or overwrite:
                eval_atrial_volume(
                    data_list=data_list,
                    output_csv=str(output_dir.joinpath("la_atrial_volume.csv")),
                )
            if not output_dir.joinpath("la_strain.csv").exists() or overwrite:
                eval_strain_lax(
                    data_list=data_list,
                    output_csv=str(output_dir.joinpath("la_strain.csv")),
                    par_dir=str(par_dir),
                )

    if args.ao:
        assert args.pressure_csv is not None
        pressure_csv = Path(args.pressure_csv).absolute()
        if distributed:
            if not output_dir.joinpath(f"aortic_area_{csv_file.stem}.csv").exists() or overwrite:
                eval_aortic_area(
                    data_list=data_list,
                    pressure_csv=str(pressure_csv),
                    output_csv=str(output_dir.joinpath(f"aortic_area_{csv_file.stem}.csv")),
                )
        else:
            if not output_dir.joinpath("aortic_area.csv").exists() or overwrite:
                eval_aortic_area(
                    data_list=data_list,
                    pressure_csv=str(pressure_csv),
                    output_csv=str(output_dir.joinpath("aortic_area.csv")),
                )


if __name__ == '__main__':
    main()
