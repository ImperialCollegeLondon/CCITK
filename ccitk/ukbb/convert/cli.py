# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
The script downloads the cardiac MR images for a UK Biobank Application given csv-file, key-path, ukbfetch-path
"""

import os
import pandas as pd
from tqdm import tqdm
from enum import IntEnum
from pathlib import Path
import multiprocessing as mp
from argparse import ArgumentParser
from typing import List

from ccitk.ukbb.convert.biobank_utils import process_manifest, Biobank_Dataset



class UKBBFieldKey(IntEnum):
    # 20208: Long axis heart images - DICOM Heart MRI
    # 20209: Short axis heart images - DICOM Heart MRI
    # 20210: Aortic distensibilty images - DICOM Heart MRI
    la = 20208
    sa = 20209
    ao = 20210


def make_parser():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", dest="input_dir", type=str, required=True,
                        help="input dir of the downloaded zip files")
    parser.add_argument("--csv-file", dest="csv_file", type=str, required=True,
                        help="List of EIDs to download, column name eid")
    parser.add_argument("--output-dir", dest="output_dir", type=str, required=True)
    parser.add_argument("--n-thread", dest="n_thread", type=int, default=0)
    parser.add_argument("--fields", nargs="+", choices=["la", "sa", "ao"], type=str, default=["la", "sa", "ao"])
    return parser


def parse_args():
    parser = make_parser()
    return parser.parse_args()


def function(eid: str, fields: List[UKBBFieldKey], input_dir: Path, output_dir: Path):
    dicom_dir = output_dir.joinpath("dicom", f"{eid}")
    nii_dir = output_dir.joinpath("nii", f"{eid}")

    dirs = []
    zip_dir = input_dir

    for field in fields:
        if field in [UKBBFieldKey.la, UKBBFieldKey.sa]:
            if nii_dir.joinpath("la_2ch.nii.gz").exists() and \
                    nii_dir.joinpath("la_3ch.nii.gz").exists() and \
                    nii_dir.joinpath("la_4ch.nii.gz").exists() and \
                    nii_dir.joinpath("sa.nii.gz").exists():
                return

        zip_file = zip_dir.joinpath(f"{eid}_{field.value}_2_0.zip")
        if not zip_file.exists():
            continue
        nii_dir.mkdir(parents=True, exist_ok=True)
        dicom_dir.mkdir(parents=True, exist_ok=True)
        dicom_dir_dir = dicom_dir.joinpath(field.name)
        assert zip_file.exists(), f"str({zip_file}) not exist"
        os.system('unzip -o {0} -d {1}'.format(str(zip_file), str(dicom_dir_dir)))
        dirs.append(dicom_dir_dir)

    for directory in dirs:

        # Process the manifest file
        if directory.joinpath('manifest.cvs').exists():
            os.system('cp {0} {1}'.format(str(directory.joinpath('manifest.cvs')),
                                          str(directory.joinpath('manifest.csv'))))
        process_manifest(str(directory.joinpath('manifest.csv')),
                         str(directory.joinpath('manifest2.csv')))
        df2 = pd.read_csv(str(str(directory.joinpath('manifest2.csv'))), error_bad_lines=False)

        # Patient ID and acquisition date
        pid = df2.at[0, 'patientid']
        date = dateutil.parser.parse(df2.at[0, 'date'][:11]).date().isoformat()

        # Organise the dicom files
        # Group the files into subdirectories for each imaging series
        for series_name, series_df in df2.groupby('series discription'):
            # series_dir = os.path.join(dicom_dir, series_name)
            series_dir = directory.joinpath(series_name)
            series_dir.mkdir(parents=True, exist_ok=True)
            # if not os.path.exists(series_dir):
            #     os.mkdir(series_dir)
            series_files = [os.path.join(str(directory), x) for x in series_df['filename']]
            os.system('mv {0} {1}'.format(' '.join(series_files), series_dir))

            # Convert dicom files and annotations into nifti images
            dset = Biobank_Dataset(str(directory))
            dset.read_dicom_images()
            dset.convert_dicom_to_nifti(str(nii_dir))

    return "finished"


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)

    csv_file = Path(args.csv_file)
    output_dir = Path(args.output_dir)
    n_thread = args.n_thread
    fields = [UKBBFieldKey[f] for f in args.fields]
    df = pd.read_csv(str(csv_file))
    data_list = df['eid']

    # Download cardiac MR images for each subject
    start_idx = 0
    end_idx = len(data_list)
    pbar = tqdm(range(start_idx, end_idx))

    output_dir.mkdir(parents=True, exist_ok=True)

    if n_thread == 0:
        for i in pbar:
            eid = str(data_list[i])
            function(eid, fields, input_dir, output_dir)
    else:
        def update(*a):
            pbar.update()
        pool = mp.Pool(processes=n_thread)
        # setup multiprocessing
        for i in range(pbar.total):
            eid = str(data_list[i])
            pool.apply_async(func=function, args=(eid, fields, input_dir, output_dir), callback=update)
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()
