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
import glob
import pandas as pd
import dateutil.parser
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import shutil
from argparse import ArgumentParser
from typing import List
from ccitk.ukbb.const import UKBBFieldKey


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--csv-file", dest="csv_file", type=str, required=True, help="List of EIDs to download, column name eid")
    parser.add_argument("--key-path", dest="key_path", type=str, required=True)
    parser.add_argument("--ukbfetch-path", dest="ukbfetch_path", type=str, required=True)
    parser.add_argument("--output-dir", dest="output_dir", type=str, required=True)
    parser.add_argument("--n-thread", dest="n_thread", type=int, default=0)
    parser.add_argument("--fields", nargs="+", choices=["la", "sa", "ao"], type=str, default=["la", "sa", "ao"])

    return parser.parse_args()


def function(eid: str, fields: List[UKBBFieldKey], ukbkey: Path, ukbfetch: Path, output_dir: Path):
    output_image_dir = output_dir.joinpath("images")

    zip_dir = output_image_dir.joinpath("zip")
    zip_dir.mkdir(parents=True, exist_ok=True)

    # Create a batch file for this subject
    batch_file = output_dir.joinpath("batch", f"{eid}_batch")
    batch_file.parent.mkdir(parents=True, exist_ok=True)
    # batch_file = os.path.join(data_dir, '{0}_batch'.format(eid))
    # if not zip_dir.joinpath(f"{eid}_20208_2_0.zip").exists() or not zip_dir.joinpath(f"{eid}_20209_2_0.zip").exists():
    with open(str(batch_file), 'w') as f_batch:
        for j in fields:
            j = j.value  # 20208, 20209, or 20210
            if zip_dir.joinpath(f"{eid}_{j}_2_0.zip").exists():
                continue
            # The field ID information can be searched at http://biobank.ctsu.ox.ac.uk/crystal/search.cgi
            # 20208: Long axis heart images - DICOM Heart MRI
            # 20209: Short axis heart images - DICOM Heart MRI
            # 20210: Aortic distensibilty images - DICOM Heart MRI
            # 2.0 means the 2nd visit of the subject, the 0th data item for that visit.
            # As far as I know, the imaging scan for each subject is performed at his/her 2nd visit.
            field = '{0}-2.0'.format(j)
            f_batch.write('{0} {1}_2_0\n'.format(eid, j))

        # Download the data using the batch file
        # ukbfetch = os.path.join(util_dir, 'ukbfetch')
        # print('Downloading data for subject {} ...'.format(eid))
        command = '{0} -b{1} -a{2}'.format(ukbfetch, str(batch_file), ukbkey)
        os.system('{0} -b{1} -a{2}'.format(ukbfetch, str(batch_file), ukbkey))
        print("Download finished")
        # Unpack the data
        for f in Path(__file__).parent.glob('{0}_*.zip'.format(eid)):
            shutil.move(str(f), str(zip_dir.joinpath(f.name)))
        print("Move finished")

        batch_file.unlink()

    return "finished"


def main():
    args = parse_args()
    # Where the data will be downloaded
    csv_file = Path(args.csv_file)
    key_path = Path(args.key_path)
    ukbfetch_path = Path(args.ukbfetch_path)
    output_dir = Path(args.output_dir)
    n_thread = args.n_thread
    fields = [int(UKBBFieldKey[f]) for f in args.fields]
    df = pd.read_csv(str(csv_file))
    data_list = df['eid']

    # Download cardiac MR images for each subject
    start_idx = 0
    end_idx = len(data_list)
    pbar = tqdm(range(start_idx, end_idx))

    if n_thread == 0:
        for i in pbar:
            eid = str(data_list[i])
            function(eid, fields, key_path, ukbfetch_path, output_dir)
    else:
        def update(*a):
            pbar.update()
        pool = mp.Pool(processes=n_thread)
        # setup multiprocessing
        for i in range(pbar.total):
            eid = str(data_list[i])
            pool.apply_async(func=function, args=(eid, fields, key_path, ukbfetch_path, output_dir), callback=update)
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()
