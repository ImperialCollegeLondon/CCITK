# Copyright 2018, Wenjia Bai. All Rights Reserved.
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
import os
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from ccitk.ukbb.analyze.cardiac_utils import aorta_pass_quality_control
from typing import List
from pathlib import Path


def eval_aortic_area(data_list: List[Path], pressure_csv: str, output_csv: str):

    # Read the spreadsheet for blood pressure information
    # Use central blood pressure provided by the Vicorder software
    # [1] Steffen E. Petersen et al. UK Biobank’s cardiovascular magneticresonance protocol. JCMR, 2016.
    # Aortic distensibility represents the relative change in area of the aorta per unit pressure,
    # taken here as the "central pulse pressure".
    #
    # The Vicorder software calculates values for central blood pressure by applying a previously described
    # brachial-to-aortic transfer function. What I observed from the data and Figure 5 in the SOP pdf
    # (https://biobank.ctsu.ox.ac.uk/crystal/docs/vicorder_in_cmri.pdf) is that after the transfer, the DBP
    # keeps the same as the brachial DBP, but the SBP is different.

    # df_info = pd.read_csv(pressure_csv, header=[0, 1], index_col=0)
    # central_pp = df_info['Central pulse pressure during PWA'][['12678-2.0', '12678-2.1']].mean(axis=1)
    print("Begin loading csv")
    df_info = pd.read_csv(pressure_csv, header=[0], index_col=0)
    print(df_info)
    central_pp = df_info[['12678-2.0', '12678-2.1']].mean(axis=1)
    print("Finished loading csv")
    # Discard central blood pressure < 10 mmHg
    central_pp[central_pp < 10] = np.nan

    # data_path = data_dir
    # data_list = sorted(os.listdir(data_path))
    table = []
    processed_list = []
    for data in tqdm(data_list):
        data_dir = str(data)
        # data_dir = os.path.join(data_path, data)
        image_name = os.path.join(data_dir, 'ao.nii.gz')
        seg_name = os.path.join(data_dir, 'seg_ao.nii.gz')

        if os.path.exists(image_name) and os.path.exists(seg_name):
            print(data)

            # Read image
            nim = nib.load(image_name)
            dx, dy = nim.header['pixdim'][1:3]
            area_per_pixel = dx * dy
            image = nim.get_data()

            # Read segmentation
            nim = nib.load(seg_name)
            seg = nim.get_data()

            if not aorta_pass_quality_control(image, seg):
                continue

            # Measure the maximal and minimal area for the ascending aorta and descending aorta
            val = {}
            for l_name, l in [('AAo', 1), ('DAo', 2)]:
                val[l_name] = {}
                A = np.sum(seg == l, axis=(0, 1, 2)) * area_per_pixel
                val[l_name]['max area'] = A.max()
                val[l_name]['min area'] = A.min()
                val[l_name]['distensibility'] = (A.max() - A.min()) / (A.min() * central_pp.loc[int(data.name)]) * 1e3

            line = [val['AAo']['max area'], val['AAo']['min area'], val['AAo']['distensibility'],
                    val['DAo']['max area'], val['DAo']['min area'], val['DAo']['distensibility']]
            table += [line]
            processed_list += [data.name]

    # Save the spreadsheet for the measures
    df = pd.DataFrame(table, index=processed_list,
                      columns=['AAo max area (mm2)', 'AAo min area (mm2)', 'AAo distensibility (10-3 mmHg-1)',
                               'DAo max area (mm2)', 'DAo min area (mm2)', 'DAo distensibility (10-3 mmHg-1)'])
    df.to_csv(output_csv)
