# Copyright 2019, Wenjia Bai. All Rights Reserved.
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
# ============================================================================
import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from typing import List
from .cardiac_utils import cine_2d_sa_motion_and_strain_analysis, sa_pass_quality_control, evaluate_wall_thickness


def eval_ventricular_volume(data_list: List[Path], output_csv: str):
    # data_path = data_dir
    # data_list = sorted(os.listdir(data_path))
    table = []
    processed_list = []
    for data in tqdm(data_list):
        # data_dir = os.path.join(data_path, data)
        data_dir = str(data)
        image_name = '{0}/sa.nii.gz'.format(data_dir)
        seg_name = '{0}/seg_sa.nii.gz'.format(data_dir)

        if os.path.exists(image_name) and os.path.exists(seg_name):
            print(data.name)

            # Image
            nim = nib.load(image_name)
            pixdim = nim.header['pixdim'][1:4]
            volume_per_pix = pixdim[0] * pixdim[1] * pixdim[2] * 1e-3
            density = 1.05

            # Heart rate
            duration_per_cycle = nim.header['dim'][4] * nim.header['pixdim'][4]
            heart_rate = 60.0 / duration_per_cycle

            # Segmentation
            seg = nib.load(seg_name).get_data()

            frame = {}
            frame['ED'] = 0
            vol_t = np.sum(seg == 1, axis=(0, 1, 2)) * volume_per_pix
            frame['ES'] = np.argmin(vol_t)

            val = {}
            for fr_name, fr in frame.items():
                # Clinical measures
                val['LV{0}V'.format(fr_name)] = np.sum(seg[:, :, :, fr] == 1) * volume_per_pix
                val['LV{0}M'.format(fr_name)] = np.sum(seg[:, :, :, fr] == 2) * volume_per_pix * density
                val['RV{0}V'.format(fr_name)] = np.sum(seg[:, :, :, fr] == 3) * volume_per_pix

            val['LVSV'] = val['LVEDV'] - val['LVESV']
            val['LVCO'] = val['LVSV'] * heart_rate * 1e-3
            val['LVEF'] = val['LVSV'] / val['LVEDV'] * 100

            val['RVSV'] = val['RVEDV'] - val['RVESV']
            val['RVCO'] = val['RVSV'] * heart_rate * 1e-3
            val['RVEF'] = val['RVSV'] / val['RVEDV'] * 100

            line = [val['LVEDV'], val['LVESV'], val['LVSV'], val['LVEF'], val['LVCO'], val['LVEDM'],
                    val['RVEDV'], val['RVESV'], val['RVSV'], val['RVEF']]
            table += [line]
            processed_list += [str(data.name)]

    df = pd.DataFrame(table, index=processed_list,
                      columns=['LVEDV (mL)', 'LVESV (mL)', 'LVSV (mL)', 'LVEF (%)', 'LVCO (L/min)', 'LVM (g)',
                               'RVEDV (mL)', 'RVESV (mL)', 'RVSV (mL)', 'RVEF (%)'])
    df.to_csv(output_csv)


def eval_strain_sax(data_list: List[Path], output_csv: str, par_dir: str = None, start_idx: int = 0, end_idx: int = 0):
    if par_dir is None:
        par_dir = Path(__file__).parent.joinpath("par").absolute()

    # data_path = data_dir
    # data_list = sorted(os.listdir(data_path))
    n_data = len(data_list)
    end_idx = n_data if end_idx == 0 else end_idx
    table = []
    processed_list = []
    for data in tqdm(data_list[start_idx:end_idx]):
        print(data)
        # data_dir = os.path.join(data_path, data)
        data_dir = str(data)

        # Quality control for segmentation at ED
        # If the segmentation quality is low, the following functions may fail.
        seg_sa_name = '{0}/seg_sa_ED.nii.gz'.format(data_dir)
        if not os.path.exists(seg_sa_name):
            continue
        if not sa_pass_quality_control(seg_sa_name):
            continue

        # Intermediate result directory
        motion_dir = os.path.join(data_dir, 'cine_motion')
        if not os.path.exists(motion_dir):
            os.makedirs(motion_dir)

        # Perform motion tracking on short-axis images and calculate the strain
        cine_2d_sa_motion_and_strain_analysis(data_dir,
                                              par_dir,
                                              motion_dir,
                                              '{0}/strain_sa'.format(data_dir))

        # Remove intermediate files
        os.system('rm -rf {0}'.format(motion_dir))

        # Record data
        if os.path.exists('{0}/strain_sa_radial.csv'.format(data_dir)) \
                and os.path.exists('{0}/strain_sa_circum.csv'.format(data_dir)):
            df_radial = pd.read_csv('{0}/strain_sa_radial.csv'.format(data_dir), index_col=0)
            df_circum = pd.read_csv('{0}/strain_sa_circum.csv'.format(data_dir), index_col=0)
            line = [df_circum.iloc[i, :].min() for i in range(17)] + [df_radial.iloc[i, :].max() for i in range(17)]
            table += [line]
            processed_list += [data.name]

    # Save strain values for all the subjects
    df = pd.DataFrame(table, index=processed_list,
                      columns=['Ecc_AHA_1 (%)', 'Ecc_AHA_2 (%)', 'Ecc_AHA_3 (%)',
                               'Ecc_AHA_4 (%)', 'Ecc_AHA_5 (%)', 'Ecc_AHA_6 (%)',
                               'Ecc_AHA_7 (%)', 'Ecc_AHA_8 (%)', 'Ecc_AHA_9 (%)',
                               'Ecc_AHA_10 (%)', 'Ecc_AHA_11 (%)', 'Ecc_AHA_12 (%)',
                               'Ecc_AHA_13 (%)', 'Ecc_AHA_14 (%)', 'Ecc_AHA_15 (%)', 'Ecc_AHA_16 (%)',
                               'Ecc_Global (%)',
                               'Err_AHA_1 (%)', 'Err_AHA_2 (%)', 'Err_AHA_3 (%)',
                               'Err_AHA_4 (%)', 'Err_AHA_5 (%)', 'Err_AHA_6 (%)',
                               'Err_AHA_7 (%)', 'Err_AHA_8 (%)', 'Err_AHA_9 (%)',
                               'Err_AHA_10 (%)', 'Err_AHA_11 (%)', 'Err_AHA_12 (%)',
                               'Err_AHA_13 (%)', 'Err_AHA_14 (%)', 'Err_AHA_15 (%)', 'Err_AHA_16 (%)',
                               'Err_Global (%)'])
    df.to_csv(output_csv)


def eval_wall_thickness(data_list: List[Path], output_csv: str):
    # data_path = data_dir
    # data_list = sorted(os.listdir(data_path))
    table = []
    processed_list = []
    for data in tqdm(data_list):
        print(data)
        # data_dir = os.path.join(data_path, data)
        data_dir = str(data)

        # Quality control for segmentation at ED
        # If the segmentation quality is low, evaluation of wall thickness may fail.
        seg_sa_name = '{0}/seg_sa_ED.nii.gz'.format(data_dir)
        if not os.path.exists(seg_sa_name):
            print("Seg not exist", seg_sa_name)
            continue
        if not sa_pass_quality_control(seg_sa_name):
            print("Quality control failed", seg_sa_name)
            continue

        # Evaluate myocardial wall thickness
        evaluate_wall_thickness('{0}/seg_sa_ED.nii.gz'.format(data_dir),
                                '{0}/wall_thickness_ED'.format(data_dir))

        # Record data
        if os.path.exists('{0}/wall_thickness_ED.csv'.format(data_dir)):
            df = pd.read_csv('{0}/wall_thickness_ED.csv'.format(data_dir), index_col=0)
            line = df['Thickness'].values
            table += [line]
            processed_list += [data.name]

    # Save wall thickness for all the subjects
    df = pd.DataFrame(table, index=processed_list,
                      columns=['WT_AHA_1 (mm)', 'WT_AHA_2 (mm)', 'WT_AHA_3 (mm)',
                               'WT_AHA_4 (mm)', 'WT_AHA_5 (mm)', 'WT_AHA_6 (mm)',
                               'WT_AHA_7 (mm)', 'WT_AHA_8 (mm)', 'WT_AHA_9 (mm)',
                               'WT_AHA_10 (mm)', 'WT_AHA_11 (mm)', 'WT_AHA_12 (mm)',
                               'WT_AHA_13 (mm)', 'WT_AHA_14 (mm)', 'WT_AHA_15 (mm)', 'WT_AHA_16 (mm)',
                               'WT_Global (mm)'])
    df.to_csv(output_csv)
