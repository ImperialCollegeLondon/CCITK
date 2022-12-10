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
# =========================c===================================================
import os
import numpy as np
import pandas as pd
import nibabel as nib
import vtk
import math
from pathlib import Path
from ccitk.ukbb.analyze.cardiac_utils import atrium_pass_quality_control, evaluate_atrial_area_length
from ccitk.ukbb.analyze.cardiac_utils import la_pass_quality_control, cine_2d_la_motion_and_strain_analysis
from tqdm import tqdm
from typing import List


def eval_atrial_volume(data_list: List[Path], output_csv: str):
    # data_path = data_dir
    # data_list = sorted(os.listdir(data_path))
    table = []
    processed_list = []
    for data in tqdm(data_list):
        # data_dir = os.path.join(data_path, data)
        data_dir = str(data)
        seg_la_2ch_name = '{0}/seg_la_2ch.nii.gz'.format(data_dir)
        seg_la_4ch_name = '{0}/seg_la_4ch.nii.gz'.format(data_dir)
        sa_name = '{0}/sa.nii.gz'.format(data_dir)

        if os.path.exists(seg_la_2ch_name) and os.path.exists(seg_la_4ch_name) and os.path.exists(sa_name):
            print(data.name)

            # Determine the long-axis from short-axis image
            nim_sa = nib.load(sa_name)
            long_axis = nim_sa.affine[:3, 2] / np.linalg.norm(nim_sa.affine[:3, 2])
            if long_axis[2] < 0:
                long_axis *= -1

            # Measurements
            # A: area
            # L: length
            # V: volume
            # lm: landmark
            A = {}
            L = {}
            V = {}
            lm = {}

            # Analyse 2 chamber view image
            nim_2ch = nib.load(seg_la_2ch_name)
            seg_la_2ch = nim_2ch.get_data()
            T = nim_2ch.header['dim'][4]

            # Perform quality control for the segmentation
            if not atrium_pass_quality_control(seg_la_2ch, {'LA': 1}):
                print('{0} seg_la_2ch does not atrium_pass_quality_control.'.format(data.name))
                continue

            A['LA_2ch'] = np.zeros(T)
            L['LA_2ch'] = np.zeros(T)
            V['LA_2ch'] = np.zeros(T)
            lm['2ch'] = {}
            for t in range(T):
                area, length, landmarks = evaluate_atrial_area_length(seg_la_2ch[:, :, 0, t], nim_2ch, long_axis)
                if type(area) == int:
                    if area < 0:
                        continue

                A['LA_2ch'][t] = area[0]
                L['LA_2ch'][t] = length[0]
                V['LA_2ch'][t] = 8 / (3 * math.pi) * area[0] * area[0] / length[0]
                lm['2ch'][t] = landmarks

                if t == 0:
                    # Write the landmarks
                    points = vtk.vtkPoints()
                    for p in landmarks:
                        points.InsertNextPoint(p[0], p[1], p[2])
                    poly = vtk.vtkPolyData()
                    poly.SetPoints(points)
                    writer = vtk.vtkPolyDataWriter()
                    writer.SetInputData(poly)
                    writer.SetFileName('{0}/lm_la_2ch_{1:02d}.vtk'.format(data_dir, t))
                    writer.Write()

            # Analyse 4 chamber view image
            nim_4ch = nib.load(seg_la_4ch_name)
            seg_la_4ch = nim_4ch.get_data()

            # Perform quality control for the segmentation
            if not atrium_pass_quality_control(seg_la_4ch, {'LA': 1, 'RA': 2}):
                print('{0} seg_la_4ch does not atrium_pass_quality_control.'.format(data))
                continue

            A['LA_4ch'] = np.zeros(T)
            L['LA_4ch'] = np.zeros(T)
            V['LA_4ch'] = np.zeros(T)
            V['LA_bip'] = np.zeros(T)
            A['RA_4ch'] = np.zeros(T)
            L['RA_4ch'] = np.zeros(T)
            V['RA_4ch'] = np.zeros(T)
            lm['4ch'] = {}
            for t in range(T):
                area, length, landmarks = evaluate_atrial_area_length(seg_la_4ch[:, :, 0, t], nim_4ch, long_axis)
                if type(area) == int:
                    if area < 0:
                        continue

                A['LA_4ch'][t] = area[0]
                L['LA_4ch'][t] = length[0]
                V['LA_4ch'][t] = 8 / (3 * math.pi) * area[0] * area[0] / length[0]
                V['LA_bip'][t] = 8 / (3 * math.pi) * area[0] * A['LA_2ch'][t] / (0.5 * (length[0] + L['LA_2ch'][t]))

                A['RA_4ch'][t] = area[1]
                L['RA_4ch'][t] = length[1]
                V['RA_4ch'][t] = 8 / (3 * math.pi) * area[1] * area[1] / length[1]
                lm['4ch'][t] = landmarks

                if t == 0:
                    # Write the landmarks
                    points = vtk.vtkPoints()
                    for p in landmarks:
                        points.InsertNextPoint(p[0], p[1], p[2])
                    poly = vtk.vtkPolyData()
                    poly.SetPoints(points)
                    writer = vtk.vtkPolyDataWriter()
                    writer.SetInputData(poly)
                    writer.SetFileName('{0}/lm_la_4ch_{1:02d}.vtk'.format(data_dir, t))
                    writer.Write()

            # Heart rate
            duration_per_cycle = nim_4ch.header['dim'][4] * nim_4ch.header['pixdim'][4]
            heart_rate = 60.0 / duration_per_cycle

            # Record atrial volumes
            # Left atrial volume: bi-plane estimation
            # Right atrial volume: single plane estimation
            val = {}
            val['LAV_bip_max'] = np.max(V['LA_bip'])
            val['LAV_bip_min'] = np.min(V['LA_bip'])
            val['LASV_bip'] = val['LAV_bip_max'] - val['LAV_bip_min']
            val['LAEF_bip'] = val['LASV_bip'] / val['LAV_bip_max'] * 100

            val['RAV_4ch_max'] = np.max(V['RA_4ch'])
            val['RAV_4ch_min'] = np.min(V['RA_4ch'])
            val['RASV_4ch'] = val['RAV_4ch_max'] - val['RAV_4ch_min']
            val['RAEF_4ch'] = val['RASV_4ch'] / val['RAV_4ch_max'] * 100

            line = [val['LAV_bip_max'], val['LAV_bip_min'], val['LASV_bip'], val['LAEF_bip'],
                    val['RAV_4ch_max'], val['RAV_4ch_min'], val['RASV_4ch'], val['RAEF_4ch']]
            table += [line]
            processed_list += [data.name]

    df = pd.DataFrame(table, index=processed_list,
                      columns=['LAV max (mL)', 'LAV min (mL)', 'LASV (mL)', 'LAEF (%)',
                               'RAV max (mL)', 'RAV min (mL)', 'RASV (mL)', 'RAEF (%)'])
    df.to_csv(output_csv)


def eval_strain_lax(data_list: List[Path], output_csv: str, par_dir: str = None, start_idx: int = 0, end_idx: int = 0):
    if par_dir is None:
        par_dir = Path(__file__).parent.joinpath("par").absolute()
    # data_path = data_dir
    # data_list = sorted(os.listdir(data_path))
    n_data = len(data_list)
    end_idx = n_data if end_idx == 0 else end_idx
    table = []
    processed_list = []
    for data in tqdm(data_list[start_idx:end_idx]):
        print(data.name)
        # data_dir = os.path.join(data_path, data)
        data_dir = str(data)

        # Quality control for segmentation at ED
        # If the segmentation quality is low, the following functions may fail.
        seg_la_name = '{0}/seg4_la_4ch_ED.nii.gz'.format(data_dir)
        if not os.path.exists(seg_la_name):
            continue
        if not la_pass_quality_control(seg_la_name):
            continue

        # Intermediate result directory
        motion_dir = os.path.join(data_dir, 'cine_motion')
        if not os.path.exists(motion_dir):
            os.makedirs(motion_dir)

        # Perform motion tracking on short-axis images and calculate the strain
        cine_2d_la_motion_and_strain_analysis(data_dir,
                                              par_dir,
                                              motion_dir,
                                              '{0}/strain_la_4ch'.format(data_dir))

        # Remove intermediate files
        os.system('rm -rf {0}'.format(motion_dir))

        # Record data
        if os.path.exists('{0}/strain_la_4ch_longit.csv'.format(data_dir)):
            df_longit = pd.read_csv('{0}/strain_la_4ch_longit.csv'.format(data_dir), index_col=0)
            line = [df_longit.iloc[i, :].min() for i in range(7)]
            table += [line]
            processed_list += [data.name]

    # Save strain values for all the subjects
    df = pd.DataFrame(table, index=processed_list,
                      columns=['Ell_1 (%)', 'Ell_2 (%)', 'Ell_3 (%)',
                               'Ell_4 (%)', 'Ell_5 (%)', 'Ell_6 (%)',
                               'Ell_Global (%)'])
    df.to_csv(output_csv)
