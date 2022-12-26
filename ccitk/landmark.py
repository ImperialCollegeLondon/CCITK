__all__ = [
    "extract_simple_landmarks",
]

import vtk
import nibabel as nib
import numpy as np
from typing import List
from pathlib import Path


def check_labels(seg, labels):
    list_labels = []
    for i in range(5):
        if seg[seg == i].shape[0] > 0:
            list_labels.append(i)
        else:
            continue
    if all(label in list_labels for label in labels):
        flag = True
    else:
        flag = False
    return flag


def extract_simple_landmarks(segmentation: Path, output_path: Path, labels: List = None) -> Path:
    """ Extract landmarks from a LVSA nifti segmentation

    Args:
        segmentation: Path, segmentation path
        output_path: Path, output landmarks path
        labels (optional): label numbers of the structures on which you want to compute the landmarks.
                           If labels is None, then labels = [2, 3]. Labels is in range 0 to 4, inclusive.
    """

    if labels is None:
        labels = [2, 3]
    nim = nib.load(str(segmentation))
    affine = nim.affine
    seg = nim.get_data()

    if check_labels(seg, labels):
        # Extract the z axis from the nifti header
        lm = []
        z_axis = np.copy(nim.affine[:3, 2])

        # loop on all the segmentation labels of interest
        for l in labels:
            # Determine the z range
            z = np.nonzero(seg == l)[2]

            z_min, z_max = z.min(), z.max()
            z_mid = int(round(0.5 * (z_min + z_max)))

            # compute landmarks positions
            if z_axis[2] < 0:
                # z_axis starts from base
                zs = [z_min, z_mid, z_max]
            else:
                # z_axis starts from apex
                zs = [z_max, z_mid, z_min]

            for z in zs:
                seg = np.squeeze(seg)
                x, y = [np.mean(i) for i in np.nonzero(seg[:, :, z] == l)]
                #                    x, y = [np.mean(i) for i in np.nonzero(seg[:, :, z, 0] == l)]
                # this might need to be changed depending on the segmentation data structure
                p = np.dot(affine, np.array([x, y, z, 1]).reshape((4, 1)))[:3, 0]
                lm.append(p)
        # Write the landmarks
        points = vtk.vtkPoints()
        for p in lm:
            points.InsertNextPoint(p[0], p[1], p[2])
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(poly)
        writer.SetFileName(str(output_path))
        writer.Write()
    else:
        print("\n ... Error in labels")
        raise ValueError("Check labels failed when extracting landmarks.")
    return output_path
