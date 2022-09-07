import math
import mirtk
import shutil
import numpy as np
import nibabel as nib
from typing import Tuple
from pathlib import Path
from scipy.ndimage import label

from CMRSegment.segmentor.tf1 import TF1Segmentor
from CMRSegment.common.utils import rescale_intensity


class TF13DSegmentor(TF1Segmentor):
    def run(self, image: np.ndarray) -> np.ndarray:
        image = rescale_intensity(image, (1, 99))
        # Pad the image size to be a factor of 16 so that the downsample and upsample procedures
        # in the network will result in the same image size at each resolution level.
        X, Y, Z = image.shape
        n_slices = 100
        X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
        x_pre, y_pre, z_pre = int((X2 - X) / 2), int((Y2 - Y) / 2), int((Z - n_slices) / 2)
        x_post, y_post, z_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre, (Z - n_slices) - z_pre
        z1, z2 = int(Z / 2) - int(n_slices / 2), int(Z / 2) + int(n_slices / 2)
        z1_, z2_ = max(z1, 0), min(z2, Z)
        image = image[:, :, z1_: z2_]
        image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (z1_ - z1, z2 - z2_)), 'constant')
        # extend batch for network requirement
        image = np.expand_dims(image, axis=0)
        pred_segt = self._sess.run(['pred_segt:0'], feed_dict={'image:0': image, 'training:0': False})
        # Transpose and crop the segmentation to recover the original size
        pred_segt = np.squeeze(pred_segt[0], axis=0).astype(np.int16)
        # map back to original size
        pred_segt = pred_segt[x_pre:x_pre + X, y_pre:y_pre + Y, z1_ - z1:z1_ - z1 + Z]
        return pred_segt

    def execute(self, phase_path: Path, output_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        nim = nib.load(str(phase_path))
        image = nim.get_data()
        if image.ndim == 4:
            image = np.squeeze(image, axis=-1).astype(np.int16)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        if not output_path.exists() or self.overwrite:
            pred_segt = self.run(image)
            pred_segt = refined_mask(pred_segt, phase_path, output_path.parent.joinpath("tmp"))
            # map back to original size
            nim2 = nib.Nifti1Image(pred_segt, nim.affine)
            nim2.header['pixdim'] = nim.header['pixdim']
            nib.save(nim2, str(output_path))
        else:
            pred_segt = nib.load(str(output_path)).get_data()
        return image, pred_segt


def get_labels(seg, label):
    mask = np.zeros(seg.shape,dtype=np.uint8)
    mask[seg[:, :, :] == label] = 1
    return mask


def refined_mask(pred_segt: np.ndarray, phase_path: Path, tmp_dir: Path):
    nim = nib.load(str(phase_path))
    ###########################################################################
    nim2 = nib.Nifti1Image(pred_segt, nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir.joinpath(phase_path.name)
    nib.save(nim2, str(tmp_path))
    ###########################################################################
    nim = nib.load(str(tmp_path))
    lvsa_data = nim.get_data()
    lvsa_data_bin = np.where(lvsa_data > 0, 1, lvsa_data)
    labelled_mask, num_labels = label(lvsa_data_bin)
    refined_mask = lvsa_data.copy()
    minimum_cc_sum = 5000
    for labl in range(num_labels + 1):
        if np.sum(refined_mask[labelled_mask == labl]) < minimum_cc_sum:
            refined_mask[labelled_mask == labl] = 0
    final_mask = np.zeros(refined_mask.shape, dtype=np.uint8)
    lv = get_labels(refined_mask, 1)
    myo = get_labels(refined_mask, 2)
    rv = get_labels(refined_mask, 3)
    final_mask[lv[:, :, :] == 1] = 1
    final_mask[myo[:, :, :] == 1] = 2
    final_mask[rv[:, :, :] == 1] = 3
    nim2 = nib.Nifti1Image(final_mask[:, :, :], affine=np.eye(4))
    nim2.header['pixdim'] = nim.header['pixdim']
    nib.save(nim2, str(tmp_path))
    mirtk.header_tool(str(tmp_path), str(tmp_path), target=str(phase_path))
    nim = nib.load(str(tmp_path))
    pred_segt = nim.get_data()
    pred_segt = np.squeeze(pred_segt, axis=-1)
    shutil.rmtree(str(tmp_dir), ignore_errors=True)
    return pred_segt
