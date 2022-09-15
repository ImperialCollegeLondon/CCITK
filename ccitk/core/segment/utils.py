import mirtk
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import label


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

    # TODO: add to mirtk
    mirtk.header_tool(str(tmp_path), str(tmp_path), target=str(phase_path))

    nim = nib.load(str(tmp_path))
    pred_segt = nim.get_data()
    pred_segt = np.squeeze(pred_segt, axis=-1)
    shutil.rmtree(str(tmp_dir), ignore_errors=True)
    return pred_segt
