__all__ = [
    "read_nii_image",
    "rescale_intensity",
]

import numpy as np
from pathlib import Path
import nibabel as nib
from typing import Tuple


def read_nii_image(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read image from nii.gz file and return image and affine matrix (4*4)"""
    nim = nib.load(str(path))
    image = nim.get_data()
    if image.ndim == 4:
        image = np.squeeze(image, axis=-1).astype(np.int16)
    image = image.astype(np.float32)
    return image, nim.affine


def rescale_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2


def normalise_intensity(image, thres_roi=10.0):
    """ Normalise the image intensity by the mean and standard deviation """
    val_l = np.percentile(image, thres_roi)
    roi = (image >= val_l)
    mu, sigma = np.mean(image[roi]), np.std(image[roi])
    eps = 1e-6
    image2 = (image - mu) / (sigma + eps)
    return image2
