__all__ = [
    "read_nii_image",
    "rescale_intensity",
    "resize_image",
    "resize_label",
    "read_nii_label",
    "set_affine",
]

import numpy as np
from pathlib import Path
import nibabel as nib
from typing import Tuple, List, Union
from scipy.ndimage import zoom, rotate


def read_nii_image(path: Path, affine: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Read image from nii.gz file and return image and affine matrix (4*4)"""
    nim = nib.load(str(path))
    image = nim.get_data()
    if image.ndim == 4:
        image = np.squeeze(image, axis=-1).astype(np.int16)
    image = image.astype(np.float32)
    if affine:
        return image, nim.affine
    else:
        return image


def read_nii_label(label_path: Path, labels: List[int], affine: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Return (n_label, x, y, z)"""
    label, affine_matrix = read_nii_image(label_path)
    X, Y, Z = label.shape

    new_labels = []
    for i in labels:
        blank_image = np.zeros((X, Y, Z))

        blank_image[label == i] = 1
        new_labels.append(blank_image)
    label = np.array(new_labels)
    if affine:
        return label, affine_matrix
    else:
        return label


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


def resize_image(image: np.ndarray, target_shape: Tuple, order: int = 0):
    image_shape = image.shape
    factors = [float(target_shape[i]) / image_shape[i] for i in range(len(image_shape))]
    output = zoom(image, factors, order=order)
    return output


def resize_label(label: np.ndarray, target_shape: Tuple):
    # label: (3, H, W, D)
    label_shape = label.shape
    factors = [float(target_shape[i - 1]) / label_shape[i] for i in range(1, len(label_shape))]
    labels = []
    for i in range(label_shape[0]):
        output = zoom(label[i], factors, order=0)
        labels.append(output)
    label = np.array(labels)
    return label


def set_affine(from_image: Path, to_image: Path):
    """
        Set the affine matrix from from_image to to_image.
    """
    nim = nib.load(str(from_image))
    nim2 = nib.load(str(to_image))

    image = nim2.get_data()
    nim3 = nib.Nifti1Image(image, nim.affine)
    nim3.header['pixdim'] = nim.header['pixdim']
    nib.save(nim3, str(to_image))
