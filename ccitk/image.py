__all__ = [
    "read_nii_image",
    "rescale_intensity",
    "resize_image",
    "resize_label",
    "read_nii_label",
    "set_affine",
    "split_volume",
    "split_sequence",
    "categorical_dice",
    "one_hot_encode_label"
]

import numpy as np
from pathlib import Path
import nibabel as nib
from scipy.ndimage import zoom
from typing import Tuple, List, Union


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


def rescale_intensity(image, thres: List[float] = None):
    """ Rescale the image intensity to the range of [0, 1] """
    if thres is not None:
        val_l, val_h = np.percentile(image, thres)
        image2 = image
        image2[image < val_l] = val_l
        image2[image > val_h] = val_h
    else:
        val_h = np.max(image)
        val_l = np.min(image)
        image2 = image

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


def split_volume(image_name, output_name):
    """ Split an image volume into a number of slices. """
    nim = nib.load(image_name)
    Z = nim.header['dim'][3]
    affine = nim.affine
    image = nim.get_data()

    for z in range(Z):
        image_slice = image[:, :, z]
        image_slice = np.expand_dims(image_slice, axis=2)
        affine2 = np.copy(affine)
        affine2[:3, 3] += z * affine2[:3, 2]
        nim2 = nib.Nifti1Image(image_slice, affine2)
        nib.save(nim2, '{0}{1:02d}.nii.gz'.format(output_name, z))


def split_sequence(image_name, output_name):
    """ Split an image sequence into a number of time frames. """
    nim = nib.load(image_name)
    T = nim.header['dim'][4]
    affine = nim.affine
    image = nim.get_data()

    for t in range(T):
        image_fr = image[:, :, :, t]
        nim2 = nib.Nifti1Image(image_fr, affine)
        nib.save(nim2, '{0}{1:02d}.nii.gz'.format(output_name, t))


def categorical_dice(prediction: np.ndarray, truth: np.ndarray, k: int, is_one_hot: bool = False,
                     n_classes: int = None, epsilon: float = 0.001):
    """
        Dice overlap metric for label k
        prediction and truth can be one-hot encoded or integer encoded.
        If one-hot encoded, they are of the shape (K, D, H, W), or (K, H, W)
        If integer encoded, they are of the shape (D, H, W), or (H, W), and n_classes must be provided.
        epsilon is for numerical stability, to avoid 0/0
    """
    if not is_one_hot:
        assert n_classes is not None
        prediction = one_hot_encode_label(prediction, n_classes)

    A = (np.argmax(prediction, axis=1) == k)
    B = (np.argmax(truth, axis=1) == k)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B) + epsilon)


def one_hot_encode_label(label: np.ndarray, n_classes: int):
    """
        Label can be of the shape (D, H, W), or (H, W)
    """
    label_flat = label.flatten()
    n_data = len(label_flat)
    label_one_hot = np.zeros((n_data, n_classes), dtype='int16')
    label_one_hot[range(n_data), label_flat] = 1
    if len(label.shape) == 3:
        label_one_hot = label_one_hot.reshape((label.shape[0], label.shape[1], label.shape[2], n_classes))
    else:
        label_one_hot = label_one_hot.reshape((label.shape[0], label.shape[1], n_classes))
    return label_one_hot
