__all__ = [
    "read_nii_image",
    "read_nii_label",
    "normalise_intensity",
    "rescale_intensity",
    "resize_image",
    "resize_label",
    "set_affine",
    "split_volume",
    "split_sequence",
    "categorical_dice",
    "one_hot_encode_label"
]

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom
from typing import Tuple, List, Union


def read_nii_image(path: Path, affine: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """ Read image from a .nii.gz file

    Args:
        path: path of the .nii.gz file
        affine (optional): whether or not to return an affine matrix

    Returns:
        - If affine is False, returns image
        - If affine is True, returns a tuple, (image, affine)
    """
    nim = nib.load(str(path))
    image = nim.get_data()
    if image.ndim == 4:
        image = np.squeeze(image, axis=-1).astype(np.int16)
    image = image.astype(np.float32)
    if affine:
        return image, nim.affine
    else:
        return image


def read_nii_label(label_path: Path, labels: List[int], affine: bool = True) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """ ead label from a .nii.gz file

    Args:
        label_path: path to a .nii.gz file, where the value of each pixel represents the label of that pixel.
        labels: a list of integers specifying label values.
        affine (optional): whether or not to return an affine matrix

    Returns:
        - If affine is False, returns label, with shape (n_label, x, y, z)
        - If affine is True, returns a tuple, (label, affine)
    """
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


def rescale_intensity(image: np.ndarray, percentiles: List[float] = None) -> np.ndarray:
    """ Rescale the image intensity  to the range of [0, 1], according to percentile or if not provided, min max values.
    If percentiles are provided, then values outside of the percentiles will be set to the values of percentiles.

    Args:
        image: image data
        percentiles (optional): intensity values clip off percentiles, such as (1, 99)

    Return:
        Image with rescaled intensity in the range of [0, 1]
    """
    if percentiles is not None:
        val_l, val_h = np.percentile(image, percentiles)
        image2 = image
        image2[image < val_l] = val_l
        image2[image > val_h] = val_h
    else:
        val_h = np.max(image)
        val_l = np.min(image)
        image2 = image

    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2


def normalise_intensity(image: np.ndarray, thres_roi: float=10.0) -> np.ndarray:
    """ Normalise the image intensity by the mean and standard deviation

    Args:
        image: numpy array
        thres_roi (optional): default 10.0, threshold percentile,
                              mean and std are computed with values above this percentile

    Return:
        Normalised image
    """
    val_l = np.percentile(image, thres_roi)
    roi = (image >= val_l)
    mu, sigma = np.mean(image[roi]), np.std(image[roi])
    eps = 1e-6
    image2 = (image - mu) / (sigma + eps)
    return image2


def resize_image(image: np.ndarray, target_shape: Tuple, order: int = 0) -> np.ndarray:
    """ Resize nifty image

    Args:
        image: numpy array
        target_shape: tuple, such as (x, y, z)
        order (optional): integer, the order of approximation, default is 0, which is choosing the nearest value.

    Returns:
        Resized image
    """
    image_shape = image.shape
    factors = [float(target_shape[i]) / image_shape[i] for i in range(len(image_shape))]
    output = zoom(image, factors, order=order)
    return output


def resize_label(label: np.ndarray, target_shape: Tuple) -> np.ndarray:
    """ Resize nifty image

    Args:
        label: numpy array, shape (n_label, x, y, z)
        target_shape: tuple, such as (x, y, z)

    Returns:
        Resized label

    """
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
    """ Set the affine matrix of from_image to to_image.

    Args:
        from_image: path of nifty image
        to_image: path of nifty image

    Returns:
        None
    """
    nim = nib.load(str(from_image))
    nim2 = nib.load(str(to_image))

    image = nim2.get_data()
    nim3 = nib.Nifti1Image(image, nim.affine)
    nim3.header['pixdim'] = nim.header['pixdim']
    nib.save(nim3, str(to_image))


def split_volume(image_name: Union[Path, str], output_name: str):
    """ Split an image volume into a number of slices.

    Args:
        image_name: path of a nifty image
        output_name: output filename prefix

    Returns:
        None
    """
    nim = nib.load(str(image_name))
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


def split_sequence(image_name: Union[Path, str], output_name: str):
    """ Split an image sequence into a number of time frames.

    Args:
        image_name: path of a nifty image
        output_name: output filename prefix

    Returns:
        None
    """

    nim = nib.load(str(image_name))
    T = nim.header['dim'][4]
    affine = nim.affine
    image = nim.get_data()

    for t in range(T):
        image_fr = image[:, :, :, t]
        nim2 = nib.Nifti1Image(image_fr, affine)
        nib.save(nim2, '{0}{1:02d}.nii.gz'.format(output_name, t))


def categorical_dice(prediction: np.ndarray, truth: np.ndarray, k: int, is_one_hot: bool = False,
                     n_classes: int = None, epsilon: float = 0.001) -> float:
    """ Dice overlap metric for label k
    prediction and truth can be one-hot encoded or integer encoded.
    If one-hot encoded, they are of the shape (K, D, H, W), or (K, H, W)
    If integer encoded, they are of the shape (D, H, W), or (H, W), and n_classes must be provided.
    epsilon is for numerical stability, to avoid 0/0

    Args:
        prediction: numpy array
        truth: numpy array
        k: integer, label k
        is_one_hot (optional): if label maps are one-hot encoded
        n_classes (optional): this is needed if is_one_hot is false
        epsilon (optional): for numerical stability

    Returns:
        Dice score, float
    """

    if not is_one_hot:
        assert n_classes is not None
        prediction = one_hot_encode_label(prediction, n_classes)

    A = (np.argmax(prediction, axis=0) == k)
    B = (np.argmax(truth, axis=0) == k)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B) + epsilon)


def one_hot_encode_label(label: np.ndarray, n_classes: int) -> np.ndarray:
    """ Label can be of the shape (D, H, W), or (H, W)

    Args:
        label: numpy array, integer value encoded
        n_classes: number of classes

    Returns:
        one-hot encoded label, shape (n_classes, D, H, W) or (n_classes, H, W)
    """
    label_flat = label.flatten()
    n_data = len(label_flat)
    label_one_hot = np.zeros((n_data, n_classes), dtype='int16')
    label_one_hot[range(n_data), label_flat] = 1
    if len(label.shape) == 3:
        label_one_hot = label_one_hot.reshape((label.shape[0], label.shape[1], label.shape[2], n_classes))
    else:
        label_one_hot = label_one_hot.reshape((label.shape[0], label.shape[1], n_classes))
    label_one_hot = np.moveaxis(label_one_hot, -1, 0)
    return label_one_hot
