import math
from ccitk.cmr_segment.common.config import AugmentationConfig
import numpy as np
from typing import Tuple
from scipy.ndimage import zoom, rotate


def resize_image(image: np.ndarray, target_shape: Tuple, order: int):
    image_shape = image.shape
    factors = [float(target_shape[i]) / image_shape[i] for i in range(len(image_shape))]
    output = zoom(image, factors, order=order)
    return output


def central_crop_with_padding(image: np.ndarray, label: np.ndarray, output_size: Tuple):
    """
    Crop around the center of the image
    image size = (slice, weight, height)
    label size = (n_class, slice, w, h)
    """
    if image is not None:
        slice, weight, height = image.shape
    if label is not None:
        __, slice, weight, height = label.shape
    s, w, h = output_size

    if w > weight:
        if image is not None:
            image = np.pad(image, ((0, 0), (math.floor(w/2-weight/2), math.ceil(w/2-weight/2)), (0, 0)))
        if label is not None:
            label = np.pad(label, ((0, 0), (0, 0), (math.floor(w/2-weight/2), math.ceil(w/2-weight/2)), (0, 0)))
    if s > slice:
        if image is not None:
            image = np.pad(image, ((math.floor(s / 2 - slice / 2), math.ceil(s / 2 - slice / 2)), (0, 0), (0, 0)))
        if label is not None:
            label = np.pad(label, ((0, 0), (math.floor(s / 2 - slice / 2), math.ceil(s / 2 - slice / 2)), (0, 0), (0, 0)))
    if h > height:
        if image is not None:
            image = np.pad(image, ((0, 0), (0, 0), (math.floor(h/2-height/2), math.ceil(h/2-height/2))))
        if label is not None:
            label = np.pad(label, ((0, 0), (0, 0), (0, 0), (math.floor(h/2-height/2), math.ceil(h/2-height/2))))
    if image is not None:
        slice, weight, height = image.shape
    if label is not None:
        __, slice, weight, height = label.shape
    if image is not None:
        cropped_image = image[
            math.floor((slice-s)/2):math.floor((s+slice)/2),
            math.floor((weight-w)/2):math.floor((w+weight)/2),
            math.floor((height-h)/2):math.floor((h+height)/2),
        ]
    else:
        cropped_image = None
    if label is not None:
        cropped_label = label[
            :,
            math.floor((slice-s)/2):math.floor((s+slice)/2),
            math.floor((weight-w)/2):math.floor((w+weight)/2),
            math.floor((height-h)/2):math.floor((h+height)/2),
        ]
    else:
        cropped_label = None
    return cropped_image, cropped_label


def soi_crop(image: np.ndarray, label: np.ndarray, output_size: Tuple):
    """
    Crop around region of segmentation
    image size = (slice, weight, height)
    label size = (n_class, slice, w, h)
    """
    slice, weight, height = image.shape
    s, w, h = output_size
    label_s = int(np.mean(np.where(label == 1)[1]))
    label_w = int(np.mean(np.where(label == 1)[2]))
    label_h = int(np.mean(np.where(label == 1)[3]))
    lb_s = label_s - s//2
    lb_w = label_w - w//2
    lb_h = label_h - h//2

    ub_s = s + lb_s
    ub_w = w + lb_w
    ub_h = h + lb_h
    if lb_s < 0:
        lb_s = 0
        ub_s = s
        label = np.pad(label, ((0, 0), (s//2 - label_s, 0), (0, 0), (0, 0)))
    if lb_w < 0:
        lb_w = 0
        ub_w = w
        label = np.pad(label, ((0, 0), (0, 0), (w//2 - label_w, 0), (0, 0)))
    if lb_h < 0:
        lb_h = 0
        ub_h = h
        label = np.pad(label, ((0, 0), (0, 0), (0, 0), (h//2 - label_h, 0)))

    if ub_s > slice:
        label = np.pad(label, ((0, 0), (0, ub_s - slice), (0, 0), (0, 0)))
    if ub_w > weight:
        label = np.pad(label, ((0, 0), (0, 0), (0, ub_w - weight), (0, 0)))
    if ub_h > height:
        label = np.pad(label, ((0, 0), (0, 0), (0, 0), (0, ub_h - height)))

    cropped_label = label[:, lb_s: ub_s, lb_w: ub_w, lb_h: ub_h]
    cropped_image = image[lb_s: ub_s, lb_w: ub_w, lb_h: ub_h]
    return cropped_image, cropped_label


def random_crop(image: np.ndarray, label: np.ndarray, output_size: Tuple):
    """
    image size = (slice, weight, height)
    """
    slice, weight, height = image.shape
    s, w, h = output_size

    if slice > s:
        i = np.random.randint(0, int((slice - s) / 2) + 1)
    elif slice == s:
        i = 0
    else:
        i = 0
        i_ = np.random.randint(0, int((s - slice) / 2) + 1)
        image = np.pad(image, ((i_, s - slice - i_), (0, 0), (0, 0)), "constant")
        label = np.pad(label, ((0, 0), (i_, s - slice - i_), (0, 0), (0, 0)), "constant")

    if weight > w:
        j = np.random.randint(0, int((weight - w) / 2) + 1)
    elif weight == s:
        j = 0
    else:
        j = 0
        j_ = np.random.randint(0, int((w - weight) / 2) + 1)
        image = np.pad(image, ((0, 0), (j_, w - weight - j_), (0, 0)), "constant")
        label = np.pad(label, ((0, 0), (0, 0), (j_, w - weight - j_), (0, 0)), "constant")

    if height > h:
        k = np.random.randint(0, int((height - h) / 2) + 1)
    elif height == h:
        k = 0
    else:
        k = 0
        k_ = np.random.randint(0, int((h - height) / 2) + 1)
        image = np.pad(image, ((0, 0), (0, 0), (k_, h - height - k_)), "constant")
        label = np.pad(label, ((0, 0), (0, 0), (0, 0), (k_, h - height - k_)), "constant")

    cropped_image = image[i: i + s, j: j + w, k: k + h]
    cropped_label = label[:, i: i + s, j: j + w, k: k + h]

    return cropped_image, cropped_label


def random_flip(image: np.ndarray, label: np.ndarray, flip_prob: float):
    for axis in range(0, 3):
        if np.random.rand() >= flip_prob:
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis + 1)
    return image, label


def random_rotation(image: np.ndarray, label: np.ndarray, angles: Tuple[float]):
    angle = angles[0]

    rotation_angle = np.random.uniform(-angle, angle)
    image = rotate(image, rotation_angle, axes=(1, 2), order=1)
    label = rotate(label, rotation_angle, axes=(2, 3), order=0)

    return image, label


def random_scaling(image: np.ndarray, label: np.ndarray, delta_factors: Tuple[float]):
    """delta_factor = (0.2, 0.2, 0.2), which leads to scale factors of (1+-0.2, 1+-0.2, 1+-0.2)"""
    factors = []
    for idx, delta in enumerate(delta_factors):
        factors.append(np.random.uniform(1 - delta, 1 + delta))
    image = zoom(image, factors, order=1)
    labels = []
    for i in range(label.shape[0]):
        labels.append(zoom(label[i, :, :, :], factors, order=0))
    label = np.stack(labels, axis=0)
    return image, label


def random_brightness(image, max_delta):
    delta = np.random.uniform(-max_delta, max_delta)
    image = image + delta
    return image


def random_contrast(image, delta):
    lower = 1 - delta
    upper = 1 + delta
    contrast_factor = np.random.uniform(lower, upper)
    mean = np.mean(image)
    image = (image - mean) * contrast_factor + mean
    return image


def adjust_gamma(image, delta):
    gamma = np.random.uniform(1 - delta, 1 + delta)
    # image = 1 * image ** gamma
    image = np.power(image, gamma)
    return image


def random_channel_shift(image, brightness, contrast, gamma):
    image = adjust_gamma(image, gamma)
    image = np.clip(image, 0, 1)
    return image


def rescale_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2


def augment(image: np.ndarray, label: np.ndarray, config: AugmentationConfig, output_size, seed: int = None):
    """image = (slice, weight, height), label = (class, slice, weight, height)"""

    if config.channel_shift:
        image = random_channel_shift(image, config.brightness, config.contrast, config.gamma)
    image, label = random_flip(image, label, config.flip)
    # image, label = random_rotation(image, label, config.rotation_angles)
    image, label = random_scaling(image, label, config.scaling_factors)
    image, label = random_crop(image, label, output_size)
    label[label > 0.5] = 1
    label[label < 0.5] = 0
    image = rescale_intensity(image, (1.0, 99.0))

    return image, label
