import numpy as np
import nibabel as nib
from skimage.exposure import match_histograms
from pathlib import Path


def set_affine(from_image: Path, to_image: Path):
    nim = nib.load(str(from_image))
    nim2 = nib.load(str(to_image))

    image = nim2.get_data()
    nim3 = nib.Nifti1Image(image, nim.affine)
    nim3.header['pixdim'] = nim.header['pixdim']
    nib.save(nim3, str(to_image))


def extract_rv_label(segmentation_path: Path, output_path: Path, overwrite: bool = False):
    import mirtk
    if not output_path.exists() or overwrite:
        mirtk.calculate_element_wise(
            str(segmentation_path),
            "-label", 3, 4,
            set=255, pad=0,
            output=str(output_path),
        )
    return output_path


def extract_lv_label(segmentation_path: Path, output_path: Path, overwrite: bool = False):
    import mirtk
    if not output_path.exists() or overwrite:
        mirtk.calculate_element_wise(
            str(segmentation_path),
            "-label", 3, 4, set=0,
            output=str(output_path),
        )
    return output_path


def rescale_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    # val_l, val_h = np.percentile(image, thres)
    val_h = np.max(image)
    val_l = np.min(image)
    image2 = image
    # image2[image < val_l] = val_l
    # image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2


def ED_ES_histogram_matching(reference_subject, target_subject):
    reference_ed_nim = nib.load(str(reference_subject.ed_path))
    reference_ed_image = reference_ed_nim.get_data()
    # reference_ed_image = rescale_intensity(reference_ed_image)

    reference_es_nim = nib.load(str(reference_subject.es_path))
    reference_es_image = reference_es_nim.get_data()
    # reference_es_image = rescale_intensity(reference_es_image)

    target_ed_nim = nib.load(str(target_subject.ed_path))
    target_ed_image = target_ed_nim.get_data()
    # target_ed_image = rescale_intensity(target_ed_image)

    target_es_nim = nib.load(str(target_subject.es_path))
    target_es_image = target_es_nim.get_data()
    # target_es_image = rescale_intensity(target_es_image)

    matched_ed = match_histograms(target_ed_image, reference_ed_image, multichannel=False)
    matched_es = match_histograms(target_es_image, reference_es_image, multichannel=False)

    nim2 = nib.Nifti1Image(matched_ed, affine=target_ed_nim.affine)
    nim2.header['pixdim'] = target_ed_nim.header['pixdim']
    nib.save(nim2, str(target_subject.ed_path))

    nim2 = nib.Nifti1Image(matched_es, affine=target_es_nim.affine)
    nim2.header['pixdim'] = target_es_nim.header['pixdim']
    nib.save(nim2, str(target_subject.es_path))


def read_nii_segmentation(path: Path):
    """Read segmentation from nii.gz file and return seg and affine matrix (4*4)"""
    nim = nib.load(str(path))
    affine = nim.affine

    seg = nim.get_data()
    if seg.ndim == 4:
        seg = np.squeeze(seg, axis=-1).astype(np.int16)
    seg = seg.astype(np.float32)
    return seg, affine


def read_nii_image(path: Path):
    """Read image from nii.gz file and return image and affine matrix (4*4)"""
    nim = nib.load(str(path))
    image = nim.get_data()
    if image.ndim == 4:
        image = np.squeeze(image, axis=-1).astype(np.int16)
    image = image.astype(np.float32)
    return image, nim.affine
