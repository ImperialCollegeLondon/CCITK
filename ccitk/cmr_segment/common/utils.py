import nibabel as nib
from skimage.exposure import match_histograms
from pathlib import Path


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
