__all__ = [
    "segment_image",
    "segment_4d_image",
]

from pathlib import Path
from typing import Union
from ccitk.core.common.resource import CineImages, PhaseImage
from ccitk.core.common.constants import DEFAULT_MODEL_PATH

from ccitk.core.segmentor.torch import TorchSegmentor
from ccitk.core.segmentor import CineSegmentor


def segment_4d_image(
        image_path: Union[Path, str], output_dir: Union[Path, str], model_path: Union[Path, str] = None,
        auto_contrast: bool = False, resample: bool = False, enlarge: bool = False,
        overwrite: bool = False, device: Union[int, str] = "cpu"
) -> Path:
    """
    Segment 4D cardiac nifti image using 3D UNet
    :param image_path: string, or Path
        The path of a 4D cine image, or a directory containing nifti of all the phases
    :param output_dir: string, or Path
    :param model_path: string or Path, optional
        The path of the trained model. If not provided, default model will be used, which was trained on UKBB data.
    :param auto_contrast: bool
    :param resample: bool
    :param enlarge: bool
    :param overwrite: boolean
        If overwrite is False, then the segmentor will not be run if the output path exists.
        If overwrite is set to True, then the segmentor will overwrite to the output path, if exists.
    :param device: int, or str, optional
        By default, we will use CPU. If an integer n is provided, then gpu:n will be used.
    :return:
        Path, the output location.
    """
    import mirtk
    if isinstance(image_path, str):
        image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"{str(image_path)} does not exist.")
    if not image_path.is_dir():
        if auto_contrast:
            contrasted_nii_path = image_path.parent.joinpath("contrasted_4d_image.nii.gz")
            mirtk.auto_contrast(str(image_path), str(contrasted_nii_path))
            image_path = contrasted_nii_path
        mirtk.split_volume(
            str(image_path),
            str(output_dir.joinpath("sequence", "image_")),
            "-sequence",
        )
        cine = CineImages.from_dir(output_dir.joinpath("sequence"))
    else:
        cine = CineImages.from_dir(image_path)

        if auto_contrast:
            output_dir.parent.joinpath("contrasted").mkdir(exist_ok=True, parents=True)
            for image in cine:
                contrasted_nii_path = output_dir.parent.joinpath("contrasted").joinpath(image.name)
                mirtk.auto_contrast(str(image_path), str(contrasted_nii_path))
            cine = CineImages.from_dir(output_dir.parent.joinpath("contrasted"))
    images = []
    for image in cine:
        image_path = image.path
        if resample:
            output_dir.parent.joinpath("resampled").mkdir(parents=True, exist_ok=True)
            mirtk.resample_image(
                str(image.path),
                str(output_dir.parent.joinpath("contrasted").joinpath(image.name)),
                '-size', 1.25, 1.25, 2
            )
            image_path = output_dir.parent.joinpath("contrasted").joinpath(image.name)
        if enlarge:
            output_dir.parent.joinpath("enlarged").mkdir(parents=True, exist_ok=True)
            mirtk.enlarge_image(
                str(image.path),
                str(output_dir.parent.joinpath("enlarged").joinpath(image.name)),
                z=20, value=0
            )
            image_path = output_dir.parent.joinpath("enlarged").joinpath(image.name)
        images.append(PhaseImage(path=image_path, phase=image.phase))
    cine = CineImages(images)

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise ValueError(f"{str(output_dir)} is not a directory.")
    if isinstance(model_path, str):
        model_path = Path(model_path)
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    else:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file {str(model_path)} does not exist")

    segmentor = TorchSegmentor(model_path=model_path, overwrite=overwrite, device=device)
    cine_segmentor = CineSegmentor(phase_segmentor=segmentor)
    cine_segmentor.apply(
        cine=cine, output_dir=output_dir, overwrite=overwrite
    )
    return output_dir


def segment_image(
        image_path: Union[Path, str], output_path: Union[Path, str] = None, model_path: Union[Path, str] = None,
        overwrite: bool = False, device: Union[int, str] = "cpu"
) -> Path:
    """
    Segment 3D cardiac nifti image using 3D UNet
    :param image_path: string, or Path
        The path of the nifti image
    :param output_path: string, or Path, optional
        If not provided, the segmentation will be saved to the directory of image_path, under filename "seg.nii.gz"
    :param model_path: string or Path, optional
        The path of the trained model. If not provided, default model will be used, which was trained on UKBB data.
    :param overwrite: boolean
        If overwrite is False, then the segmentor will not be run if the output path exists.
        If overwrite is set to True, then the segmentor will overwrite to the output path, if exists.
    :param device: int, or str, optional
        By default, we will use CPU. If an integer n is provided, then gpu:n will be used.
    :return:
        Path, the output location.
    """
    segmentor = TorchSegmentor(model_path=model_path, overwrite=overwrite, device=device)

    if isinstance(image_path, str):
        image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file {str(image_path)} does not exist")
    if isinstance(output_path, str):
        output_path = Path(output_path)
    if isinstance(model_path, str):
        model_path = Path(model_path)
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    else:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file {str(model_path)} does not exist")

    if output_path is None:
        output_path = image_path.parent.joinpath("seg.nii.gz")
    if not output_path.exists() or overwrite:
        segmentor = TorchSegmentor(model_path=model_path, overwrite=overwrite,device=device)
        segmentor.execute(
            phase_path=image_path, output_path=output_path
        )
    return output_path
