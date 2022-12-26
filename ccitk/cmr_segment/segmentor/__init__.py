import numpy as np
from tqdm import tqdm
import nibabel as nib
from pathlib import Path
from typing import List, Tuple

from ccitk.common.resource import PhaseImage, Segmentation, CineImages


class Segmentor:
    """
    This module uses a trained 3D convolutional neural network (UNet) to segment ED/ES enlarged images.
    The code to train the network is in experiments/fcn_3d/

    It takes input

    .. code-block:: text

        /path/
            subject1/
                lvsa_SR_ED.nii.gz               ->  Enlarged ED image
                lvsa_SR_ES.nii.gz               ->  Enlarged ES image
                enlarged/
                    lvsa_0.nii.gz               ->  Enlarged phase 0 image
                    lvsa_1.nii.gz               ->  Enlarged phase 1 image
                    ...
            subject2/
                ...
    and generates output

    .. code-block:: text

        /path/
            subject1/
                seg_lvsa_SR_ED.nii.gz           ->  ED segmentation
                seg_lvsa_SR_ES.nii.gz           ->  ES segmentation
                segs/
                    lvsa_0.nii.gz               ->  Phase 0 segmentation
                    lvsa_1.nii.gz               ->  Phase 1 segmentation
                    ...
                4D_rview/
                    4Dimg.nii.gz                ->  4D image
                    4Dseg.nii.gz                ->  4D segmentation
            subject2/
                ...

    To run only the segmentor, use the following command:

    .. code-block:: text

        ccitk-cmr-segment -o /output-dir/ --data-dir /input-dir/ --segment --model-path /path-to-model --torch --segment-cine

    """
    def __init__(self, model_path: Path, overwrite: bool = False):
        self.model_path = model_path
        self.overwrite = overwrite

    def run(self, image: np.ndarray) -> np.ndarray:
        """Process image numpy array and return segmentation as np array"""
        raise NotImplementedError("Must be implemented by subclasses.")

    def apply(self, image: PhaseImage, output_path: Path) -> Segmentation:
        """Apply segmentor to PhaseImage and return Segmentation"""
        np_image, predicted = self.execute(image.path, output_path)
        return Segmentation(phase=image.phase, path=output_path)

    def execute(self, phase_path: Path, output_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Segment a 3D volume cardiac phase from phase_path, save to output_dir"""
        raise NotImplementedError("Must be implemented by subclasses.")


class CineSegmentor:
    """
    Use :class:`Segmentor` to segment each phase of the cardiac cycle.
    """
    def __init__(self, phase_segmentor: Segmentor):
        """ Cine CMR consists in the acquisition of the same slice position at different phases of the cardiac cycle."""
        self.__segmentor = phase_segmentor

    def apply(self, cine: CineImages, output_dir: Path, overwrite: bool = False) -> List[Segmentation]:
        segmentations = []
        output_dir.joinpath("segs").mkdir(parents=True, exist_ok=True)
        for idx, image in enumerate(tqdm(cine)):
            output_path = output_dir.joinpath("segs").joinpath(f"lvsa_{idx}.nii.gz")
            if overwrite or not output_path.exists():
                segmentation = self.__segmentor.apply(
                    image, output_path=output_path
                )
            else:
                segmentation = Segmentation(phase=image.phase, path=output_path)
            segmentations.append(segmentation)
        if overwrite or not output_dir.joinpath("4D_rview", "4Dseg.nii.gz").exists() or \
            not output_dir.joinpath("4D_rview", "4Dimg.nii.gz").exists():
            nim = nib.load(str(segmentations[-1].path))
            # batch * height * width * channels (=slices)
            segt_labels = np.array([seg.get_data() for seg in segmentations], dtype=np.int32)
            segt_labels = np.transpose(segt_labels, (1, 2, 3, 0))
            images = np.array([np.squeeze(image.get_data(), axis=3) for image in cine], dtype=np.float32)  # b
            images = np.transpose(images, (1, 2, 3, 0))
            nim2 = nib.Nifti1Image(segt_labels, nim.affine)
            nim2.header['pixdim'] = nim.header['pixdim']
            output_dir.joinpath("4D_rview").mkdir(exist_ok=True, parents=True)
            nib.save(nim2, str(output_dir.joinpath("4D_rview", "4Dseg.nii.gz")))
            nim2 = nib.Nifti1Image(images, nim.affine)
            nim2.header['pixdim'] = nim.header['pixdim']
            nib.save(nim2, str(output_dir.joinpath("4D_rview", "4Dimg.nii.gz")))
        return segmentations
