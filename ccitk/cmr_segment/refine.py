import mirtk
import numpy as np
from pathlib import Path

from ccitk.cmr_segment.common.constants import RESOURCE_DIR
from ccitk.resource import Segmentation, PhaseImage
from ccitk.data_table import DataTable
from ccitk.refine import refine_segmentation_with_atlases


mirtk.subprocess.showcmd = True


class SegmentationRefiner:
    """Use multi-atlas registration to refine predicted segmentation"""
    def __init__(self, csv_path: Path, n_atlas: int = None, param_path: Path = None):
        assert csv_path.exists(), "Path to csv file containing list of atlases must exist. "
        data_table = DataTable.from_csv(csv_path)
        label_paths = data_table.select_column("label_path")
        landmarks = []
        atlases = []
        for idx, path in enumerate(label_paths):
            if idx % 2 == 0:
                if Path(path).parent.joinpath("landmarks2.vtk").exists():
                    atlases.append(Path(path))
                    landmarks.append(Path(path).parent.joinpath("landmarks2.vtk"))
        print("Total {} atlases with landmarks...".format(len(atlases)))
        if n_atlas is not None:
            if n_atlas < len(atlases):
                print("Randomly choosing {} atlases...".format(n_atlas))
                indices = np.random.choice(np.arange(len(atlases)), n_atlas, replace=False)
                atlases = np.array(atlases)
                landmarks = np.array(landmarks)
                atlases = atlases[indices].tolist()
                landmarks = landmarks[indices].tolist()
                print("Total {} atlases remained...".format(len(atlases)))

        self.atlases = atlases
        self.landmarks = landmarks
        if param_path is None:
            param_path = RESOURCE_DIR.joinpath("ffd_label_1.cfg")
        self.param_path = param_path
        self.affine_param_path = RESOURCE_DIR.joinpath("segareg_2.txt")

    def run(self, subject_image: PhaseImage, subject_seg: Segmentation, subject_landmarks: Path, output_dir: Path,
            n_top: int, force: bool) -> Segmentation:
        output_path = output_dir.joinpath(subject_seg.path.stem + "_refined.nii.gz")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        refine_segmentation_with_atlases(
            atlases_label=self.atlases,
            atlases_landmark=self.landmarks,
            subject_segmentation=subject_seg.path,
            subject_image=subject_image.path,
            subject_landmarks=subject_landmarks,
            affine_parin=self.affine_param_path,
            ffd_parin=self.param_path,
            output_path=output_path,
            n_top=n_top,
            overwrite=force,
            phase=str(subject_image.phase),
        )

        return Segmentation(
            path=output_path,
            phase=subject_seg.phase
        )
