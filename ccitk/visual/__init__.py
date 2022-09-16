from .warp_labels import visualize_warped_labels_motion
from .mesh_contour import visualize_mesh_contour_motion
from pathlib import Path
import numpy as np

__all__ = [
    "visualize_warped_labels_motion",
    "visualize_mesh_contour_motion",
    "plot_nii_gz"
]


def plot_nii_gz(file_path: Path):
    from vedo import Plotter
    vp = Plotter()
    image = vp.load(str(file_path))
    vp.show(image)
    vp.close()
