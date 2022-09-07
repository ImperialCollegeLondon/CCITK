from pathlib import Path
from vedo import Plotter
import numpy as np


def plot_nii_gz(file_path: Path):
    vp = Plotter()
    image = vp.load(str(file_path))
    print(np.max(image.getDataArray()), np.sum(image.getDataArray() == 3), np.sum(image.getDataArray() == 2), np.sum(image.getDataArray() == 1), np.sum(image.getDataArray() == 0), np.sum(image.getDataArray() == 255), np.mean(image.getDataArray()[image.getDataArray()>0]))
    # vp.show(image)
    # vp.close()
