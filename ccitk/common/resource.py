import os
import numpy as np
from enum import Enum
import nibabel as nib
from pathlib import Path

from typing import List, Union


class Phase(Enum):
    """
    Enum for indicating cardica phases, ED or ES
    """
    ED = "ED"
    ES = "ES"

    def __str__(self):
        return self.value


class Resource:
    """
    Resource base class, a wrapper around pathlib.Path
    """
    def __init__(self, path: Path):
        self.path = path

    def __str__(self):
        return str(self.path)

    def __getattr__(self, name: str) -> object:
        # pathlib.Path attributes
        for key in ("stem", "name", "suffix", "suffixes"):
            if name == key:
                return getattr(self.path, key)
        raise AttributeError("{} has no attribute named '{}'".format(type(self).__name__, name))

    def exists(self):
        return self.path.exists()


class RVMesh:
    """
    RV Mesh, containing meshes blood pool (rv) and epicardium (optional)
    """
    def __init__(self, rv: Path, epicardium: Path = None):
        self.rv = rv
        self.epicardium = epicardium

    @classmethod
    def from_dir(cls, dir: Path, phase: Phase):
        rv = Path(dir.joinpath(f"RV_{phase}.vtk"))
        epi = Path(dir.joinpath(f"RVepi_{phase}.vtk"))
        return cls(rv, epi)

    def exists(self):
        return self.rv.exists()

    def check_valid(self):
        if self.exists():
            return True
        if not self.rv.exists():
            raise FileNotFoundError(f"RV mesh does not exists at {self.rv}.")


class LVMesh:
    """
    LV Mesh, containing three meshes, epicardium, endocardium, and myocardium
    """
    def __init__(self, epicardium: Path, endocardium: Path, myocardium: Path):
        self.epicardium = epicardium
        self.endocardium = endocardium
        self.myocardium = myocardium

    @classmethod
    def from_dir(cls, dir: Path, phase: Phase):
        epi = Path(dir.joinpath(f"LVepi_{phase}.vtk"))
        endo = Path(dir.joinpath(f"LVendo_{phase}.vtk"))
        myo = Path(dir.joinpath(f"LVmyo_{phase}.vtk"))
        return cls(epi, endo, myo)

    def exists(self):
        return self.endocardium.exists() and self.epicardium.exists() and self.myocardium.exists()

    def check_valid(self):
        if self.exists():
            return True
        if not self.endocardium.exists():
            raise FileNotFoundError(f"LV endo mesh does not exists at {self.endocardium}.")
        if not self.epicardium.exists():
            raise FileNotFoundError(f"LV epi mesh does not exists at {self.epicardium}.")
        if not self.myocardium.exists():
            raise FileNotFoundError(f"LV myo mesh does not exists at {self.myocardium}.")


class CardiacMesh:
    """
    Mesh for the whole heart, containing lv mesh and rv mesh
    """
    def __init__(self, rv: RVMesh, lv: LVMesh, phase: Union[Phase, str, int]):
        self.rv = rv
        self.lv = lv
        if isinstance(phase, str):
            phase = Phase[phase]
        self.phase = phase

    @classmethod
    def from_dir(cls, dir: Path, phase: Union[Phase, str, int]):
        rv = RVMesh.from_dir(dir, phase)
        lv = LVMesh.from_dir(dir, phase)
        return cls(rv, lv, phase)

    def exists(self):
        return self.rv.exists() and self.lv.exists()

    def check_valid(self):
        if self.exists():
            return True
        self.rv.check_valid()
        self.lv.check_valid()


class ImageResource(Resource):
    """

    """
    def get_data(self) -> np.ndarray:
        nim = nib.load(str(self.path))
        seg = nim.get_data()
        return seg


class Segmentation(ImageResource):
    """

    """
    def __init__(self, path: Path, phase: Union[Phase, str, int]):
        self.path = path
        self.phase = phase
        super().__init__(path)


class NiiData(ImageResource):
    """

    """
    @classmethod
    def from_dir(cls, dir: Path):
        assert dir.is_dir(), "{} is not a directory.".format(str(dir))
        nii_path = dir.joinpath("LVSA.nii.gz")
        return cls(nii_path)


class PhaseImage(ImageResource):
    """

    """
    def __init__(self, path: Path, phase: Union[Phase, int]):
        self.path = path
        self.phase = phase
        super().__init__(path)

    @classmethod
    def from_dir(cls, dir: Path, phase: Union[Phase, int]):
        assert dir.is_dir(), "{} is not a directory.".format(str(dir))
        filename = None
        if phase == Phase.ED and filename is None:
            filename = "lvsa_ED.nii.gz"
        if phase == Phase.ES and filename is None:
            filename = "lvsa_ES.nii.gz"
        if type(phase) is int:
            filename = "lvsa_{}.nii.gz".format(phase)
        assert filename is not None
        nii_path = dir.joinpath(filename)
        return cls(nii_path, phase)

    def __repr__(self):
        return "PhaseImage(path={}, phase={})".format(self.path, self.phase)


class CineImages:
    """

    """
    def __init__(self, images: List[PhaseImage]):
        self.images = images

    @classmethod
    def from_dir(cls, dir: Path):
        assert dir.is_dir(), "{} is not a directory.".format(str(dir))
        images = []
        for idx, phase_name in enumerate(os.listdir(str(dir))):
            phase_path = dir.joinpath(phase_name)
            image = PhaseImage(path=phase_path, phase=idx)
            images.append(image)
        return cls(images)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return iter(self.images)

    def __getitem__(self, index: int):
        return self.images[index]
