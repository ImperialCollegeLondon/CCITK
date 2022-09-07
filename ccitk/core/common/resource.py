import os
import shutil
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import nibabel as nib


class Phase(Enum):
    ED = "ED"
    ES = "ES"

    def __str__(self):
        return self.value


@dataclass
class Mesh:
    phase: Union[Phase, str, int]
    dir: Path

    @property
    def rv(self):
        return self.dir.joinpath(f"RV_{self.phase}.vtk")

    @property
    def rv_epi(self):
        return self.dir.joinpath(f"RVepi_{self.phase}.vtk")

    @property
    def lv_endo(self):
        return self.dir.joinpath(f"LVendo_{self.phase}.vtk")

    @property
    def lv_epi(self):
        return self.dir.joinpath(f"LVepi_{self.phase}.vtk")

    @property
    def lv_myo(self):
        return self.dir.joinpath(f"LVmyo_{self.phase}.vtk")

    def exists(self):
        return self.rv.exists() and self.rv_epi.exists() and self.lv_endo.exists() and self.lv_epi.exists()\
               and self.lv_myo.exists()

    def check_valid(self):
        if self.exists():
            return True
        if not self.rv.exists():
            raise FileNotFoundError(f"RV mesh does not exists at {self.rv}.")
        if not self.rv_epi.exists():
            raise FileNotFoundError(f"RV epi mesh does not exists at {self.rv_epi}.")
        if not self.lv_endo.exists():
            raise FileNotFoundError(f"LV endo mesh does not exists at {self.lv_endo}.")
        if not self.lv_epi.exists():
            raise FileNotFoundError(f"LV epi mesh does not exists at {self.lv_epi}.")
        if not self.lv_myo.exists():
            raise FileNotFoundError(f"LV myo mesh does not exists at {self.lv_myo}.")


class Resource:
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


class MeshResource(Resource):
    pass


class RVMesh:
    def __init__(self, mesh: MeshResource, epicardium: MeshResource):
        self.rv = mesh
        self.epicardium = epicardium

    @classmethod
    def from_dir(cls, dir: Path, phase: Phase):
        rv = MeshResource(dir.joinpath(f"RV_{phase}.vtk"))
        epi = MeshResource(dir.joinpath(f"RVepi_{phase}.vtk"))
        return cls(rv, epi)

    def exists(self):
        return self.rv.exists()

    def check_valid(self):
        if self.exists():
            return True
        if not self.rv.exists():
            raise FileNotFoundError(f"RV mesh does not exists at {self.rv}.")
        if not self.epicardium.exists():
            raise FileNotFoundError(f"RV epi mesh does not exists at {self.epicardium}.")


class LVMesh:
    def __init__(self, epicardium: MeshResource, endocardium: MeshResource, myocardium: MeshResource):
        self.epicardium = epicardium
        self.endocardium = endocardium
        self.myocardium = myocardium

    @classmethod
    def from_dir(cls, dir: Path, phase: Phase):
        epi = MeshResource(dir.joinpath(f"LVepi_{phase}.vtk"))
        endo = MeshResource(dir.joinpath(f"LVendo_{phase}.vtk"))
        myo = MeshResource(dir.joinpath(f"LVmyo_{phase}.vtk"))
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


class PhaseMesh:
    def __init__(self, rv: RVMesh, lv: LVMesh, phase: Phase):
        self.rv = rv
        self.lv = lv
        self.phase = phase

    @classmethod
    def from_dir(cls, dir: Path, phase: Phase):
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
    def get_data(self) -> np.ndarray:
        nim = nib.load(str(self.path))
        seg = nim.get_data()
        return seg


class Segmentation(ImageResource):
    def __init__(self, path: Path, phase: Union[Phase, str, int]):
        self.path = path
        self.phase = phase
        super().__init__(path)


class NiiData(ImageResource):
    @classmethod
    def from_dir(cls, dir: Path):
        assert dir.is_dir(), "{} is not a directory.".format(str(dir))
        nii_path = dir.joinpath("LVSA.nii.gz")
        return cls(nii_path)


class PhaseImage(ImageResource):
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


@dataclass
class Template:
    dir: Path

    @property
    def landmark(self):
        return self.dir.joinpath("landmarks2.vtk")

    def rv(self, phase: Phase):
        return self.dir.joinpath(f"RV_{phase}.vtk")

    def lv_endo(self, phase: Phase):
        return self.dir.joinpath(f"LVendo_{phase}.vtk")

    def lv_epi(self, phase: Phase):
        return self.dir.joinpath(f"LVepi_{phase}.vtk")

    def lv_myo(self, phase: Phase):
        return self.dir.joinpath(f"LVmyo_{phase}.vtk")

    def vtk_rv(self, phase: Phase):
        return self.dir.joinpath(f"vtk_RV_{phase}.nii.gz")

    def vtk_lv(self, phase: Phase):
        return self.dir.joinpath(f"vtk_LV_{phase}.nii.gz")

    def check_valid(self):
        for phase in [Phase.ED, Phase.ES]:
            if not self.rv(phase).exists():
                raise FileNotFoundError(f"RV {phase} template does not exists at {self.rv(phase)}.")
            if not self.lv_endo(phase).exists():
                raise FileNotFoundError(f"LV endo {phase} template does not exists at {self.lv_endo(phase)}.")
            if not self.lv_epi(phase).exists():
                raise FileNotFoundError(f"LV epi {phase} template does not exists at {self.lv_epi(phase)}.")
            if not self.lv_myo(phase).exists():
                raise FileNotFoundError(f"LV myo {phase} template does not exists at {self.lv_myo(phase)}.")
