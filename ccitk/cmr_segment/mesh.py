import shutil
from pathlib import Path

from ccitk.common.resource import Segmentation, CardiacMesh
from ccitk.mesh import extract_mesh_from_segmentation


class MeshExtractor:
    def __init__(self, iso_value: int = 120, blur: int = 2, overwrite: bool = False):
        self.iso_value = iso_value
        self.blur = blur
        self.overwrite = overwrite

    def run(self, segmentation: Segmentation, output_dir: Path) -> CardiacMesh:
        temp_dir = output_dir.joinpath("temp")
        if self.overwrite:
            if temp_dir.exists():
                shutil.rmtree(str(temp_dir), ignore_errors=True)
            if output_dir.exists():
                shutil.rmtree(str(output_dir), ignore_errors=True)
        mesh = CardiacMesh.from_dir(dir=output_dir, phase=segmentation.phase)
        if not mesh.exists() or self.overwrite:
            self.rv(segmentation, mesh.rv.rv)
            self.rv_epi(segmentation, mesh.rv.epicardium)
            self.lv_endo(segmentation, mesh.lv.endocardium)
            self.lv_epi(segmentation, mesh.lv.epicardium)
            self.lv_myo(segmentation, mesh.lv.myocardium)
        # if temp_dir.exists():
        #     shutil.rmtree(str(temp_dir), ignore_errors=True)
        mesh.check_valid()
        return mesh

    def rv(self, segmentation: Segmentation, output_path: Path):
        if not output_path.exists() or self.overwrite:
            output_path = extract_mesh_from_segmentation(
                segmentation=segmentation.path,
                output_path=output_path,
                labels=[3, 4],
                iso_value=self.iso_value,
                blur=self.blur,
                overwrite=self.overwrite,
            )

    def rv_epi(self, segmentation: Segmentation, output_path: Path):
        if not output_path.exists() or self.overwrite:
            output_path = extract_mesh_from_segmentation(
                segmentation=segmentation.path,
                output_path=output_path,
                labels=[3, 4],
                iso_value=self.iso_value,
                blur=self.blur,
                overwrite=self.overwrite,
            )

    def lv_endo(self, segmentation: Segmentation, output_path: Path):
        if not output_path.exists() or self.overwrite:
            output_path = extract_mesh_from_segmentation(
                segmentation=segmentation.path,
                output_path=output_path,
                labels=[1],
                iso_value=self.iso_value,
                blur=self.blur,
                overwrite=self.overwrite,
            )

    def lv_epi(self, segmentation: Segmentation, output_path: Path):
        if not output_path.exists() or self.overwrite:
            output_path = extract_mesh_from_segmentation(
                segmentation=segmentation.path,
                output_path=output_path,
                labels=[1, 2],
                iso_value=self.iso_value,
                blur=self.blur,
                overwrite=self.overwrite,
            )

    def lv_myo(self, segmentation: Segmentation, output_path: Path):
        if not output_path.exists() or self.overwrite:
            output_path = extract_mesh_from_segmentation(
                segmentation=segmentation.path,
                output_path=output_path,
                labels=[2],
                iso_value=self.iso_value,
                blur=self.blur,
                overwrite=self.overwrite,
            )