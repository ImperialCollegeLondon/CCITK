import shutil
from pathlib import Path

from ccitk.mesh import extract_mesh_from_segmentation
from ccitk.common.resource import Segmentation, CardiacMesh


class MeshExtractor:
    """
    Extractor extracts 3D triangular mesh from 3D segmentations using marching cubes algorithms. It also extracts landmarks.

    It takes input from

    .. code-block:: text

        /path/
            subject1/
                seg_lvsa_SR_ED.nii.gz           ->  ED segmentation
                seg_lvsa_SR_ES.nii.gz           ->  ES segmentation
            subject2/
                ...
    or from

    .. code-block:: text
        /path/
            subject1/
                refine/
                    seg_lvsa_SR_ED.nii_refined.nii.gz       ->  Refined ED segmentation
                    seg_lvsa_SR_ES.nii_refined.nii.gz       ->  Refined ES segmentation
            subject2/
                ...

    and generates output

    .. code-block:: text

        /path/
            subject1/
                landmark_ED.vtk                 ->  ED landmarks
                landmark_ES.vtk                 ->  ES landmarks
                mesh/
                    LVendo_ED.vtk               ->  LV Endo ED mesh
                    LVendo_ES.vtk               ->  LV Endo ES mesh
                    LVepi_ED.vtk                ->  LV Epi ED mesh
                    LVepi_ES.vtk                ->  LV Epi ES mesh
                    LVmyo_ED.vtk                ->  LV Myo ED mesh
                    LVmyo_ES.vtk                ->  LV Myo ES mesh
                    RV_ED.vtk                   ->  RV ED mesh
                    RV_ES.vtk                   ->  RV ES mesh
                    RVepi_ED.vtk                ->  RV Epi ED mesh
                    RVepi_ES.vtk                ->  RV Epi ES mesh
            subject2/
                ...

    To run only the extractor, use one of the following commands:

    .. code-block:: text

        ccitk-cmr-segment -o /output-dir/ --data-dir /input-dir/ --extract
    """
    def __init__(self, iso_value: int = 120, blur: int = 2, overwrite: bool = False):
        self.iso_value = iso_value
        self.blur = blur
        self.overwrite = overwrite

    def run(self, segmentation: Segmentation, output_dir: Path) -> CardiacMesh:
        """Extract mesh from segmentation

        Args:
            segmentation: segmentation object containing the path
            output_dir: output directory

        Returns:
            mesh containing mesh paths for lv and rv

        """
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
        """Extract RV blood pool mesh"""
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
        """Extract RV epicardium mesh"""
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
        """Extract LV endocardium mesh"""
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
        """Extract LV epicardium mesh"""
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
        """Extract LV myocardium mesh"""
        if not output_path.exists() or self.overwrite:
            output_path = extract_mesh_from_segmentation(
                segmentation=segmentation.path,
                output_path=output_path,
                labels=[2],
                iso_value=self.iso_value,
                blur=self.blur,
                overwrite=self.overwrite,
            )
