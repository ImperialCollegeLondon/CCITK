import shutil
import logging
from pathlib import Path

from ccitk.common.resource import CardiacMesh, Segmentation
from ccitk.cmr_segment.common.utils import extract_lv_label, extract_rv_label
from ccitk.register import register_cardiac_phases

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("CMRSegment.coregister")


# TODO: multi process
class Coregister:
    """
    Coregister registers each subject's ED/ES mesh with an atlas and transform each subject to the same atlas space.

    It takes input from

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

    and generates output

    .. code-block:: text

        /path/
            subject1/
                registration/
                    temp/                           ->  Temporary files
                    debug/                          ->  Intermediate meshes
                    landmarks.dof.gz
                    rigid/                          ->  Meshes after rigid transformation
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
                    nonrigid/                       -> Meshes after nonrigid transformation
                        LVendo_ED.vtk               ->  LV Endo ED mesh
                        LVendo_ES.vtk               ->  LV Endo ES mesh
                        LVepi_ED.vtk                ->  LV Epi ED mesh
                        LVepi_ES.vtk                ->  LV Epi ES mesh
                        LVmyo_ED.vtk                ->  LV Myo ED mesh
                        LVmyo_ES.vtk                ->  LV Myo ES mesh
                        RV_ED.vtk                   ->  RV ED mesh
                        RV_ES.vtk                   ->  RV ES mesh

            subject2/
                ...

    To run only the coregister, use the following command:

    .. code-block:: text

        ccitk-cmr-segment -o /output-dir/ --data-dir /input-dir/ --coregister --template-dir
        /template-dir --param-dir /param-dir

    Default template dir is ``ccitk/cmr_segment/resource/params``

    Default param dir is ``ccitk/cmr_segment/resource/``


    """
    def __init__(self, template_dir: Path, param_dir: Path, overwrite: bool = False):
        self.template_dir = template_dir
        self.template_landmarks = self.template_dir.joinpath("landmarks2_old.vtk")

        segareg_path = param_dir.joinpath("segareg.txt")
        segreg_path = param_dir.joinpath("segreg.txt")
        spnreg_path = param_dir.joinpath("spnreg.txt")

        if not segreg_path.exists():
            raise FileNotFoundError(f"segreg.txt does not exist at {segreg_path}")
        if not spnreg_path.exists():
            raise FileNotFoundError(f"spnreg_path.txt does not exist at {spnreg_path}")
        if not segareg_path.exists():
            raise FileNotFoundError(f"segareg_path.txt does not exist at {segareg_path}")
        self.segareg_path = segareg_path
        self.segreg_path = segreg_path
        self.spnreg_path = spnreg_path
        self.logger = LOGGER
        self.overwrite = overwrite

    def run(self, mesh: CardiacMesh, segmentation: Segmentation, landmark_path: Path, output_dir: Path):
        template_mesh = CardiacMesh.from_dir(self.template_dir, mesh.phase)
        template_lv_label = self.template_dir.joinpath(f"LV_{mesh.phase}.nii.gz")
        template_rv_label = self.template_dir.joinpath(f"RV_{mesh.phase}.nii.gz")
        if not landmark_path.exists():
            raise FileNotFoundError(
                f"Landmark file does not exist at {landmark_path}. "
                f"To generate landmark, please run landmark extractor first."
            )
        try:
            mesh.check_valid()
        except FileNotFoundError as e:
            self.logger.error(f"Mesh does not exist. To generate mesh, please run mesh extractor first.")
            raise e
        if not segmentation.path.exists():
            self.logger.error(f"Segmentation does not exist at {segmentation.path}. To generate segmenation, "
                              f"please run segmentor first.")
        temp_dir = output_dir.joinpath("temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        if self.overwrite:
            if temp_dir.exists():
                shutil.rmtree(str(temp_dir), ignore_errors=True)
            if output_dir.exists():
                shutil.rmtree(str(output_dir), ignore_errors=True)
        rv_label = extract_rv_label(
            segmentation_path=segmentation.path,
            output_path=temp_dir.joinpath(f"vtk_RV_{segmentation.phase}.nii.gz")
        )

        lv_label = extract_lv_label(
            segmentation_path=segmentation.path,
            output_path=temp_dir.joinpath(f"vtk_LV_{segmentation.phase}.nii.gz"),
        )

        transformed_mesh, dof = register_cardiac_phases(
            fixed_mesh=template_mesh,
            fixed_landmarks=self.template_landmarks,
            moving_mesh=mesh,
            moving_landmarks=landmark_path,
            affine_parin=self.segareg_path,
            ffd_parin=self.segreg_path,
            output_dir=output_dir,
            overwrite=self.overwrite,
            ds=8,
        )
        return transformed_mesh

    # def compute_wall_thickness(self, mesh: PhaseMesh, output_dir: Path):
    #     fr = mesh.phase
    #     output_lv_thickness = output_dir.joinpath("wt", f"LVmyo_{fr}.vtk")
    #     output_rv_thickness = output_dir.joinpath("wt", f"RV_{fr}.vtk")
    #     output_lv_thickness.parent.mkdir(parents=True, exist_ok=True)
    #
    #     if not output_lv_thickness.exists() or self.overwrite:
    #         mirtk.evaluate_distance(
    #             str(mesh.lv.endocardium),
    #             str(mesh.lv.epicardium),
    #             str(output_lv_thickness),
    #             name="WallThickness",
    #         )
    #     if not output_rv_thickness.exists() or self.overwrite:
    #         mirtk.evaluate_distance(
    #             str(mesh.rv.rv),
    #             str(mesh.rv.epicardium),
    #             str(output_rv_thickness),
    #             name="WallThickness",
    #         )
    #     if not output_dir.joinpath("rv_{}_wallthickness.txt".format(fr)).exists() or self.overwrite:
    #         mirtk.convert_pointset(
    #             str(output_rv_thickness),
    #             str(output_dir.joinpath("rv_{}_wallthickness.txt".format(fr))),
    #         )
    #     if not output_dir.joinpath("lv_myo{}_wallthickness.txt".format(fr)).exists() or self.overwrite:
    #         mirtk.convert_pointset(
    #             str(output_lv_thickness),
    #             str(output_dir.joinpath("lv_myo{}_wallthickness.txt".format(fr))),
    #         )

    # def compute_curvature(self, mesh: PhaseMesh, output_dir: Path):
    #     fr = mesh.phase
    #     output_lv_curv = output_dir.joinpath("curv", f"LVmyo_{fr}.vtk")
    #     output_rv_curv = output_dir.joinpath("curv", f"RV_{fr}.vtk")
    #     output_rv_curv.parent.mkdir(parents=True, exist_ok=True)
    #
    #     if not output_lv_curv.exists() or self.overwrite:
    #         mirtk.calculate_surface_attributes(
    #             str(mesh.lv.myocardium),
    #             str(output_lv_curv),
    #             smooth_iterations=64,
    #         )
    #
    #     if not output_rv_curv.exists() or self.overwrite:
    #         mirtk.calculate_surface_attributes(
    #             str(mesh.rv.rv),
    #             str(output_rv_curv),
    #             smooth_iterations=64,
    #         )
    #     if not output_dir.joinpath("rv_{}_curvature.txt".format(fr)).exists() or self.overwrite:
    #         mirtk.convert_pointset(
    #             str(output_rv_curv),
    #             str(output_dir.joinpath("rv_{}_curvature.txt".format(fr))),
    #         )
    #     if not output_dir.joinpath("lv_myo{}_curvature.txt".format(fr)).exists() or self.overwrite:
    #         mirtk.convert_pointset(
    #             str(output_lv_curv),
    #             str(output_dir.joinpath("lv_myo{}_curvature.txt".format(fr))),
    #         )
