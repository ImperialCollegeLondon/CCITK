import shutil
import logging
from pathlib import Path

from ccitk.resource import PhaseMesh, Segmentation, Template, RVMesh, LVMesh
from ccitk.cmr_segment.common.utils import extract_lv_label, extract_rv_label
from ccitk.register import register_cardiac_phases

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("CMRSegment.coregister")


# TODO: multi process
class Coregister:
    def __init__(self, template_dir: Path, param_dir: Path, overwrite: bool = False):
        self.template = Template(dir=template_dir)
        self.template.check_valid()
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

    def run(self, mesh: PhaseMesh, segmentation: Segmentation, landmark_path: Path, output_dir: Path):
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
        template_mesh = PhaseMesh(
            rv=RVMesh(
                mesh=self.template.rv(mesh.phase),
                epicardium=self.template.rv(mesh.phase),
            ),
            lv=LVMesh(
                endocardium=self.template.lv_endo(mesh.phase),
                epicardium=self.template.lv_epi(mesh.phase),
                myocardium=self.template.lv_myo(mesh.phase)
            ),
            phase=mesh.phase
        )
        transformed_mesh, dof = register_cardiac_phases(
            fixed_mesh=template_mesh,
            fixed_landmarks=self.template.landmark,
            fixed_rv_label=self.template.vtk_rv(mesh.phase),
            fixed_lv_label=self.template.vtk_lv(mesh.phase),
            moving_mesh=mesh,
            moving_landmarks=landmark_path,
            moving_rv_label=rv_label,
            moving_lv_label=lv_label,
            affine_parin=self.segareg_path,
            ffd_parin=self.spnreg_path,
            output_dir=output_dir,
            overwrite=self.overwrite
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
