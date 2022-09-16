import mirtk
import shutil
import numpy as np
from tqdm import tqdm
from typing import List
from pathlib import Path
from ccitk.resource import CineImages, Template, Phase, PhaseMesh, MeshResource, Segmentation
from ccitk.motion import warp_label, forward_motion, backward_motion, average_forward_backward_motion, phase_mesh_motion
from ccitk.register import register_cardiac_phases


class Landmarks:
    def to_list(self):
        raise NotImplementedError()

    def __iter__(self):
        return iter(self.to_list())

    def __len__(self):
        raise len(self.to_list())


class LVLandmarks(Landmarks):
    def __init__(self, top: np.ndarray, circle: List[np.ndarray], bottom: np.ndarray):
        self.top = top.copy()
        self.circle = circle.copy()
        self.bottom = bottom.copy()

    def to_list(self) -> List[np.ndarray]:
        l = [self.top]
        for p in self.circle:
            l.append(p)
        l.append(self.bottom)
        return l


class RVLandmarks(Landmarks):
    def __init__(self, mid: List[np.ndarray], bottom: np.ndarray):
        self.mid = mid
        self.bottom = bottom

    def to_list(self):
        l = self.mid.copy()
        l.append(self.bottom)
        return l


class SubjectLandmarks(Landmarks):
    def __init__(self, lv_landmarks: LVLandmarks, rv_landmarks: RVLandmarks, path: Path = None):
        self.lv_landmarks = lv_landmarks
        self.rv_landmarks = rv_landmarks
        self.path = path

    def to_list(self):
        """

        Returns:
            [LV_top, LV_circle, LV_bottom, RV_mid, RV_bottom]
        """
        lvs = self.lv_landmarks.to_list().copy()
        for p in self.rv_landmarks.to_list():
            lvs.append(p)
        return lvs


class MotionTracker:
    def __init__(self, param_dir: Path, template_dir: Path, ffd_motion_cfg: Path = None):
        self.param_dir = param_dir
        if ffd_motion_cfg is None:
            ffd_motion_cfg = self.param_dir.joinpath("ffd_motion_2.cfg")
        self.ffd_motion_cfg = ffd_motion_cfg
        self.ffd_refine_cfg = self.param_dir.joinpath("ffd_refine.cfg")
        self.template = Template(dir=template_dir)

    def run(self, cine: CineImages, ed_segmentation: Segmentation, landmarks: Path, ED_mesh: PhaseMesh,
            output_dir: Path, overwrite: bool = False):
        output_dir.mkdir(parents=True, exist_ok=True)
        dof_dir = output_dir.joinpath("dof")
        dof_dir.mkdir(parents=True, exist_ok=True)
        forward_compose_dofs = forward_motion(
            images=cine.images,
            output_dir=dof_dir,
            parin=self.ffd_motion_cfg,
            compose_spacing=10,
            overwrite=overwrite,
        )
        backward_compose_dofs = backward_motion(
            images=cine.images,
            output_dir=dof_dir,
            parin=self.ffd_motion_cfg,
            compose_spacing=10,
            overwrite=overwrite,
        )
        combine_dofs = average_forward_backward_motion(
            forward_compose_dofs=forward_compose_dofs,
            backward_compose_dofs=backward_compose_dofs,
            output_dir=dof_dir,
            overwrite=overwrite,
        )

        # Warp labels
        output_dir.joinpath("seg").mkdir(parents=True, exist_ok=True)
        if not output_dir.joinpath("seg").joinpath(f"lvsa_00.nii.gz").exists() or overwrite:
            shutil.copy(str(ed_segmentation.path), str(output_dir.joinpath("seg").joinpath(f"lvsa_00.nii.gz")))
        for fr in tqdm(range(1, len(cine))):
            if not output_dir.joinpath("seg").joinpath(f"lvsa_{fr:02d}.nii.gz").exists() or overwrite:
                warp_label(
                    reference_label=ed_segmentation.path,
                    output_path=output_dir.joinpath("seg").joinpath(f"lvsa_{fr:02d}.nii.gz"),
                    dofin=combine_dofs[fr],
                    invert=True
                )

        transformed_atlas_mesh, ffd_out = register_cardiac_phases(
            fixed_mesh=ED_mesh,
            fixed_landmarks=landmarks.path,
            moving_mesh=self.template,
            moving_landmarks=self.template.landmark,
            affine_parin=self.affine_parin,
            ffd_parin=self.ffd_parin,
            output_dir=output_dir.joinpath("register"),
            ds=20,
            rigid=True,
            overwrite=overwrite,
        )
        phase_motion = phase_mesh_motion(
            reference_mesh=transformed_atlas_mesh,
            motion_dofs=combine_dofs,
            output_dir=output_dir.joinpath("VTK"),
            overwrite=overwrite,
        )
        lv_endo_vtks = phase_motion["lv"]["endo"]
        lv_epi_vtks = phase_motion["lv"]["epi"]
        lv_myo_vtks = phase_motion["lv"]["myo"]
        rv_vtks = phase_motion["rv"]["rv"]
        rv_epi_vtks = phase_motion["rv"]["epi"]

        txt_dir = output_dir.joinpath("TXT")

        self.convert_vtks_to_txts(
            vtks=lv_endo_vtks,
            output_dir=txt_dir.joinpath("LV_endo"),
            overwrite=overwrite,
        )

        self.convert_vtks_to_txts(
            vtks=lv_epi_vtks,
            output_dir=txt_dir.joinpath("LV_epi"),
            overwrite=overwrite,
        )

        self.convert_vtks_to_txts(
            vtks=lv_myo_vtks,
            output_dir=txt_dir.joinpath("LV_myo"),
            overwrite=overwrite,
        )

        self.convert_vtks_to_txts(
            vtks=rv_vtks,
            output_dir=txt_dir.joinpath("RV"),
            overwrite=overwrite,
        )

        self.convert_vtks_to_txts(
            vtks=rv_epi_vtks,
            output_dir=txt_dir.joinpath("RV_epi"),
            overwrite=overwrite,
        )

    @staticmethod
    def convert_vtks_to_txts(vtks: List[Path], output_dir: Path, overwrite: bool = False):
        # Convert vtks to text files
        print("\n ...   Convert vtks to text files")
        output_dir.mkdir(parents=True, exist_ok=True)
        txts = []
        for fr in tqdm(range(0, len(vtks))):
            txt = output_dir.joinpath("fr{:02d}.txt".format(fr))
            if not txt.exists() or overwrite:
                mirtk.convert_pointset(
                    str(vtks[fr]),
                    str(txt),
                )
            txts.append(txt)
        return vtks, txts
