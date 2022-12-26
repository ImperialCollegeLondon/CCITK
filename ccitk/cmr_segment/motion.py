__all__ = [
    "MotionTracker",
]
import mirtk
import shutil
import numpy as np
from tqdm import tqdm
from typing import List
from ccitk.common.resource import CineImages, CardiacMesh, Path, Segmentation
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
    """
    Compute motion by averaging forward motion and backward motion,
    transform atlas to subject's ED mesh in subject space,
    apply motion to the transformed atlas, then register each one back to atlas space.

    It takes input from

    .. code-block:: text

        /path/
            subject1/
                landmark_ED.vtk                         ->  ED landmarks
                refine/
                    seg_lvsa_SR_ED.nii_refined.nii.gz   ->  Refined ED segmentation
                enlarged/
                    lvsa_0.nii.gz                       ->  Enlarged phase 0 image
                    lvsa_1.nii.gz                       ->  Enlarged phase 1 image
                    ...
                mesh/
                    LVendo_ED.vtk                       ->  LV Endo ED mesh
                    LVepi_ED.vtk                        ->  LV Epi ED mesh
                    LVmyo_ED.vtk                        ->  LV Myo ED mesh
                    RV_ED.vtk                           ->  RV ED mesh
                    RVepi_ED.vtk                        ->  RV Epi ED mesh
            subject2/
                ...
    and generates output

    .. code-block:: text

        /path/
            subject1/
                motion/
                    dof/                                        -> motion transformation files
                        ffd_00_to_00.dof.gz                     -> avergaed, composed motion from frame 00 to 00
                        ffd_00_to_01.dof.gz                     -> avergaed, composed motion from frame 00 to 01
                        ffd_00_to_02.dof.gz                     -> avergaed, composed motion from frame 00 to 02
                        ...
                        ffd_forward_00_to_01.dof.gz             -> forward motion from frame 00 to 01
                        ffd_forward_01_to_02.dof.gz             -> forward motion from frame 01 to 02
                        ...
                        ffd_comp_forward_00_to_02.dof.gz        -> forward composed motion from frame 00 to 02
                        ffd_comp_forward_00_to_03.dof.gz        -> forward composed motion from frame 00 to 03
                        ...
                        ffd_backward_00_to_24.dof.gz            -> backward motion from frame 00 to 24
                        ffd_backward_02_to_01.dof.gz            -> backward motion from frame 02 to 01
                        ...
                        ffd_comp_backward_00_to_01.dof.gz       -> backward composed motion from frame 00 to 01
                        ffd_comp_backward_00_to_02.dof.gz       -> backward composed motion from frame 00 to 02
                        ...
                    register/                                   -> Intermediate registration results when transforming atlas
                                                                   to match subject ED mesh
                        lm/                                     -> Atlas mesh after registeration using landmarks
                            mesh/
                                LVendo_ED.vtk
                                LVepi_ED.vtk
                                LVmyo_ED.vtk
                                RV_ED.vk
                            landmarks.dof.gz
                        rigid/                                  -> Atlas mesh after rigid registeration
                            mesh/
                            lv_rigid_ED.dof.gz
                            rigid_ED.dof.gz
                            rv_rigid_ED.dof.gz
                        affine/                                 -> Atlas mesh after affine registeration
                            mesh/
                            rv_affine_ED.dof.gz
                        ffd/                                    -> Atlas mesh after ffd registeration
                            mesh/
                            lv_ffd_ED.dof.gz
                            rv_ffd_ED.dof.gz
                    seg/                                        -> Warped ED label according to motion
                        lvsa_00.nii.gz
                        lvsa_01.nii.gz
                        ...
                    TXT/                                        -> motion of transformed atlas mesh in atlas space, txt format
                        LV_endo/
                            fr00.txt
                            fr01.txt
                            ...
                        LV_epi/
                        LV_myo/
                        RV/
                    VTK/                                        -> motion of transformed atlas mesh in atlas space, vtk format
                        LV_endo/
                            fr00.vtk
                            fr01.vtk
                            ...
                        LV_epi/
                        LV_myo/
                        RV/
            subject2/
                ...

    To run only the coregister, use the following command:

    .. code-block:: text

        ccitk-cmr-segment -o /output-dir/ --data-dir /input-dir/ --track-motion --template-dir /template-dir
        --param-dir /param-dir --ffd-motion-cfg /path-to-cfg


    Default template dir is ``ccitk/cmr_segment/resource/params``

    Default param dir is ``ccitk/cmr_segment/resource/``

    Default ffd motion cfg is ``ccitk/cmr_segment/resource/ffd_motion_2.cfg``


    """
    def __init__(self, param_dir: Path, template_dir: Path, ffd_motion_cfg: Path = None):
        self.param_dir = param_dir
        if ffd_motion_cfg is None:
            ffd_motion_cfg = self.param_dir.joinpath("ffd_motion_2.cfg")
        self.ffd_motion_cfg = ffd_motion_cfg
        self.ffd_refine_cfg = self.param_dir.joinpath("ffd_refine.cfg")
        self.template_dir = template_dir
        self.template_landmarks = self.template_dir.joinpath("landmarks2_old.vtk")
        # self.template = Template(dir=template_dir)

    def run(self, cine: CineImages, ed_segmentation: Segmentation, landmarks: Path, ED_mesh: CardiacMesh,
            output_dir: Path, overwrite: bool = False):

        template_ED_mesh = CardiacMesh.from_dir(self.template_dir, "ED")

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
                    dofin=combine_dofs[fr - 1],
                    invert=True
                )

        transformed_atlas_mesh, ffd_out = register_cardiac_phases(
            fixed_mesh=ED_mesh,
            fixed_landmarks=landmarks,
            moving_mesh=template_ED_mesh,
            moving_landmarks=self.template_landmarks,
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
        # rv_epi_vtks = phase_motion["rv"]["epi"]

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

        # self.convert_vtks_to_txts(
        #     vtks=rv_epi_vtks,
        #     output_dir=txt_dir.joinpath("RV_epi"),
        #     overwrite=overwrite,
        # )

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
