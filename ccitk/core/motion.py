import shutil
from pathlib import Path
from ccitk.core.common.resource import CineImages, Template, Phase, PhaseMesh, MeshResource, Segmentation
from tqdm import tqdm
import mirtk
from typing import List
import numpy as np


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


class DOFs:
    def __init__(self, length: int):
        self._list = [None for _ in range(length)]

    def __getitem__(self, phase: int) -> Path:
        idx = phase - 1
        if self._list[idx] is not None:
            return self._list[idx]
        else:
            raise ValueError(f"DOF {idx} is not set.")

    def __setitem__(self, phase: int, dof_path: Path):
        idx = phase - 1
        self._list[idx] = dof_path

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)


class MotionTracker:
    def __init__(self, param_dir: Path, template_dir: Path, ffd_motion_cfg: Path = None):
        self.param_dir = param_dir
        if ffd_motion_cfg is None:
            ffd_motion_cfg = self.param_dir.joinpath("ffd_motion_2.cfg")
        self.ffd_motion_cfg = ffd_motion_cfg
        self.ffd_refine_cfg = self.param_dir.joinpath("ffd_refine.cfg")
        self.template = Template(dir=template_dir)

    def run(self, cine: CineImages, ed_segmentation: Segmentation, landmarks: SubjectLandmarks, ED_mesh: PhaseMesh,
            output_dir: Path, overwrite: bool = False):
        # Forward image registration
        forward_dofs = DOFs(len(cine) - 1)
        output_dir.mkdir(parents=True, exist_ok=True)
        dof_dir = output_dir.joinpath("dof")
        dof_dir.mkdir(parents=True, exist_ok=True)
        for fr in tqdm(range(1, len(cine))):

            target = cine[fr - 1]
            source = cine[fr]
            par = self.ffd_motion_cfg
            dof_output = dof_dir.joinpath("ffd_forward_{:02d}_to_{:02d}.dof.gz".format(fr-1, fr))
            if not dof_output.exists() or overwrite:
                mirtk.register(
                    str(target),
                    str(source),
                    parin=str(par),
                    dofout=str(dof_output)
                )
            forward_dofs[fr] = dof_output

        # Compose inter-frame transformation fields #
        print("\n ...  Compose forward inter-frame transformation fields")
        forward_compose_dofs = DOFs(len(cine) - 1)
        forward_compose_dofs[1] = forward_dofs[1]
        for fr in tqdm(range(2, len(cine))):
            dof_out = dof_dir.joinpath("ffd_comp_forward_00_to_{:02d}.dof.gz".format(fr))
            dofs = [str(forward_dofs[k]) for k in range(1, fr+1)]

            if not dof_out.exists() or overwrite:
                mirtk.compose_dofs(
                    *dofs,
                    str(dof_out),
                    approximate=None,
                    spacing=1,
                    # target=str(ed_segmentation.path),
                )
            forward_compose_dofs[fr] = dof_out

        # Backward image registration
        backward_dofs = DOFs(len(cine) - 1)
        for fr in tqdm(range(len(cine) - 1, 0, -1)):
            target_fr = (fr + 1) % len(cine)
            source_fr = fr
            target = cine[target_fr]
            source = cine[source_fr]
            par = self.ffd_motion_cfg
            dof_output = dof_dir.joinpath("ffd_backward_{0:02d}_to_{1:02d}.dof.gz".format(target_fr, source_fr))
            if not dof_output.exists() or overwrite:
                mirtk.register(
                    str(target),
                    str(source),
                    parin=str(par),
                    dofout=str(dof_output)
                )
            backward_dofs[fr] = dof_output
        # Compose inter-frame transformation fields #
        print("\n ...  Compose backward inter-frame transformation fields")
        backward_compose_dofs = DOFs(len(cine) - 1)
        backward_compose_dofs[len(cine) - 1] = backward_dofs[len(cine) - 1]
        for fr in tqdm(range(len(cine) - 2, 0, -1)):
            dof_out = dof_dir.joinpath("ffd_comp_backward_00_to_{:02d}.dof.gz".format(fr))
            dofs = [str(backward_dofs[k]) for k in range(len(cine) - 1, fr - 1, -1)]
            if not dof_out.exists() or overwrite:
                mirtk.compose_dofs(
                    *dofs,
                    str(dof_out),
                    approximate=None,
                    spacing=1,
                    # target=str(ed_segmentation.path),
                )
            backward_compose_dofs[fr] = dof_out

        # Average ffd motion fields
        # combine_dofs = DOFs(len(cine) - 1)
        # for fr in tqdm(range(1, len(cine))):
        #     dof_forward = forward_compose_dofs[fr]
        #     dof_backward = backward_compose_dofs[fr]
        #     weight_forward = float(len(cine) - fr) / len(cine)
        #     weight_backward = float(fr) / len(cine)
        #     dof_combine = dof_dir.joinpath("ffd_00_to_{:02d}.dof.gz".format(fr))
        #
        #     if not dof_combine.exists() or overwrite:
        #         mirtk.average_3d_ffd(
        #             "2",
        #             str(dof_forward),
        #             weight_forward,
        #             str(dof_backward),
        #             weight_backward,
        #             str(dof_combine),
        #             verbose=None,
        #         )
        #     combine_dofs[fr] = dof_combine
        combine_dofs = forward_compose_dofs
        # combine_dofs = forward_compose_dofs
        # Refine motion fields
        # Composition of inter-frame motion fields can lead to accumulative errors.
        # At this step, we refine the motion fields by re-registering the n-th frame with the ED frame.
        # refine_dofs = {}
        # print("\n ...  Refine motion fields")
        # for fr in tqdm(range(2, len(cine))):
        #     target = cine[0]
        #     source = cine[fr]
        #     dofin = compose_dofs[fr]
        #     dofout = dof_dir.joinpath("ffd_00_to_{:02d}.dof.gz".format(fr))
        #
        #     if not dofout.exists():
        #         mirtk.register(
        #             str(target),
        #             str(source),
        #             parin=str(self.ffd_refine_cfg),
        #             dofin=str(dofin),
        #             dofout=str(dofout)
        #         )
        #     refine_dofs[fr] = dofout
        # refine_dofs[1] = forward_dofs[1]

        # Warp labels
        output_dir.joinpath("seg").mkdir(parents=True, exist_ok=True)
        if not output_dir.joinpath("seg").joinpath(f"lvsa_00.nii.gz").exists() or overwrite:
            shutil.copy(str(ed_segmentation.path), str(output_dir.joinpath("seg").joinpath(f"lvsa_00.nii.gz")))
        for fr in tqdm(range(1, len(cine))):
            if not output_dir.joinpath("seg").joinpath(f"lvsa_{fr:02d}.nii.gz").exists() or overwrite:
                mirtk.transform_image(
                    str(ed_segmentation),
                    str(output_dir.joinpath("seg").joinpath(f"lvsa_{fr:02d}.nii.gz")),
                    "-invert", "-v",
                    interp="NN",
                    dofin=combine_dofs[fr]
                )

        landmark_init_dof = dof_dir.joinpath("landmarks.dof.gz")
        if not landmark_init_dof.exists() or overwrite:
            mirtk.register(
                str(landmarks.path),
                str(self.template.landmark),
                model="Rigid",
                dofout=str(landmark_init_dof),
            )
        # register_landmarks(
        #     target=landmarks.path,
        #     source=self.template.landmark,
        #     weights=[1] + [1] * len(landmarks.lv_landmarks.circle) + [1] + [1, 1, 1, 1, 1, 1, 1] + [5],
        #     # weights=None,
        #     output_path=landmark_init_dof,
        # )
        mirtk.transform_points(
            str(self.template.landmark),
            str(dof_dir.joinpath("template_lm.vtk")),
            dofin=str(landmark_init_dof),
            invert=None,
        )

        vtk_dir = output_dir.joinpath("VTK")
        txt_dir = output_dir.joinpath("TXT")
        temp_dir = output_dir.joinpath("temp")

        lv_endo_frame_0 = self.transform_mesh(
            target_mesh=ED_mesh.lv.endocardium,
            source_mesh=self.template.lv_endo(Phase.ED),
            landmark_init_dof=landmark_init_dof,
            output_dir=vtk_dir.joinpath("LV_endo"),
            temp_dir=temp_dir.joinpath("LV_endo"),
            overwrite=overwrite,
        )

        lv_epi_frame_0 = self.transform_mesh(
            source_mesh=self.template.lv_epi(Phase.ED),
            target_mesh=ED_mesh.lv.epicardium,
            landmark_init_dof=landmark_init_dof,
            output_dir=vtk_dir.joinpath("LV_epi"),
            temp_dir=temp_dir.joinpath("LV_epi"),
            overwrite=overwrite,
        )

        rv_frame_0 = self.transform_mesh(
            source_mesh=self.template.rv(Phase.ED),
            target_mesh=ED_mesh.rv.rv,
            landmark_init_dof=landmark_init_dof,
            output_dir=vtk_dir.joinpath("RV"),
            temp_dir=temp_dir.joinpath("RV"),
            overwrite=overwrite,
        )

        # Transform the mesh
        print("\n ...   Transform the LV endo mesh")
        lv_endo_vtks = self.motion_mesh(
            frame_0_mesh=lv_endo_frame_0,
            motion_dofs=combine_dofs,
            output_dir=vtk_dir.joinpath("LV_endo"),
            overwrite=overwrite
        )

        # Transform the mesh
        print("\n ...   Transform the LV epi mesh")
        lv_epi_vtks = self.motion_mesh(
            frame_0_mesh=lv_epi_frame_0,
            motion_dofs=combine_dofs,
            output_dir=vtk_dir.joinpath("LV_epi"),
            overwrite=overwrite
        )

        # Transform the mesh
        print("\n ...   Transform the RV mesh")
        rv_vtks = self.motion_mesh(
            frame_0_mesh=rv_frame_0,
            motion_dofs=combine_dofs,
            output_dir=vtk_dir.joinpath("RV"),
            overwrite=overwrite
        )

        self.convert_vtks_to_txts(
            vtks=lv_endo_vtks,
            motion_dofs=combine_dofs,
            output_dir=txt_dir.joinpath("LV_endo"),
            overwrite=overwrite
        )

        self.convert_vtks_to_txts(
            vtks=lv_epi_vtks,
            motion_dofs=combine_dofs,
            output_dir=txt_dir.joinpath("LV_epi"),
            overwrite=overwrite
        )

        self.convert_vtks_to_txts(
            vtks=rv_vtks,
            motion_dofs=combine_dofs,
            output_dir=txt_dir.joinpath("RV"),
            overwrite=overwrite
        )

    @staticmethod
    def transform_mesh(target_mesh: MeshResource, source_mesh: MeshResource, landmark_init_dof: Path,
                       output_dir: Path, overwrite: bool = False, temp_dir: Path = None) -> MeshResource:
        output_dir.mkdir(parents=True, exist_ok=True)
        if temp_dir is None:
            temp_dir = output_dir.joinpath("temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        dof_dir = temp_dir.joinpath("dof")
        dof_dir.mkdir(parents=True, exist_ok=True)
        rigid_dof = dof_dir.joinpath("srreg.dof.gz")
        affine_dof = dof_dir.joinpath("sareg.dof.gz")
        nonrigid_dof = dof_dir.joinpath("snreg.dof.gz")
        temp_lm_vtk = MeshResource(temp_dir.joinpath("lm.vtk"))
        temp_srreg_vtk = MeshResource(temp_dir.joinpath("srreg.vtk"))
        transformed_vtk = MeshResource(output_dir.joinpath("fr00.vtk"))
        mirtk.transform_points(
            str(source_mesh),
            str(temp_lm_vtk),
            dofin=str(landmark_init_dof),
            invert=None
        )

        if not rigid_dof.exists() or overwrite:
            mirtk.register_points(
                "-t", str(target_mesh),  # subject
                "-s", str(source_mesh),  # template
                "-symmetric",
                dofin=str(landmark_init_dof),
                dofout=str(rigid_dof),
            )
        if not temp_srreg_vtk.exists() or overwrite:
            mirtk.transform_points(
                str(source_mesh),
                str(temp_srreg_vtk),
                dofin=str(rigid_dof),
                invert=None,
            )

        if not affine_dof.exists() or overwrite:
            mirtk.register_points(
                "-t", str(temp_srreg_vtk),
                "-s", str(target_mesh),
                "-symmetric",
                model="Affine",
                dofout=str(affine_dof),
            )

        if not nonrigid_dof.exists() or overwrite:
            mirtk.register(
                str(temp_srreg_vtk),
                str(target_mesh),
                "-par", "Point set distance correspondence", "CP",
                ds=20,
                model="FFD",
                dofin=str(affine_dof),
                dofout=str(nonrigid_dof),
            )

        if not transformed_vtk.exists() or overwrite:
            mirtk.transform_points(
                str(temp_srreg_vtk),
                str(transformed_vtk),
                dofin=str(nonrigid_dof),
            )
        return transformed_vtk

    @staticmethod
    def motion_mesh(frame_0_mesh: MeshResource, motion_dofs: DOFs, output_dir: Path, overwrite: bool = False):
        output_dir.mkdir(parents=True, exist_ok=True)
        vtks = [frame_0_mesh]
        for fr in tqdm(range(1, len(motion_dofs) + 1)):
            vtk = output_dir.joinpath("fr{:02d}.vtk".format(fr))
            if not vtk.exists() or overwrite:
                mirtk.transform_points(
                    str(frame_0_mesh),
                    str(vtk),
                    dofin=str(motion_dofs[fr]),
                )
            vtks.append(vtk)
        return vtks

    @staticmethod
    def convert_vtks_to_txts(vtks: List[Path], motion_dofs: DOFs, output_dir: Path, overwrite: bool = False):
        # Convert vtks to text files
        print("\n ...   Convert vtks to text files")
        output_dir.mkdir(parents=True, exist_ok=True)
        txts = []
        for fr in tqdm(range(0, len(motion_dofs) + 1)):
            txt = output_dir.joinpath("fr{:02d}.txt".format(fr))
            if not txt.exists() or overwrite:
                mirtk.convert_pointset(
                    str(vtks[fr]),
                    str(txt),
                )
            txts.append(txt)
        return vtks, txts
