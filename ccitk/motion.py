__all__ = [
    "warp_label",
    "forward_motion",
    "backward_motion",
    "average_forward_backward_motion",
    "mesh_motion",
    "phase_mesh_motion",
]

import shutil

import mirtk
from pathlib import Path
from typing import List
from tqdm import tqdm
from ccitk.common.resource import CardiacMesh


def warp_label(reference_label: Path, output_path: Path, dofin: Path, invert: bool = False):
    kwargs = {}
    if invert:
        kwargs["invert"] = None
    mirtk.transform_image(
        str(reference_label),
        str(output_path),
        dofin=str(dofin),
        interp="NN",
        **kwargs,
    )
    return output_path


def forward_motion(images: List[Path], output_dir: Path, parin: Path, compose_spacing: int = 10,
                   overwrite: bool = False) -> List[Path]:
    """
        Return:
            list of motion, ordering from 00 to 01, to 00 to N, where N is the number of the last frame.
    """
    assert output_dir.is_dir()
    forward_dofs = []
    for fr in tqdm(range(1, len(images))):

        target = images[fr - 1]
        source = images[fr]
        dof_output = output_dir.joinpath("ffd_forward_{:02d}_to_{:02d}.dof.gz".format(fr - 1, fr))
        if not dof_output.exists() or overwrite:
            mirtk.register(
                str(target),
                str(source),
                parin=str(parin),
                dofout=str(dof_output)
            )
        forward_dofs.append(dof_output)

    # Compose inter-frame transformation fields #
    print("\n ...  Compose forward inter-frame transformation fields")
    forward_compose_dofs = [forward_dofs[0]]
    for fr in tqdm(range(2, len(images))):
        dof_out = output_dir.joinpath("ffd_comp_forward_00_to_{:02d}.dof.gz".format(fr))
        dofs = [str(forward_dofs[k]) for k in range(0, fr)]

        if not dof_out.exists() or overwrite:
            mirtk.compose_dofs(
                *dofs,
                str(dof_out),
                approximate=None,
                spacing=compose_spacing,
            )
        forward_compose_dofs.append(dof_out)
    return forward_compose_dofs


def backward_motion(images: List[Path], output_dir: Path, parin: Path, compose_spacing: int = 10,
                    overwrite: bool = False) -> List[Path]:
    """
        Return:
            list of motion, ordering from 00 to 01, to 00 to N, where N is the number of the last frame.
    """
    # Backward image registration
    backward_dofs = {}
    for fr in tqdm(range(len(images) - 1, 0, -1)):
        target_fr = (fr + 1) % len(images)
        source_fr = fr
        target = images[target_fr]
        source = images[source_fr]
        dof_output = output_dir.joinpath("ffd_backward_{0:02d}_to_{1:02d}.dof.gz".format(target_fr, source_fr))
        if not dof_output.exists() or overwrite:
            mirtk.register(
                str(target),
                str(source),
                parin=str(parin),
                dofout=str(dof_output)
            )
        backward_dofs[fr] = dof_output
    # Compose inter-frame transformation fields #
    print("\n ...  Compose backward inter-frame transformation fields")
    backward_compose_dofs = {}
    backward_compose_dofs[len(images) - 1] = backward_dofs[len(images) - 1]
    for fr in tqdm(range(len(images) - 2, 0, -1)):
        dof_out = output_dir.joinpath("ffd_comp_backward_00_to_{:02d}.dof.gz".format(fr))
        dofs = [str(backward_dofs[k]) for k in range(len(images) - 1, fr - 1, -1)]
        if not dof_out.exists() or overwrite:
            mirtk.compose_dofs(
                *dofs,
                str(dof_out),
                approximate=None,
                spacing=compose_spacing,
            )
        backward_compose_dofs[fr] = dof_out
    backward_compose_dofs = [backward_compose_dofs[k] for k in sorted(backward_dofs.keys())]
    return backward_compose_dofs


def average_forward_backward_motion(
        forward_compose_dofs: List[Path], backward_compose_dofs: List[Path],
        output_dir: Path, overwrite: bool = False
) -> List[Path]:
    assert len(forward_compose_dofs) == len(backward_compose_dofs)
    combine_dofs = []
    cine_length = len(forward_compose_dofs) + 1
    for fr in tqdm(range(0, cine_length - 1)):
        dof_forward = forward_compose_dofs[fr]
        dof_backward = backward_compose_dofs[fr]
        weight_forward = float(cine_length - fr) / cine_length
        weight_backward = float(fr) / cine_length
        dof_combine = output_dir.joinpath("ffd_00_to_{:02d}.dof.gz".format(fr))

        if not dof_combine.exists() or overwrite:
            mirtk.average_3d_ffd(
                "2",
                str(dof_forward),
                weight_forward,
                str(dof_backward),
                weight_backward,
                str(dof_combine),
                verbose=None,
            )
        combine_dofs.append(dof_combine)
    return combine_dofs


def mesh_motion(reference_mesh: Path, motion_dofs: List[Path], output_dir: Path,
                file_prefix: str = "fr", overwrite: bool = False) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    vtks = [reference_mesh]
    vtk = output_dir.joinpath("{}{:02d}.vtk".format(file_prefix, 0))
    if not vtk.exists() or overwrite:
        shutil.copy(str(reference_mesh), str(vtk))
    for fr in tqdm(range(1, len(motion_dofs) + 1)):
        vtk = output_dir.joinpath("{}{:02d}.vtk".format(file_prefix, fr))
        if not vtk.exists() or overwrite:
            mirtk.transform_points(
                str(reference_mesh),
                str(vtk),
                dofin=str(motion_dofs[fr - 1]),
            )
        vtks.append(vtk)
    return vtks


def phase_mesh_motion(
        reference_mesh: CardiacMesh, motion_dofs: List[Path], output_dir: Path,
        file_prefix: str = "fr", overwrite: bool = False
):
    lv_endo_vtks = mesh_motion(
        reference_mesh=reference_mesh.lv.endocardium,
        motion_dofs=motion_dofs,
        output_dir=output_dir.joinpath("LV_endo"),
        file_prefix=file_prefix,
        overwrite=overwrite,
    )
    lv_epi_vtks = mesh_motion(
        reference_mesh=reference_mesh.lv.epicardium,
        motion_dofs=motion_dofs,
        output_dir=output_dir.joinpath("LV_epi"),
        file_prefix=file_prefix,
        overwrite=overwrite,
    )
    lv_myo_vtks = mesh_motion(
        reference_mesh=reference_mesh.lv.myocardium,
        motion_dofs=motion_dofs,
        output_dir=output_dir.joinpath("LV_myo"),
        file_prefix=file_prefix,
        overwrite=overwrite,
    )
    rv_vtks = mesh_motion(
        reference_mesh=reference_mesh.rv.rv,
        motion_dofs=motion_dofs,
        output_dir=output_dir.joinpath("RV"),
        file_prefix=file_prefix,
        overwrite=overwrite,
    )
    # rv_epi_vtks = mesh_motion(
    #     reference_mesh=reference_mesh.rv.epicardium,
    #     motion_dofs=motion_dofs,
    #     output_dir=output_dir.joinpath("RV_epi"),
    #     file_prefix=file_prefix,
    #     overwrite=overwrite,
    # )

    return {
        "lv": {"endo": lv_endo_vtks, "epi": lv_epi_vtks, "myo": lv_myo_vtks},
        "rv": {"rv": rv_vtks},
    }
