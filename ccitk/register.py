__all__ = [
    "register_landmarks",
    "register_points",
    "register_points_ICP",
    "register_cardiac_phases",
    "register_labels_affine",

    "transform_phase_mesh",
    "transform_mesh_using_landmarks",
    "transform_mesh_rigid",
    "transform_mesh_affine",
    "transform_mesh_ffd",
]
import mirtk
from pathlib import Path
from typing import Union, List, Dict
import vtk
import numpy as np

from ccitk.image import set_affine
from ccitk.common.resource import CardiacMesh

"""
    Points
"""


def register_landmarks(fixed: Path, moving: Path, output_path: Path, use_mirtk: bool = False, overwrite: bool = False):
    """

    """
    if not overwrite and output_path.exists():
        return output_path

    if not use_mirtk:
        from trimesh.registration import procrustes
        poly = vtk.vtkPolyData()
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(fixed))
        reader.Update()
        fixed = reader.GetOutput()
        fixed = np.array(fixed.GetPoints().GetData())

        poly = vtk.vtkPolyData()
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(moving))
        reader.Update()
        moving = reader.GetOutput()
        moving = np.array(moving.GetPoints().GetData())

        matrix, transform, cost = procrustes(
            moving, fixed, reflection=False, translation=True, scale=False, return_cost=True
        )  # matrix transform a to b
        # save matrix to dofs
        matrix = matrix.tolist()
        output_dir = output_path.parent
        output_dir.mkdir(exist_ok=True, parents=True)
        temp_txt = output_dir.joinpath(output_path.stem + ".txt")
        with open(str(temp_txt), "w") as file:
            for array in matrix:
                array = [str(a) for a in array]
                file.write(" ".join(array) + "\n")
        mirtk.convert_dof(
            temp_txt,
            output_path,
            input_format="aladin",
            output_format="mirtk",
        )
        temp_txt.unlink()
    else:
        mirtk.register(
            str(moving),  # target
            str(fixed),  # source
            model="Rigid",
            dofout=str(output_path),
        )
    return output_path


def register_points_ICP(moving: Union[Path, List[Path]], fixed: Union[Path, List[Path]], output_path: Path,
                        model: str = None, symmetric: bool = False, dofin: Path = None, overwrite: bool = False):

    """
        Register point sets or surfaces by iteratively approximating the residual target registration error
        given current point correspondences and transformation estimate using iterative closest points (ICP)
        algorithm.
    """
    if model is None:
        model = "Rigid"
    assert model in ["Rigid", "Similarity", "Affine"], \
        f"Model must be either Rigid, Similarity or Affine, not {model}."
    kwargs = {}
    if dofin is not None:
        kwargs["dofin"] = str(dofin)
    if symmetric:
        kwargs["symmetric"] = None
    kwargs["csp"] = None  # Using -closest-surface-point option
    if isinstance(moving, Path):
        assert isinstance(fixed, Path), \
            f"Moving and fixed should both be path or both be list. Fixed {fixed}, moving {moving}"
        if not output_path.exists() or overwrite:
            mirtk.register_points(
                t=str(moving),  # target
                s=str(fixed),  # source
                model=model,
                dofout=str(output_path),
                **kwargs,
            )
        return output_path
    assert len(moving) == len(fixed), "Moving and fixed must have the same length."
    args = []
    for m, f in zip(moving, fixed):
        args.append("-t")
        args.append(str(m))
        args.append("-s")
        args.append(str(f))
    if not output_path.exists() or overwrite:
        mirtk.register_points(
            *args,
            model=model,
            dofout=str(output_path),
            **kwargs,
        )
    return output_path


def register_points(
        moving: Union[Path, List[Path]], fixed: Union[Path, List[Path]], output_path: Path,
        model: str, dofin: Path = None, parin: Path = None, ds: int = None, overwrite: bool = False
):

    assert model in ["Rigid", "Affine", "FFD"], \
        f"Model must be either Rigid, Affine, or FFD, not {model}."
    if output_path.exists() and not overwrite:
        return output_path
    # Register atlas to subject
    kwargs = {}
    if dofin is not None:
        kwargs["dofin"] = str(dofin)
    if ds is not None:
        kwargs["ds"] = ds
    if parin is not None:
        kwargs["parin"] = str(parin)
    if isinstance(moving, Path):
        assert isinstance(fixed, Path), \
            f"Moving and fixed should both be path or both be list. Fixed {fixed}, moving {moving}"
        kwargs["par"] = ["Point set distance correspondence", "CP"]

        mirtk.register(
            str(moving),  # target
            str(fixed),  # source
            model=model,
            dofout=str(output_path),
            **kwargs
        )
        return output_path
    assert len(moving) == len(fixed), "Moving and fixed must have the same length."
    kwargs["par"] = ["Energy function", "PCD(T o P(1:2:end), P(2:2:end))"]

    args = []
    for m, f in zip(moving, fixed):
        args.append(str(m))
        args.append(str(f))

    mirtk.register(
        *args,
        model=model,
        dofout=str(output_path),
        **kwargs
    )
    return output_path


"""
    Images
"""


def register_labels_affine(
        fixed_label: Path, moving_label: Path, labels: List,
        output_path: Path, parin: Path, dofin: Path = None, overwrite: bool = False,
):

    moving_labels = []
    fixed_labels = []
    temp_dir = output_path.parent.joinpath("temp_labels")
    temp_dir.mkdir(parents=True, exist_ok=True)
    for tag, label_path in zip(["moving", "fixed"], [moving_label, fixed_label]):
        for label in labels:
            path = temp_dir.joinpath(
                tag, Path(label_path.stem).stem + f"_{label}.nii.gz"
            )
            temp_dir.joinpath(tag).mkdir(parents=True, exist_ok=True)
            if tag == "moving":
                moving_labels.append(path)
            else:
                fixed_labels.append(path)
            if not path.exists():
                mirtk.calculate_element_wise(
                    str(label_path),
                    opts=[
                        ("binarize", label, label),
                        ("out", str(path))
                    ],
                )
            set_affine(label_path, path)

    if not output_path.exists() or overwrite:
        kwargs = {}
        if dofin is not None:
            kwargs["dofin"] = str(dofin)
        mirtk.register(
            *[str(path) for path in fixed_labels],  # target
            *[str(path) for path in moving_labels],  # source
            dofout=str(output_path),
            parin=str(parin),
            model="Affine",
            **kwargs
        )
    return output_path


"""
    Cardiac
"""


def transform_phase_mesh(
        mesh: CardiacMesh, output_mesh: Union[CardiacMesh, Path], dofin: Union[Path, Dict],
        overwrite: bool = False, invert: bool = False,
) -> CardiacMesh:
    if isinstance(output_mesh, Path):
        output_mesh.mkdir(parents=True, exist_ok=True)
        assert output_mesh.is_dir()
        output_mesh = CardiacMesh.from_dir(output_mesh, phase=mesh.phase)
    kwargs = {}
    if invert:
        kwargs["invert"] = None
    if isinstance(dofin, dict):
        lv_dofin = dofin["lv"]
        rv_dofin = dofin["rv"]
    else:
        lv_dofin = dofin
        rv_dofin = dofin
    if not output_mesh.exists() or overwrite:
        if mesh.rv.rv.exists():
            mirtk.transform_points(
                str(mesh.rv.rv),
                str(output_mesh.rv.rv),
                dofin=str(rv_dofin),
                **kwargs,
            )
        if mesh.rv.epicardium.exists():
            mirtk.transform_points(
                str(mesh.rv.epicardium),
                str(output_mesh.rv.epicardium),
                dofin=str(rv_dofin),
                **kwargs,
            )

        if mesh.lv.myocardium.exists():
            mirtk.transform_points(
                str(mesh.lv.myocardium),
                str(output_mesh.lv.myocardium),
                dofin=str(lv_dofin),
                **kwargs,
            )
        if mesh.lv.endocardium.exists():
            mirtk.transform_points(
                str(mesh.lv.endocardium),
                str(output_mesh.lv.endocardium),
                dofin=str(lv_dofin),
                **kwargs,
            )
        if mesh.lv.epicardium.exists():
            mirtk.transform_points(
                str(mesh.lv.epicardium),
                str(output_mesh.lv.epicardium),
                dofin=str(lv_dofin),
                **kwargs,
            )
    return output_mesh


def transform_mesh_using_landmarks(
        moving_mesh: Union[CardiacMesh, Path], fixed_landmarks: Path, moving_landmarks: Path, output_dir: Path,
        mirtk: bool = False, overwrite: bool = False,
):
    dofout = output_dir.joinpath("landmarks.dof.gz")
    dofout.parent.mkdir(parents=True, exist_ok=True)
    if not dofout.exists() or overwrite:
        dofout = register_landmarks(
            fixed=fixed_landmarks,
            moving=moving_landmarks,
            output_path=dofout,
            use_mirtk=mirtk,
            overwrite=overwrite,
        )
    output_dir.joinpath("mesh").mkdir(exist_ok=True, parents=True)
    if isinstance(moving_mesh, CardiacMesh):
        moving_mesh = transform_phase_mesh(
            mesh=moving_mesh,
            output_mesh=output_dir.joinpath("mesh"),
            dofin=dofout,
            overwrite=overwrite,
        )
    else:
        if not output_dir.joinpath(moving_mesh.name).exists() or overwrite:
            mirtk.transform_points(
                str(moving_mesh),
                str(output_dir.joinpath(moving_mesh.name)),
                dofin=str(dofout),
            )
            moving_mesh = output_dir.joinpath(moving_mesh.name)
    return moving_mesh, dofout


def transform_mesh_rigid(
        fixed_mesh: Union[CardiacMesh, Path], moving_mesh: Union[CardiacMesh, Path], output_dir: Path,
        dofin: Path = None, symmetric: bool = True, overwrite: bool = False,
):
    assert all([isinstance(fixed_mesh, CardiacMesh), isinstance(moving_mesh, CardiacMesh)]) or \
           all([isinstance(fixed_mesh, Path), isinstance(moving_mesh, Path)])
    if isinstance(fixed_mesh, CardiacMesh):
        phase = moving_mesh.phase
        rigid_dof = register_points_ICP(
            fixed=[fixed_mesh.rv.rv, fixed_mesh.lv.endocardium, fixed_mesh.lv.epicardium],
            moving=[moving_mesh.rv.rv, moving_mesh.lv.endocardium, moving_mesh.lv.epicardium],
            output_path=output_dir.joinpath(f"rigid_{phase}.dof.gz"),
            symmetric=symmetric,
            dofin=dofin,
            overwrite=overwrite,
        )

        lv_rigid_dof = register_points_ICP(
            fixed=[fixed_mesh.lv.endocardium, fixed_mesh.lv.epicardium],
            moving=[moving_mesh.lv.endocardium, moving_mesh.lv.epicardium],
            dofin=rigid_dof,
            output_path=output_dir.joinpath(f"lv_rigid_{phase}.dof.gz"),
            symmetric=symmetric,
            overwrite=overwrite,
        )

        rv_rigid_dof = register_points_ICP(
            fixed=fixed_mesh.rv.rv,
            moving=moving_mesh.rv.rv,
            dofin=rigid_dof,
            output_path=output_dir.joinpath(f"rv_rigid_{phase}.dof.gz"),
            symmetric=symmetric,
            overwrite=overwrite,
        )

        moving_mesh = transform_phase_mesh(
            mesh=moving_mesh,
            output_mesh=output_dir.joinpath("mesh"),
            dofin={"lv": lv_rigid_dof, "rv": rv_rigid_dof},
            overwrite=overwrite,
        )
        return moving_mesh, {"lv": lv_rigid_dof, "rv": rv_rigid_dof}
    rigid_dof = register_points_ICP(
        fixed=fixed_mesh,
        moving=moving_mesh,
        output_path=output_dir.joinpath(f"rigid.dof.gz"),
        symmetric=symmetric,
        dofin=dofin,
        overwrite=overwrite,
    )
    if not output_dir.joinpath(moving_mesh.name).exists() or overwrite:
        mirtk.transform_points(
            str(moving_mesh),
            str(output_dir.joinpath(moving_mesh.name)),
            dofin=str(rigid_dof),
        )
    return moving_mesh, rigid_dof


def transform_mesh_affine(
        fixed_mesh: Union[CardiacMesh, Path], moving_mesh: Union[CardiacMesh, Path],
        output_dir: Path, parin: Path = None, dofin: Union[Dict, Path] = None,
        use_ICP: bool = False, overwrite: bool = False,
):
    assert all([isinstance(fixed_mesh, CardiacMesh), isinstance(moving_mesh, CardiacMesh)]) or \
           all([isinstance(fixed_mesh, Path), isinstance(moving_mesh, Path)]), \
        f"Moving mesh is {moving_mesh}, fixed mesh is {fixed_mesh}"
    if isinstance(fixed_mesh, CardiacMesh):
        phase = moving_mesh.phase
        lv_affine_dof = output_dir.joinpath(f"lv_affine_{phase}.dof.gz")
        rv_affine_dof = output_dir.joinpath(f"rv_affine_{phase}.dof.gz")
        if dofin is not None:
            if isinstance(dofin, dict):
                lv_dofin = dofin["lv"]
                rv_dofin = dofin["rv"]
            else:
                assert isinstance(dofin, Path)
                lv_dofin = dofin
                rv_dofin = dofin
        else:
            lv_dofin = None
            rv_dofin = None

        if not use_ICP:
            register_points(
                fixed=fixed_mesh.lv.epicardium,
                moving=moving_mesh.lv.epicardium,
                model="Affine",
                dofin=lv_dofin,
                parin=parin,
                output_path=lv_affine_dof,
                overwrite=overwrite,
            )

            register_points(
                fixed=fixed_mesh.rv.rv,
                moving=moving_mesh.rv.rv,
                model="Affine",
                dofin=rv_dofin,
                parin=parin,
                output_path=rv_affine_dof,
                overwrite=overwrite,
            )
        else:
            register_points_ICP(
                fixed=[fixed_mesh.lv.epicardium, fixed_mesh.lv.endocardium, fixed_mesh.rv.rv],
                moving=[moving_mesh.lv.epicardium, moving_mesh.lv.epicardium, moving_mesh.rv.rv],
                model="Affine",
                dofin=rv_dofin,
                output_path=rv_affine_dof,
                overwrite=overwrite,
                symmetric=True,
            )
            lv_affine_dof = rv_affine_dof

        moving_mesh = transform_phase_mesh(
            mesh=moving_mesh,
            output_mesh=output_dir.joinpath("mesh"),
            dofin={"lv": rv_affine_dof, "rv": rv_affine_dof},
            overwrite=overwrite,
        )
        return moving_mesh, {"lv": lv_affine_dof, "rv": rv_affine_dof}
    assert isinstance(dofin, Path)
    moving_mesh = register_points(
        fixed=fixed_mesh,
        moving=moving_mesh,
        model="Affine",
        dofin=dofin,
        parin=parin,
        output_path=output_dir.joinpath(f"lv_affine.dof.gz"),
        overwrite=overwrite,
    )
    return moving_mesh, output_dir.joinpath(f"lv_affine.dof.gz")


def transform_mesh_ffd(
        fixed_mesh: CardiacMesh, moving_mesh: CardiacMesh, output_dir: Path, parin: Path = None,
        dofin: Dict = None, ds: int = 8, overwrite: bool = False,

):
    # using_label = all(
    #     [
    #         fixed_lv_label is not None, fixed_rv_label is not None,
    #         moving_rv_label is not None, moving_lv_label is not None,
    #     ]
    # )
    # assert all([fixed_lv_label is not None, fixed_rv_label is not None,
    #            moving_rv_label is not None, moving_lv_label is not None]) or \
    #        all([fixed_lv_label is None, fixed_rv_label is None,
    #            moving_rv_label is None, moving_lv_label is None])
    phase = moving_mesh.phase
    lv_dofin = None if dofin is None else dofin["lv"]
    rv_dofin = None if dofin is None else dofin["rv"]
    lv_ffd_dof = output_dir.joinpath(f"lv_ffd_{phase}.dof.gz")
    rv_ffd_dof = output_dir.joinpath(f"rv_ffd_{phase}.dof.gz")
    if not rv_ffd_dof.exists() or overwrite:
        # if using_label:
        #     kwargs = {}
        #     if rv_dofin is not None:
        #         kwargs["dofin"] = rv_dofin
        #     rv_ffd_label_dof = output_dir.joinpath(f"rv_ffd_label_{phase}.dof.gz")
        #     if not rv_ffd_label_dof.exists() or overwrite:
        #         mirtk.register(
        #             str(moving_rv_label),  # target
        #             str(fixed_rv_label),  # source,
        #             model="FFD",
        #             dofout=str(rv_ffd_label_dof),
        #             parin=str(parin),
        #             **kwargs,
        #         )
        #     rv_dofin = rv_ffd_label_dof
        rv_ffd_dof = register_points(
            fixed=fixed_mesh.rv.rv,
            moving=moving_mesh.rv.rv,
            ds=ds,
            model="FFD",
            dofin=rv_dofin,
            parin=parin,
            output_path=rv_ffd_dof,
        )
    if not lv_ffd_dof.exists() or overwrite:
        # if using_label:
        #     kwargs = {}
        #     if lv_dofin is not None:
        #         kwargs["dofin"] = lv_dofin
        #     lv_ffd_label_dof = output_dir.joinpath(f"lv_ffd_label_{phase}.dof.gz")
        #     if not lv_ffd_label_dof.exists() or overwrite:
        #         mirtk.register(
        #             str(moving_lv_label),  # target
        #             str(fixed_lv_label),  # source,
        #             model="FFD",
        #             dofout=str(lv_ffd_label_dof),
        #             parin=str(parin),
        #             **kwargs,
        #         )
        #     lv_dofin = lv_ffd_label_dof
        lv_ffd_dof = register_points(
            fixed=[fixed_mesh.lv.endocardium, fixed_mesh.lv.epicardium],
            moving=[moving_mesh.lv.endocardium, moving_mesh.lv.epicardium],
            ds=ds,
            model="FFD",
            dofin=lv_dofin,
            parin=parin,
            output_path=lv_ffd_dof,
        )
    output_mesh = transform_phase_mesh(
        mesh=moving_mesh,
        output_mesh=output_dir.joinpath("mesh"),
        dofin={"lv": lv_ffd_dof, "rv": rv_ffd_dof},
        overwrite=overwrite,
    )
    return output_mesh, {"lv": lv_ffd_dof, "rv": rv_ffd_dof}


def register_cardiac_phases(
        fixed_mesh: CardiacMesh,  fixed_landmarks: Path,
        moving_mesh: CardiacMesh, moving_landmarks: Path,
        output_dir: Path, affine_parin: Path = None, ffd_parin: Path = None,
        ds: int = 8, rigid: bool = False, overwrite: bool = False,
):
    phase = moving_mesh.phase
    temp_dir = output_dir.joinpath("temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    lm_dir = output_dir.joinpath("lm")
    lm_dir.mkdir(parents=True, exist_ok=True)
    # Landmark registration
    moving_mesh, rigid_dof = transform_mesh_using_landmarks(
        moving_mesh=moving_mesh,
        fixed_landmarks=fixed_landmarks,
        moving_landmarks=moving_landmarks,
        output_dir=lm_dir,
        mirtk=False,
        overwrite=overwrite,
    )
    rigid_dof = {"lv": rigid_dof, "rv": rigid_dof}
    if rigid:
        rigid_dir = output_dir.joinpath("rigid")
        rigid_dir.mkdir(parents=True, exist_ok=True)
        moving_mesh, rigid_dof = transform_mesh_rigid(
            fixed_mesh=fixed_mesh,
            moving_mesh=moving_mesh,
            output_dir=rigid_dir,
            dofin=rigid_dof["lv"],
            symmetric=True,
            overwrite=overwrite,
        )

    # Affine registration
    affine_dir = output_dir.joinpath("affine")
    affine_dir.mkdir(parents=True, exist_ok=True)
    moving_mesh, affine_dof = transform_mesh_affine(
        fixed_mesh=fixed_mesh,
        moving_mesh=moving_mesh,
        parin=affine_parin,
        output_dir=affine_dir,
        use_ICP=True,
        overwrite=overwrite,
    )

    # FFD registration
    ffd_dir = output_dir.joinpath("ffd")
    ffd_dir.mkdir(parents=True, exist_ok=True)
    moving_mesh, ffd_dof = transform_mesh_ffd(
        fixed_mesh=fixed_mesh,
        moving_mesh=moving_mesh,
        parin=ffd_parin,
        output_dir=ffd_dir,
        ds=ds,
        overwrite=overwrite,
    )

    return moving_mesh, ffd_dof
