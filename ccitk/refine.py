__all__ = [
    "select_top_similar_atlases",
    "refine_segmentation_with_atlases",
]

import mirtk
import numpy as np
from typing import List
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple

from ccitk.image import set_affine
from ccitk.register import register_landmarks, register_labels_affine


mirtk.subprocess.showcmd = True


def select_top_similar_atlases(
        atlases_label: List[Path], atlases_landmark: List[Path],
        subject_image: Path, subject_label: Path, subject_landmarks: Path, parin: Path,
        output_dir: Path, n_top: int = 5, overwrite: bool = False
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Select top similar atlases, according to subject segmentation and landmark

    Args:
        atlases_label: list of paths to atlas labels
        atlases_landmark: list of paths to atlas landmarks
        subject_image: subject image path
        subject_label: subject label path
        subject_landmarks: subject landmarks path
        parin: affine registration param path
        output_dir: output directory to store the intermediate results
        n_top (optional): number of top similar atlases to be selected
        overwrite (optional): whether to overwrite intermediate results if exist

    Returns:
        A tuple of three elements, (top_similar_atlases, top_atlas_dofs, top_atlas_landmarks).
        Each is a list of paths, with length equals n_top.

    """

    assert len(atlases_label) == len(atlases_landmark)
    nmi = []

    top_similar_atlases = []

    n_atlases = len(atlases_label)

    output_dofs = []
    top_atlas_dofs = []
    top_atlas_landmarks = []
    output_dir.joinpath("nmi").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("dof").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("atlas").mkdir(parents=True, exist_ok=True)

    for i in range(n_atlases):
        affine_dof = output_dir.joinpath("dof", f"shapeaffine_{i}.dof.gz")
        if not affine_dof.exists() or overwrite:
            lm_dof = register_landmarks(
                fixed=atlases_landmark[i],
                moving=subject_landmarks,
                output_path=output_dir.joinpath("dof", f"shapelandmarks_{i}.dof.gz"),
                use_mirtk=False,
                overwrite=overwrite,
            )
            # Affine registration using landmark as initialisation
            # Split label maps into separate binary masks
            new_atlas_path = output_dir.joinpath("atlas", f"{i}", atlases_label[i].name)
            if not new_atlas_path.exists() or overwrite:
                output_dir.joinpath("atlas", f"{i}").mkdir(parents=True, exist_ok=True)
                mirtk.calculate_element_wise(
                    str(atlases_label[i]),
                    "-label", 3, 4,
                    set=3,
                    output=str(new_atlas_path),
                )
                set_affine(atlases_label[i], new_atlas_path)
            atlases_label[i] = new_atlas_path

            mirtk.transform_image(
                str(new_atlas_path),
                str(new_atlas_path.parent.joinpath("atlas_label_init.nii.gz")),
                dofin=str(lm_dof),
                target=str(subject_image),
                interp="NN",
            )
            mirtk.transform_points(
                str(atlases_landmark[i]),
                str(new_atlas_path.parent.joinpath(f"atlas_lm_init.vtk")),
                "-invert",
                dofin=str(lm_dof),
            )

            mirtk.transform_points(
                str(subject_landmarks),
                str(new_atlas_path.parent.joinpath(f"subject_lm_init.vtk")),
                dofin=str(lm_dof),
            )

            affine_dof = register_labels_affine(
                fixed_label=subject_label,
                moving_label=new_atlas_path.parent.joinpath("atlas_label_init.nii.gz"),
                labels=[1, 2, 3],
                output_path=output_dir.joinpath("dof", f"shapeaffine_{i}.dof.gz"),
                parin=parin,
                dofin=lm_dof,
                overwrite=overwrite,
            )

        if not output_dir.joinpath("nmi", f"shapenmi_{i}.txt").exists() or overwrite:
            mirtk.evaluate_similarity(
                str(subject_label),  # target
                str(atlases_label[i]),  # source
                Tbins=64,
                Sbins=64,
                dofin=str(affine_dof),  # source image transformation
                table=str(output_dir.joinpath("nmi", f"shapenmi_{i}.txt")),
            )
        output_dofs.append(affine_dof)

        if output_dir.joinpath("nmi", f"shapenmi_{i}.txt").exists():
            similarities = np.genfromtxt(str(output_dir.joinpath("nmi", f"shapenmi_{i}.txt")), delimiter=",")
            nmi += [similarities[1, 8]]
        else:
            nmi += [0]

    if n_top < n_atlases:
        sorted_indexes = np.array(nmi).argsort()[::-1]
        for i in range(n_top):
            top_similar_atlases += [atlases_label[sorted_indexes[i]]]
            top_atlas_dofs += [output_dofs[sorted_indexes[i]]]
            top_atlas_landmarks += [atlases_landmark[sorted_indexes[i]]]
    else:
        top_similar_atlases = atlases_label
        top_atlas_dofs = output_dofs
        top_atlas_landmarks = atlases_landmark

    return top_similar_atlases, top_atlas_dofs, top_atlas_landmarks


def refine_segmentation_with_atlases(
        atlases_label: List[Path], atlases_landmark: List[Path],
        subject_image: Path, subject_segmentation: Path, subject_landmarks: Path, phase: str,
        affine_parin: Path, ffd_parin: Path, output_path: Path, n_top: int = 5, overwrite: bool = False,
):
    """
    Refine segmentation by selecting top similar atlases to the subject and registering each to the subject and using
    majority voting to fuse the labels.

    Args:
        atlases_label: list of paths to atlas labels
        atlases_landmark: list of paths to atlas landmarks
        subject_image: subject image path
        subject_segmentation: subject label path
        subject_landmarks: subject landmark path
        phase: subject phase, ED or ES
        affine_parin: affine registration param path
        ffd_parin: ffd registration param path
        output_path: output path of the refined segmentation
        n_top (optional): number of top similar atlases to be selected
        overwrite (optional): whether to overwrite intermediate results if exist

    Returns:
        output path of the refined segmentation
    """
    top_atlases, top_dofs, top_lm = select_top_similar_atlases(
        atlases_label=atlases_label,
        atlases_landmark=atlases_landmark,
        subject_image=subject_image,
        subject_label=subject_segmentation,
        subject_landmarks=subject_landmarks,
        parin=affine_parin,
        output_dir=output_path.parent.joinpath("select"),
        n_top=n_top,
        overwrite=overwrite,
    )
    atlas_labels = []
    tmp_dir = output_path.parent.joinpath("temp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i, (atlas, dof) in enumerate(zip(top_atlases, top_dofs)):

        label_path = tmp_dir.joinpath(f"seg_affine_{i}_{phase}.nii.gz")
        if not label_path.exists() or overwrite:
            mirtk.transform_image(
                str(atlas),
                str(label_path),
                dofin=str(dof),  # Transformation that maps atlas to subject
                target=str(subject_image),
                interp="NN",
            )
            set_affine(subject_image, label_path)

            # Transform points for debugging
            mirtk.transform_points(
                str(top_lm[i]),
                str(tmp_dir.joinpath(f"lm_affine_{i}_{phase}.vtk")),
                "-invert",
                dofin=str(dof),
            )

        if not tmp_dir.joinpath(f"shapeffd_{i}_{str(phase)}.dof.gz").exists() or overwrite:
            mirtk.register(
                str(subject_segmentation),  # target
                str(atlas),  # source
                parin=str(ffd_parin),
                dofin=str(dof),
                dofout=tmp_dir.joinpath(f"shapeffd_{i}_{phase}.dof.gz")
            )

        label_path = tmp_dir.joinpath(f"seg_{i}_{phase}.nii.gz")
        if not label_path.exists() or overwrite:
            mirtk.transform_image(
                str(atlas),
                str(label_path),
                dofin=str(tmp_dir.joinpath(f"shapeffd_{i}_{phase}.dof.gz")),
                target=str(subject_image),
                interp="NN",
            )
        dof = tmp_dir.joinpath(f"shapeffd_{i}_{phase}.dof.gz")
        # Transform points for debugging
        mirtk.transform_points(
            str(top_lm[i]),
            str(tmp_dir.joinpath(f"lm_ffd_{i}_{phase}.vtk")),
            "-invert",
            dofin=str(dof),
        )
        atlas_labels.append(label_path)

    # apply label fusion
    labels = sitk.VectorOfImage()

    for label_path in atlas_labels:
        label = sitk.ReadImage(str(label_path), imageIO="NiftiImageIO", outputPixelType=sitk.sitkUInt8)
        labels.push_back(label)
    voter = sitk.LabelVotingImageFilter()
    voter.SetLabelForUndecidedPixels(0)
    fused_label = voter.Execute(labels)
    sitk.WriteImage(
        fused_label, str(output_path), imageIO="NiftiImageIO"
    )
    return output_path
