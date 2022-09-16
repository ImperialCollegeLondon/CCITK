__all__ = [
    "select_top_similar_atlases",
    "refine_segmentation_with_atlases",
]

import mirtk
import numpy as np
from typing import List
import SimpleITK as sitk
from pathlib import Path
from ccitk.image import set_affine
from ccitk.register import register_landmarks, register_labels_affine


mirtk.subprocess.showcmd = True


def select_top_similar_atlases(
        atlases_label: List[Path], atlases_landmark: List[Path],
        subject_image: Path, subject_label: Path, subject_landmarks: Path, parin: Path,
        output_dir: Path, n_top: int = 5, overwrite: bool = False
):
    """Select top similar atlases, according to subject segmentation and landmark"""
    assert len(atlases_label) == len(atlases_landmark)
    nmi = []

    top_similar_atlases = []

    n_atlases = len(atlases_label)

    output_dofs = []
    top_atlas_dofs = []
    top_atlas_landmarks = []

    for i in range(n_atlases):
        affine_dof = output_dir.joinpath("dof", f"shapeaffine_{i}.dof.gz")
        affine_dof.parent.mkdir(parents=True, exist_ok=True)
        if not affine_dof.exists() or overwrite:
            lm_dof = register_landmarks(
                fixed=subject_landmarks,
                moving=atlases_landmark[i],
                output_path=output_dir.joinpath("dof", f"shapelandmarks_{i}.dof.gz"),
                mirtk=False,
                overwrite=overwrite
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
