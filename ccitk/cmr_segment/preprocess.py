import os
import mirtk
import shutil

import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Iterable, Tuple
from ccitk.common.resource import NiiData, CineImages, PhaseImage, Phase


class DataPreprocessor:
    """Find subjects in data_dir, find ED/ES phases, and split into a sequence of phases"""
    def run(self, data_dir: Path, output_dir: Path, overwrite: bool = False, use_irtk: bool = True,
            do_cine: bool = False, do_contrast: bool = True, preprocess_flip: bool = False)\
            -> Iterable[Tuple[PhaseImage, PhaseImage, CineImages, Path]]:
        do_contrast = True
        for idx, subject_dir in enumerate(sorted(os.listdir(str(data_dir)))):
            subject_output_dir = output_dir.joinpath(subject_dir)
            if not data_dir.joinpath(subject_dir).is_dir():
                continue
            nii_data = NiiData.from_dir(dir=data_dir.joinpath(subject_dir))
            if not nii_data.exists():
                continue

            subject_output_dir.mkdir(exist_ok=True, parents=True)
            print(subject_dir)
            if overwrite or not NiiData.from_dir(dir=subject_output_dir).exists():
                if nii_data != subject_output_dir:
                    shutil.copy(str(nii_data), str(subject_output_dir))
            ed_image = PhaseImage.from_dir(subject_output_dir, phase=Phase.ED)
            es_image = PhaseImage.from_dir(subject_output_dir, phase=Phase.ES)
            print("ED ES image:\n\t{}\n\t{}".format(repr(ed_image), repr(es_image)))
            contrasted_nii_path = subject_output_dir.joinpath("contrasted_{}".format(nii_data.name))

            # Flip nii data
            if preprocess_flip:
                nim = nib.load(str(nii_data))
                image = nim.get_data()
                image = np.flip(image, axis=1)

                nim2 = nib.Nifti1Image(image, nim.affine)
                nim2.header['pixdim'] = nim.header['pixdim']
                nib.save(nim2, str(subject_output_dir.joinpath("LVSA_flipped.nii.gz")))
                nii_data = subject_output_dir.joinpath("LVSA_flipped.nii.gz")
            if not ed_image.exists() or not es_image.exists() or not contrasted_nii_path.exists():
                print(' Detecting ED/ES phases {}...'.format(str(nii_data)))
                if not use_irtk:
                    if do_contrast:
                        mirtk.auto_contrast(str(nii_data), str(contrasted_nii_path))
                        nii_data = contrasted_nii_path
                    mirtk.detect_cardiac_phases(
                        str(nii_data), output_ed=str(ed_image.path), output_es=str(es_image.path)
                    )
                else:
                    if do_contrast:
                        command = 'autocontrast '\
                                  f'{str(nii_data)} '\
                                  f'{str(contrasted_nii_path)}'
                        print(command)
                        subprocess.call(command, shell=True)
                        nii_data = contrasted_nii_path

                    command = 'cardiacphasedetection '\
                              f'{str(nii_data)} '\
                              f'{str(ed_image.path)} '\
                              f'{str(es_image.path)}'
                    print(command)
                    subprocess.call(command, shell=True)
                print('  Found ED/ES phases ...')

            if not ed_image.exists() or not es_image.exists():
                print(" ED {0} or ES {1} does not exist. Skip.".format(ed_image, es_image))
                continue

            # resample and enlarge ED/ES image
            ed_image = self.resample_image(
                ed_image,
                output_path=subject_output_dir.joinpath(f"lvsa_SR_ED_resampled.nii.gz"),
                overwrite=overwrite,
                use_irtk=use_irtk
            )
            enlarged_ed_image = self.enlarge_image(
                ed_image,
                output_path=subject_output_dir.joinpath(f"lvsa_SR_ED.nii.gz"),
                overwrite=overwrite,
                use_irtk=use_irtk,
            )
            es_image = self.resample_image(
                es_image,
                output_path=subject_output_dir.joinpath(f"lvsa_SR_ES_resampled.nii.gz"),
                overwrite=overwrite,
                use_irtk=use_irtk,
            )
            enlarged_es_image = self.enlarge_image(
                es_image,
                output_path=subject_output_dir.joinpath(f"lvsa_SR_ES.nii.gz"),
                overwrite=overwrite,
                use_irtk=use_irtk,
            )
            enlarged_images = []
            if do_cine:
                gray_phase_dir = subject_output_dir.joinpath("gray_phases")
                gray_phase_dir.mkdir(parents=True, exist_ok=True)
                cine = CineImages.from_dir(gray_phase_dir)
                if overwrite or len(cine) == 0:
                    print(" ... Split sequence")
                    if not use_irtk:
                        mirtk.split_volume(
                            str(nii_data), "{}/lvsa_".format(str(gray_phase_dir)), "-sequence"
                        )
                    else:
                        command = f'splitvolume {str(nii_data)} '\
                                  f'{str(gray_phase_dir)}/lvsa_ -sequence'
                        print(command)
                        subprocess.call(command, shell=True)
                    cine = CineImages.from_dir(gray_phase_dir)

                # resample and enlarge gray phases
                for idx, image in enumerate(cine):
                    image = self.resample_image(
                        image,
                        output_path=subject_output_dir.joinpath("resampled").joinpath(f"lvsa_{idx:02d}.nii.gz"),
                        overwrite=overwrite,
                        use_irtk=use_irtk
                    )
                    image = self.enlarge_image(
                        image,
                        output_path=subject_output_dir.joinpath("enlarged").joinpath(f"lvsa_{idx:02d}.nii.gz"),
                        overwrite=overwrite,
                        use_irtk=use_irtk,
                    )
                    enlarged_images.append(image)
            else:
                cine = CineImages([])
            enlarged_cine = CineImages(enlarged_images)
            yield enlarged_ed_image, enlarged_es_image, enlarged_cine, cine, subject_output_dir

    @staticmethod
    def resample_image(image: PhaseImage, output_path: Path, overwrite: bool = False, use_irtk: bool = True) -> PhaseImage:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        if overwrite or not output_path.exists():
            if not use_irtk:
                mirtk.resample_image(str(image.path), str(output_path), '-size', 1.25, 1.25, 2)
            else:
                command = 'resample ' \
                    f'{str(image.path)} ' \
                    f'{str(output_path)} ' \
                    '-size 1.25 1.25 2'
                print(command)
                subprocess.call(command, shell=True)
        return PhaseImage(path=output_path, phase=image.phase)

    @staticmethod
    def enlarge_image(image: PhaseImage, output_path: Path, overwrite: bool = False, use_irtk: bool = True) -> PhaseImage:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        if overwrite or not output_path.exists():
            if not use_irtk:
                mirtk.enlarge_image(str(image.path), str(output_path), z=20, value=0)
            else:
                command = 'enlarge_image ' \
                    f'{str(image.path)} ' \
                    f'{str(output_path)} ' \
                    '-z 20 -value 0'
                print(command)
                subprocess.call(command, shell=True)
        return PhaseImage(path=output_path, phase=image.phase)
