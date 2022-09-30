import mirtk
from pathlib import Path
from ccitk.cmr_segment.preprocess import DataPreprocessor
from ccitk.cmr_segment.segmentor import CineSegmentor
from ccitk.cmr_segment.register import Coregister
from ccitk.cmr_segment.mesh import MeshExtractor

from ccitk.landmark import extract_simple_landmarks
from ccitk.cmr_segment.config import PipelineConfig
from ccitk.common.resource import Segmentation, CardiacMesh, Phase
from ccitk.cmr_segment.motion import MotionTracker
from ccitk.cmr_segment.refine import SegmentationRefiner
from ccitk.image import set_affine


class CMRPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, data_dir: Path):
        preprocessor = DataPreprocessor()
        if self.config.extract:
            mesh_extractor = MeshExtractor(
                iso_value=self.config.extract_config.iso_value,
                blur=self.config.extract_config.blur,
                overwrite=self.config.overwrite
            )
        if self.config.coregister:
            coregister = Coregister(
                template_dir=self.config.coregister_config.template_dir,
                param_dir=self.config.coregister_config.param_dir,
                overwrite=self.config.overwrite
            )
        if self.config.segment:
            if not self.config.segment_config.torch:
                from ccitk.cmr_segment.segmentor.tf1.HR import TF13DSegmentor

                hr_segmentor = TF13DSegmentor(
                    model_path=self.config.segment_config.model_path, overwrite=self.config.overwrite
                )
                hr_segmentor.__enter__()
            else:
                from ccitk.cmr_segment.segmentor.torch import TorchSegmentor
                hr_segmentor = TorchSegmentor(
                    model_path=self.config.segment_config.model_path, overwrite=self.config.overwrite,
                    device=self.config.segment_config.device
                )
            if self.config.segment_config.segment_cine:
                cine_segmentor = CineSegmentor(phase_segmentor=hr_segmentor)
        if self.config.refine:
            segmentation_corrector = SegmentationRefiner(
                csv_path=self.config.refine_config.csv_path,
                n_atlas=self.config.refine_config.n_atlas,
            )
        if self.config.track_motion:
            motion_tracker = MotionTracker(
                template_dir=self.config.motion_tracker_config.template_dir,
                param_dir=self.config.motion_tracker_config.param_dir,
                ffd_motion_cfg=self.config.motion_tracker_config.ffd_motion_cfg,
            )
        subjects = preprocessor.run(
            data_dir=data_dir,
            output_dir=self.config.output_dir,
            overwrite=self.config.overwrite,
            use_irtk=self.config.use_irtk,
            do_cine=self.config.do_cine,
            preprocess_flip=self.config.preprocess_flip
        )
        for ed_image, es_image, enlarged_cine, cine, output_dir in subjects:
            if self.config.segment and self.config.do_cine:
                print("Segmenting all {} cine images...".format(len(enlarged_cine)))
                cine_segmentations = cine_segmentor.apply(enlarged_cine, output_dir=output_dir, overwrite=self.config.overwrite)
            else:
                cine_segmentations = [
                    Segmentation(
                        path=output_dir.joinpath("segs").joinpath(f"lvsa_{idx}.nii.gz"), phase=idx
                    ) for idx in range(len(enlarged_cine))
                ]

            meshes = []
            segmentations = []
            landmark_path = output_dir.joinpath("landmarks.vtk".format(str(ed_image.phase)))

            for phase_image in [ed_image, es_image]:
                if self.config.segment:
                    print("Segmenting {} image...".format(phase_image.phase))
                    if self.config.overwrite or not output_dir.joinpath(f"seg_lvsa_SR_{phase_image.phase}.nii.gz").exists():
                        segmentation = hr_segmentor.apply(
                            phase_image, output_path=output_dir.joinpath(f"seg_lvsa_SR_{phase_image.phase}.nii.gz")
                        )
                    else:
                        segmentation = Segmentation(
                            phase=phase_image.phase, path=output_dir.joinpath(f"seg_lvsa_SR_{phase_image.phase}.nii.gz")
                        )
                else:
                    segmentation = Segmentation(
                        phase=phase_image.phase, path=output_dir.joinpath(f"seg_lvsa_SR_{phase_image.phase}.nii.gz")
                    )

                if self.config.refine and self.config.refine_config.is_lr_seg:
                    mirtk.resample_image(
                        str(segmentation.path),
                        str(output_dir.joinpath(Path(segmentation.path.stem).stem + "_resampled.nii.gz")),
                        '-size', 1.25, 1.25, 2
                    )
                    mirtk.enlarge_image(
                        str(output_dir.joinpath(Path(segmentation.path.stem).stem + "_resampled.nii.gz")),
                        str(output_dir.joinpath(Path(segmentation.path.stem).stem + "_enlarged.nii.gz")),
                        z=20, value=0
                    )
                    segmentation.path = output_dir.joinpath(Path(segmentation.path.stem).stem + "_enlarged.nii.gz")
                    set_affine(from_image=phase_image.path, to_image=segmentation.path)

                if self.config.extract:
                    print("Extracting landmarks from {} segmentation...".format(phase_image.phase))
                    if (not landmark_path.exists() or self.config.overwrite) and phase_image.phase == Phase.ED:
                        try:
                            # landmark_path = extract_landmarks(
                            #     segmentation.path, output_path=landmark_path, labels=[2, 3]
                            # )
                            landmark_path = extract_simple_landmarks(
                                segmentation.path, output_path=landmark_path,
                            )

                        except ValueError:
                            pass
                if self.config.refine:
                    print("Refining")
                    if landmark_path.exists() and segmentation.exists():
                        segmentation = segmentation_corrector.run(
                            subject_image=phase_image,
                            subject_seg=segmentation,
                            subject_landmarks=landmark_path,
                            output_dir=output_dir.joinpath("refine"),
                            n_top=self.config.refine_config.n_top_atlas,
                            force=self.config.overwrite,
                        )
                    if self.config.refine_config.is_lr_seg:
                        try:
                            # landmark_path = extract_landmarks(
                            #     segmentation.path, output_path=landmark_path, labels=[2, 3]
                            # )
                            landmark_path = extract_simple_landmarks(
                                segmentation.path, output_path=landmark_path,
                            )
                        except ValueError:
                            pass
                if self.config.extract:
                    mesh = mesh_extractor.run(segmentation, output_dir.joinpath("mesh"))
                else:
                    mesh = CardiacMesh.from_dir(output_dir.joinpath("mesh"), phase=phase_image.phase)

                segmentations.append(segmentation)
                meshes.append(mesh)

                if self.config.coregister:
                    print("Coregistering")
                    if landmark_path.exists() and mesh.exists() and segmentation.exists():
                        coregister.run(
                            mesh, segmentation, landmark_path, output_dir=output_dir.joinpath("registration")
                        )

            if self.config.track_motion:
                print("Tracking motion")
                motion_tracker.run(
                    cine=enlarged_cine,
                    ed_segmentation=segmentations[0],
                    landmarks=landmark_path,
                    ED_mesh=meshes[0],
                    output_dir=output_dir.joinpath("motion"),
                    overwrite=self.config.overwrite,
                )
