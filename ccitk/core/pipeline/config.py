import dataclasses
from pathlib import Path
from argparse import ArgumentParser
from CMRSegment.common.constants import ROOT_DIR, LIB_DIR


@dataclasses.dataclass
class PipelineModuleConfig:
    overwrite: bool = False


@dataclasses.dataclass
class SegmentorConfig(PipelineModuleConfig):
    model_path: Path = None
    segment_cine: bool = True
    torch: bool = True
    device: int = 0

    def __post_init__(self):
        if self.model_path is None:
            raise TypeError("__init__ missing 1 required argument: 'model_path'")


@dataclasses.dataclass
class RefinerConfig(PipelineModuleConfig):
    csv_path: Path = None
    n_top_atlas: int = 7
    n_atlas: int = 500
    is_lr_seg: bool = False  # If the segmentation input for refiner is 2D low resolution

    def __post_init__(self):
        if self.csv_path is None:
            raise TypeError("__init__ missing 1 required argument: 'csv_path'")


@dataclasses.dataclass
class MeshExtractorConfig(PipelineModuleConfig):
    iso_value: int = 120
    blur: int = 2


@dataclasses.dataclass
class CoregisterConfig(PipelineModuleConfig):
    param_dir: Path = None
    template_dir: Path = None

    def __post_init__(self):
        if self.param_dir is None:
            self.param_dir = LIB_DIR.joinpath("CMRSegment", "resource")
        if self.template_dir is None:
            self.template_dir = ROOT_DIR.joinpath("input", "params")


@dataclasses.dataclass
class MotionTrackerConfig(PipelineModuleConfig):
    param_dir: Path = None
    template_dir: Path = None
    ffd_motion_cfg: Path = None

    def __post_init__(self):
        if self.param_dir is None:
            self.param_dir = LIB_DIR.joinpath("CMRSegment", "resource")
        if self.template_dir is None:
            self.template_dir = ROOT_DIR.joinpath("input", "params")
        if self.ffd_motion_cfg is None:
            self.ffd_motion_cfg = LIB_DIR.joinpath("CMRSegment", "resource", "ffd_motion_2.cfg")


class PipelineConfig:
    def __init__(
        self,
        segment: bool,
        refine: bool,
        extract: bool,
        coregister: bool,
        track_motion: bool,
        output_dir: Path,
        overwrite: bool = False,
        do_cine: bool = False,
        model_path: Path = None,
        torch: bool = True,
        device: int = 0,
        refine_csv_path: Path = None,
        refine_n_top: int = 7,
        refine_n_atlas: int = 500,
        refine_is_lr_seg: bool = False,
        iso_value: int = 120,
        blur: int = 2,
        param_dir: Path = None,
        template_dir: Path = None,
        ffd_motion_cfg: Path = None,
        use_irtk: bool = False,
    ):
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.do_cine = do_cine
        if segment:
            self.segment_config = SegmentorConfig(
                model_path=model_path,
                overwrite=overwrite,
                torch=torch,
                device=device,
            )
            self.segment = True
        else:
            self.segment = False
            self.segment_config = None
        if refine:
            assert refine_csv_path is not None, "Need to provide atlas directory if using refine."
            self.refine = True
            self.refine_config = RefinerConfig(
                csv_path=refine_csv_path,
                n_top_atlas=refine_n_top,
                n_atlas=refine_n_atlas,
                is_lr_seg=refine_is_lr_seg,
            )
        else:
            self.refine = False
            self.refine_config = None
        if extract:
            self.extract_config = MeshExtractorConfig(
                iso_value=iso_value, blur=blur,  overwrite=overwrite
            )
            self.extract = True
        else:
            self.extract = False
            self.extract_config = None
        if coregister:
            self.coregister_config = CoregisterConfig(
                overwrite=overwrite, param_dir=param_dir, template_dir=template_dir
            )
            self.coregister = True
        else:
            self.coregister = False
            self.coregister_config = None
        self.use_irtk = use_irtk

        if track_motion:
            self.motion_tracker_config = MotionTrackerConfig(
                overwrite=overwrite, param_dir=param_dir, template_dir=template_dir, ffd_motion_cfg=ffd_motion_cfg,
            )
            self.track_motion = True
        else:
            self.track_motion = False
            self.motion_tracker_config = None

    @staticmethod
    def argument_parser() -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument("-o", "--output-dir", dest="output_dir", required=True, type=str)
        parser.add_argument("--overwrite", dest="overwrite", action="store_true")
        parser.add_argument("--irtk", dest="use_irtk", action="store_true")
        parser.add_argument("--do-cine", action="store_true", help="Preprocess and segment cine")

        parser.add_argument(
            "--segment", dest="segment", action="store_true",
            help="Run segmentor module to segment image."
        )
        parser.add_argument(
            "--refine", dest="refine", action="store_true",
            help="Run refiner module to refine segmentation using multi-altas registration."
        )
        parser.add_argument(
            "--extract", dest="extract", action="store_true",
            help="Extract meshes from segmentations."
        )
        parser.add_argument("--coregister", dest="coregister", action="store_true")
        parser.add_argument("--track-motion", dest="track_motion", action="store_true")

        segment_parser = parser.add_argument_group("segment")
        segment_parser.add_argument("--model-path", dest="model_path", default=None, type=str)
        segment_parser.add_argument("--torch", dest="torch", action="store_true")
        segment_parser.add_argument("--device", dest="device", default=0, type=int)

        refine_parser = parser.add_argument_group("refine")
        refine_parser.add_argument(
            "--csv-path", dest="refine_csv_path", type=str, default=None,
            help="Path to a csv file where all the atlases will be used for refinement."
        )
        refine_parser.add_argument(
            "--n-top", dest="refine_n_top", type=int, default=7,
            help="Number of top similar atlases, selected for refinement"
        )
        refine_parser.add_argument(
            "--n-atlas", dest="refine_n_atlas", type=int, default=500,
            help="Number of atlases in total for for refinement"
        )
        refine_parser.add_argument("--is-lr-seg", dest="is_lr_seg", action="store_true")

        extract_parser = parser.add_argument_group("extract")
        extract_parser.add_argument("--iso-value", dest="iso_value", default=120, type=int)
        extract_parser.add_argument("--blur", dest="blur", default=2, type=int)

        coregister_parser = parser.add_argument_group("coregister")
        coregister_parser.add_argument("--template-dir", dest="template_dir", default=None, type=str)
        coregister_parser.add_argument("--param-dir", dest="param_dir", default=None, type=str)
        coregister_parser.add_argument("--ffd-motion-cfg", dest="ffd_motion_cfg", default=None, type=str)

        return parser

    @classmethod
    def from_args(cls, args):
        return cls(
            segment=args.segment,
            refine=args.refine,
            extract=args.extract,
            coregister=args.coregister,
            track_motion=args.track_motion,
            output_dir=Path(args.output_dir),
            overwrite=args.overwrite,
            do_cine=args.do_cine,
            model_path=Path(args.model_path) if args.model_path is not None else None,
            torch=args.torch,
            device=args.device,
            refine_csv_path=Path(args.refine_csv_path) if args.refine_csv_path is not None else None,
            refine_n_top=args.refine_n_top,
            refine_n_atlas=args.refine_n_atlas,
            refine_is_lr_seg=args.is_lr_seg,
            iso_value=args.iso_value,
            blur=args.blur,
            param_dir=Path(args.param_dir) if args.param_dir is not None else None,
            template_dir=Path(args.template_dir) if args.template_dir is not None else None,
            ffd_motion_cfg=Path(args.ffd_motion_cfg) if args.ffd_motion_cfg is not None else None,
            use_irtk=args.use_irtk,
        )
