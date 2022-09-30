import dataclasses
from pathlib import Path
from pyhocon import ConfigTree, ConfigFactory
from ccitk.cmr_segment.common.constants import RESOURCE_DIR, ROOT_DIR
from tempfile import gettempdir
from datetime import datetime
from typing import List, Tuple

DATA_CONF_PATH = RESOURCE_DIR.joinpath("data.conf")
DATA_CONF = ConfigFactory.parse_file(str(DATA_CONF_PATH))


def get_conf(conf: ConfigTree, group: str = "", key: str = "", default=None):
    if group:
        key = ".".join([group, key])
    return conf.get(key, default)


@dataclasses.dataclass
class ImageLabelFormat:
    image: str
    label: str


@dataclasses.dataclass
class DatasetConfig:
    name: str
    dir: Path
    image_label_format: ImageLabelFormat
    dataframe_path: Path
    size: int = None

    @classmethod
    def from_conf(cls, name: str, mode: str, mount_prefix: Path):
        """From data.conf"""
        name_size = name.split(":")
        if len(name_size) > 1:
            size = int(name_size[1])
        else:
            size = None
        name = name_size[0]
        dir = mount_prefix.joinpath(get_conf(DATA_CONF, group=name, key="dir"))
        if mode == "2D":
            format = ImageLabelFormat(
            image=get_conf(DATA_CONF, group=name, key="2D.image_format"),
            label=get_conf(DATA_CONF, group=name, key="2D.label_format")
            )
            dataframe_path = get_conf(DATA_CONF, group=name, key="2D.dataframe", default=dir.joinpath("2D.csv"))
        elif mode == "3D":
            if get_conf(DATA_CONF, group=name, key="3D") is not None:
                format = ImageLabelFormat(
                    image=get_conf(DATA_CONF, group=name, key="3D.image_format"),
                    label=get_conf(DATA_CONF, group=name, key="3D.label_format"),
                )
                dataframe_path = get_conf(DATA_CONF, group=name, key="3D.dataframe", default=dir.joinpath("3D.csv"))
            else:
                raise ValueError()
        else:
            raise ValueError()

        return cls(
            name=name,
            dir=dir,
            image_label_format=format,
            dataframe_path=dataframe_path,
            size=size
        )


@dataclasses.dataclass
class DataConfig:
    mount_prefix: Path
    training_datasets: List[str]
    extra_validation_datasets: List[str] = None
    data_mode: str = "2D"
    validation_split: float = 0.2
    renew_dataframe: bool = False
    augmentation_prob: float = 0.5

    def __post_init__(self):
        if self.extra_validation_datasets is None:
            self.extra_validation_datasets = []

    @classmethod
    def from_conf(cls, conf_path: Path):
        """From train.conf"""
        conf = ConfigFactory.parse_file(str(conf_path))
        mount_prefix = Path(get_conf(conf, group="data", key="mount_prefix"))
        training_datasets = get_conf(conf, group="data", key="training_datasets")
        extra_validation_datasets = get_conf(conf, group="data", key="extra_validation_datasets")

        data_mode = get_conf(conf, group="data", key="data_mode")
        validation_split = get_conf(conf, group="data", key="validation_split")
        renew_dataframe = get_conf(conf, group="data", key="renew_dataframe", default=False)
        augmentation_prob = get_conf(conf, group="data", key="augmentation_prob")

        return cls(
            mount_prefix=mount_prefix,
            training_datasets=training_datasets,
            extra_validation_datasets=extra_validation_datasets,
            data_mode=data_mode,
            validation_split=validation_split,
            renew_dataframe=renew_dataframe,
            augmentation_prob=augmentation_prob,
        )


@dataclasses.dataclass
class ExperimentConfig:
    experiment_dir: Path = None
    batch_size: int = 32
    num_epochs: int = 100
    gpu: bool = False
    device: int = 0
    num_workers: int = 0
    pin_memory: bool = False
    n_inference: int = 10
    seed: int = 1024

    def __post_init__(self):
        if self.experiment_dir is None:
            self.experiment_dir = Path(gettempdir()).joinpath("CMRSegment", "experiments")
        time_now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.experiment_dir = self.experiment_dir.joinpath(time_now)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_conf(cls, user_conf_path: Path):
        pass


@dataclasses.dataclass
class AugmentationConfig:
    rotation_angles: Tuple[float] = (30, 30, 30)
    scaling_factors: Tuple[float] = (0.1, 0.2, 0.2)
    flip: float = 0.5

    channel_shift: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    gamma: float = 0.2

    @classmethod
    def from_conf(cls, conf_path: Path):
        conf = ConfigFactory.parse_file(str(conf_path))
        rotation_angles = get_conf(conf, group="augmentation", key="rotation_angles")
        scaling_factors = get_conf(conf, group="augmentation", key="scaling_factors")
        flip = get_conf(conf, group="augmentation", key="flip")
        channel_shift = get_conf(conf, group="augmentation", key="channel_shift")
        brightness = get_conf(conf, group="augmentation", key="brightness")
        contrast = get_conf(conf, group="augmentation", key="contrast")
        gamma = get_conf(conf, group="augmentation", key="gamma")
        return cls(
            rotation_angles=rotation_angles,
            scaling_factors=scaling_factors,
            flip=flip,
            channel_shift=channel_shift,
            brightness=brightness,
            contrast=contrast,
            gamma=gamma,
        )
