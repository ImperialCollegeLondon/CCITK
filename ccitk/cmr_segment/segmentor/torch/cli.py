import torch
import shutil
from pathlib import Path
from pyhocon import ConfigFactory
from argparse import ArgumentParser

from ccitk.resource import PhaseImage
from ccitk.cmr_segment.common.config import get_conf
from ccitk.cmr_segment.segmentor.torch.network import UNet
from ccitk.cmr_segment.segmentor.torch import TorchSegmentor


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-path", dest="model_path", type=str, default=None)
    parser.add_argument("-c", "--checkpoint-path", dest="checkpoint_path", type=str, default=None)
    parser.add_argument("-n", "--network-conf", dest="network_conf_path", default=None, type=str)

    parser.add_argument("-i", "--input-path", dest="input_path", type=str, required=True, help="Input path of a nii.gz file.")
    parser.add_argument("--phase", dest="phase", default="phase", type=str)

    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str, required=True)
    parser.add_argument("-d", "--device", dest="device", default=0, type=int)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.model_path is not None and args.checkpoint_path is not None:
        raise ValueError("Must input either a model path or a checkpoint path.")
    if args.model_path is None and args.checkpoint_path is None:
        raise ValueError("Must input either a model path or a checkpoint path.")
    if args.checkpoint_path is not None:
        if args.network_conf_path is None:
            raise ValueError("If a checkpoint path is provided, network conf must be provided too.")
        train_conf = ConfigFactory.parse_file(str(Path(args.network_conf_path)))
        network = UNet(
            in_channels=get_conf(train_conf, group="network", key="in_channels"),
            n_classes=get_conf(train_conf, group="network", key="n_classes"),
            n_filters=get_conf(train_conf, group="network", key="n_filters"),
        )
        checkpoint = torch.load(str(Path(args.checkpoint_path)), map_location=torch.device(args.device))
        network.load_state_dict(checkpoint)
        network.cuda(args.device)
        torch.save(network, str(output_dir.joinpath("inference_model.pt")))
        model_path = output_dir.joinpath("inference_model.pt")
    else:
        model_path = Path(args.model_path)
    segmentor = TorchSegmentor(model_path=model_path, device=args.device, overwrite=args.overwrite)
    shutil.copy(str(Path(args.input_path)), str(output_dir))
    segmentation = segmentor.apply(
        image=PhaseImage(phase=args.phase, path=Path(args.input_path)),
        output_path=output_dir.joinpath("seg.nii.gz")
    )


if __name__ == '__main__':
    main()
