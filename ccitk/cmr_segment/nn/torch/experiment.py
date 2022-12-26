import torch
from ccitk.cmr_segment.nn.torch.data import TorchDataset
from ccitk.cmr_segment.nn.torch.data import MultiDataLoader
from ccitk.cmr_segment.nn.torch.augmentation import augment
from ccitk.cmr_segment.nn.torch import prepare_tensors

from ccitk.cmr_segment.nn.torch.loss import TorchLoss
from torch.optim.optimizer import Optimizer
from ccitk.cmr_segment.common.config import ExperimentConfig
from tqdm import tqdm
from typing import Iterable, Tuple, List, Callable
from torch.utils.tensorboard import SummaryWriter
import logging
from argparse import ArgumentParser
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)


class Experiment:
    def __init__(
        self,
        config: ExperimentConfig,
        network: torch.nn.Module,
        training_sets: List[TorchDataset],
        validation_sets: List[TorchDataset],
        extra_validation_sets: List[TorchDataset],
        loss: TorchLoss,
        optimizer: Optimizer,
        other_validation_metrics: List = None,
        tensor_board: SummaryWriter = None,
        inference_func: Callable = None,
        logger=None,
    ):
        self.config = config
        self.network = network
        self.training_sets = training_sets
        self.validation_sets = validation_sets
        self.extra_validation_sets = extra_validation_sets
        self.optimizer = optimizer
        self.loss = loss
        self.other_validation_metrics = other_validation_metrics if other_validation_metrics is not None else []
        self.tensor_board = tensor_board or SummaryWriter(str(self.config.experiment_dir.joinpath("tb_runs")))
        self.logger = logger or logging.getLogger("CMRSegment.nn.torch.Experiment")
        self.inference_func = inference_func
        self.set_device()

    def set_device(self):
        if self.config.gpu:
            if self.config.device is None or isinstance(self.config.device, str):
                device = 0
            else:
                device = self.config.device
        else:
            device = "cpu"
        if self.config.gpu:
            self.network.cuda(device=device)
        return self

    def train(self):
        self.network.train()
        train_data_loader = MultiDataLoader(
            *self.training_sets,
            batch_size=self.config.batch_size,
            sampler_cls="random",
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        set = False
        for epoch in range(self.config.num_epochs):
            self.network.train()
            self.logger.info("{}: starting epoch {}/{}".format(datetime.now(), epoch, self.config.num_epochs))
            self.loss.reset()
            # if epoch > 10 and not set:
            #     self.optimizer.param_groups[0]['lr'] /= 10
            #     print("-------------Learning rate: {}-------------".format(self.optimizer.param_groups[0]['lr']))
            #     set = True

            # train loop
            pbar = tqdm(enumerate(train_data_loader))
            n = 0
            for idx, (inputs, outputs) in pbar:
                inputs = prepare_tensors(inputs, self.config.gpu, self.config.device)
                outputs = prepare_tensors(outputs, self.config.gpu, self.config.device)
                predicted = self.network(inputs)
                loss = self.loss.cumulate(predicted, outputs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description(
                    "{:.2f} --- {}".format((idx + 1) / len(train_data_loader), self.loss.description())
                )
                n += inputs.shape[0]
            self.logger.info(f"{n} data processed.")
            self.logger.info("Epoch finished !")
            val_metrics = self.eval(self.loss.new(), *self.other_validation_metrics, datasets=self.validation_sets)
            self.logger.info("Validation loss: {}".format(val_metrics[0].description()))
            if val_metrics[1:]:
                self.logger.info("Other metrics on validation set.")
                for metric in val_metrics[1:]:
                    self.logger.info("{}".format(metric.description()))
            self.tensor_board.add_scalar("loss/training/{}".format(self.loss.document()), self.loss.log(), epoch)
            self.tensor_board.add_scalar(
                "loss/validation/loss_{}".format(val_metrics[0].document()), val_metrics[0].log(), epoch
            )
            for metric in val_metrics[1:]:
                self.tensor_board.add_scalar(
                    "other_metrics/validation/{}".format(metric.document()), metric.avg(), epoch
                )

            # eval extra validation sets
            if self.extra_validation_sets:
                for val in self.extra_validation_sets:
                    val_metrics = self.eval(
                        self.loss.new(), *self.other_validation_metrics, datasets=[val]
                    )
                    self.logger.info(
                        "Extra Validation loss on dataset {}: {}".format(val.name, val_metrics[0].description())
                    )
                    if val_metrics[1:]:
                        self.logger.info("Other metrics on extra validation set.")
                        for metric in val_metrics[1:]:
                            self.logger.info("{}".format(metric.description()))
                    self.tensor_board.add_scalar(
                        "loss/extra_validation_{}/loss_{}".format(val.name, val_metrics[0].document()),
                        val_metrics[0].log(), epoch
                    )
                    for metric in val_metrics[1:]:
                        self.tensor_board.add_scalar(
                            "other_metrics/extra_validation_{}/{}".format(val.name, metric.document()),
                            metric.avg(), epoch
                        )

            checkpoint_dir = self.config.experiment_dir.joinpath("checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            output_path = checkpoint_dir.joinpath("CP_{}.pth".format(epoch))
            torch.save(self.network.state_dict(), str(output_path))
            self.logger.info("Checkpoint {} saved at {}!".format(epoch, str(output_path)))
            if self.inference_func is not None:
                self.inference(epoch)

    def eval(self, *metrics: TorchLoss, datasets: List[TorchDataset]) -> Tuple[TorchLoss]:
        """Evaluate on validation set with training loss function if none provided"""
        val_data_loader = MultiDataLoader(
            *datasets,
            batch_size=self.config.batch_size,
            sampler_cls="sequential",
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        self.network.eval()
        if not isinstance(metrics, Iterable) and isinstance(metrics, TorchLoss):
            metrics = [metrics]
        for metric in metrics:
            metric.reset()
        for idx, (inputs, outputs) in enumerate(val_data_loader):
            inputs = prepare_tensors(inputs, self.config.gpu, self.config.device)
            preds = self.network(inputs)
            for metric in metrics:
                outputs = prepare_tensors(outputs, self.config.gpu, self.config.device)
                metric.cumulate(preds, outputs)
        return metrics

    def inference(self, epoch: int):
        output_dir = self.config.experiment_dir.joinpath("inference").joinpath("CP_{}".format(epoch))
        for val in self.validation_sets:
            indices = np.random.choice(len(val.image_paths), self.config.n_inference)
            for idx in indices:
                image_path = val.image_paths[idx]
                label_path = val.label_paths[idx]
                output_dir.joinpath(val.name, image_path.parent.stem).mkdir(exist_ok=True, parents=True)
                image = val.get_image_tensor_from_index(idx)
                label = val.get_label_tensor_from_index(idx)
                image = torch.unsqueeze(image, 0)
                image = prepare_tensors(image, self.config.gpu, self.config.device)
                self.logger.info("Inferencing for {} dataset, image {}.".format(val.name, idx))

                self.inference_func(
                    image, label, image_path, self.network, output_dir.joinpath(val.name, image_path.parent.stem)
                )
        for val in self.extra_validation_sets:
            indices = np.random.choice(len(val.image_paths), self.config.n_inference)
            for idx in indices:
                image_path = val.image_paths[idx]
                label_path = val.label_paths[idx]
                output_dir.joinpath(val.name, image_path.parent.stem).mkdir(exist_ok=True, parents=True)
                image = val.get_image_tensor_from_index(idx)
                label = val.get_label_tensor_from_index(idx)
                image = torch.unsqueeze(image, 0)
                image = prepare_tensors(image, self.config.gpu, self.config.device)
                self.logger.info("Inferencing for {} dataset, image {}.".format(val.name, idx))

                self.inference_func(
                    image, label, image_path, self.network, output_dir.joinpath(val.name, image_path.parent.stem)
                )

    @staticmethod
    def parser() -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument("-i", "--datasets", dest="dataset_names", required=True, nargs="+", type=str)
        parser.add_argument("--mount-prefix", dest="mount_prefix", type=str, required=True)
        parser.add_argument("-o", "--experiment-dir", dest="experiment_dir", type=str, default=None)
        parser.add_argument("--data-mode", dest="data_mode", type=str, default="2D")

        parser.add_argument("-v", "--validation-split", dest="validation_split", type=float, default=0.2)
        parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=10)
        parser.add_argument("-n", "--num-epochs", dest="num_epochs", type=int, default=10)
        parser.add_argument("-lr", dest="learning_rate", type=float, default=0.0001)
        parser.add_argument("-g", "--gpu", action="store_true", dest="gpu", default=False, help="use cuda")
        parser.add_argument("--device", dest="device", type=int, default=0)
        return parser
