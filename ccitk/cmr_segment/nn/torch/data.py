import os
from pathlib import Path
from ccitk.core.common.config import DatasetConfig, DataConfig, AugmentationConfig
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from typing import List, Tuple
from ccitk.core.common.data_table import DataTable
from scipy.ndimage import zoom
from ccitk.core.nn.torch.augmentation import augment, random_crop, central_crop_with_padding
from tqdm import tqdm


def construct_training_validation_dataset(
    data_config: DataConfig, crop_size: Tuple[int], voxel_size: Tuple[int], output_dir: Path, is_3d: bool = False,
    augmentation_config: AugmentationConfig = None, seed: int = None,
) -> Tuple[List["Torch2DSegmentationDataset"], List["Torch2DSegmentationDataset"], List["Torch2DSegmentationDataset"]]:
    training_set_configs = [
        DatasetConfig.from_conf(name, mode=data_config.data_mode, mount_prefix=data_config.mount_prefix)
        for name in data_config.training_datasets
    ]

    extra_val_set_configs = [
        DatasetConfig.from_conf(name, mode=data_config.data_mode, mount_prefix=data_config.mount_prefix)
        for name in data_config.extra_validation_datasets
    ]
    training_sets = []
    validation_sets = []
    for config in training_set_configs:
        train, val = train_val_dataset_from_config(
            dataset_config=config,
            augmentation_config=augmentation_config,
            augmentation_prob=data_config.augmentation_prob,
            validation_split=data_config.validation_split,
            crop_size=crop_size,
            voxel_size=voxel_size,
            is_3d=is_3d,
            renew_dataframe=data_config.renew_dataframe,
            seed=seed,
            output_dir=output_dir,
        )
        training_sets.append(train)
        if val is not None:
            validation_sets.append(val)
    extra_val_sets = []

    for config in extra_val_set_configs:
        __, val = train_val_dataset_from_config(
            dataset_config=config,
            validation_split=data_config.validation_split,
            crop_size=crop_size,
            voxel_size=voxel_size,
            is_3d=is_3d,
            only_val=True,
            renew_dataframe=data_config.renew_dataframe,
            seed=seed,
            output_dir=output_dir,
        )
        extra_val_sets.append(val)
    return training_sets, validation_sets, extra_val_sets


def train_val_dataset_from_config(dataset_config: DatasetConfig, validation_split: float, crop_size: Tuple[int],
                                  voxel_size: Tuple[int], is_3d: bool, output_dir: Path, only_val: bool = False,
                                  renew_dataframe: bool = False, seed: int = None,
                                  augmentation_config: AugmentationConfig = None, augmentation_prob: float = 0):
    if dataset_config.dataframe_path.exists():
        print("Dataframe {} exists.".format(dataset_config.dataframe_path))
    if not dataset_config.dataframe_path.exists() or renew_dataframe:
        generate_dataframe(dataset_config)
    image_paths, label_paths = read_dataframe(dataset_config.dataframe_path)
    c = list(zip(image_paths, label_paths))
    random.shuffle(c)
    shuffled_image_paths, shuffled_label_paths = zip(*c)
    print("Dataset {} has {} images.".format(dataset_config.name, len(shuffled_image_paths)))
    if dataset_config.size is None:
        size = len(shuffled_image_paths)
    else:
        size = dataset_config.size

    if not only_val:
        train_image_paths = image_paths[:int((1 - validation_split) * size)]
        val_image_paths = image_paths[int((1 - validation_split) * size):size]

        train_label_paths = label_paths[:int((1 - validation_split) * size)]
        val_label_paths = label_paths[int((1 - validation_split) * size):size]
        print("Selecting {} trainig images, {} validation images.".format(len(train_image_paths), len(val_image_paths)))
        train_set = Torch2DSegmentationDataset(
            name=dataset_config.name,
            image_paths=train_image_paths,
            label_paths=train_label_paths,
            augmentation_prob=augmentation_prob,
            augmentation_config=augmentation_config,
            voxel_size=voxel_size, crop_size=crop_size, is_3d=is_3d, seed=seed,
            output_dir=output_dir.joinpath("train"),
        )

    else:
        train_set = None
        val_image_paths = image_paths[:size]
        val_label_paths = label_paths[:size]
        print("Selecting {} validation images.".format(len(val_image_paths)))
    if val_image_paths:
        val_set = Torch2DSegmentationDataset(
            name=dataset_config.name,
            image_paths=val_image_paths,
            label_paths=val_label_paths,
            voxel_size=voxel_size, crop_size=crop_size, is_3d=is_3d, seed=seed,
            output_dir=output_dir.joinpath("val"),
        )
    else:
        val_set = None
    return train_set, val_set


class TorchDataset(Dataset):
    def __init__(self, name: str, image_paths: List[Path], label_paths: List[Path]):
        assert len(image_paths) == len(label_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.name = name

    def __len__(self):
        return len(self.image_paths)

    def sequential_loader(self, batch_size: int, n_workers: int = 0, pin_memory: bool = False) -> DataLoader:
        return DataLoader(
            self, batch_size=batch_size, sampler=SequentialSampler(self), num_workers=n_workers, pin_memory=pin_memory
        )

    def random_loader(self, batch_size: int, n_workers: int = 0, pin_memory: bool = False) -> DataLoader:
        return DataLoader(
            self, batch_size=batch_size, sampler=RandomSampler(self), num_workers=n_workers, pin_memory=pin_memory
        )

    def export(self, output_path: Path):
        """Save paths to csv and config"""
        data_table = DataTable(columns=["image_path", "label_path"], data=zip(self.image_paths, self.label_paths))
        data_table.to_csv(output_path)
        return output_path


def rescale_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2


def resize_image(image: np.ndarray, target_shape: Tuple, order: int):
    image_shape = image.shape
    factors = [float(target_shape[i]) / image_shape[i] for i in range(len(image_shape))]
    output = zoom(image, factors, order=order)
    return output


def resize_label(label: np.ndarray, target_shape: Tuple, order: int):
    # label: (3, H, W, D)
    label_shape = label.shape
    factors = [float(target_shape[i - 1]) / label_shape[i] for i in range(1, len(label_shape))]
    labels = []
    for i in range(label_shape[0]):
        output = zoom(label[i], factors, order=order)
        labels.append(output)
    label = np.array(labels)
    return label


class Torch2DSegmentationDataset(TorchDataset):
    def __init__(self, name: str, image_paths: List[Path], label_paths: List[Path],
                 crop_size: Tuple[int], voxel_size: Tuple[int], augmentation_prob: float = 0,
                 augmentation_config: AugmentationConfig = None, is_3d: bool = False, seed: int = None,
                 output_dir: Path = None):
        super().__init__(name, image_paths, label_paths)
        # crop size: (H, W, D)
        self.crop_size = crop_size
        self.voxel_size = voxel_size
        self.is_3d = is_3d
        self.augmentation_prob = augmentation_prob
        self.augmentation_config = augmentation_config
        self.seed = seed
        self.output_dir = output_dir

    @staticmethod
    def read_image(image_path: Path) -> np.ndarray:
        image = nib.load(str(image_path)).get_data()
        if image.ndim == 4:
            image = np.squeeze(image, axis=-1).astype(np.int16)
        image = image.astype(np.float32)
        X, Y, Z = image.shape
        image = np.transpose(image, (2, 0, 1))  # needed for FCN3D
        image = rescale_intensity(image, (1.0, 99.0))
        return image

    @staticmethod
    def read_label(label_path: Path) -> np.ndarray:
        label = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
        label = np.transpose(label, axes=(2, 1, 0))
        if label.ndim == 4:
            label = np.squeeze(label, axis=-1).astype(np.int16)
        label = label.astype(np.float32)
        label[label == 4] = 3

        X, Y, Z = label.shape

        labels = []
        for i in range(1, 4):
            blank_image = np.zeros((X, Y, Z))

            blank_image[label == i] = 1
            labels.append(blank_image)
        label = np.array(labels)
        label = np.transpose(label, (0, 3, 1, 2))  # needed for FCN3D
        return label

    def get_image_tensor_from_index(self, index: int) -> torch.Tensor:
        image = self.read_image(self.image_paths[index])
        if self.is_3d:
            image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).float()
        return image

    def get_label_tensor_from_index(self, index: int):
        label = self.read_label(self.label_paths[index])
        label = torch.from_numpy(label).float()
        return label

    def __getitem__(self, index: int):

        label = self.read_label(self.label_paths[index])
        image = self.read_image(self.image_paths[index])
        original_image_shape = image.shape
        # if self.augmentation_prob > 0 and self.augmentation_config is not None:
        #     image, label = augment(
        #         image, label, self.augmentation_config,
        #         self.crop_size,
        #         seed=self.seed
        #     )
        # else:
        #     image, label = random_crop(image, label, self.crop_size)
        image, label = central_crop_with_padding(image, label, self.crop_size)

        image = resize_image(image, self.voxel_size, 0)
        label = resize_label(label, self.voxel_size, 0)

        self.save(image, label, index)

        if self.is_3d:
            image = np.expand_dims(image, 0)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        return image, label

    def test(self, index):
        label = self.read_label(self.label_paths[index])
        image = self.read_image(self.image_paths[index])
        image, label = augment(
            image, label, self.augmentation_config,
            self.crop_size,
            seed=self.seed
        )
        image = resize_image(image, self.voxel_size, 0)
        label = resize_label(label, self.voxel_size, 0)
        return image, label

    def test_save(self, index, image, label, augmented_image, augmented_label):
        nim = nib.load(str(self.image_paths[index]))
        image = np.transpose(image, [1, 2, 0])
        nim2 = nib.Nifti1Image(image, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        output_path = self.output_dir.joinpath(self.name, "image_{}.nii.gz".format(index))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nim2, str(output_path))

        augmented_image = np.transpose(augmented_image, [1, 2, 0])
        nim2 = nib.Nifti1Image(augmented_image, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        output_path = self.output_dir.joinpath(self.name, "image_augmented_{}.nii.gz".format(index))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nim2, str(output_path))

        final_label = np.zeros((label.shape[1], label.shape[2], label.shape[3]))
        for i in range(label.shape[0]):
            final_label[label[i, :, :, :] == 1.0] = i + 1
        final_label = np.transpose(final_label, [1, 2, 0])
        nim2 = nib.Nifti1Image(final_label, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        output_path = self.output_dir.joinpath(self.name, "label_{}.nii.gz".format(index))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nim2, str(output_path))

        final_label = np.zeros((augmented_label.shape[1], augmented_label.shape[2], augmented_label.shape[3]))
        for i in range(augmented_label.shape[0]):
            final_label[augmented_label[i, :, :, :] == 1.0] = i + 1
        final_label = np.transpose(final_label, [1, 2, 0])
        nim2 = nib.Nifti1Image(final_label, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        output_path = self.output_dir.joinpath(self.name, "label_augmented_{}.nii.gz".format(index))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nim2, str(output_path))

    def save(self, image: np.ndarray, label: np.ndarray, index: int):
        if index % 100 == 0:
            nim = nib.load(str(self.image_paths[index]))
            image = np.transpose(image, [1, 2, 0])
            nim2 = nib.Nifti1Image(image, nim.affine)
            nim2.header['pixdim'] = nim.header['pixdim']
            output_path = self.output_dir.joinpath(self.name, "image_{}.nii.gz".format(index))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nim2, str(output_path))

            final_label = np.zeros((label.shape[1], label.shape[2], label.shape[3]))
            for i in range(label.shape[0]):
                final_label[label[i, :, :, :] == 1.0] = i + 1
            final_label = np.transpose(final_label, [1, 2, 0])
            nim2 = nib.Nifti1Image(final_label, nim.affine)
            nim2.header['pixdim'] = nim.header['pixdim']
            output_path = self.output_dir.joinpath(self.name, "label_{}.nii.gz".format(index))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nim2, str(output_path))

    @staticmethod
    def crop_image(image, cx, cy, size):
        """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
        X, Y = image.shape[:2]
        r = int(size / 2)
        x1, x2 = cx - r, cx + r
        y1, y2 = cy - r, cy + r
        x1_, x2_ = max(x1, 0), min(x2, X)
        y1_, y2_ = max(y1, 0), min(y2, Y)
        # Crop the image
        crop = image[x1_: x2_, y1_: y2_]
        # Pad the image if the specified size is larger than the input image size
        if crop.ndim == 3:
            crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)), 'constant')
        elif crop.ndim == 4:
            crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)), 'constant')
        elif crop.ndim == 2:
            crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_)), 'constant')
        else:
            print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
            exit(0)
        return crop

    @staticmethod
    def crop_3D_image(image, cx, cy, size_xy, cz, size_z):
        """ Crop a 3D image using a bounding box centred at (cx, cy, cz) with specified size """
        X, Y, Z = image.shape[:3]
        rxy = int(size_xy / 2)
        r_z = int(size_z / 2)
        x1, x2 = cx - rxy, cx + rxy
        y1, y2 = cy - rxy, cy + rxy
        z1, z2 = cz - r_z, cz + r_z
        x1_, x2_ = max(x1, 0), min(x2, X)
        y1_, y2_ = max(y1, 0), min(y2, Y)
        z1_, z2_ = max(z1, 0), min(z2, Z)
        # Crop the image
        crop = image[x1_: x2_, y1_: y2_, z1_: z2_]
        # Pad the image if the specified size is larger than the input image size
        if crop.ndim == 3:
            crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (z1_ - z1, z2 - z2_)), 'constant')
        elif crop.ndim == 4:
            crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (z1_ - z1, z2 - z2_), (0, 0)), 'constant')
        else:
            print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
            exit(0)
        return crop


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class MultiDataLoader:
    def __init__(self, *datasets: Dataset, batch_size: int, sampler_cls: str, num_workers: int = 0,
                 pin_memory: bool = False):
        if sampler_cls == "random":
            sampler_cls = RandomSampler
        else:
            sampler_cls = SequentialSampler
        self.loaders = [
            DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler_cls(dataset),
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=worker_init_fn
            ) for dataset in datasets
        ]
        self.loader_iters = []

    def __iter__(self):
        iters = []
        for loader in self.loaders:
            iters.append(iter(loader))
        self.loader_iters = iters
        return self

    def __next__(self):
        all_inputs = []
        all_outputs = []
        all_stops = []
        for loader_iter in self.loader_iters:
            try:
                inputs, outputs = next(loader_iter)
            except StopIteration:
                all_stops.append(True)
            else:
                all_stops.append(False)
                all_inputs.append(inputs)
                all_outputs.append(outputs)
        if all(all_stops):
            raise StopIteration

        if isinstance(all_inputs[0], tuple) or isinstance(all_inputs[0], list):
            all_inputs = zip(*all_inputs)
            all_inputs = (torch.cat(inputs) for inputs in all_inputs)
        else:
            all_inputs = torch.cat(all_inputs, dim=0)

        if isinstance(all_outputs[0], tuple) or isinstance(all_outputs[0], list):
            all_outputs = zip(*all_outputs)
            all_outputs = (torch.cat(outputs) for outputs in all_outputs)
        else:
            all_outputs = torch.cat(all_outputs, dim=0)
        return all_inputs, all_outputs

    def __len__(self):
        lengths = [len(loader) for loader in self.loaders]
        return max(lengths)


def generate_dataframe(dataset_config: DatasetConfig):
    image_paths = []
    label_paths = []
    paths = sorted(os.listdir(str(dataset_config.dir)))
    print("{} paths found. ".format(len(paths)))
    wrong_image_paths = []
    wrong_label_paths = []
    wrong_image_shapes = []
    wrong_label_shapes = []
    for path in tqdm(paths):
        # for phase in ["ED", "ES"]:
        for phase in ["ED"]:
            path = dataset_config.dir.joinpath(path)
            image_path = path.joinpath(dataset_config.image_label_format.image.format(phase=phase))
            label_path = path.joinpath(dataset_config.image_label_format.label.format(phase=phase))
            if image_path.exists() and label_path.exists():
                image = nib.load(str(image_path)).get_data()
                image = np.squeeze(image, axis=-1).astype(np.int16)
                label = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
                label = np.transpose(label, axes=(2, 1, 0))
                if image.shape == label.shape:
                    image_paths.append(path.joinpath(dataset_config.image_label_format.image.format(phase=phase)))
                    label_paths.append(path.joinpath(dataset_config.image_label_format.label.format(phase=phase)))
                else:
                    print(image_path)
                    wrong_image_paths.append(path.joinpath(dataset_config.image_label_format.image.format(phase=phase)))
                    wrong_label_paths.append(path.joinpath(dataset_config.image_label_format.label.format(phase=phase)))
                    wrong_image_shapes.append(image.shape)
                    wrong_label_shapes.append(label.shape)

        data_table = DataTable(columns=["image_path", "label_path"], data=zip(image_paths, label_paths))
        data_table.to_csv(dataset_config.dataframe_path)

        data_table = DataTable(columns=["image_path", "label_path", "image_shape", "label_shape"],
                               data=zip(wrong_image_paths, wrong_label_paths, wrong_image_shapes, wrong_label_shapes))
        data_table.to_csv(dataset_config.dataframe_path.parent.joinpath("misshape.csv"))
    return dataset_config.dataframe_path


def read_dataframe(dataframe_path: Path):
    data_table = DataTable.from_csv(dataframe_path)
    image_paths = data_table.select_column("image_path")
    label_paths = data_table.select_column("label_path")
    image_paths = [Path(path) for path in image_paths]
    label_paths = [Path(path) for path in label_paths]
    return image_paths, label_paths
