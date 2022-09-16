import math
import torch
import numpy as np
import nibabel as nib
from typing import Tuple
from pathlib import Path
from ccitk.cmr_segment.segmentor import Segmentor
from ccitk.cmr_segment.segmentor.utils import refined_mask
from ccitk.cmr_segment.nn.torch.data import rescale_intensity
from ccitk.cmr_segment.nn.torch import prepare_tensors


class TorchSegmentor(Segmentor):
    def __init__(self, model_path: Path, overwrite: bool = False, resize_size: Tuple = None, device: int = 0):
        super().__init__(model_path, overwrite)
        self.model = torch.load(str(model_path)).cuda(device)
        self.model.eval()
        if resize_size is None:
            resize_size = (128, 128, 64)
        self.resize_size = resize_size
        self.device = device

    def run(self, image: np.ndarray) -> np.ndarray:
        image = torch.from_numpy(image).float()
        image = torch.unsqueeze(image, 0)
        image = prepare_tensors(image, True, self.device)
        predicted = self.model(image)
        predicted = torch.sigmoid(predicted)
        # print("sigmoid", torch.mean(predicted).item(), torch.max(predicted).item())
        predicted = (predicted > 0.5).float()
        # print("0.5", torch.mean(predicted).item(), torch.max(predicted).item())
        predicted = predicted.cpu().detach().numpy()
        predicted = np.squeeze(predicted, axis=0)
        # map back to original size
        final_predicted = np.zeros((image.shape[2], image.shape[3], image.shape[4]))
        # print(predicted.shape, final_predicted.shape)

        for i in range(predicted.shape[0]):
            final_predicted[predicted[i, :, :, :] > 0.5] = i + 1
        # image = nim.get_data()
        final_predicted = np.transpose(final_predicted, [1, 2, 0])
        return final_predicted

    def execute(self, phase_path: Path, output_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        nim = nib.load(str(phase_path))
        image = nim.get_data()
        if image.ndim == 4:
            image = np.squeeze(image, axis=-1).astype(np.int16)
        image = image.astype(np.float32)
        # resized_image = resize_image(image, (self.resize_size[0], self.resize_size[1], self.resize_size[2]), 0)
        # Crop image to 128, 128, 64
        X, Y, Z = image.shape
        n_slices = 96
        X2, Y2 = int(math.ceil(X / 32.0)) * 32, int(math.ceil(Y / 32.0)) * 32
        x_pre, y_pre, z_pre = int((X2 - X) / 2), int((Y2 - Y) / 2), int((Z - n_slices) / 2)
        x_post, y_post, z_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre, (Z - n_slices) - z_pre
        z1, z2 = int(Z / 2) - int(n_slices / 2), int(Z / 2) + int(n_slices / 2)
        z1_, z2_ = max(z1, 0), min(z2, Z)
        image = image[:, :, z1_: z2_]
        image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (z1_ - z1, z2 - z2_)), 'constant')
        # print("Padded image shape: {}".format(image.shape))
        image = np.transpose(image, (2, 0, 1))
        image = rescale_intensity(image, (1.0, 99.0))
        image = np.expand_dims(image, 0)

        # resized_image = np.transpose(resized_image, (2, 0, 1))
        # resized_image = rescale_intensity(resized_image, (1.0, 99.0))
        # resized_image = np.expand_dims(resized_image, 0)
        predicted = self.run(image)
        # predicted = resize_image(predicted, image.shape, 0)
        # print("Predicted shape: {}".format(predicted.shape))
        predicted = predicted[x_pre:x_pre + X, y_pre:y_pre + Y, z1_ - z1:z1_ - z1 + Z]
        if int(Z / 2) - int(n_slices / 2) > 0:
            if Z % 2 == 1:
                d = 1
            else:
                d = 0
            predicted = np.pad(
                predicted,
                ((0, 0), (0, 0), (int(Z / 2) - int(n_slices / 2), int(Z / 2) - int(n_slices / 2) + d)),
                'constant'
            )
        predicted = refined_mask(predicted, phase_path, output_path.parent.joinpath("tmp"))
        # print("Predicted shape after cropping: {}".format(predicted.shape))
        nim2 = nib.Nifti1Image(predicted, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        nib.save(nim2, str(output_path))
        image = np.squeeze(image, 0)
        image = image[x_pre:x_pre + X, y_pre:y_pre + Y, z1_ - z1:z1_ - z1 + Z]
        return image, predicted
