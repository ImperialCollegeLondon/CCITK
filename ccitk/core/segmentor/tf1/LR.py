import numpy as np
import nibabel as nib
import math
from CMRSegment.common.utils import rescale_intensity
from pathlib import Path
from CMRSegment.segmentor.tf1 import TF1Segmentor


class TF12DSegmentor(TF1Segmentor):
    def run(self, image: np.ndarray) -> np.ndarray:
        X, Y = image.shape
        image = np.expand_dims(image, axis=2)
        # Intensity rescaling
        image = rescale_intensity(image, (1, 99))
        # Pad the image size to be a factor of 16 so that the downsample and upsample procedures
        # in the network will result in the same image size at each resolution level.
        X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
        x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
        x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
        image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0)), "constant")
        # Transpose the shape to NXYC
        image = np.transpose(image, axes=(2, 0, 1)).astype(np.float32)
        image = np.expand_dims(image, axis=-1)
        # Evaluate the networ
        prob, pred = self._sess.run(["prob:0", "pred:0"], feed_dict={"image:0": image, "training:0": False})
        # Transpose and crop the segmentation to recover the original size
        pred = np.transpose(pred, axes=(1, 2, 0))
        pred = pred[x_pre: x_pre + X, y_pre: y_pre + Y]
        pred = np.squeeze(pred, axis=-1).astype(np.int16)
        return pred

    def execute(self, phase_path: Path, output_path: Path):
        nim = nib.load(str(phase_path))
        image = nim.get_data()
        imageOrg = np.squeeze(image, axis=-1).astype(np.int16)
        tmp = imageOrg
        X, Y, Z = image.shape[:3]
        # print('  Segmenting {0} frame ...'.format(fr))
        for slice in range(Z):
            pred = self.run(imageOrg[:, :, slice])
            tmp[:, :, slice] = pred
        pred = tmp
        pred[pred == 3] = 4
        nim2 = nib.Nifti1Image(pred, nim.affine)
        nim2.header["pixdim"] = nim.header["pixdim"]
        nib.save(nim2, str(output_path))
        return image, pred


#
# from CMRSegment.preprocessor import DataPreprocessor
# from CMRSegment.common.constants import ROOT_DIR, MODEL_DIR
#
# preprocessor = DataPreprocessor(force_restart=False)
# model_path = MODEL_DIR.joinpath("LR", "FCN_sa")
# reference_center = "genscan"
# subjects = preprocessor.run(data_dir=ROOT_DIR.joinpath("data", reference_center))
# reference_subject = subjects[0]
# # , "ukbb", "sheffield"
# for center in ["singapore_hcm", "singapore_lvsa", "sheffield", "ukbb"]:
#     subjects = preprocessor.run(data_dir=ROOT_DIR.joinpath("data", center))
#     for subject in subjects:
#         ED_ES_histogram_matching(reference_subject, subject)
#         deeplearningseg_LR(model_path, subject)
