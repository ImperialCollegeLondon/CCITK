__all__ = [
    "visualize_warped_labels_motion",
]
import os
import imageio
from pathlib import Path
from matplotlib import pyplot as plt
from ccitk.image import read_nii_image, read_nii_label, rescale_intensity
import numpy as np
from typing import List


def visualize_warped_labels_motion(image_frames_dir: Path, seg_phases_dir: Path, output_dir: Path, slice_numbers: List[int]):

    frames = [file for file in os.listdir(str(image_frames_dir)) if file.endswith(".nii.gz")]
    seg_frames = [file for file in os.listdir(str(seg_phases_dir)) if file.endswith(".nii.gz")]
    assert len(frames) == len(seg_frames)
    image_frames = [image_frames_dir.joinpath(name) for name in frames]
    seg_phases = [seg_phases_dir.joinpath(name) for name in seg_frames]
    output_dir.joinpath("img").mkdir(parents=True, exist_ok=True)
    imgs = {}
    for slice_num in slice_numbers:
        imgs[slice_num] = []

    for idx, seg_path, image_path in zip(range(len(seg_phases) - 1), seg_phases, image_frames[1:]):
        image_3d, __ = read_nii_image(image_path)
        seg_3d, __ = read_nii_label(seg_path, labels=[1, 2, 3])
        image_3d = rescale_intensity(image_3d)
        image_3d_rgb = np.stack([image_3d, image_3d, image_3d], axis=3)
        seg_3d_rgb = np.zeros((seg_3d.shape[1], seg_3d.shape[2], seg_3d.shape[3], 3))
        print(seg_3d.shape)
        seg_3d_rgb[seg_3d[0] == 1] = [255, 0, 0]
        seg_3d_rgb[seg_3d[1] == 1] = [0, 255, 0]
        seg_3d_rgb[seg_3d[2] == 1] = [0, 0, 255]
        print(image_3d_rgb.shape, image_3d.shape)
        for slice_num in slice_numbers:
            image = image_3d_rgb[:, :, slice_num, :]
            seg = seg_3d_rgb[:, :, slice_num, :]
            plt.imshow(image)
            plt.imshow(seg, alpha=0.5)
            plt.savefig(str(output_dir.joinpath("img", f"img_{slice_num}_{idx}.png")))
            plt.close()
            imgs[slice_num].append(output_dir.joinpath("img", f"img_{slice_num}_{idx}.png"))

    for slice_num in slice_numbers:
        with imageio.get_writer(output_dir.joinpath(f"warp_labels_slice_{slice_num}.gif"), mode="I") as writer:
            for file in imgs[slice_num]:
                image = imageio.imread_v2(str(file))
                writer.append_data(image)
    return output_dir
