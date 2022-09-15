import os
import imageio
from pathlib import Path
from matplotlib import pyplot as plt
from CMRSegment.common.utils import read_nii_image, read_nii_segmentation, rescale_intensity
import numpy as np


image_frames_dir = Path("P:\\sheffield\\results_contrast_enlarge\\phd001\\enlarged")
seg_phases_dir = Path("P:\\sheffield\\results_contrast_enlarge\\phd001\\motion\\seg")
output_dir = Path("P:\\sheffield\\results_contrast_enlarge\\phd001\\warp_labels")

# image_frames_dir = Path("P:\\ukbb\\results\\1035470\\enlarged")
# seg_phases_dir = Path("P:\\ukbb\\results\\1035470\\motion\\seg")
# output_dir = Path("P:\\ukbb\\results\\1035470\\warp_labels")
slice_nums = [40, 50, 60]

frames = [f"lvsa_{idx:02d}.nii.gz" for idx in range(len(os.listdir(str(image_frames_dir))))]
seg_frames = [f"lvsa_{idx:02d}.nii.gz" for idx in range(len(os.listdir(str(image_frames_dir))))]

image_frames = [image_frames_dir.joinpath(name) for name in frames]
# seg_phases = os.listdir(str(seg_phases_dir))
seg_phases = [seg_phases_dir.joinpath(name) for name in seg_frames]
output_dir.joinpath("img").mkdir(parents=True, exist_ok=True)
imgs = {}
for slice_num in slice_nums:
    imgs[slice_num] = []

for idx, seg_path, image_path in zip(range(len(seg_phases) - 1), seg_phases, image_frames[1:]):
    print(idx, image_path, seg_path)
    image_3d, __ = read_nii_image(image_path)
    seg_3d, __ = read_nii_segmentation(seg_path)
    image_3d = rescale_intensity(image_3d, (0, 255))
    image_3d_rgb = np.stack([image_3d, image_3d, image_3d], axis=3)
    seg_3d_rgb = np.zeros((seg_3d.shape[0], seg_3d.shape[1], seg_3d.shape[2], 3))
    seg_3d_rgb[seg_3d == 1] = [255, 0, 0]
    seg_3d_rgb[seg_3d == 2] = [0, 255, 0]
    seg_3d_rgb[seg_3d == 3] = [0, 0, 255]
    for slice_num in slice_nums:
        image = image_3d_rgb[:, :, slice_num, :]
        seg = seg_3d_rgb[:, :, slice_num, :]
        plt.imshow(image)
        plt.imshow(seg, alpha=0.5)
        plt.savefig(str(output_dir.joinpath("img", f"img_{slice_num}_{idx}.png")))
        plt.close()
        imgs[slice_num].append(output_dir.joinpath("img", f"img_{slice_num}_{idx}.png"))

for slice_num in slice_nums:
    with imageio.get_writer(output_dir.joinpath(f"warp_labels_slice_{slice_num}.gif"), mode="I") as writer:
        for file in imgs[slice_num]:
            image = imageio.imread_v2(str(file))
            writer.append_data(image)
