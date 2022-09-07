import numpy as np
import nibabel as nib
import scipy.ndimage.measurements as measure


def get_largest_cc(binary):
    """ Get the largest connected component in the foreground. """
    cc, n_cc = measure.label(binary)
    max_n = -1
    max_area = 0
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area > max_area:
            max_area = area
            max_n = n
    largest_cc = (cc == max_n)
    return largest_cc


def remove_small_cc(binary, thres=10):
    """ Remove small connected component in the foreground. """
    cc, n_cc = measure.label(binary)
    binary2 = np.copy(binary)
    for n in range(1, n_cc + 1):
        area = np.sum(cc == n)
        if area < thres:
            binary2[cc == n] = 0
    return binary2


def padding(input_A_name, input_B_name, output_name, value_in_B, value_output):
    nim = nib.load(input_A_name)
    image_A = nim.get_data()
    image_B = nib.load(input_B_name).get_data()
    image_A[image_B == value_in_B] = value_output
    nim2 = nib.Nifti1Image(image_A, nim.affine)
    nib.save(nim2, output_name)


def auto_crop_image(input_name, output_name, reserve):
    nim = nib.load(input_name)
    image = nim.get_data()
    X, Y, Z = image.shape[:3]

    # Detect the bounding box of the foreground
    idx = np.nonzero(image > 0)
    x1, x2 = idx[0].min() - reserve, idx[0].max() + reserve + 1
    y1, y2 = idx[1].min() - reserve, idx[1].max() + reserve + 1
    z1, z2 = idx[2].min() - reserve, idx[2].max() + reserve + 1
    x1, x2 = max(x1, 0), min(x2, X)
    y1, y2 = max(y1, 0), min(y2, Y)
    z1, z2 = max(z1, 0), min(z2, Z)
    print('Bounding box')
    print('  bottom-left corner = ({},{},{})'.format(x1, y1, z1))
    print('  top-right corner = ({},{},{})'.format(x2, y2, z2))

    # Crop the image
    image = image[x1:x2, y1:y2, z1:z2]

    # Update the affine matrix
    affine = nim.affine
    affine[:3, 3] = np.dot(affine, np.array([x1, y1, z1, 1]))[:3]
    nim2 = nib.Nifti1Image(image, affine)
    nib.save(nim2, output_name)


def split_volume(image_name, output_name):
    """ Split an image volume into a number of slices. """
    nim = nib.load(image_name)
    Z = nim.header['dim'][3]
    affine = nim.affine
    image = nim.get_data()

    for z in range(Z):
        image_slice = image[:, :, z]
        image_slice = np.expand_dims(image_slice, axis=2)
        affine2 = np.copy(affine)
        affine2[:3, 3] += z * affine2[:3, 2]
        nim2 = nib.Nifti1Image(image_slice, affine2)
        nib.save(nim2, '{0}{1:02d}.nii.gz'.format(output_name, z))


def split_sequence(image_name, output_name):
    """ Split an image sequence into a number of time frames. """
    nim = nib.load(image_name)
    T = nim.header['dim'][4]
    affine = nim.affine
    image = nim.get_data()

    for t in range(T):
        image_fr = image[:, :, :, t]
        nim2 = nib.Nifti1Image(image_fr, affine)
        nib.save(nim2, '{0}{1:02d}.nii.gz'.format(output_name, t))


def make_sequence(image_names, dt, output_name):
    """ Combine a number of time frames into one image sequence. """
    nim = nib.load(image_names[0])
    affine = nim.affine
    X, Y, Z = nim.header['dim'][1:4]
    T = len(image_names)
    image = np.zeros((X, Y, Z, T))

    for t in range(T):
        image[:, :, :, t] = nib.load(image_names[t]).get_data()

    nim2 = nib.Nifti1Image(image, affine)
    nim2.header['pixdim'][4] = dt
    nib.save(nim2, output_name)


def np_categorical_dice(pred, truth, k):
    """ Dice overlap metric for label k """
    A = (pred == k).astype(np.float32)
    B = (truth == k).astype(np.float32)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))