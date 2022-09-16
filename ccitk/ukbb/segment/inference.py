# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import time
import math
import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
import tensorflow.compat.v1 as tf
from ccitk.image import rescale_intensity
from ccitk.image import normalise_intensity

tf.disable_v2_behavior()


def segment_sa_la(data_dir: Path, seq_name: str, model_path: Path, seg4: bool = False, process_seq: bool = True,
                  save_seg: bool = True, output_dir: Path = None):
    """ Deployment parameters """

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Import the computation graph and restore the variable values
        saver = tf.train.import_meta_graph('{0}.meta'.format(str(model_path)))
        saver.restore(sess, '{0}'.format(str(model_path)))

        print('Start deployment on the data set ...')
        start_time = time.time()

        # Process each subject subdirectory
        data_list = sorted(os.listdir(str(data_dir)))
        processed_list = []
        table_time = []
        for data in tqdm(data_list):
            print(data)
            subject_dir = data_dir.joinpath(data)
            if output_dir is None:
                output_subject_dir = subject_dir
            else:
                output_subject_dir = output_dir.joinpath(data)
                output_subject_dir.mkdir(parents=True, exist_ok=True)

            if seq_name == 'la_4ch' and seg4:
                seg_name = output_subject_dir.joinpath(f"seg4_{seq_name}.nii.gz")
            else:
                seg_name = output_subject_dir.joinpath(f"seg_{seq_name}.nii.gz")
            if seg_name.exists():
                continue

            if process_seq:
                # Process the temporal sequence
                image_name = subject_dir.joinpath(f"{seq_name}.nii.gz")

                if not image_name.exists():
                    print('  Directory {0} does not contain an image with file '
                          'name {1}. Skip.'.format(subject_dir, image_name.name))
                    continue

                # Read the image
                print('  Reading {} ...'.format(image_name))
                nim = nib.load(str(image_name))
                image = nim.get_data()
                X, Y, Z, T = image.shape
                orig_image = image

                print('  Segmenting full sequence ...')
                start_seg_time = time.time()

                # Intensity rescaling
                image = rescale_intensity(image, (1, 99))

                # Prediction (segmentation)
                pred = np.zeros(image.shape)

                # Pad the image size to be a factor of 16 so that the
                # downsample and upsample procedures in the network will
                # result in the same image size at each resolution level.
                X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0), (0, 0)), 'constant')

                # Process each time frame
                for t in range(T):
                    # Transpose the shape to NXYC
                    image_fr = image[:, :, :, t]
                    image_fr = np.transpose(image_fr, axes=(2, 0, 1)).astype(np.float32)
                    image_fr = np.expand_dims(image_fr, axis=-1)

                    # Evaluate the network
                    prob_fr, pred_fr = sess.run(['prob:0', 'pred:0'],
                                                feed_dict={'image:0': image_fr, 'training:0': False})

                    # Transpose and crop segmentation to recover the original size
                    pred_fr = np.transpose(pred_fr, axes=(1, 2, 0))
                    pred_fr = pred_fr[x_pre:x_pre + X, y_pre:y_pre + Y]
                    pred[:, :, :, t] = pred_fr

                seg_time = time.time() - start_seg_time
                print('  Segmentation time = {:3f}s'.format(seg_time))
                table_time += [seg_time]
                processed_list += [data]

                # ED frame defaults to be the first time frame.
                # Determine ES frame according to the minimum LV volume.
                k = {}
                k['ED'] = 0
                if seq_name == 'sa' or (seq_name == 'la_4ch' and seg4):
                    k['ES'] = np.argmin(np.sum(pred == 1, axis=(0, 1, 2)))
                else:
                    k['ES'] = np.argmax(np.sum(pred == 1, axis=(0, 1, 2)))
                print('  ED frame = {:d}, ES frame = {:d}'.format(k['ED'], k['ES']))

                # Save the segmentation
                if save_seg:
                    print('  Saving segmentation ...')
                    nim2 = nib.Nifti1Image(pred, nim.affine)
                    nim2.header['pixdim'] = nim.header['pixdim']
                    if seq_name == 'la_4ch' and seg4:
                        seg_name = output_subject_dir.joinpath(f"seg4_{seq_name}.nii.gz")
                    else:
                        seg_name = output_subject_dir.joinpath(f"seg_{seq_name}.nii.gz")

                    nib.save(nim2, str(seg_name))

                    for fr in ['ED', 'ES']:
                        nib.save(nib.Nifti1Image(orig_image[:, :, :, k[fr]], nim.affine),
                                 str(output_subject_dir.joinpath(f"{seq_name}_{fr}.nii.gz")))
                        if seq_name == 'la_4ch' and seg4:
                            seg_name = output_subject_dir.joinpath(f"seg4_{seq_name}_{fr}.nii.gz")
                        else:
                            seg_name = output_subject_dir.joinpath(f"seg_{seq_name}_{fr}.nii.gz")

                        nib.save(nib.Nifti1Image(pred[:, :, :, k[fr]], nim.affine), str(seg_name))
            else:
                # Process ED and ES time frames
                image_ED_name = subject_dir.joinpath(f"{seq_name}_ED.nii.gz")
                image_ES_name = subject_dir.joinpath(f"{seq_name}_ES.nii.gz")
                if not image_ED_name.exists() or not image_ES_name.exists():
                    print('  Directory {0} does not contain an image with '
                          'file name {1} or {2}. Skip.'.format(subject_dir, image_ED_name.name, image_ES_name.name))
                    continue

                measure = {}
                for fr in ['ED', 'ES']:
                    image_name = subject_dir.joinpath(f"{seq_name}_{fr}.nii.gz")

                    # Read the image
                    print('  Reading {} ...'.format(image_name))
                    nim = nib.load(str(image_name))
                    image = nim.get_data()
                    X, Y = image.shape[:2]
                    if image.ndim == 2:
                        image = np.expand_dims(image, axis=2)

                    print('  Segmenting {} frame ...'.format(fr))
                    start_seg_time = time.time()

                    # Intensity rescaling
                    image = rescale_intensity(image, (1, 99))

                    # Pad the image size to be a factor of 16 so that
                    # the downsample and upsample procedures in the network
                    # will result in the same image size at each resolution
                    # level.
                    X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                    x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                    x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                    image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0)), 'constant')

                    # Transpose the shape to NXYC
                    image = np.transpose(image, axes=(2, 0, 1)).astype(np.float32)
                    image = np.expand_dims(image, axis=-1)

                    # Evaluate the network
                    prob, pred = sess.run(['prob:0', 'pred:0'],
                                          feed_dict={'image:0': image, 'training:0': False})

                    # Transpose and crop the segmentation to recover the original size
                    pred = np.transpose(pred, axes=(1, 2, 0))
                    pred = pred[x_pre:x_pre + X, y_pre:y_pre + Y]

                    seg_time = time.time() - start_seg_time
                    print('  Segmentation time = {:3f}s'.format(seg_time))
                    table_time += [seg_time]
                    processed_list += [data]

                    # Save the segmentation
                    if save_seg:
                        print('  Saving segmentation ...')
                        nim2 = nib.Nifti1Image(pred, nim.affine)
                        nim2.header['pixdim'] = nim.header['pixdim']
                        if seq_name == 'la_4ch' and seg4:
                            seg_name = output_subject_dir.joinpath(f"seg4_{seq_name}_{fr}.nii.gz")
                        else:
                            seg_name = output_subject_dir.joinpath(f"seg_{seq_name}_{fr}.nii.gz")

                        nib.save(nim2, str(seg_name))

        if process_seq:
            print('Average segmentation time = {:.3f}s per sequence'.format(np.mean(table_time)))
        else:
            print('Average segmentation time = {:.3f}s per frame'.format(np.mean(table_time)))
        process_time = time.time() - start_time
        print('Including image I/O, CUDA resource allocation, '
              'it took {:.3f}s for processing {:d} subjects ({:.3f}s per subjects).'.format(
            process_time, len(processed_list), process_time / len(processed_list)))


def segment_ao(data_dir: str, model: str, model_path: str, seq_name: str, time_step: int = 1,
               process_seq: bool = True, save_seg: bool = True, z_score: bool = False,
               weight_R: int = 5, weight_r: float = 0.1):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Import the computation graph and restore the variable values
        saver = tf.train.import_meta_graph('{0}.meta'.format(model_path))
        saver.restore(sess, '{0}'.format(model_path))

        print('Start evaluating on the test set ...')
        start_time = time.time()

        # Process each subject subdirectory
        data_list = sorted(os.listdir(data_dir))
        processed_list = []
        table = []
        for data in tqdm(data_list):
            print(data)
            subject_dir = os.path.join(data_dir, data)

            if process_seq:
                # Process the temporal sequence
                image_name = '{0}/{1}.nii.gz'.format(subject_dir, seq_name)

                if not os.path.exists(image_name):
                    print('  Directory {0} does not contain an image with file name {1}. '
                          'Skip.'.format(subject_dir, os.path.basename(image_name)))
                    continue

                # Read the image
                print('  Reading {} ...'.format(image_name))
                nim = nib.load(image_name)
                dx, dy, dz, dt = nim.header['pixdim'][1:5]
                area_per_pixel = dx * dy
                image = nim.get_data()
                X, Y, Z, T = image.shape
                orig_image = image

                print('  Segmenting full sequence ...')
                start_seg_time = time.time()

                # Intensity normalisation
                if z_score:
                    image = normalise_intensity(image, 10.0)
                else:
                    image = rescale_intensity(image, (1.0, 99.0))

                # Probability (segmentation)
                n_class = 3
                prob = np.zeros((X, Y, Z, T, n_class), dtype=np.float32)

                # Pad the image size to be a factor of 16 so that the downsample and upsample procedures
                # in the network will result in the same image size at each resolution level.
                # X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                X2, Y2 = 256, 256
                x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0), (0, 0)), 'constant')

                # Process each time frame
                if model == 'UNet':
                    # For each time frame
                    for t in range(T):
                        # Transpose the shape to NXYC
                        image_fr = image[:, :, :, t]
                        image_fr = np.transpose(image_fr, axes=(2, 0, 1)).astype(np.float32)
                        image_fr = np.expand_dims(image_fr, axis=-1)

                        # Evaluate the network
                        # prob_fr: NXYC
                        prob_fr = sess.run('prob:0',
                                           feed_dict={'image:0': image_fr, 'training:0': False})

                        # Transpose and crop to recover the original size
                        # prob_fr: XYNC
                        prob_fr = np.transpose(prob_fr, axes=(1, 2, 0, 3))
                        prob_fr = prob_fr[x_pre:x_pre + X, y_pre:y_pre + Y]
                        prob[:, :, :, t, :] = prob_fr
                elif model == 'UNet-LSTM' or model == 'Temporal-UNet':
                    time_window = weight_R * 2 - 1
                    rad = int((time_window - 1) / 2)
                    weight = np.zeros((1, 1, 1, T, 1))

                    w = []
                    for t in range(time_window):
                        d = abs(t - rad)
                        if d <= weight_R:
                            w_t = pow(1 - float(d) / weight_R, weight_r)
                        else:
                            w_t = 0
                        w += [w_t]

                    w = np.array(w)
                    w = np.reshape(w, (1, 1, 1, time_window, 1))

                    # For each time frame after a time_step
                    for t in range(0, T, time_step):
                        # Get the frames in the time window
                        t1 = t - rad
                        t2 = t + rad
                        idx = []
                        for i in range(t1, t2 + 1):
                            if i < 0:
                                idx += [i + T]
                            elif i >= T:
                                idx += [i - T]
                            else:
                                idx += [i]

                        # image_idx: NTXYC
                        image_idx = image[:, :, :, idx]
                        image_idx = np.transpose(image_idx, axes=(2, 3, 0, 1)).astype(np.float32)
                        image_idx = np.expand_dims(image_idx, axis=-1)

                        # Evaluate the network
                        # Curious: can we deploy the LSTM model more efficiently by utilising the state variable?
                        # Currently, we have to feed all the time frames in the time window and we can not just
                        # feed one time frame, because the LSTM is an unrolled model in the dataflow graph.
                        # It needs all the input from the time window.
                        # prob_idx: NTXYC
                        prob_idx = sess.run('prob:0',
                                            feed_dict={'image:0': image_idx, 'training:0': False})

                        # Transpose and crop the segmentation to recover the original size
                        # prob_idx: XYNTC
                        prob_idx = np.transpose(prob_idx, axes=(2, 3, 0, 1, 4))

                        # Tile the overlapping probability maps
                        prob[:, :, :, idx] += prob_idx[x_pre:x_pre + X, y_pre:y_pre + Y] * w
                        weight[:, :, :, idx] += w

                    # Average probability
                    prob /= weight
                else:
                    print('Error: unknown model {0}.'.format(model))
                    exit(0)

                # Segmentation
                pred = np.argmax(prob, axis=-1).astype(np.int32)

                # Save the segmentation
                if save_seg:
                    print('  Saving segmentation ...')
                    nim2 = nib.Nifti1Image(pred, nim.affine)
                    nim2.header['pixdim'] = nim.header['pixdim']
                    nib.save(nim2, '{0}/seg_{1}.nii.gz'.format(subject_dir, seq_name))

                seg_time = time.time() - start_seg_time
                print('  Segmentation time = {:3f}s'.format(seg_time))
                processed_list += [data]
            else:
                if model == 'UNet-LSTM':
                    print('UNet-LSTM does not support frame-wise segmentation. '
                          'Please use the -process_seq flag.')
                    exit(0)

                # Process ED and ES time frames
                image_ED_name = '{0}/{1}_{2}.nii.gz'.format(subject_dir, seq_name, 'ED')
                image_ES_name = '{0}/{1}_{2}.nii.gz'.format(subject_dir, seq_name, 'ES')
                if not os.path.exists(image_ED_name) or not os.path.exists(image_ES_name):
                    print('  Directory {0} does not contain an image with file name {1} or {2}. '
                          'Skip.'.format(subject_dir, os.path.basename(image_ED_name), os.path.basename(image_ES_name)))
                    continue

                measure = {}
                for fr in ['ED', 'ES']:
                    image_name = '{0}/{1}_{2}.nii.gz'.format(subject_dir, seq_name, fr)

                    # Read the image
                    # image: XYZ
                    print('  Reading {} ...'.format(image_name))
                    nim = nib.load(image_name)
                    dx, dy, dz, dt = nim.header['pixdim'][1:5]
                    area_per_pixel = dx * dy
                    image = nim.get_data()
                    X, Y = image.shape[:2]

                    print('  Segmenting {} frame ...'.format(fr))
                    start_seg_time = time.time()

                    # Intensity normalisation
                    if z_score:
                        image = normalise_intensity(image, 10.0)
                    else:
                        image = rescale_intensity(image, (1.0, 99.0))

                    # Pad the image size to be a factor of 16 so that the downsample and upsample procedures
                    # in the network will result in the same image size at each resolution level.
                    X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                    x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                    x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                    image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0)), 'constant')

                    # Transpose the shape to NXYC
                    # image: NXY
                    image = np.transpose(image, axes=(2, 0, 1)).astype(np.float32)
                    # image: NXYC
                    image = np.expand_dims(image, axis=-1)

                    # Evaluate the network
                    # pred: NXY
                    prob, pred = sess.run(['prob:0', 'pred:0'],
                                          feed_dict={'image:0': image, 'training:0': False})

                    # Transpose and crop the segmentation to recover the original size
                    pred = np.transpose(pred, axes=(1, 2, 0))
                    pred = pred[x_pre:x_pre + X, y_pre:y_pre + Y]

                    seg_time = time.time() - start_seg_time
                    print('  Segmentation time = {:3f}s'.format(seg_time))

                    # Save the segmentation
                    if save_seg:
                        print('  Saving segmentation ...')
                        nim2 = nib.Nifti1Image(pred, nim.affine)
                        nim2.header['pixdim'] = nim.header['pixdim']
                        nib.save(nim2, '{0}/seg_{1}_{2}.nii.gz'.format(subject_dir, seq_name, fr))

                processed_list += [data]

        process_time = time.time() - start_time
        print('Including image I/O, CUDA resource allocation, '
              'it took {:.3f}s for processing {:d} subjects ({:.3f}s per subjects).'.format(
            process_time, len(processed_list), process_time / len(processed_list)))
