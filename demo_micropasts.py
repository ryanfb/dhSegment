#!/usr/bin/env python

import tensorflow as tf
from dh_segment.loader import LoadedModel
from dh_segment.post_processing import boxes_detection, binarization, PAGE
from skimage import img_as_ubyte
from tqdm import tqdm
from glob import glob
import numpy as np
import os
import cv2
from imageio import imread, imsave

def page_make_binary_mask(probs: np.ndarray, threshold: float=-1) -> np.ndarray:
    """
    Computes the binary mask of the detected Page from the probabilities outputed by network
    :param probs: array with values in range [0, 1]
    :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
    :return: binary mask
    """

    mask = binarization.thresholding(probs, threshold)
    mask = binarization.cleaning_binary(mask, size=5)
    return mask


def format_quad_to_string(quad):
    """
    Formats the corner points into a string.
    :param quad: coordinates of the quadrilateral
    :return:
    """
    s = ''
    for corner in quad:
        s += '{},{},'.format(corner[0], corner[1])
    return s[:-1]


if __name__ == '__main__':

    # If the model has been trained load the model, otherwise use the given model
    model_dir = 'micropasts/micropasts_model/export'
    if not os.path.exists(model_dir):
        model_dir = 'micropasts/model/'

    input_files = glob('micropasts/micropasts/test_a1/images/*')

    output_dir = 'micropasts/processed_images'
    os.makedirs(output_dir, exist_ok=True)

    # Store coordinates of page in a .txt file
    txt_coordinates = ''

    with tf.Session():  # Start a tensorflow session
        # Load the model
        m = LoadedModel(model_dir, predict_mode='filename')

        for filename in tqdm(input_files, desc='Processed files'):
            basename = os.path.splitext(os.path.basename(filename))[0]
            if not os.path.isfile(os.path.join(output_dir, '{}_bin_upscaled.png'.format(basename))):
                # For each image, predict each pixel's label
                prediction_outputs = m.predict(filename)
                probs = prediction_outputs['probs'][0]
                original_shape = prediction_outputs['original_shape']
                probs = probs[:, :, 1]  # Take only class '1' (class 0 is the background, class 1 is the page)
                probs = probs / np.max(probs)  # Normalize to be in [0, 1]

                # Binarize the predictions
                page_bin = page_make_binary_mask(probs)

                # Upscale to have full resolution image (cv2 uses (w,h) and not (h,w) for giving shapes)
                bin_upscaled = cv2.resize(img_as_ubyte(page_bin),
                                         tuple(original_shape[::-1]), interpolation=cv2.INTER_NEAREST)

                imsave(os.path.join(output_dir, '{}_bin_upscaled.png'.format(basename)), bin_upscaled)
                imsave(os.path.join(output_dir, '{}_page_bin.png'.format(basename)), img_as_ubyte(page_bin))
                imsave(os.path.join(output_dir, '{}_probs.png'.format(basename)), img_as_ubyte(probs))
            else:
                print('Skipping: ' + basename)
