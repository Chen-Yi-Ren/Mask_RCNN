import os
import sys
import argparse
import numpy as np
import coco
import utils
import model as modellib
#from classes import get_class_names, InferenceConfig
from ast import literal_eval as make_tuple
import imageio
import visualize
import random
from brands import get_class_names, InferenceConfig

# Creates a color layer and adds Gaussian noise.
# For each pixel, the same noise value is added to each channel
# to mitigate hue shfting.


def draw_mosaic(image):
    mosaic_range = [8, 16]
    row, col, ch = image.shape
    half_patch = np.random.randint(mosaic_range[0], mosaic_range[1]+1, 1)[0]
    img_out = image.copy()

    for i in range(half_patch, row - 1 - half_patch, half_patch):
        for j in range(half_patch, col - 1 - half_patch, half_patch):
            k1 = random.random() - 0.5
            k2 = random.random() - 0.5
            m = np.floor(k1 * (half_patch * 2 + 1))
            n = np.floor(k2 * (half_patch * 2 + 1))
            h = int((i + m) % row)
            w = int((j + n) % col)
            img_out[i - half_patch:i + half_patch, j - half_patch:j + half_patch, :] = image[h, w, :]
    return img_out


def create_noisy_color(image, color):
    color_mask = np.full(shape=(image.shape[0], image.shape[1], 3),
                         fill_value=color)

    noise = np.random.normal(0, 25, (image.shape[0], image.shape[1]))
    noise = np.repeat(np.expand_dims(noise, axis=2), repeats=3, axis=2)
    mask_noise = np.clip(color_mask + noise, 0., 255.)
    return mask_noise


# Helper function to allow both RGB triplet + hex CL input

def string_to_rgb_triplet(triplet):

    if '#' in triplet:
        # http://stackoverflow.com/a/4296727
        triplet = triplet.lstrip('#')
        _NUMERALS = '0123456789abcdefABCDEF'
        _HEXDEC = {v: int(v, 16)
                   for v in (x + y for x in _NUMERALS for y in _NUMERALS)}
        return (_HEXDEC[triplet[0:2]], _HEXDEC[triplet[2:4]],
                _HEXDEC[triplet[4:6]])

    else:
        # https://stackoverflow.com/a/9763133
        triplet = make_tuple(triplet)
        return triplet


def person_blocker(args, outfile):

    # Required to load model, but otherwise unused
    ROOT_DIR = os.getcwd()
    COCO_MODEL_PATH = args.model or os.path.join(ROOT_DIR, "mask_rcnn_brands.h5")

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")  # Required to load model

    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Load model and config
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    image = imageio.imread(args.image)

    # Create masks for all objects
    results = model.detect([image], verbose=0)
    r = results[0]

    if args.labeled:
        position_ids = ['[{}]'.format(x)
                        for x in range(r['class_ids'].shape[0])]
        #visualize.display_instances(image, r['rois'],
        #                            r['masks'], r['class_ids'],
        #                            get_class_names(), position_ids, outfile=outfile)
        return r, position_ids
        #sys.exit()
    else:
        # Filter masks to only the selected objects
        objects = np.array(get_class_names()[1:])

        # Object IDs:
        if np.all(np.chararray.isnumeric(objects)):
            object_indices = objects.astype(int)
        # Types of objects:
        else:
            selected_class_ids = np.flatnonzero(np.in1d(get_class_names(),
                                                        objects))
            object_indices = np.flatnonzero(
                np.in1d(r['class_ids'], selected_class_ids))

        if len(object_indices) != 0:
            mask_selected = np.sum(r['masks'][:, :, object_indices], axis=2)
        else:
            mask_selected = np.zeros((image.shape[0], image.shape[1]))

        # Replace object masks with noise
        #mask_color = string_to_rgb_triplet(args.color)
        image_masked = image.copy()
        mosaic_img = draw_mosaic(image)
        #noisy_color = create_noisy_color(image, mask_color)
        #print('object_indices', object_indices)
        #print('mask_selected', mask_selected.shape)
        #print('image_masked', image_masked.shape)
        #print('noisy_color', noisy_color.shape)
        #image_masked[mask_selected > 0] = noisy_color[mask_selected > 0]
        image_masked[mask_selected > 0] = mosaic_img[mask_selected > 0]

        imageio.imwrite(outfile, image_masked)
