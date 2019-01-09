"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import math
import random
import numpy as np
import cv2
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


def get_class_names():
    return np.array(['BG', 'HP', 'adidas_symbol', 'adidas_text', 'aldi', 'apple', 'becks_symbol',
                     'becks_text', 'bmw', 'carlsberg_symbol', 'carlsberg_text', 'chimay_symbol',
                     'chimay_text', 'cocacola', 'corona_symbol', 'corona_text', 'dhl',
                     'erdinger_symbol', 'erdinger_text', 'esso_symbol', 'esso_text', 'fedex',
                     'ferrari', 'ford', 'fosters_symbol', 'fosters_text', 'google',
                     'guinness_symbol', 'guinness_text', 'heineken', 'milka', 'nvidia_symbol',
                     'nvidia_text', 'paulaner_symbol', 'paulaner_text', 'pepsi_symbol',
                     'pepsi_text', 'rittersport', 'shell', 'singha_symbol', 'singha_text',
                     'starbucks', 'stellaartois_symbol', 'stellaartois_text', 'texaco',
                     'tsingtao_symbol', 'tsingtao_text', 'ups'])


class BrandsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "brands"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 47  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class BrandsDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_brands(self, datafile='filelist-logosonly.txt'):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """

        brands = get_class_names()
        # Add classes
        for i in range(1,len(brands)):
            self.add_class('brands', (i), brands[i])

        file_path = '/home/cv107/FlickrLogos_47/train/'
        dataset = file_path + datafile
        dataset = open(dataset)

        for i, line in enumerate(dataset):
            line = line.strip()
            img_path = file_path + line

            A, B, C = img_path.split('.')
            label_path = A + '.' + B + '.gt_data.txt'
            
            label_file = open(label_path,'r')
            label = []
            for j , line2 in enumerate(label_file):
                line2 = line2.strip()
                x1, y1, x2, y2, class_id, dummy, mask, diff, trun = line2.split(' ')
                label.append(int(class_id))

            label = np.array(label)
            self.add_image(source='brands',
                            image_id=i,
                            path=img_path,
                            label_path=label_path,
                            brands=label)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        img_path = info['path']
        image = skimage.io.imread(img_path)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "brands":
            return info["brands"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        label_path = info['label_path']
        img_path = info['path']

        A, B, C = img_path.split('.')

        label_file = open(label_path)
        mask = []
        for i, line in enumerate(label_file):
            line = line.strip()
            x1, y1, x2, y2, class_id, dummy, mask0, diff, trun = line.split(' ')
            mask_path =  A + '.' + B + '.' + mask0 + '.png'
            mask_img = skimage.io.imread(mask_path).astype(np.bool)
            mask.append(mask_img)
            '''
            if i == 0:
                mask_img = skimage.io.imread(mask_path)
            else:
                buffer = skimage.io.imread(mask_path)
                mask_img = mask_img + buffer
            '''
        mask = np.stack(mask, axis=-1)
        '''
        height = mask_img.shape[0]
        width = mask_img.shape[1]

        for i in range(height):
            for j in range(width):
                if mask_img[i,j] > 255:
                    mask_img[i,j] = 255
        '''
        #brands = info['brands']
        return mask, info['brands']
