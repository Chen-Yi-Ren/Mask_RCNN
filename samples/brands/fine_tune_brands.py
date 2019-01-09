import os
import sys
import math
import random
import numpy as np
import cv2
import argparse
import matplotlib
import matplotlib.pyplot as plt
from brands import BrandsConfig, BrandsDataset

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class InferenceConfig(BrandsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def training(config, MODEL_DIR, init_with):
   # Training dataset
    dataset_train = BrandsDataset()
    dataset_train.load_brands()
    dataset_train.prepare()

    dataset_val = BrandsDataset()
    dataset_val.load_brands(datafile='val.txt')
    dataset_val.prepare()


    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    # Which weights to start with?
    #init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
    
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=50, 
                layers='3+')
    '''
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10,
                epochs=2, 
                layers="all")
    '''
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_brands.h5")
    model.keras_model.save_weights(model_path)
    return 0


def inference(inference_config, MODEL_DIR, img):
    model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    model_path = os.path.join(ROOT_DIR, "mask_rcnn_brands.h5")
    #model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    results = model.detect([img], verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset_val.class_names, r['scores'], ax=get_ax())
    return 0





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model for training')
    parser.add_argument('--infer', help='run offline inference instead of training', type=int)
    parser.add_argument('--dir', help='set direction.', type=str)
    args = parser.parse_args()


    MODEL_DIR = '../../'

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.infer:
        print('#########################################')
        print('inference')
        config = InferenceConfig()
        img = 0
        gg = inference(config, MODEL_DIR, img)
        print('#########################################')
        print('inference end')
    else:
        print('#########################################')
        print('training')
        config = BrandsConfig()
        gg = training(config, MODEL_DIR, 'coco')
        print('#########################################')
        print('training end')

