
#!/Users/wilat/Anaconda3/envs/tf-gpu
"""
Mask R-CNN
Configurations and data loading code for MS COCO.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
MIT License
Copyright (c) 2019 Adam Kelly
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import sys
from numpy import random
import math
import json
import re
import time
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from classGraph import *
from classVideo import *
from evaluate_IoU import *
from processing import *
# Root directory of the project
ROOT_DIR = 'C:\\Users\\wilat\\Downloads\\Mask_RCNN-master\\Mask_RCNN-master'

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.visualize import display_images
from mrcnn.model import log
sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


############################################################
#  Configurations
############################################################


class CocoSynthConfig(Config):
    """Configuration for training on the box_synthetic dataset.
    Derives from the base Config class and overrides specific values.
    """
    # Give the configuration a recognizable name
    NAME = "cocosynth_dataset"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # BATCH_SIZE = 100
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 7 box types

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 198

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 85
    BATCH_SIZE = 100
    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet101'

    LEARNING_RATE = 0.001
    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 


############################################################
#  Inference Configurations
############################################################


class InferenceConfig(CocoSynthConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
    DETECTION_MIN_CONFIDENCE = 0.80


############################################################
#  Video Inference Configurations
############################################################


class VideoInferenceConfig(CocoSynthConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640
    IMAGE_SHAPE = [640, 480, 3]
    DETECTION_MIN_CONFIDENCE = 0.80

 
############################################################
#  Dataset
############################################################


class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """
    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
                
    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids


############################################################
#  Display Dataset
############################################################


def display_datasets(dataset_train, dataset_val):

    print("Training dataset\nImages: {}\nClasses: {}".format(len(dataset_train.image_ids), dataset_train.class_names))
    print("Validation dataset\nImages: {}\nClasses: {}".format(len(dataset_val.image_ids), dataset_val.class_names))

    for name, dataset in [('training', dataset_train), ('validation', dataset_val)]:
        print(f'Displaying examples from {name} dataset:')
    
        image_ids = np.random.choice(dataset.image_ids, 3)
        for image_id in image_ids:
            image = dataset.load_image(image_id)
            mask, class_ids = dataset.load_mask(image_id)
            visualize.display_top_masks(image, mask, class_ids, dataset.class_names)


############################################################
#  Compute VOC-style Average Precision
############################################################


def compute_batch_ap(image_ids, dataset, config, threshold = 0.5):
    APs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                              r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold = threshold)
        APs.append(AP)
    return APs


if __name__ == '__main__':

    SettingPath = os.path.join(os.path.dirname(__file__), 'SettingModelPath.json')
    with open(SettingPath) as json_file:
        data = json.load(json_file)

    command = data['command']
    train_path = data['train_path']
    val_path = data['val_path']
    _model = data['_model']
    epoch = data['epoch']
    read_video_path = data['read_video_path']
    save_video_path = data['save_video_path']
    video_save_name = data['video_save_name']

    print("Command: ", command)
    assert os.path.exists(train_path), 'train_path does not exist.)'
    print("train_path: ", train_path)
    assert os.path.exists(val_path), 'val_path does not exist.)'
    print("val_path: ", val_path)
    print("Model_path: ", _model)
    print("Epoch: ", epoch)
    

    dataset_train = CocoLikeDataset()
    dataset_train.load_data(train_path + '\\' + 'coco_instances.json',
                            train_path + '\\' + 'images')
    dataset_train.prepare()

    dataset_val = CocoLikeDataset()
    dataset_val.load_data(val_path + '\\' + 'coco_instances.json',
                        val_path + '\\' + 'images')
    dataset_val.prepare()

    
    # Configurations
    if command == "train":
        config = CocoSynthConfig()
    elif command == "video":
        config = VideoInferenceConfig()
    else:
        config = InferenceConfig()

    config.display()

    # Create model
    if command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=MODEL_DIR)
    else:
        # Create model in inference mode
        model = modellib.MaskRCNN(mode="inference", 
                                config=config,model_dir=MODEL_DIR)
            
        if _model == "last":
            model_path = model.find_last()
        else:
            model_path = str(Path(ROOT_DIR) / "logs" / "{}".format(_model)) #

        # Load trained weights (fill in path to trained weights here)
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)

    if command == "train":
        print("training...")
        # Which weights to start with?
        init_with = "coco"  #coco, or last
        if init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            model.load_weights(COCO_MODEL_PATH, by_name=True,
                            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                        "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            model.load_weights(model.find_last(), by_name=True)
        
        # Train the head branches
        # Passing layers="heads" freezes all layers except the head
        # layers. You can also pass a regular expression to select
        # which layers to train by name pattern.
        start_train = time.time()
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE, 
                    epochs= int(epoch), 
                    layers='heads')
        end_train = time.time()
        minutes = round((end_train - start_train) / 60, 2)
        print(f'Training took {minutes} minutes')


    elif command == "evaluate":
        import skimage
        print("Validation process...")
        DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

        # Create model in inference mode
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                    config=config)

        # Load trained weights (fill in path to trained weights here)
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)
        
        #Image files for test and evaluate model.
        real_test_dir = val_path + "\\images"

        image_paths = []
        for filename in os.listdir(real_test_dir):
            if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
                image_paths.append(os.path.join(real_test_dir, filename))

        evaluate = evaluation(val_path)
        for image_path in image_paths:
            img = skimage.io.imread(image_path)
            img_arr = np.array(img)
            start = time.time()
            filenameList = image_path.split("\\")
            print("Name: ", filenameList[-1])
            # if filenameList[-1] != '217.png' and filenameList[-1] != '294.png' and filenameList[-1] != '297.png':
            results = model.detect([img_arr], verbose=1)
            r = results[0]
            print("Time: ", time.time() - start)
        
            if r['rois'].tolist() == []:
                print("This images is emtry!!!\n")
                predFormat = evaluate.resultFormat(filenameList[-1], r['rois'], r['class_ids'])
                predFormat[filenameList[-1]] = {'1' : [], '2' : []}
                print("pred: ", predFormat)
            else:
                predFormat = evaluate.resultFormat(filenameList[-1], r['rois'], r['class_ids'])
            evaluate.main(filenameList[-1],  predFormat)
            # if filenameList[-1] == '1.png' or filenameList[-1] == '10.png' or filenameList[-1] == '250.png' or filenameList[-1] == '44.png' or filenameList[-1] == '100.png' or filenameList[-1] == '200.png' or filenameList[-1] == '150.png':
            #     visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
            #                 dataset_train.class_names, r['scores'], figsize=(8,8))
            # else:
                # pass

        evaluate.saveJson()
        

    elif command == "video":
        print("video process...")
        ############################################################
        #  Setting path of video.
        ############################################################
        video_file = Path(read_video_path)
        assert os.path.exists(video_file), 'video_file does not exist.)'

        video_save_dir = Path(save_video_path)
        assert os.path.exists(video_save_dir), 'video_save_dir does not exist.)'

        video_save_dir.mkdir(exist_ok=True)

        vid_name = video_save_dir / "output_{}.mp4".format(video_save_name)
        v_format="FMP4"
        fourcc = cv2.VideoWriter_fourcc(*v_format)
        print('Writing output video to: ' + str(vid_name))

        ############################################################
        #  Perform Inference on Video
        ############################################################
        #param : model, dataset_train, video_file, video_save_dir, vid_name, v_format, fourcc
        vidGen = videoGenarator(model, dataset_val, video_file, video_save_dir, vid_name, v_format, fourcc)
        vidGen.main(saveImg = True)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(command))

# command = "video"
# readVideo = "E:/videoII/Test/Sequence 02_17.mp4"
# writeVideo = "C:/Users/wilat/OneDrive/Desktop/images_II"
# train_path = "C:/Users/wilat/OneDrive/Desktop/validation/train/write"
# val_path = "C:/Users/wilat/OneDrive/Desktop/Real_val"
# _model = "cocosynth_dataset20191015T0907\mask_rcnn_cocosynth_dataset_0100.h5"
# epoch = 10
# read_video_path = "E:\\videoII\\Test\\Sequence 02_17.mp4"
# save_video_path = "C:\\Users\\wilat\\OneDrive\\Desktop\\images_II"
# video_save_name = "Test"
# mainDetection(command, train_path, val_path, _model, epoch, read_video_path, save_video_path, video_save_name)