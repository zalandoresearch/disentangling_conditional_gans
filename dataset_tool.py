# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import glob
import argparse
import threading
import six.moves.queue as Queue
import traceback
import numpy as np
import tensorflow as tf
import PIL.Image
import pickle
from tqdm import *

import tfutil
import dataset

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    exit(1)

#----------------------------------------------------------------------------

class TFRecordExporter:

    def __init__(self, tfrecord_dir):

        self.tfrecord_dir       = tfrecord_dir
        
        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

        tfr_file = os.path.join(self.tfrecord_dir, 'dataset.tfrecords')

        self.tfr_writer = tf.python_io.TFRecordWriter(tfr_file, tfr_opt)

        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)

        assert(os.path.isdir(self.tfrecord_dir))
        
    def close(self):
            
        self.tfr_writer.close()

    def add_image_mask_color(self, image, mask, color):

        ex = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()])),
            'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask.tostring()])),
            'color': tf.train.Feature(bytes_list=tf.train.BytesList(value=[color.tostring()])),
            'image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
            'mask_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=mask.shape)),
            'color_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=color.shape)),
            }))

        self.tfr_writer.write(ex.SerializeToString())
            
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

#----------------------------------------------------------------------------

def read_image(path, resolution, antialias):

    pil_image = PIL.Image.open(path).resize((resolution, resolution), antialias)

    image = np.asarray(pil_image)

    channels = image.shape[2] if image.ndim == 3 else 1

    if channels == 1:
        image = image[np.newaxis, :, :] # HW => CHW
    else:
        image = image.transpose(2, 0, 1) # HWC => CHW      

    return image

#----------------------------------------------------------------------------

def create_from_images(image_path, tfrecords_save_path, dataset_resolution):

    image_dir = os.path.join(image_path, "images")
    mask_dir = os.path.join(image_path, "masks")
    color_path = os.path.join(image_path, "average_colors.pkl")

    print(f"Creating dataset at {tfrecords_save_path}")

    print('Loading images from "%s"' % image_dir)
    image_filenames = sorted(glob.glob(os.path.join(image_dir, '*')))
    if len(image_filenames) == 0:
        error('No input images found')
    print(f"{len(image_filenames)} images found.")
        
    img = np.asarray(PIL.Image.open(image_filenames[0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')

    try:
        labels = pickle.load(open(color_path, "rb"))
    except:
        error(f'Color file was not found at path {color_path}')
    
    with TFRecordExporter(tfrecords_save_path) as tfr:

        for i, image_path in tqdm(enumerate(image_filenames), total=len(image_filenames)):

            basename = os.path.basename(image_path)

            mask_path = os.path.join(mask_dir, basename)

            image = read_image(image_path, dataset_resolution, PIL.Image.ANTIALIAS)

            mask = read_image(mask_path, dataset_resolution, PIL.Image.NEAREST)

            color = labels[basename]

            tfr.add_image_mask_color(image, mask, color)

#----------------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 4:
        error('Argument error. Example usage --> python dataset_tool.py /path/to/read/images /path/to/save/tfrecords resolution')

    program_name, image_path, tfrecords_save_path, dataset_resolution = sys.argv

    dataset_resolution = int(dataset_resolution)

    create_from_images(image_path, tfrecords_save_path, dataset_resolution)

#----------------------------------------------------------------------------
