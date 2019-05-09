from PIL import Image
import tensorflow as tf
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils.tfrecordfeatures import *
from preprocess import decode_image, raw_images_to_array
import numpy
import matplotlib.pyplot as plt


def display_data(file):

    print('display')

    gen = tf.python_io.tf_record_iterator(file)
    for data_i, string_record in enumerate(gen):
        result = tf.train.Example.FromString(string_record)
        features = result.features.feature

        # binary map: 0 for free space, 255 for walls
        bi_map = decode_image(features['map_wall'].bytes_list.value[0])

        # true states
        # (x, y, theta). x,y: pixel coordinates; theta: radians
        # coordinates index the map as a numpy array: map[x, y]
        true_states = features['states'].bytes_list.value[0]
        true_states = numpy.frombuffer(true_states, numpy.float32).reshape((-1, 3))

        # odometry
        # each entry is true_states[i+1]-true_states[i].
        # last row is always [0,0,0]
        odometry = features['odometry'].bytes_list.value[0]
        odometry = numpy.frombuffer(odometry, numpy.float32).reshape((-1, 3))

        # observations are enceded as a list of png images
        rgb = raw_images_to_array(list(features['rgb'].bytes_list.value))
        depth = raw_images_to_array(list(features['depth'].bytes_list.value))

        print ("True states (first three)")
        print (true_states[:3])

        print ("Odometry (first three)")
        print (odometry[:3])

        # note: when printed as an image, map should be transposed
        plt.figure()
        plt.imshow(bi_map)
        print(rgb.shape)
        plt.figure()
        plt.imshow(rgb[0])

        plt.show()
