from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import shutil
import cv2
import numpy
import argparse
import os
import sys
sys.path.append('./pfnet-master')
from PIL import Image
import tensorflow as tf


from preprocess import decode_image, raw_images_to_array

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def tf_bytelist_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

i=0
def display_data(file):

    print('display')

    gen = tf.python_io.tf_record_iterator(file)
    for data_i, string_record in enumerate(gen):
        result = tf.train.Example.FromString(string_record)
        features = result.features.feature

        # binary map: 0 for free space, 255 for walls
        bi_map = decode_image(features['map'].bytes_list.value[0])

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

        print("Plot map and first observation")
        cv2.namedWindow('binarymap')
        cv2.imshow('testwindow', bi_map)
        cv2.waitKey(0)
        # note: when printed as an image, map should be transposed
        # plt.figure()
        # plt.imshow(bi_map)
        print(rgb.shape)
        plt.figure()
        plt.imshow(rgb[0])

        plt.show()


def transform_to_tfrecord(fname,reading_path):
    global  i
    for trajectory in os.listdir(os.path.join(reading_path, 'data')):
        with tf.python_io.TFRecordWriter(fname + '/tfrecord' + str(i)+ '.tfrecord') as writer:
            bi_map=cv2.imread(os.path.normcase(os.path.join(reading_path,'bimap.png')))
            map_temp=cv2.imencode('.png',numpy.array( bi_map))[1].tostring()     #change into bytes
            map_feature=_bytes_feature(map_temp)

            bi_object_map=cv2.imread(os.path.join(reading_path,'object_bimap.png'))
            map_temp=cv2.imencode('.png',numpy.array( bi_object_map))[1].tostring()     #change into bytes
            object_map_feature=_bytes_feature(map_temp)

            raw_map=cv2.imread(os.path.join(reading_path,'raw_map.png'))
            map_temp=cv2.imencode('.png',numpy.array( raw_map))[1].tostring()          #change into bytes
            raw_map_feature=_bytes_feature(map_temp)


            # transform image into string type

            rgb_path=os.path.join(reading_path,'data',trajectory,'rgb')
            depth_path=os.path.join(reading_path,'data',trajectory,'depth')
            rgb_list=[]
            depth_list=[]
            if not os.path.exists(rgb_path) and not  os.path.exists(depth_path):
                continue

            for rgb in os.listdir(rgb_path):
                image=cv2.imread(os.path.join(rgb_path,rgb))
                rgb_list.append(cv2.imencode('.png',image)[1].tostring())
            for depth in os.listdir(depth_path):
                image=cv2.imread(os.path.join(depth_path,depth))
                depth_list.append(cv2.imencode('.png',image)[1].tostring())


            states=[]

            with open(os.path.normcase(os.path.join(reading_path, 'data', trajectory,'information.txt')),'r',errors='ignore') as fp:
                for state in fp:
                    states.append(eval(state))

            states.append((0,0,0)) #padding (0,0,0) at the end to ensure that the odometry is in same length
            odometry=[ list(map(lambda x: x[0] - x[1], zip( states[i+1], states[i])))  for i in range(len(states)-1)]   # each entry is states[i+1]-states[i].
            feature_odometry=_bytes_feature(numpy.array(odometry).tostring())
            feature_states = _bytes_feature(numpy.array(states).tostring())


            example = tf.train.Example(
                      features=tf.train.Features(
                          feature={
                              'map': map_feature,
                              'object_map':object_map_feature,
                              'raw_map': raw_map_feature,
                              'states': feature_states,
                              'odometry': feature_odometry ,
                              'rgb': tf_bytelist_feature(rgb_list),
                              'depth':tf_bytelist_feature(depth_list)
                          }))
            writer.write(example.SerializeToString())
            print('tfrecord'+str(i)+'done')
            i = i + 1


def main():
    global i
    if os.path.isdir('tfdataset'):
        shutil.rmtree('tfdataset')
    os.mkdir('tfdataset')
    writing_path = './tfdataset'

    for model in os.listdir(os.getcwd()):
        if not os.path.exists(model + '/floor0'):
            # they have generate floor map
            continue
        for floor in os.listdir('./' + model):
            if not os.path.exists(os.path.join(model, floor, 'data')):
                # they have generate the rgb data
                continue

            reading_path = './' + model + '/' + floor
            print('generate tfrecord of '+str(floor)+str( model))

            transform_to_tfrecord(writing_path, reading_path)




if __name__ == '__main__':
    #main()

    display_data('./tfdataset/tfrecord26.tfrecord')




