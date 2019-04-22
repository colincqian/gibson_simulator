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

from utils.tfrecordfeatures import *
from preprocess import decode_image, raw_images_to_array

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def tf_bytelist_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def valid_checking(reading_path):
    #make sure that two maps,rgb,depth,information.txt exist
    c1=os.path.exists(os.path.join(reading_path,'bimap.png'))
    c2=os.path.exists(os.path.join(reading_path,'object_bimap.png'))
    c3 = os.path.exists(os.path.join(reading_path, 'raw_map.png'))
    if c1 and c2 and c3:
        for trajectory in os.listdir(os.path.join(reading_path, 'data')):
            c4=os.path.exists(os.path.join(reading_path, 'data', trajectory, 'rgb'))
            c5=os.path.exists(os.path.join(reading_path, 'data', trajectory, 'depth'))
            c6=os.path.exists(os.path.join(reading_path, 'data', trajectory,'information.txt'))
            if c4 and c5 and c6:
                continue
            else:
                return False
        return True
    else:
        return False






def transform_to_tfrecord(fname):
    with tf.python_io.TFRecordWriter(fname + '/tfrecord'+ '.tfrecord') as writer:

        for model in os.listdir(os.getcwd()):
            if not os.path.exists(model + '/floor0'):
                # they have generate floor map
                continue
            for floor in os.listdir('./' + model):
                if not os.path.exists(os.path.join(model, floor, 'data')):
                    # they have generate the rgb data
                    continue

                reading_path = './' + model + '/' + floor
                if not valid_checking(reading_path):
                    continue

                for trajectory in os.listdir(os.path.join(reading_path, 'data')):
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
                        image=cv2.resize(image,(56,56))
                        rgb_list.append(cv2.imencode('.png',image)[1].tostring())
                    for depth in os.listdir(depth_path):
                        image=cv2.imread(os.path.join(depth_path,depth))
                        image = cv2.resize(image, (56, 56))
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
                print('tfrecord'+'done')



def main():
    if os.path.isdir('tfdataset'):
        shutil.rmtree('tfdataset')
    os.mkdir('tfdataset')
    writing_path = './tfdataset'



    transform_to_tfrecord(writing_path)




if __name__ == '__main__':
    main()






