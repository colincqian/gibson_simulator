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
    with tf.python_io.TFRecordWriter(fname + '/train'+ '.tfrecords') as writer:

        for model in os.listdir(os.getcwd())[5:]:
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
                    bi_map=cv2.cvtColor(bi_map,cv2.COLOR_RGB2GRAY)
                    bi_map=cv2.bitwise_not(bi_map);assert len(bi_map.shape)==2  
                    map_temp=cv2.imencode('.png',bi_map)[1].tostring()     #change into bytes
                    map_feature=_bytes_feature(map_temp)

                    # bi_object_map=cv2.imread(os.path.join(reading_path,'object_bimap.png'))
                    # map_temp=cv2.imencode('.png',numpy.array( bi_object_map))[1].tostring()     #change into bytes
                    # object_map_feature=_bytes_feature(map_temp)

                    raw_map=cv2.imread(os.path.join(reading_path,'raw_map.png'))
                    ret, thresh1 = cv2.threshold(raw_map, 5, 255, cv2.THRESH_BINARY_INV)
                    map_temp=cv2.imencode('.png',numpy.array(thresh1))[1].tostring()          #change into bytes
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
                        # import pdb
                        # pdb.set_trace()
                        rgb_list.append(cv2.imencode('.png',image)[1].tostring())
                    for depth in os.listdir(depth_path):
                        image=cv2.imread(os.path.join(depth_path,depth))
                        image = cv2.resize(image, (56, 56))
                        image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
                        depth_list.append(cv2.imencode('.png',image)[1].tostring())


                    states=[]

                    with open(os.path.normcase(os.path.join(reading_path, 'data', trajectory,'information.txt')),'r',errors='ignore') as fp:
                        for state in fp:
                            states.append(eval(state))
                    states.append((0,0,0)) #padding (0,0,0) at the end to ensure that the odometry is in same length
                    odometry=[[states[i+1][0]-states[i][0],states[i+1][1]-states[i][1],states[i+1][2]-states[i][2]] for i in range(len(states)-1)]
                    # feature_odometry=_bytes_feature(numpy.array(odometry,'f').tostring())
                    # feature_states = _bytes_feature(numpy.array(states,'f').tostring())
                    states_str_list=[numpy.array(states,'f').tostring() ]
                    odometry_str_list=[numpy.array(odometry,'f').tostring() ]
                    feature_states=tf_bytelist_feature(states_str_list)
                    feature_odometry=tf_bytelist_feature(odometry_str_list)

                    example = tf.train.Example(
                              features=tf.train.Features(
                                  feature={
                                      'map_wall': map_feature,
                                      'map_door': raw_map_feature,
                                      'states': feature_states,
                                      'odometry': feature_odometry ,
                                      'rgb': tf_bytelist_feature(rgb_list),
                                      'depth':tf_bytelist_feature(depth_list)
                                  }))
                    writer.write(example.SerializeToString())
                    print(model,floor,trajectory)

                #print('train'+'done')

            


    with tf.python_io.TFRecordWriter(fname + '/valid'+ '.tfrecords') as writer:

        for model in os.listdir(os.getcwd())[:5]:
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
                    bi_map=cv2.cvtColor(bi_map,cv2.COLOR_RGB2GRAY)
                    bi_map=cv2.bitwise_not(bi_map);assert len(bi_map.shape)==2  
                    map_temp=cv2.imencode('.png',bi_map)[1].tostring()     #change into bytes
                    map_feature=_bytes_feature(map_temp)

                    # bi_object_map=cv2.imread(os.path.join(reading_path,'object_bimap.png'))
                    # map_temp=cv2.imencode('.png',numpy.array( bi_object_map))[1].tostring()     #change into bytes
                    # object_map_feature=_bytes_feature(map_temp)

                    raw_map=cv2.imread(os.path.join(reading_path,'raw_map.png'))
                    ret, thresh1 = cv2.threshold(raw_map, 5, 255, cv2.THRESH_BINARY_INV)
                    map_temp=cv2.imencode('.png',numpy.array(thresh1))[1].tostring()          #change into bytes
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
                        image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
                        depth_list.append(cv2.imencode('.png',image)[1].tostring())


                    states=[]

                    with open(os.path.normcase(os.path.join(reading_path, 'data', trajectory,'information.txt')),'r',errors='ignore') as fp:
                        for state in fp:
                            states.append(eval(state))
                    states.append((0,0,0)) #padding (0,0,0) at the end to ensure that the odometry is in same length
                    odometry=[[states[i+1][0]-states[i][0],states[i+1][1]-states[i][1],states[i+1][2]-states[i][2]] for i in range(len(states)-1)]
                    # feature_odometry=_bytes_feature(numpy.array(odometry,'f').tostring())
                    # feature_states = _bytes_feature(numpy.array(states,'f').tostring())
                    states_str_list=[numpy.array(states,'f').tostring() ]
                    odometry_str_list=[numpy.array(odometry,'f').tostring() ]
                    feature_states=tf_bytelist_feature(states_str_list)
                    feature_odometry=tf_bytelist_feature(odometry_str_list)


                    example = tf.train.Example(
                              features=tf.train.Features(
                                  feature={
                                      'map_wall': map_feature,
                                      'map_door': raw_map_feature,
                                      'states': feature_states,
                                      'odometry': feature_odometry ,
                                      'rgb': tf_bytelist_feature(rgb_list),
                                      'depth':tf_bytelist_feature(depth_list)
                                  }))
                    writer.write(example.SerializeToString())
                    print(model,floor,trajectory)
                    
                
            
                #print('validation'+'done')

            


def main():
    if os.path.isdir('tfdataset'):
        shutil.rmtree('tfdataset')
    os.mkdir('tfdataset')
    writing_path = './tfdataset'



    transform_to_tfrecord(writing_path)




if __name__ == '__main__':
    main()






