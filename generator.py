import sys
import matplotlib.pyplot as plt
import meshcut
import numpy as np
import random
import cv2
import os
import random
from io import StringIO
import PIL
import dbscan
import shutil

def img_fill(im_in, starting_point):
    '''
    arg:
        im_in:a numpy array reprsenting the image
        starting point: the marked original agent position which should be indoor


    return:
        a numpy array representing floodfilled image
    '''


    im_in=im_in.astype(np.uint8)
    # Copy the thresholded image.
    im_floodfill = im_in.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, starting_point, 255)


    return  im_floodfill


def load_obj(fn):
    verts = []
    faces = []
    with open(fn) as f:
        for line in f:
            if line[:2] == 'v ':
                verts.append(list(map(float, line.strip().split()[1:4])))
            if line[:2] == 'f ':
                face = [int(item.split('/')[0]) for item in line.strip().split()[-3:]]
                faces.append(face)
    verts = np.array(verts)
    faces = np.array(faces) - 1
    return verts, faces

def floodfill_adjustment(image,raw_map,origin):
    kernel_size=1
    while (image[0][0] != 0 or image[-1][0] != 0 or image[0][-1] != 0 or image[-1][-1] != 0):
        #if any of the four corner is white, revealing that floodfill fails
        kernel_size+=1
        blur_image=cv2.blur(raw_map,(kernel_size,kernel_size))
        _,image=cv2.threshold(img_fill(blur_image, origin), 230, 255,cv2.THRESH_BINARY)
        print('adjuestment made!')
    return image


def get_balck_pixel(image):
    '''

    :param image: numpy array
    :return:  return the coordinate in numpy array of the original (0,0) and (0.5,0.5)
    '''
    coordinate=[]
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x,y]<=0.1:
                coordinate.append((x,y))
                image[x,y]=1
    #get the pixel with largest x minimum y and minimum x with largest y
    print(coordinate)
    coordinate.sort(key=lambda x: (-x[0],x[1])) #descending for 1st element and ascending for 2nd element
    zero_co=coordinate[0]
    coordinate.sort(key=lambda x: (-x[1],x[0]))#descending for 2nd ele ascending for 1st ele
    one_co=coordinate[0]

    return zero_co,one_co

def transform_real_to_cv(x,y,min_point,max_point,resolution):
    '''

    :param x:  x
    :param y:  y in real model
    :param min_point:  the min_point for the map initializing
    :param max_point:  the max_point for the map initializing
    :param resolution: resolution set for the map
    :return: (x,y) the coordinate in the map
    '''
    x_min=min_point[0]
    y_max=max_point[1]
    x_o=(1-x_min)/resolution+x/resolution
    y_o=(1+y_max)/resolution-y/resolution
    return (int(x_o),int(y_o))



def line_map(superp,path,writing_path,multi_floor,object_map=False,draw=False,resolution=0.04):
    '''
    generate the line map
    :param superp:  how many superpostion for the line map
    :param path:    reading and saving path
    :param draw:    whether to display the map
    :param resolution:  set default to be 0.4
    :return: the significant parameters that influence the map
    '''
    _,_,floor_height,robot_h,min_height=multi_floor

    fn = path + 'mesh_z_up.obj'
    verts, faces = load_obj(fn)
#    robot_h = np.min(verts[:, -1]) + 0.5  # your robot height  #probably using this to optimize the algo


    plt.figure(0)
    x_max_list=[];y_max_list=[]
    x_min_list=[];y_min_list=[]

    cross_section = meshcut.cross_section(verts, faces, plane_orig=(0, 0, floor_height), plane_normal=(0, 0, 1));
    for item in cross_section:
        #create the map

        x_max, y_max = np.max(item[:,:2], axis=0)  # find the largest x and y
        x_min, y_min = np.min(item[:,:2], axis=0)  # find the minimum x and y
        x_max_list.append(x_max);y_max_list.append(y_max)
        x_min_list.append(x_min);y_min_list.append(y_min)
        min_point=(np.min(x_min_list),np.min(y_min_list))
        max_point=(np.max(x_max_list),np.max(y_max_list))

        raw_map=np.zeros((int((max_point[1]-min_point[1])/resolution)+int(2/resolution)
                           ,int((max_point[0]-min_point[0])/resolution)+int(2/resolution)))  #generate the map

    if object_map==True:
        rawmap_with_obj = raw_map.copy()
        z=min_height
        cross_section = meshcut.cross_section(verts, faces, plane_orig=(0, 0, z), plane_normal=(0, 0, 1))
        for item in cross_section:
            for i in range(len(item) - 1):
                (x1, y1) = transform_real_to_cv(item[i, 0], item[i, 1], min_point, max_point, resolution)
                (x2, y2) = transform_real_to_cv(item[i + 1, 0], item[i + 1, 1], min_point, max_point, resolution)
                cv2.line(rawmap_with_obj, (x1, y1), (x2, y2), 150, 2)
        return z,rawmap_with_obj,max_point,min_point,resolution



    for i in range(superp):
        z = min_height+(i+1)*(robot_h-min_height)/superp   #make a croos section between floor height and robot_h
        cross_section = meshcut.cross_section(verts, faces, plane_orig=(0, 0, z), plane_normal=(0, 0, 1))
        for item in cross_section:
            for i in range(len(item) - 1):
                (x1,y1)=transform_real_to_cv(item[i,0],item[i,1],min_point,max_point,resolution)
                (x2,y2)=transform_real_to_cv(item[i+1,0],item[i+1,1],min_point,max_point,resolution)

                cv2.line(raw_map,(x1,y1),(x2,y2),150,2)




    if draw==True:
        cv2.namedWindow('testwindow')
        cv2.imshow('testwindow',raw_map)
        cv2.waitKey(0)
    cv2.imwrite(writing_path+'raw_map.png',raw_map)
    return robot_h,raw_map,max_point,min_point,resolution


def map_generator(T,superp,path,writing_path,multi_floor,draw=False):

    '''
    This function generate the binary map in the folder with a txt file 'map_para.txt' saving some important parameter
    T:the number of maps superpositioning for the binary map
    path:path to read the mesh file
    writing_path: path to write the maps and save map para
    superp: number of superposition for line map

    '''
    #get the indoor coordinate from multifloor
    x0,y0,_,_,_=multi_floor

    _,object_map,_,_,_=line_map(superp, path, writing_path, multi_floor,object_map=True, draw=draw, resolution=0.03)


    binary_map_list=[]
    for it in range(T):
        z,map,max_point,min_point,resolution = line_map(superp,path,writing_path,multi_floor,draw=draw,resolution=0.03)
    #    #flood fill the map
        origin=transform_real_to_cv(x0,y0,min_point,max_point,resolution)
        ret, im_bi = cv2.threshold(img_fill(map, origin), 230, 255,cv2.THRESH_BINARY)

        im_bi = floodfill_adjustment(im_bi, map, origin)

        if draw==True:
            cv2.namedWindow('testwindow')
            cv2.imshow('testwindow', im_bi)
            cv2.waitKey(0)
        binary_map_list.append(im_bi)
    sp_bimap=binary_map_list[0]
    for i in range(len(binary_map_list)-1):
        sp_bimap=cv2.bitwise_or(sp_bimap,binary_map_list[i+1])
    if draw == True:
        cv2.namedWindow('testwindow')
        cv2.imshow('testwindow', sp_bimap)
        cv2.waitKey(0)
    cv2.imwrite(writing_path+'bimap.png',sp_bimap)
    print('bimap ready!')
    with open(writing_path+'map_para.txt','w') as f:
        f.write(str(max_point)+"\n"+ str(min_point)+"\n"+"("+str(resolution)+")"+"\n"+"("+str(z)+")")

    origin = transform_real_to_cv(x0, y0, min_point, max_point, resolution)
    ret, obj_bimap = cv2.threshold(img_fill(object_map, origin), 230, 255, cv2.THRESH_BINARY)

    obj_bimap = floodfill_adjustment(obj_bimap, object_map, origin)

    cv2.imwrite(writing_path + 'object_bimap.png', obj_bimap)
    print('object bimap ready!')



def main(T,superp,path,draw=False):
    #calculate how many floors are there in this model
    coordinate_information=dbscan.get_floor(path,0.3,7)
    for i,pose in enumerate(  coordinate_information):
        if os.path.isdir(path+'floor'+str(i)):
            shutil.rmtree(path+'floor'+str(i))

        writing_path=path+'floor'+str(i)+'/'
        os.mkdir(path+'floor'+str(i))
        map_generator(T,superp,path,writing_path,pose)






if __name__ == "__main__":
    #test case
    main(1,1,'model/Newfields/')

