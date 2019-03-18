import matplotlib.pyplot as plt
import numpy as np
import random
import meshcut
import cv2
#from gibson.envs.husky_env import  HuskyNavigateEnv
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
import sys
import shutil

'''
version 1.3 using opencv to solve all the problems
'''

PATH='Allensville/'


def transform_real_to_cv(x,y,map_para):
    '''

    :param x:  x
    :param y:  y in real model
    :param min_point:  the min_point for the map initializing
    :param max_point:  the max_point for the map initializing
    :param resolution: resolution set for the map
    :return: (x,y） the coordinate in the map
    '''
    max_point=map_para[0]
    min_point=map_para[1]
    resolution=map_para[2]
    x_min=min_point[0]
    y_max=max_point[1]
    x_o=(1-x_min)/resolution+x/resolution
    y_o=(1+y_max)/resolution-y/resolution
    return (int(x_o),int(y_o))

def transform_cv_to_real(x,y,map_para):
    '''

    :param x:  x
    :param y: y  in opencv coord
    :param map_para: a list containing 3 elements: max_point,min_point,resolution
    :return:  return the transfered coordinates in the real world
    '''
    max_point=map_para[0]
    min_point=map_para[1]
    resolution=map_para[2]
    x_min=min_point[0]
    y_max=max_point[1]
    x_o=resolution*x+x_min-1
    y_o=1+y_max-resolution*y
    return x_o,y_o

def euler_to_quaternion(yaw, pitch, roll): #（Z,Y,X)
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
        yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
        yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)

    return [qw, qx, qy, qz]

def quaternion_to_euler(qw,qx,qy,qz):
    sinr_cosp = +2.0 * (qw * qx+ qy * qz)
    cosr_cosp = +1.0 - 2.0 * (qx * qx + qy * qy)
    roll=math.atan2(sinr_cosp,cosr_cosp)
    sinp = +2.0 * (qw * qy - qz * qx)
    if math.fabs(sinp)>=1:
        pitch=math.copysign(math.pi/2,sinp)
    else:
        pitch=math.asin(sinp)
    siny_cosp = +2.0 * (qw * qz + qx * qy)
    cosy_cosp = +1.0 - 2.0 * (qy * qy + qz * qz)
    yaw=math.atan2(siny_cosp,cosy_cosp)
    return yaw,pitch,roll #Z,Y,X

def generate_random_point(pre_pos,orn,step,p_change_orientation=0.5):
    '''

    :param pre_pos:  (x,y,z) the previous position ,set z to be the height of robot
            orn: previous orientation in euler form
            p_change..: the possiblity to change orientation instead of moving forward
    :return:  position and Orientation


    tip:
       euler form theta is the angle between the x postive axis and anticlockwisely rotated ray
    '''
    p=np.random.binomial(1,p_change_orientation) #possibility to change orientation
    if p==1:
        '''
        change orientation
        '''
        (x0,y0,z0)=orn     #previous orientation
        x = 3.14;y = 0;z = random.uniform(0, 2 * np.pi)  # euler rotation
        # print('delta_theta',z-z0)
        return pre_pos,(x,y,z)
    else:
        '''
        moving forward
        '''
        x,y,z=pre_pos
        _,_,theta=orn
        distance=step#step*random.uniform(0.5,1)
        X= x + distance*math.cos(theta)
        Y = y + distance*math.sin(theta)
        Z= z
        # print('delta_x',X-x)
        # print('delta_y',Y-y)

        return (X,Y,Z),orn





def is_in_room(bi_map,position):
    '''

    :param  bi_map: numpy array representing the image
            position: a tuple. the position in the binary map (opencv coordinate)
    :return:  Boolean: True when it is in the room
    '''
    (x,y)=position
    if bi_map[y,x]!=0 : #in white space==in the room numpy coordinate
        return True
    else:
        return False
order=0
def get_rgb(pos,orn,path):
    '''

    :param pos: (x,y,z) the postion in the model
    :param orn: (x,y,z) the orientation in the model euler orientation
    :return: None. save the image named after the position and orientation
    '''

    # (x,y,z)=orn
    # (qw, qx, qy, qz) = euler_to_quaternion(x, y, z)
    # env.robot.reset_new_pose(pos,[qw,qx,qy,qz])
    # action=0
    # global order
    # obs, _, _, _ = env.step(action)
    # plt.imsave(path+'rgb/'+order+'.png', obs['rgb_filled'])
    # cv2.imwrite()
    # plt.imsave(path + 'depth/' + order + '.png', obs['depth'])
    # order+=1

def load_obj(fn):
    '''

    :param fn: load the file containing teh verts and faces
    :return: verts and faces
    '''
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



def not_collide(bi_map,start,end):
    '''
    make judgement of whether the movement of robot has collided with the object
    :param bi_map:  the numpy array that represent the binary map
    :param start:   the startin coodinates
    :param end:     the ending coordinates
    :return:   Boolean True if it is collided
    '''
    #start end are both opencv coordinate
    y_list=sorted([start[1],end[1]]) #ascending
    x_list=sorted([start[0],end[0]])
    for x in range(x_list[0]-1,x_list[1]+1):
        for y in range(y_list[0]-1,y_list[1]+1):
            if bi_map[y,x]==0:
                return False
    return True


def documenting(path,pos,orn):
    '''

    saving the postion and orientation in the specific path
    '''

    with open(path+'information.txt','a') as f:
        f.write(str((pos[0],pos[1],orn[2]))+'\n')




def trajectory_generator(start,orn,bi_map,map_para,path,count=1):
    '''

    :param map_para: a list consisting : max_point, min_point and resolution
           start: (x,y)  the origin point of the model
           bi_map: the binary map generated by the function map_generator
    :return: None
    '''
    step=0.2  #usually set the step between 0.2 to 0.5
    start_x,start_y=transform_real_to_cv(start[0],start[1],map_para)
    pos_new,orn_new=generate_random_point(start,orn,step)
    end_x,end_y= transform_real_to_cv(pos_new[0], pos_new[1], map_para)
    in_bound= all([bool(end_x < bi_map.shape[1]) , bool(end_x >= 0) ,bool(end_y < bi_map.shape[0]) , bool(end_y >=0)])

    if in_bound and is_in_room(bi_map,(end_x,end_y)) and not_collide(bi_map,(start_x,start_y),(end_x,end_y)):
        get_rgb(pos_new,orn_new,path)                   #get rgb of that position
        cv2.line(bi_map,(start_x,start_y),(end_x,end_y),150) #plot trajectory
        documenting(path,pos_new,orn_new)               #save position and orientation
        return pos_new,orn_new
    elif count==30:                                     #prevent it from too many recursion
        return 0,0
    else:
        return trajectory_generator(start,orn, bi_map, map_para,path,count=count+1)

# def is_indoor(coord, image):
#     x0,y0=coord
#     for y in range(image.shape[1]):
#         if image[x0,y]==0:   #row check
#             break            #meet black break
#     else:
#         # all white
#         return False
#
#     for x in range(image.shape[0]):
#         if image[x,y0]==0:    #column check
#             break
#     else:
#         return False
#
#     return True



def generate_indoor_starting_point(image,number):

    '''
    This function generate a series of starting point according to the map
    :param image:  numpy array representing the image
    :param number: number of the starting points
    :return: a list of tuples representing the starting point(in the opencv coordinate)
    '''
    def is_in_door(coord,image):
        x0,y0=coord
        if image[y0,x0]==0:
            return False
        for x in range(x0,image.shape[1]):
            if image[y0,x]==0:
                break
        else:
            return False

        for x in range(0,x0):
            if image[y0,x]==0:
                break
        else:
            return False

        for y in range(y0,image.shape[0]):
            if image[y,x0]==0:
                break
        else:
            return False


        for y in range(y0):
            if image[y,x0]==0:
                break
        else:
            return False

        return True
    output=[]
    while (len(output)!=number):
        x=np.random.randint(image.shape[1])
        y=np.random.randint(image.shape[0])
        if is_in_door((x,y),image):
            output.append((x,y))
    return output

    #
    # row,col =np.nonzero(image)
    # assert(number<len(row))
    #
    # points={(x,y) for y,x in zip(row,col)   #consider the different coordinate between numpy and opencv
    #
    # #make an additional judgement of whether the starting point is indoor
    # sample_points=random.sample(points,number)

    return sample_points

if __name__ == "__main__":
    # Here is the test value
    # NUMBER_OF_STARTINGPOINT=200   #number of trajectory
    # NUMBER_OF_ACTION=300          #number of steps taken
    NUMBER_OF_STARTINGPOINT=int(sys.argv[1])#number of trajectory
    NUMBER_OF_ACTION=int(sys.argv[2]  )     #number of steps taken
    for folder in os.listdir(os.getcwd()):  #e.g. allensville
        if os.path.isfile(folder):
            continue
        for floor in os.listdir('./'+folder):    #e.g. floor1
            if not os.path.exists('./'+folder+'/'+ floor+'/bimap.png'):
                continue
            else:
                PATH='./'+folder+'/'+ floor+'/'
                print('PATH',PATH)
                bimap=cv2.imread(PATH+'bimap.png',cv2.IMREAD_GRAYSCALE)
                DATA=[]
                with open(PATH+'map_para.txt','r') as fp:#read the map para from map para
                    for line in fp:
                        DATA.append(eval(line))   #max_point, min_point,resolution,z
                print("map parameters loaded!!")

                starting_point=generate_indoor_starting_point(bimap,NUMBER_OF_STARTINGPOINT)
                print('starting point loaded!!')

                max_point,min_point,resolution,z_position=DATA
                print('max',max_point,'min:',min_point,'resolution:',resolution,'z_postion:',z_position)
                map_para=[max_point,min_point,resolution]

                # env = HuskyNavigateEnv(config = 'husky_navigate_enjoy.yaml')
                # env.reset()

                if os.path.isdir(PATH + 'data'):
                    shutil.rmtree(PATH + 'data')

                os.mkdir(PATH + 'data')

                for (x,y) in starting_point:
                    (x,y)=transform_cv_to_real(x,y,map_para) #transform into real coordinate



                    os.mkdir(PATH+'data/'+str((x,y)))
                    RGB_PATH=PATH+'data/'+str((x,y))+'/'    #create the trajectory dir
                    os.mkdir(RGB_PATH+'rgb')                                  #create rgb dir
                    os.mkdir(RGB_PATH + 'depth')

                    pos=(x,y,z_position)
                    orn=(3.14,0,random.uniform(0, 2 * np.pi))

                    for n in range(NUMBER_OF_ACTION):
                        pos,orn=trajectory_generator(pos,orn,bimap,map_para,RGB_PATH)
                        if pos==0:   #abandon the trajectory which trap the robot
                            shutil.rmtree(PATH+'data/'+str((x,y)))
                            break
                cv2.imwrite(PATH+'tra_bin.png',bimap)










