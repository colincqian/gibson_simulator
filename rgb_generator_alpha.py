import matplotlib.pyplot as plt
import numpy as np
import random
import meshcut
import cv2
from gibson.envs.husky_env import  HuskyNavigateEnv
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
import sys
import shutil
import pybullet as p

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

def generate_random_point(pre_pos,orn,step,p_change_orientation=0.3):
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
        x = 3.14;y = 0;z = random.uniform(-np.pi, np.pi)  # euler rotation
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
        res=get_present_pos_with_robotheight((X,Y,Z))
        if res is not None:
            return res,orn
        else:
            return generate_random_point(pre_pos, orn, step)






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
def get_rgb(path):
    '''

    :param pos: (x,y,z) the postion in the model
    :param orn: (x,y,z) the orientation in the model euler orientation
    :return: None. save the image named after the position and orientation
    '''

    # (x,y,z)=orn
    # (qw, qx, qy, qz) = euler_to_quaternion(x, y, z)
    # env.robot.reset_new_pose(pos,[qw,qx,qy,qz])
    action=0
    global order
    obs, _, _, _ = env.step(action)
    rgb_image=obs['rgb_filled']#cv2.resize(obs['rgb_filled'],(56,56))
    depth_image=obs['depth']#cv2.resize(obs['depth'],(56,56))
    plt.imsave(path+'rgb/'+str(order)+'.png', rgb_image)
    cv2.imwrite(path + 'depth/' + str(order) + '.png', depth_image)
    order+=1

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
    x_max,y_max=bi_map.shape
    for x in range(x_list[0],x_list[1]):
        for y in range(y_list[0],y_list[1]):
            if y>=x_max or x>=y_max or x<0 or y<0:
                return False
            if bi_map[y,x]==0:
                return False
    return True


def pybullet_collision_checking(pos_new,orn_new):
    '''

    get contact object id and return boolean
    True: not collide
    False: collision happens
    '''
    pos_pre=env.robot.get_position()
    orn_pre=env.robot.get_orientation()

    (x,y,z)=orn_new
    (qw, qx, qy, qz) = euler_to_quaternion(x, y, z)
    env.robot.reset_new_pose(pos_new, [qw, qx, qy, qz])


    obj_id=env.robot.robot_ids;ground_id,=env.ground_ids
    id2=p.getClosestPoints(obj_id[0],ground_id[1],0.3)#id=p.getContactPoints(obj_id[0],ground_id[1])
    if len(id2)>=20:
        #collision happened
        #go back to previuos state
        env.robot.reset_new_pose(pos_pre, orn_pre)
        return False

    return True




def documenting(path,pos,orn):
    '''

    saving the postion and orientation in the specific path
    '''
    #pos=env.robot.get_position()
    with open(path+'information.txt','a') as f:
        f.write(str((pos[0],pos[1],orn[2]))+'\n')

def modify_config_file(model_name,config_file):
    import ruamel.yaml
    yaml = ruamel.yaml.YAML()
    with open(config_file) as fp:
        data=yaml.load(fp)
        data['model_id']=model_name

    with open(config_file,'w') as f:
        yaml.dump(data,f)

def bi_map_collision_checking(bi_map,start_x,start_y,end_x,end_y):
    '''
    collision checking method based on the bimap
    Another method is the pybullet collsion check
    :return:
    '''
    in_bound = all([bool(end_x < bi_map.shape[1]), bool(end_x >= 0), bool(end_y < bi_map.shape[0]), bool(end_y >= 0)])
    return in_bound and is_in_room(bi_map,(end_x,end_y)) and not_collide(bi_map,(start_x,start_y),(end_x,end_y))

def get_present_pos_with_robotheight(pos):
    '''

    :param coord: (x,y,z) the coord in real world
    :return: the new adjusted pos
    '''
    _, _, _, hit_position, _ = p.rayTest(pos, [pos[0], pos[1], pos[2] - 2])[0]
    return (hit_position[0],hit_position[1],hit_position[2]+ROBOT_HEIGHT)

        





def pybullet_path_checking(start,end):
    '''

    :param start: starting coordinate in real world
    :return: True:no collision with object in path
             False: collide with object in path
    '''
    pos=end
    dz=2
    dense_of_ray=1#describ how dense the ray for collision detection
    _,_,_,hit_position,_=p.rayTest(start,[start[0],start[1],start[2]-dz])[0]
    height=start[2]-hit_position[2]
    start_pos_array=[ [start[0],start[1],hit_position[2]+i*height/dense_of_ray]  for i in range(1,dense_of_ray+1)]

    _, _, _, end_hit_position, _ = p.rayTest(pos, [pos[0], pos[1], pos[2] - dz])[0]
    end_height=pos[2]-end_hit_position[2]
    end_pos_array=[[pos[0],pos[1],end_hit_position[2]+i*end_height/dense_of_ray]   for i in range(1,dense_of_ray+1)]

    result=p.rayTestBatch(start_pos_array,end_pos_array)
    for r in result:
        if r[1]==-1:
            #no collision
            continue
        else:
            return False

    return True

def setting_initial_point(pos,orn):

    (x,y,z)=orn
    (qw, qx, qy, qz) = euler_to_quaternion(x, y, z)
    env.robot.reset_new_pose(pos, [qw, qx, qy, qz])

    obj_id=env.robot.robot_ids;ground_id,=env.ground_ids
    id=p.getClosestPoints(obj_id[0],ground_id[1],0.3)
    if len(id)>=15:
        return False
    return True

    




def comprehensive_collision_checking(bi_map,pos_new,orn_new,start_x,start_y,end_x,end_y):
    condition1=bi_map_collision_checking(bi_map,start_x,start_y,end_x,end_y)
    condition2=pybullet_collision_checking(pos_new,orn_new)
    if condition1 and condition2:
        return True
    if not condition1 and condition2:
        #bi_map[end_y,end_x]=1 #change the status ot that point to be a reachable one
        return True
    if condition1 and not condition2:
        #bi_map[end_y,end_x]=0
        return False
    if not condition1 and not condition2:
        return False



def trajectory_generator(start,orn,bi_map,map_para,path,count=1):
    '''

    :param map_para: a list consisting : max_point, min_point and resolution
           start: (x,y)  the origin point of the model
           bi_map: the binary map generated by the function map_generator
    :return: None
    '''
    step=0.1  #usually set the step between 0.2 to 0.5
    start_x,start_y=transform_real_to_cv(start[0],start[1],map_para)
    pos_new,orn_new=generate_random_point(start,orn,step)
    end_x,end_y= transform_real_to_cv(pos_new[0], pos_new[1], map_para)
    if pybullet_path_checking(start,pos_new) and comprehensive_collision_checking(bi_map,pos_new,orn_new,start_x,start_y,end_x,end_y):
        get_rgb(path)                   #get rgb of that position
        documenting(path,(end_x,end_y),orn_new)               #save position and orientation
        cv2.line(bi_map, (start_x, start_y), (end_x, end_y), 255)
        return tuple(env.robot.get_position()),orn_new
    elif count==10:                                     #prevent it from too many recursion
        print('***********abandon path********************************')
        return 0,0
    else:
        print('**********recursion depth',str(count+1),'********************')
        return trajectory_generator(start,orn, bi_map, map_para,path,count=count+1)



def generate_indoor_starting_point(image,number,z):

    '''
    This function generate a series of starting point according to the map
    :param image:  numpy array representing the image
    :param number: number of the starting points
    :return: a list of tuples representing the starting point(in the opencv coordinate)
    '''
    row,col =np.nonzero(image)
    #assert(number<len(row))
    orn = (3.14, 0, random.uniform(0, 2 * np.pi))
    points={(x,y) for y,x in zip(row,col) if pybullet_collision_checking((x,y,z),orn)} #consider the different coordinate between numpy and opencv

    #make an additional judgement of whether the starting point is indoor
    if number >= len(points):
        print('not enough space for starting point!')
        yield from points
    else:
        while(points):
            sample_points=random.sample(points,1);points.remove(sample_points[0])
            yield from sample_points

if __name__=="__main__":
    NUMBER_OF_STARTINGPOINT=int(sys.argv[1])#number of trajectory
    NUMBER_OF_ACTION=int(sys.argv[2]  )     #number of steps taken
    ROBOT_HEIGHT=float(sys.argv[3] )        #the height of the robot
    if len(sys.argv) == 4:  # without setting the number of maps
        LIMITED_MODEL = -1  # go through all the model
    else:
        LIMITED_MODEL = int(sys.argv[4])  # set the number of maps

    for folder in os.listdir(os.getcwd()):  #e.g. allensville
        if os.path.isfile(folder):
            continue
        if not os.path.exists('./'+folder+'/floor0'):
            continue
        if LIMITED_MODEL==0:
            break
	
        modify_config_file(folder,"husky_navigate_enjoy.yaml")   #modify the model id of config file
        env = HuskyNavigateEnv(config = 'husky_navigate_enjoy.yaml')
        env.reset()
	
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



                max_point,min_point,resolution,z_position=DATA
                print('max',max_point,'min:',min_point,'resolution:',resolution,'z_postion:',z_position)
                map_para=[max_point,min_point,resolution]

                if os.path.isdir(PATH + 'data'):
                    shutil.rmtree(PATH + 'data')
                print('starting point loaded!!')
                os.mkdir(PATH + 'data')
                number_of_trajectory=NUMBER_OF_STARTINGPOINT
                for (x,y) in generate_indoor_starting_point(bimap,NUMBER_OF_STARTINGPOINT,z_position):
                    if number_of_trajectory==0:
                        #make sure that it create enough trajectory
                        break
                    order=0
                    (x,y)=transform_cv_to_real(x,y,map_para) #transform into real coordinate

                    os.mkdir(PATH+'data/'+str((x,y)))
                    RGB_PATH=PATH+'data/'+str((x,y))+'/'    #create the trajectory dir
                    os.mkdir(RGB_PATH+'rgb')                                  #create rgb dir
                    os.mkdir(RGB_PATH + 'depth')

                    pos=get_present_pos_with_robotheight((x,y,z_position))
                    orn=(3.14,0,random.uniform(0, 2 * np.pi))
                    if not setting_initial_point(pos,orn):
                        continue

                    #p.setGravity(0,0,0)#Set gravity in three axis to be 0
                    obj_id = env.robot.robot_ids
                    ground_id, = env.ground_ids
                    enableCollision=0  #1: ENABLE 0:DISALE
                    #p.setCollisionFilterPair(obj_id[0],ground_id[0],-1,-1,enableCollision)

                    for n in range(NUMBER_OF_ACTION):
                        pos,orn=trajectory_generator(pos,orn,bimap,map_para,RGB_PATH)
                        if pos==0:   #abandon the trajectory which trap the robot
                            shutil.rmtree(PATH+'data/'+str((x,y)))
                            break
                    else:
                        number_of_trajectory-=1
                #cv2.imwrite(PATH+'tra_bin.png',bimap)
                cv2.imwrite(PATH+'bi_map_adj.png',bimap)
                print('model:',folder,'floor:',floor,'done!')
        LIMITED_MODEL-=1

        env.close()
	
    print('done')











