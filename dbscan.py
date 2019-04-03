import csv
import random
import statistics

def read_camera_pose(fn):
    '''

    :param fn:
    :return: a list consisting of the heights of camera poses
    '''
    height=[]
    position=[]
    with open(fn,'r') as f:
        reader=csv.reader(f)
        for line in reader:
            position.append((eval(line[1]),eval(line[2])))
            height.append([eval(line[3])])
    return height,position


def distance(x1,x2):
    # 1D data
    s=0
    for X0,X1 in zip(x1,x2):
        s+=(X0-X1)**2
    return s**0.5

def get_neighbor(data,dataset,epsilon):
    '''

    :param data:     the data you want get neighbor
    :param dataset: the whole dataset of heights
    :param epsilon: the threshold distance set for neighbors
    :return:  a list of neighbors index include the center data itself
    '''
    result=[]
    for i,d in enumerate(dataset):
        if distance(d,data)<epsilon:
            result.append(i)
    return result

def DBSCAN(dataSet , epsilon , minPts):
    '''
    
    :param dataSet: 
    :param epsilon:  the distance from the core object
    :param minPts:   the minimum number for a valid cluster
    :return: a dictionary. key: the cluster ID  value: index of the data belongs to that cluster
    '''
    coreObjs = {}
    C = {}
    n = len(dataSet)
    #find all the core object :key is the index of core object and value is index in the neighborhood
    for i in range(n):
        neibor = get_neighbor(dataSet[i] , dataSet , epsilon)
        if len(neibor)>=minPts:  #judge whether the neighborhood is valid
            coreObjs[i] = neibor #save that index into the dictionary

    oldCoreObjs = coreObjs.copy()
    k = 0#initialize cluster number
    un_visited = list(range(n))
    while len(coreObjs)>0:
        OldNotAccess = []
        OldNotAccess.extend(un_visited)

        #get a random core for coreobjects
        cores = coreObjs.keys()
        randNum = random.randint(0,len(cores)-1)
        cores=list(cores)
        core = cores[randNum]


        queue = []
        queue.append(core)
        un_visited.remove(core)

        while len(queue)>0:
            q = queue[0]
            del queue[0]
            if q in oldCoreObjs.keys() :
                delte = [val for val in oldCoreObjs[q] if val in un_visited]# get all the reachable  and unvisited neibors in delte
                queue.extend(delte) #treat them as the next core
                un_visited = [val for val in un_visited if val not in delte]#update the unvisited list

        #add the cluster number
        k += 1
        C[k] = [val for val in OldNotAccess if val not in un_visited]
        for x in C[k]:
            if x in coreObjs.keys():
                del coreObjs[x]
    return C
def get_floor(path,epsilon,minpoints):
    '''

    :param path:   the folder where the camera_poses.csv file lies
    :param epsilon:
    :param minpoints:
    :return:   a list of triple. Each of them represent the coordinate of one floor and the maximum height
    '''
    dataSet,position=read_camera_pose(path+"camera_poses.csv")
    dic = DBSCAN(dataSet, epsilon, minpoints)
    print('number of floor:', len(dic))
    output=[]
    for key,value in dic.items():
        sorted_value=sorted(value,key=lambda x: dataSet[x][0])
        median_index=sorted_value[len(sorted_value)//2]    #get the median index whose value would be the floor height
        floor_height=dataSet[median_index]
        height_max=dataSet[sorted_value[-1]]
        height_min=dataSet[sorted_value[0]]
        floor_indoor_position=position[median_index]
        output.append((floor_indoor_position[0],floor_indoor_position[1],floor_height[0],height_max[0],height_min[0]))

    return output











if __name__ == "__main__":
    print(get_floor('./',0.3,7))



