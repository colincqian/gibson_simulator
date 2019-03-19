# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:02:14 2018

@author: colin qian
"""
import matplotlib.pyplot as plt
import sys
import os
import generator


BINARYMAP_SUPERPOSITION=int(sys.argv[1])# more superpostion means bettre removing the object indoor
LINEMAP_SUPERPOSITION=int(sys.argv[2] ) #more super postion here means more accurate description of the map
if len(sys.argv)==3:  #without setting the number of maps
       limited_map=-1             #go through all the model
else:
       limited_map=int(sys.argv[3] )     # set the number of maps

for file in os.listdir(os.getcwd())[::-1]:
    path=file+'/'
    if limited_map==0:
        break
    if os.path.exists(path+'mesh_z_up.obj'):
        print('start generating : '+file)
        generator.main(BINARYMAP_SUPERPOSITION,LINEMAP_SUPERPOSITION,path)        #call the function to generate the map
    else:
        continue
    print('map of '+file+'  generated')
    limited_map=limited_map-1
print('DONE')



