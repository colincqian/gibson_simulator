# Application of GIBSON ENVIRONMENT for robot motion planning

Thanks to the gibson environment developed by StanfordVL, we are able to use that to solve robot motion planning problem.

## Dataset
----
You can down load the dataset from https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md. The folder ' model ' contains the 
tiny dataset of gibson environment


## Map generation
-----
The file `generator.py` can display and generate the maps of different floors for all the model. The map will by default saved under the folder of
`./[model name]/[floori]`  (i is the floor number)

You can run the demo and display the map by the following command:

       python generator.py              
    
    
The file `map.py`will serve to generate the map for customized number of maps. You can use the following command:

        python map.py 1 3 20             
    
 To generate the map for 20 models with 1 superposition of binary map and 3 superpostion of raw map.
 You will find 3 maps in your `  <model name>/floor<i>  ` folder and a `  map_para.txt    `file containing the map parameters for future use.
 
 
 #### parameter explanation :
    argv[1]: The number of superposition for binary map,more superposition means better remove the object indoor
    argv[2]: The number of superposition for raw map, more superpostion means higher accuracy to describe the model
    argv[3](optional): Determining how many models is required to generate maps. The default would run through all the models
    
 ## Image render
 -----
 File `rgb_generator_alpha` is used to rendered the rgb image and depth image. For convience, we only consider the case when using the docker
 provided by the gibson environment. You can follow the QUICK INSTALLATION(DOCKER) on https://github.com/StanfordVL/GibsonEnv. After going through
 the installation on that website, you can start the docker by the following command:
 
      xhost +local:root   
      docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v <host path to model folder>:/root/mount/gibson/gibson/assets/dataset gibson
 
 Then you will get an interactive shell. You have to move ` rgb_generator_alpha.py ` to the dir of the model before generating the 
 image.You can use the following command:
 

        pyhton rgb_generator_alpha.py 50 100                   
  
  To generate the image with 50 trajectory for each model and 100 actions in each trajectory. The rgb images and depth images are saved
  under the folder named after the starting point of the trajectory in the ` ./model/<model name>/data ` folder with a ` information.txt `
  file documenting the state information. 
  
  #### parameter explanation :
    argv[1]: The number of trajectory in each map.
    argv[2]: The number of steps need to take in each trajectory
 
    
  

    








