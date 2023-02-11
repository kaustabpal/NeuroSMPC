from matplotlib import pyplot as plt

from matplotlib import pyplot as plt
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import pickle
import os

def run():
    np.set_printoptions(suppress=True)
    dataset_dir = "storm/"
    npy_dataset_dir = "storm_npy/"
    files = os.listdir(dataset_dir)

    for i in range(len(files)):
        obs_pos = []
        file_name = dataset_dir + "data_" + str(i) + ".pkl"
        with open(file_name, "rb") as f:
            data = pickle.load(f)

        obstable_array = data["obstable_array"]
        # print(obstable_array[:,:,2])
        g_path = data["g_path"]
        print(obstable_array[:,:,0].shape)
        for j in range(obstable_array[:,:,0].shape[0]):
            for k in range(obstable_array[:,:,0].shape[1]):
                if(obstable_array[j,k,0] ==255):
                    obs_pos.append([j,k])
                    
        np.savez_compressed(npy_dataset_dir + "data_" + str(i),obstacles = obs_pos, g_path=g_path )

        # For Aditya
        # write a function f(obstacle_array, g_path) that returns frenet_obs_array, frenet_g_path
        
        # For Kaustab
        # Run STORM on frenet_obs_array, frenet_g_path
        # Write center_line cost function
        # Do hierarchical elimination
        
        # For Aditya
        # Write a function that takes STORM trajectory and converts it into g_path frame
        
        
        # print(g_path.shape)
        # print(g_path)
        # quit()
        # plt.imshow(obstable_array[:,:,0])
        # plt.show()
        # plt.imshow(obstable_array[:,:,1])
        # plt.show()
        # plt.imshow(obstable_array[:,:,2])
        # plt.show()
        # quit()


if __name__=='__main__':
    run()
