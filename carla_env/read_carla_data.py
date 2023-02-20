#!/usr/bin/env python

from carla_env.carla_env_controls import CarEnv
import numpy as np

import time

import cv2
import matplotlib.pyplot as plt

import pickle
import os

def run():
    np.set_printoptions(suppress=True)
    dataset_dir = "/Users/kaustabpal/Downloads/02-17-2023_23-21-26" #"/scratch/parth.shah/carla_manual/02-10-2023_04:28:41"
    
    files = os.listdir(dataset_dir)
    
    print(files)

    i = 0
    while True:
        file_name = dataset_dir + "/storm/data_" + str(i) + ".pkl"
        with open(file_name, "rb") as f:
            data = pickle.load(f)

        obstable_array = data["obstable_array"]
        g_path = data["g_path"]
        speed = data["speed"]
        # print(type(g_path), g_path.shape)
        print(speed)
        print(data["speed"])
        quit()
        
        obstacle_array = np.reshape(obstable_array, (256,256,1))
        # print(obstable_array.shape)
        # exit()
        cv2.imshow("obstacle_array", obstacle_array)
        cv2.waitKey(10)
        # plt.imshow(obstable_array)
        # plt.show()
        i += 1
    # plt.imshow(g_path)
    # plt.show()


if __name__=='__main__':
    run()
