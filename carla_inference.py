#!/usr/bin/env python

from carla_env.carla_env_wpts import CarEnv

import numpy as np
import matplotlib.pyplot as plt
import cv2

import time
import signal
import os

from DeXBee import DeXBee

def run():
    # Setting up planner
    planner = DeXBee()
    
    # Setting up Carla
    print("setting up Carla-Gym")
    env = CarEnv()
    print("Starting loop")
    obs = env.reset()
    
    while True:
        obstacle_array = env.obstacle_bev
        global_path = env.next_g_path
        current_speed = env.speed_ego
        bev = env.bev
        
        tic = time.time()
        best_path, best_controls = planner.generate_path(obstacle_array, global_path, current_speed)
        toc = time.time()

        print("----------------------------------")
        print("Time taken: ", toc-tic)

        best_path = np.array(best_path)
        best_controls = np.array(best_controls)

        # swap first and second column of best_path
        best_path[:,[0, 1]] = best_path[:,[1, 0]]

        target_speed = best_controls[0,0]
        # print("Best path: ", best_path)
        # print("Best path length: ", len(best_path))
        # print("Best Path shape: ", best_path.shape)

        # print("----------------------------------")

        # print("Best controls: ", best_controls)
        # print("Best controls length: ", len(best_controls))
        # print("Best controls shape: ", best_controls.shape)

        # print("Target speed: ", target_speed, best_controls[1,0])

        obs, reward, done, info = env.step(best_path, target_speed=target_speed)
        env.render()

        # scale = 256 / 30
        # # Scaling the points to the image size
        # path_in_image_scale = best_path[:, :2]*scale + 256//2

        # # Plotting the path
        # global_path_relative_near = global_path_relative_near[:, [1, 0]]
        # # Flipping Y-Axis
        # global_path_relative_near[:, 1] = -global_path_relative_near[:, 1]

        # # Scaling the points to the image size
        # global_path_pixel_coords = global_path_relative_near[:, :2]*scale + img_size[0]// 2
        # global_path_pixel_coords = np.clip(np.intp(global_path_pixel_coords[:, :2]), 0, 255)


        bev = env.bev
        obstable_array = env.obstacle_bev
        g_path = env.next_g_path
        speed = env.speed_ego

        # cv2.imshow("BEV", bev)
        # q = cv2.waitKey(100)
        # if q == ord('q'):
        #     break



if __name__=='__main__':
    run()


