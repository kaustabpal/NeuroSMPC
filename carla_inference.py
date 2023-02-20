#!/usr/bin/env python

# from carla_env.carla_env_wpts_obs import CarEnv
from carla_env.carla_env_expts import CarEnv

import numpy as np
import matplotlib.pyplot as plt
import cv2

import time
import signal
import os

from DeXBee import DeXBee

from datetime import datetime
import pickle

now = datetime.now()

dataset_dir = "data/experiments/" + now.strftime("%m-%d-%Y_%H:%M:%S") + "/"
temp_dir = "data/temp/"

save_data = True

def handler(signum, frame):
    if not save_data:
        print("Ok....exiting now")
        exit(1)
    msg = "Ctrl-c was pressed. Do you want to save? y/n "
    print(msg)
    res = input()
    if res == 'y':
        print("Saving the current data to ", dataset_dir)
        os.makedirs(dataset_dir + "bev", exist_ok=True)
        os.makedirs(dataset_dir + "storm", exist_ok=True)
        os.makedirs(dataset_dir + "dexbee", exist_ok=True)

        os.system('scp -r ' + temp_dir + " " + dataset_dir)

    os.system('rm -rf ' + temp_dir)

    print("Ok....exiting now")
    exit(1)

signal.signal(signal.SIGINT, handler)

def run():
    # Setting up planner
    planner = DeXBee()
    
    os.makedirs(temp_dir + "bev", exist_ok=True)
    os.makedirs(temp_dir + "storm", exist_ok=True)
    os.makedirs(temp_dir + "dexbee", exist_ok=True)

    # Setting up Carla
    print("setting up Carla-Gym")
    env = CarEnv('env_config.json')
    print("Starting loop")
    obs = env.reset()
    
    i = 0
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


        if env.bev is None:
            continue

        bev = env.bev
        obstable_array = env.obstacle_bev
        g_path = env.next_g_path
        speed = env.speed_ego
        left_lane = env.left_lane_coords
        right_lane = env.right_lane_coords
        dyn_obs = env.dyn_obs_poses
        num_obs = len(env.dyn_obs_poses)

        data = {
            "obstable_array": obstable_array,
            "g_path": g_path,
            "speed": speed,
            "left_lane": left_lane,
            "right_lane": right_lane,
            "dyn_obs": dyn_obs,
            "num_obs": num_obs
        }
        
        if save_data:
            file_name = temp_dir + "storm/data_" + str(i).zfill(2) + ".pkl"
            with open(file_name, "wb") as f:
                pickle.dump(data, f)

            file_name = temp_dir + "bev/bev_" + str(i).zfill(2) + ".jpg"
            cv2.imwrite(file_name, bev)

            file_name = temp_dir + "dexbee/plot_" + str(i).zfill(2) + ".jpg"
            planner.save_plot(file_name)            

        i += 1
        # cv2.imshow("BEV", bev)
        # q = cv2.waitKey(100)
        # if q == ord('q'):
        #     break


if __name__=='__main__':
    run()


