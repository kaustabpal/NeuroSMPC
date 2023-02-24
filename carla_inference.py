#!/usr/bin/env python

# from carla_env.carla_env_wpts_obs import CarEnv
from carla_env.carla_env_expts import CarEnv

import numpy as np
import matplotlib.pyplot as plt
import cv2

import time
import signal
import os

from planners.LocalPlanner import LocalPlanner

from datetime import datetime
import pickle

from pprint import pprint

np.set_printoptions(precision=3, suppress=True)

EXPT_NAME = "NuroMPPI_temporal_2-1"

dataset_dir = "data/experiments/" + EXPT_NAME + "/"
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
        os.makedirs(dataset_dir + "data", exist_ok=True)
        os.makedirs(dataset_dir + "planner", exist_ok=True)
        os.makedirs(dataset_dir + "run_info", exist_ok=True)
        os.makedirs(dataset_dir + "god_view", exist_ok=True)
        
        os.system('scp -r ' + temp_dir + "/* " + dataset_dir)

    print("Ok....exiting now")
    exit(1)

signal.signal(signal.SIGINT, handler)

def run():
    # Setting up planner
    planner_type = EXPT_NAME.split("_")[0]
    expt_type = EXPT_NAME.split("_")[1]
    expt_ver = EXPT_NAME.split("_")[2]

    print("Using planner: ", planner_type)
    planner = LocalPlanner(planner=planner_type, expt_type = expt_type)
    
    os.system('rm -rf ' + temp_dir)

    os.makedirs(temp_dir + "bev", exist_ok=True)
    os.makedirs(temp_dir + "data", exist_ok=True)
    os.makedirs(temp_dir + "planner", exist_ok=True)
    os.makedirs(temp_dir + "run_info", exist_ok=True)
    os.makedirs(temp_dir + "god_view", exist_ok=True)
    
    config_file = "config/expt_" + expt_ver + "_config.json"
    env = CarEnv(config_file)
    obs = env.reset()
    
    x = input("Press s to start")
    if x == 's':
        print("Starting the simulation")
    else:
        print("Exiting")
        exit(1)

    i = 0

    ego_path = []
    ego_velocities = []
    target_velocities = []
    controls = []
    compute_times = []
    collisions = []

    temp_best_path = np.array([[0, 0] for i in range(100)])
    while True:
        print("Loop ", i)
        obstacle_array = env.obstacle_bev
        global_path = env.next_g_path
        current_speed = env.ego_speed
        bev = env.bev
        
        tic = time.time()
        best_path, best_controls, status = planner.generate_path(obstacle_array, global_path, current_speed)
        toc = time.time()
        compute_time = toc - tic
        compute_times.append(compute_time)
        
        if status != -1:
            best_path = np.array(best_path)
            best_controls = np.array(best_controls)

            # swap first and second column of best_path
            best_path[:,[0, 1]] = best_path[:,[1, 0]]

            target_speed = best_controls[0,0]
            ego_pose = env.ego_pose
            ego_path.append(ego_pose)   # [x,y,z,yaw,pitch,roll]
            target_velocities.append(target_speed)
            ego_velocities.append(current_speed)
        
            best_path = best_path[5:, :]    
        else:
            best_path = None
            target_speed = -1
        
        obs, reward, done, state, action = env.step(best_path, target_speed=target_speed)
        env.render()

        controls.append(action)
        
        collisions.append(0)
        if done:
            collisions.pop()
            collisions.append(1)
            print("Collision detected - Exiting")
            break

        if env.bev is None:
            continue

        bev = env.bev
        obstable_array = env.obstacle_bev
        g_path = env.next_g_path
        speed = env.ego_speed
        left_lane = env.left_lane_coords
        right_lane = env.right_lane_coords
        dyn_obs = env.dyn_obs_poses
        num_obs = len(env.dyn_obs_poses)
        god_view = env.third_person_view

        data = {
            "obstable_array": obstable_array,
            "g_path": g_path,
            "speed": speed,
            "left_lane": left_lane,
            "right_lane": right_lane,
            "dyn_obs": dyn_obs,
            "num_obs": num_obs
        }
        
        run_info = {
            "ego_path": ego_path,
            "ego_velocities": ego_velocities,
            "target_velocities": target_velocities,
            "controls": controls,
            "compute_times": compute_times,
            "collisions": collisions
        }

        if save_data:
            file_name = temp_dir + "data/data_" + str(i).zfill(2) + ".pkl"
            with open(file_name, "wb") as f:
                pickle.dump(data, f)

            file_name = temp_dir + "bev/bev_" + str(i).zfill(2) + ".jpg"
            cv2.imwrite(file_name, bev)

            file_name = temp_dir + "planner/plot_" + str(i).zfill(2) + ".jpg"
            planner.save_plot(file_name)

            file_name = temp_dir + "run_info/run_info.pkl"
            with open(file_name, "wb") as f:
                pickle.dump(run_info, f)

            file_name = temp_dir + "god_view/god_view_" + str(i).zfill(2) + ".jpg"
            cv2.imwrite(file_name, god_view)

        pprint(planner.time_info)

        i += 1
    
    handler(1,1)


if __name__=='__main__':
    run()


