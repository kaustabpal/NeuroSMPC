#!/usr/bin/env python

from carla.carla_env_wpts import CarEnv

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
        print("Loop - ", i)

        bev = env.obstacle_bev
        global_path = env.next_g_path
        current_speed = env.speed_ego

        best_path, best_controls = planner.generate_path(bev, global_path, current_speed)
        
        best_path = np.array(best_path)
        best_controls = np.array(best_controls)

        target_speed = best_controls[0,0]
        print("Best path: ", best_path)
        print("Best path length: ", len(best_path))
        print("Best Path shape: ", best_path.shape)

        print("----------------------------------")

        print("Best controls: ", best_controls)
        print("Best controls length: ", len(best_controls))
        print("Best controls shape: ", best_controls.shape)

        cv2.imshow("BEV", bev)
        cv2.waitKey(0)
        
        exit(1)
        obs, reward, done, info = env.step(best_path, target_speed=target_speed)
        env.render()

        bev = env.bev
        obstable_array = env.obstacle_bev
        g_path = env.next_g_path
        speed = env.speed_ego

        cv2.imshow("BEV", bev)
        q = cv2.waitKey(100)
        if q == ord('q'):
            break

        i+=1 


if __name__=='__main__':
    run()


