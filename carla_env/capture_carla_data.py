#!/usr/bin/env python

from carla_env_controls import CarEnv


import numpy as np
import matplotlib.pyplot as plt
import cv2

import pickle

import time
import os

import rospy
from geometry_msgs.msg import Twist

from datetime import datetime

action = [0., 0.]

import signal
import time

now = datetime.now()
dataset_dir = "/scratch/parth.shah/carla_manual/" + now.strftime("%m-%d-%Y_%H:%M:%S") + "/"


temp_dir = "/scratch/parth.shah/temp/"

def handler(signum, frame):
    msg = "Ctrl-c was pressed. Do you want to save? y/n "
    print(msg)
    res = input()
    if res == 'y':
        print("Saving the current data to ", dataset_dir)
        os.system('scp -r ' + temp_dir + " " + dataset_dir)

    os.system('rm -rf ' + temp_dir)

    print("Ok....exiting now")
    exit(1)

signal.signal(signal.SIGINT, handler)

def teleop_callbak(data):
    global action
    steer = -data.angular.z
    vel = data.linear.x
    
    # print("Steer: ", steer, "Vel: ", vel)

    steer = np.clip(steer / 30, -1, 1) 

    action = [vel, steer]

def run():

    temp_dir = "/scratch/parth.shah/temp/"
    os.makedirs(temp_dir + "bev", exist_ok=True)
    os.makedirs(temp_dir + "storm", exist_ok=True)

    # os.makedirs(dataset_dir + "bev", exist_ok=True)
    # os.makedirs(dataset_dir + "storm", exist_ok=True)

    print("setting up Carla-Gym")
    env = CarEnv()

    print("Starting loop")
    obs = env.reset()
    i = 0
    
    times = []
    while not rospy.is_shutdown():
        print("Loop - ", i)

        # action = [0.3, -0.02]
        global action
        obs, reward, done, info = env.step(action)
        env.render()

        ### Testing BEV
        # tic = time.time()
        bev = env.bev
        obstable_array = env.obstacle_bev
        g_path = env.next_g_path
        speed = env.speed_ego

        data = {
            "obstable_array": obstable_array,
            "g_path": g_path,
            "speed": speed
        }

        with open(temp_dir + "storm/data_" + str(i) + ".pkl", "wb") as f:
            pickle.dump(data, f)

        file_name = temp_dir + "bev/bev_{}.jpg".format(i)

        cv2.imwrite(file_name, bev)

        cv2.imshow("BEV", bev)
        cv2.waitKey(100)

        i+=1 
        # if done: break


if __name__=='__main__':
    rospy.init_node('talker', anonymous=True)

    rospy.Subscriber("/cmd_vel", Twist, teleop_callbak)

    run()


