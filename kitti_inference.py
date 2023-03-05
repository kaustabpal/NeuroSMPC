#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import argparse
import os
import yaml
import signal

from kitti.laserscan import LaserScan, SemLaserScan
from kitti.utils import parse_calibration, parse_poses

from planners.LocalPlanner import LocalPlanner

np.set_printoptions(precision=3, suppress=True)

EXPT_NAME = "NuroMPPI_3-4"

dataset_dir = "data/kitti_inference/"
temp_dir = "data/temp/"

save_data = True

def handler(signum, frame):
    # if not save_data:
    #     print("Ok....exiting now")
    #     exit(1)
    # msg = "Ctrl-c was pressed. Do you want to save? y/n "
    # print(msg)
    # res = input()
    # if res == 'y':
    #     print("Saving the current data to ", dataset_dir)
    #     os.makedirs(dataset_dir + "bev", exist_ok=True)
    #     os.makedirs(dataset_dir + "data", exist_ok=True)
    #     os.makedirs(dataset_dir + "planner", exist_ok=True)
    #     os.makedirs(dataset_dir + "run_info", exist_ok=True)
    #     os.makedirs(dataset_dir + "god_view", exist_ok=True)
        
    #     os.system('scp -r ' + temp_dir + "/* " + dataset_dir)

    # print("Ok....exiting now")
    exit(1)

signal.signal(signal.SIGINT, handler)

class KittiInfernce:
    def __init__(self, dataset_path, sequence_num, config_path):
        self.dataset_path = dataset_path
        self.sequence_num = sequence_num

        self.CFG = yaml.safe_load(open(config_path, 'r'))

        scan_paths = os.path.join(self.dataset_path, "sequences",
                            self.sequence_num, "velodyne")
        if os.path.isdir(scan_paths):
            print("Sequence folder exists! Using sequence from %s" % scan_paths)
        else:
            print("Sequence folder doesn't exist! Exiting...")
            quit()

        self.scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(scan_paths)) for f in fn]
        self.scan_names.sort()

        label_paths = os.path.join(self.dataset_path, "sequences",
                            self.sequence_num, "labels")
        if os.path.isdir(label_paths):
            print("Labels folder exists! Using labels from %s" % label_paths)
        else:
            print("Labels folder doesn't exist! Exiting...")
            quit()        
        self.label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_paths)) for f in fn]
        self.label_names.sort()

        self.color_dict = self.CFG["color_map"]
        self.nclasses = len(self.color_dict)
        self.scan = SemLaserScan(self.nclasses, self.color_dict, project=True)

        calib_file = os.path.join(self.dataset_path, "sequences",
                            self.sequence_num, "calib.txt")
        self.calibration = parse_calibration(calib_file)

        poses_file = os.path.join(self.dataset_path, "sequences",
                            self.sequence_num, "poses.txt")
        self.poses = np.array(parse_poses(poses_file, self.calibration))
        
        self.data_length = self.poses.shape[0]
        self.offset = 0

    def read_scan(self, offset):
        self.scan.open_scan(self.scan_names[offset])
        self.scan.open_label(self.label_names[offset])
        self.scan.colorize()

    def generate_occ_map(self, img_size, sensing_range, z_max):
        assert img_size[0] == img_size[1], "BEV should be square"
        scale = img_size[0] / (2 * sensing_range)
        
        scan_pts = np.copy(self.scan.points)
        scan_labels = np.copy(self.scan.sem_label)

        # correcting axes-directions
        scan_pts[:, 0] = -scan_pts[:, 0]
        scan_pts[:, 1] = -scan_pts[:, 1]


        road_mask = (scan_labels == 40) | (scan_labels == 60) | (scan_labels == 49) | (scan_labels == 1)
        obs_pts = scan_pts[np.bitwise_not(road_mask)]

        x_mask = np.abs(obs_pts[:, 0]) < sensing_range
        y_mask = np.abs(obs_pts[:, 1]) < sensing_range
        z_mask = obs_pts[:, 2] < z_max
        vicinity_mask = x_mask & y_mask & z_mask
        obs_pts_near = obs_pts[vicinity_mask]

        obs_pts_near_coords = obs_pts_near[:, :2] * scale + img_size[0] // 2
        obs_pts_near_coords = np.clip(np.intp(obs_pts_near_coords), 0, 255)

        occupancy_map = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        occupancy_map[obs_pts_near_coords[:, 0], obs_pts_near_coords[:, 1]] = 255

        return occupancy_map

    def fetch_data(self, offset):
        print("Offset - ", offset)
        self.read_scan(offset)
        img_size = (256, 256)
        sensing_range = 15
        scale = img_size[0] / (2* sensing_range)

        occ_map = self.generate_occ_map(img_size, sensing_range, 0)

        global_path = []

        i, dist = 0, 0
        curr_pose = self.poses[offset]
        # print(np.linalg.inv(curr_pose))

        xy = [curr_pose[0, 3], curr_pose[1, 3]]
        # global_path.append(xy)
        while dist < 30:
            pose = self.poses[offset + i]   
            next_xy = [pose[0, 3], pose[1, 3]]

            relative_dist = np.linalg.norm(np.array(xy) - np.array(next_xy))
            if relative_dist > 1:
                dist += relative_dist
                xy = next_xy
                
                pose_transformed = np.linalg.inv(curr_pose) @ pose

                rel_xy = [pose_transformed[0, 3], pose_transformed[1, 3]]
                global_path.append(rel_xy)

            i += 1

        global_path = np.array(global_path)
        global_path[:, 1] = - global_path[:, 1]
        global_path_in_image_scale = global_path*scale + img_size[0]//2

        return occ_map, global_path_in_image_scale

    def step(self):
        offset = self.offset
        self.offset += 1

        print("Step - ", offset)
        return self.fetch_data(offset)

def run():
    planner_type = EXPT_NAME.split("_")[0]
    expt_ver = EXPT_NAME.split("_")[1]
    os.makedirs(dataset_dir + "planner", exist_ok=True)
    
    print("Using planner: ", planner_type)
    planner = LocalPlanner(planner=planner_type)
 
    kitti = KittiInfernce("/scratch/kitti-semantic/", "00", "config/semantic-kitti.yaml")
    kitti.offset = 0
    for i in range(kitti.data_length):
        occ_map, global_path = kitti.step()

        print(global_path)
        planner.generate_path(occ_map, global_path, 0)

        file_name = dataset_dir + "planner/plot_" + str(i).zfill(2) + ".png"
        planner.save_plot(file_name)

if __name__ == "__main__":
    run()