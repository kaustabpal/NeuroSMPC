#!/usr/bin/env python

import rospy

import numpy as np
import cv2
from matplotlib import pyplot as plt

import torch
from torch import nn

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Twist, PointStamped

from sensor_msgs.point_cloud2 import read_points_list

from lli_msgs.msg import CarState

import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler, euler_from_matrix, quaternion_matrix
from car.road_net import MLP
from car.transforms import build_se3_transform, get_rpy_from_odom_orientation
from car.transforms import T_cinit_map, T_vel_cam

from DeXBee import DeXBee

import time
import pickle
from datetime import datetime
import os

import pdb

class LocalPlanner:
    def __init__(self):
        # self.rate = rospy.Rate(1)

        self.gpath_received = False
        self.global_path = None
        self.gpath = None
        self.obstacle_cloud = None

        self.log_flags = {
            'no_gpath' : False,
            'tf_error' : False,
            'target_reached' : False
        } 

        self.subscribers_init()
        self.publishers_init()

        self.num_wpts_gpath = 100 

        self.model = MLP()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = "/home/aditya/Documents/dexbee_car/model_weights/road-net/mlp_model.state_dict"
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))
        self.model.to(self.device)

        self.ego_dim = [1.8, 1.8]
        self.img_size = [256, 256]

        self.car_state = None
        self.car_state_ts = None

        self.pcd = None
        self.pcd_ts = None
        self.pcd_ts_ros = None

        self.odom = None
        self.odom_ts = None

        self.save_data = False
        self.save_dir = "/home/aditya/Documents/dexbee_car/data/" + datetime.now().strftime("%m-%d-%Y_%H:%M:%S") + "/"
        if self.save_data:
            os.makedirs(self.save_dir + "storm", exist_ok=True)
            os.makedirs(self.save_dir + "bev", exist_ok=True)
            os.makedirs(self.save_dir + "plot", exist_ok=True)

        self.dexbee = DeXBee()

        self.visualize = False
        if self.visualize:
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5))

    def subscribers_init(self):
        rospy.loginfo("Creating LP Subscribers")
        self.global_plan_sub = rospy.Subscriber(
            '/global_planner/plan', Path, self.global_path_callback)
        
        # self.obstacle_cloud_listener = rospy.Subscriber('/obstacle_cloud', PointCloud2, callback=self.obstacle_cloud_callback)
        self.pcd_listener = rospy.Subscriber('/velodyne_points', PointCloud2, callback=self.pcd_callback)

        self.odom_listener = rospy.Subscriber('/odom', Odometry,
                         self.odom_aftmap_callback)

        self.state_listener = rospy.Subscriber('/ll_state', CarState, self.car_state_callback)

        self.tf_listener = tf.TransformListener()

    def publishers_init(self):
        rospy.loginfo("Creating LP Publishers")
        self.waypoints_pub = rospy.Publisher(
            "/local_planner/waypoints", Path, queue_size=1, latch=True)
        self.waypoints_info_pub = rospy.Publisher(
            "/local_planner/info", Twist, queue_size=1, latch=True)

    def global_path_callback(self, global_path):
        rospy.loginfo("received new global path")
        self.global_path = global_path
        gpath = []
        for wpt_ in enumerate(global_path.poses):
            wpt = wpt_[1]
            gpath.append([wpt.pose.position.x, wpt.pose.position.y, euler_from_quaternion((wpt.pose.orientation.x, wpt.pose.orientation.y, wpt.pose.orientation.z, wpt.pose.orientation.w))[2]])
        self.gpath = np.array(gpath)
        self.gpath_received = True

    def pcd_callback(self, pcd_msg):
        self.pcd = pcd_msg
        self.pcd_ts = time.time()
        self.pcd_ts_ros = pcd_msg.header.stamp
    
    def car_state_callback(self, msg):
        self.car_state = msg

    def odom_aftmap_callback(self,msg):
        self.odom_nomap_seq = msg.header.seq
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        roll, pitch, yaw = get_rpy_from_odom_orientation(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        odom_tf = build_se3_transform([position.x, position.y, position.z, roll, pitch, yaw])
        self.odom = odom_tf
        self.odom_ts = time.time()

    def publish_local_plan(self, path_x, path_y, path_yaw):
        waypoints = Path()

        waypoints.header = Header()
        waypoints.header.stamp = rospy.Time.now()
        waypoints.header.frame_id = 'map'
 
        for idx in range(len(path_x)):
            wpt = PoseStamped()
            wpt.header = waypoints.header
            
            wpt.pose.position.x = path_x[idx]
            wpt.pose.position.y = path_y[idx]

            wpt.pose.orientation.x, wpt.pose.orientation.y, wpt.pose.orientation.z, wpt.pose.orientation.w = quaternion_from_euler(0, 0, path_yaw[idx])

            waypoints.poses.append(wpt)
        
        self.waypoints_pub.publish(waypoints)

    def publish_local_plan_info(self, safe_speed, curvature, nearest_obstacle_dist):
        waypoints_info = Twist()
        waypoints_info.linear.x = safe_speed
        waypoints_info.angular.z = curvature
        waypoints_info.angular.x = nearest_obstacle_dist
        self.waypoints_info_pub.publish(waypoints_info)

    def planning_loop(self):
        i = 0
        tic, toc = None, None
        while not rospy.is_shutdown():
            toc = time.time()
            if tic is not None and toc - tic > 0.00001:
                print("---")
                print("Planning time: {}".format(toc - tic))
                print("--------------------------------"*2)
            tic =  time.time()

            if not self.gpath_received:
                if not self.log_flags['no_gpath']:
                    rospy.loginfo("No global path received yet")
                    self.log_flags['no_gpath'] = True
                # rospy.loginfo("No global path received yet")
                self.log_flags['no_gpath'] = False 
                continue

            if self.pcd_ts is None or (time.time() - self.pcd_ts) > 0.5:
                # rospy.loginfo("no obstacle cloud {}".format( time.time() - self.pcd_ts if self.pcd_ts is not None else 0))
                continue
            
            if self.odom_ts is None or (time.time() - self.odom_ts) > 0.6:
                # rospy.loginfo("no odom - {}".format(time.time() - self.odom_ts if self.odom_ts is not None else 0))
                continue
            
            if self.car_state is None:
                # rospy.loginfo("no car state - {}".format(time.time() - self.car_state_ts if self.car_state_ts is not None else 0))
                continue

            try:
                (trans, rot) = self.tf_listener.lookupTransform(
                    'map', 'base_link', rospy.Time(0))
                curr_pose = np.array([ trans[0], trans[1], euler_from_quaternion(rot)[2] ])
                
                pose = [trans, euler_from_quaternion(rot)]

                tf_matrix = np.array(quaternion_matrix(rot))
                trans = np.array(trans)

                tf_matrix[:3, 3] = trans[:3].T  
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                if not self.log_flags['tf_error']:
                    rospy.logwarn('Failed to find current pose')
                    self.log_flags['tf_error'] = False
                continue
            self.log_flags['tf_error'] = False

            global_path = self.global_path
            gpath = self.gpath

            angles = np.abs(gpath[:, 2] - curr_pose[2])
            angles[angles > np.pi] -= 2*np.pi
            angles = np.abs(angles)
            dist = np.linalg.norm(gpath[:, :2] - curr_pose[:2], axis=1)

            actual_closest_wp_idx = np.argmin(dist)
            dist_ = dist[actual_closest_wp_idx]

            dist[angles > 1.57] = np.inf
            closest_wp_idx = np.argmin(dist)
            closest_wp_dist = dist[closest_wp_idx]

            if abs(closest_wp_dist) > 5:
                print("No gpath bug", closest_wp_dist)
                continue

            segmented_gpath = []
            segmented_global_path = []
            
            start_idx = closest_wp_idx
            end_idx = closest_wp_idx + self.num_wpts_gpath if ((closest_wp_idx + self.num_wpts_gpath) < len(gpath)) else len(gpath)
            
            for idx in range(start_idx, end_idx):
                pt = global_path.poses[idx]
                curr_pt = np.array([[pt.pose.position.x, pt.pose.position.y, pt.pose.position.z, 1]]).T              

                transformed_pt = np.linalg.pinv(tf_matrix) @ curr_pt


                segmented_gpath.append([transformed_pt[0, 0], transformed_pt[1, 0], 0])
                
            segmented_gpath = np.array(segmented_gpath)

            # print(segmented_gpath)
            # if len(segmented_global_path) <= 20:
            #     rospy.loginfo("Target reached")
            #     continue

            pcd_points = np.array(read_points_list(self.pcd))[:, :3]
            pcd_points = np.hstack([pcd_points, np.ones((pcd_points.shape[0], 1))])

            pcd_points_old = pcd_points.copy()

            pcd_points = (T_cinit_map@self.odom@T_vel_cam @ pcd_points.T).T

            # obstacle_points = model
            tic_gnd = time.time()
            pcd_tensor = torch.tensor(pcd_points[:, :3], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                network_output = self.model(pcd_tensor)[:,0].detach().cpu()
            toc_gnd = time.time()
            print("Ground segmentation time: {}".format(toc_gnd - tic_gnd))

            road_pcd = pcd_points_old[:,:3][network_output<0.5]
            not_road_pcd = pcd_points_old[:,:3][network_output>=0.5]
            
            # z_mask = not_road_pcd[:, 2] < 0.5
            # obstacle_pcd = not_road_pcd[z_mask]
            obstacle_pcd = not_road_pcd

            if obstacle_pcd.shape[0] == 0:
                rospy.loginfo("No obstacles")
                continue
            
            tic_bev = time.time()
            bev, obstacle_bev, g_path_img = self.generate_bev(obstacle_pcd, segmented_gpath, self.ego_dim, self.img_size, 15, 0)
            ego_speed = self.car_state.velocity
            toc_bev = time.time()
            print("BEV time: {}".format(toc_bev - tic_bev))

            data = {
                "obstable_array": obstacle_bev,
                "g_path": g_path_img,
                "speed": ego_speed
            }

            file_name = None
            if self.save_data:
                with open(self.save_dir + "storm/data_" + str(i) + ".pkl", "wb") as f:
                    pickle.dump(data, f)

                file_name = self.save_dir + "bev/bev_{}.jpg".format(i)
                cv2.imwrite(file_name, bev)

                file_name = self.save_dir + "plot/plot_{}.jpg".format(i)


            if self.visualize:
                self.ax[0].imshow(bev)
                self.ax[1].imshow(obstacle_bev)
                plt.show()

            tic1 = time.time()
            best_path, best_controls = self.dexbee.generate_path(obstacle_bev, g_path_img[:, :2], ego_speed, file_name)
            toc1 = time.time()
            print("dexbee time", toc1 - tic1)
            i += 1 

    def generate_bev(self, obs_pcd, global_path, ego_dim, img_size, obs_range = 15, z_max = 0):
        assert img_size[0] == img_size[1], "BEV should be square"
        
        scale = img_size[0] / (2 * obs_range)
        
        # Creating a blank image
        bev = np.ones([img_size[0], img_size[0], 3], dtype=np.uint8)
        bev[:, :] = [0, 0, 0]

        ### Obstacle Layer
        # finding the points that are not road
        obstacle_pcd = np.copy(obs_pcd)

        # finding the points that are in the vicinity of the ego
        x_mask = np.abs(obstacle_pcd[:, 0]) < obs_range
        y_mask = np.abs(obstacle_pcd[:, 1]) < obs_range
        z_mask = obstacle_pcd[:, 2] < z_max
        vicinity_mask = x_mask & y_mask & z_mask

        obstacle_pcd_near = obstacle_pcd[vicinity_mask]
        obstacle_pcd_near[:, 0] = -obstacle_pcd_near[:, 0]
        obstacle_pcd_near[:, 1] = -obstacle_pcd_near[:, 1]


        # scaling the points to the image size
        obstacle_pcd_near_scaled = obstacle_pcd_near[:, :2]*scale + img_size[0]//2
        obs_pixel_coords = np.clip(np.intp(obstacle_pcd_near_scaled[:, :2]), 0, 255)

        # drawing the points on the image
        bev_obs = np.copy(bev)
        bev_obs.shape
        bev_obs[obs_pixel_coords[:, 0], obs_pixel_coords[:, 1]] = [255, 255, 255]

        '''
        # Dilating the points to make them visible
        kernel = np.ones((5, 5), np.uint8)
        bev_obs = cv2.dilate(bev_obs, kernel, iterations=1)
        '''

        g_path = global_path.copy()
        # Scaling the points to the image size
        g_path_in_image_scale = g_path[:, :2]*scale + img_size[0]//2
 

        ### Ego Layer
        ego_dim_x, ego_dim_y = ego_dim[0], ego_dim[1] # (in meters) # dimension are for tesla model 3 (in Carla)
        
        # Creatng a rectangle for the ego
        rot_rect = ((127, 127), (ego_dim_x * scale, ego_dim_y * scale), 0)
        box = cv2.boxPoints(rot_rect)
        
        # Drawing the rectangle on the image
        bev_ego = np.copy(bev)
        bev_ego = cv2.drawContours(bev_ego, [box.astype(int)], 0, [255, 255, 255], -1)

        # ### GlobalPath Layer
        # global_path_tf = self.global_path_tf
        
        # # Transforming the global path to the ego's frame
        # ego_tf = np.array(self.ego.get_transform().get_matrix())
        # global_path_relative_tf = np.linalg.pinv(ego_tf) @ global_path_tf

        # # Extracting XYZ coords
        # global_path_relative_coords = global_path_relative_tf[:, 0:3, 3]

        # Finding the points that are in the vicinity of the ego
        x_mask = np.abs(g_path[:, 0]) < obs_range
        y_mask = np.abs(g_path[:, 1]) < obs_range
        vicinity_mask = x_mask & y_mask
        global_path_relative_near = g_path[vicinity_mask]

        # # # Correcting Axis-Order
        global_path_relative_near = global_path_relative_near[:, [1, 0]]
        # Flipping Y-Axis
        global_path_relative_near[:, 1] = -global_path_relative_near[:, 1]
        global_path_relative_near[:, 0] = -global_path_relative_near[:, 0]

        # Scaling the points to the image size
        global_path_pixel_coords = global_path_relative_near[:, :2]*scale + img_size[0]// 2
        global_path_pixel_coords = np.clip(np.intp(global_path_pixel_coords[:, :2]), 0, 255)

        # Drawing the path on the image
        bev_global_path = np.copy(bev)
        if global_path_pixel_coords.shape[0] > 1:
            bev_global_path = cv2.polylines(bev_global_path, [global_path_pixel_coords], False, color=(255, 255, 255), thickness=3)

         ### Combining the layers
        obs_mask = bev_obs[:, :, 0] == 255
        ego_mask = bev_ego[:, :, 0] == 255
        global_path_mask = bev_global_path[:, :, 0] == 255

        bev[obs_mask] = [255, 255, 255] # Obstacles - White
        bev[global_path_mask] = [0, 0, 255] # Global Path - Blue
        bev[ego_mask] = [255, 0, 0] # Ego - Red

        obstacle_bev = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        obstacle_bev[obs_mask] = 255 # Obstacles - White
        
        return bev, obstacle_bev, np.array(g_path_in_image_scale)


if __name__ == '__main__':
    rospy.init_node('local_planner', anonymous=True)
    lp = LocalPlanner()
    lp.planning_loop()
