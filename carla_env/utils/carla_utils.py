#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Summary of useful helper functions for scenarios
"""

import math
from math import cos, sin
import shapely.geometry
import shapely.affinity
import carla
import numpy as np

from agents.tools.misc import vector
from agents.navigation.local_planner import RoadOption

from locale import currency
from os import POSIX_SPAWN_CLOSE
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import cv2

import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.patches as pt
import matplotlib.collections
import scipy
import time

import carla
import sys

import scipy.io
import math
import random as rnd


def generate_target_waypoint_list_multilane(waypoint, change='left',  # pylint: disable=too-many-return-statements
                                            distance_same_lane=10, distance_other_lane=25,
                                            total_lane_change_distance=25, check=True,
                                            lane_changes=1, step_distance=2):
    """
    This methods generates a waypoint list which leads the vehicle to a parallel lane.
    The change input must be 'left' or 'right', depending on which lane you want to change.

    The default step distance between waypoints on the same lane is 2m.
    The default step distance between the lane change is set to 25m.

    @returns a waypoint list from the starting point to the end point on a right or left parallel lane.
    The function might break before reaching the end point, if the asked behavior is impossible.
    """

    plan = []
    plan.append((waypoint, RoadOption.LANEFOLLOW))  # start position

    option = RoadOption.LANEFOLLOW

    # Same lane
    distance = 0
    while distance < distance_same_lane:
        next_wps = plan[-1][0].next(step_distance)
        if not next_wps:
            return None, None
        next_wp = next_wps[0]
        distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
        plan.append((next_wp, RoadOption.LANEFOLLOW))

    if change == 'left':
        option = RoadOption.CHANGELANELEFT
    elif change == 'right':
        option = RoadOption.CHANGELANERIGHT
    else:
        # ERROR, input value for change must be 'left' or 'right'
        return None, None

    lane_changes_done = 0
    lane_change_distance = total_lane_change_distance / lane_changes

    # Lane change
    while lane_changes_done < lane_changes:

        # Move forward
        next_wps = plan[-1][0].next(lane_change_distance)
        if not next_wps:
            return None, None
        next_wp = next_wps[0]

        # Get the side lane
        if change == 'left':
            if check and str(next_wp.lane_change) not in ['Left', 'Both']:
                return None, None
            side_wp = next_wp.get_left_lane()
        else:
            if check and str(next_wp.lane_change) not in ['Right', 'Both']:
                return None, None
            side_wp = next_wp.get_right_lane()

        if not side_wp or side_wp.lane_type != carla.LaneType.Driving:
            return None, None

        # Update the plan
        plan.append((side_wp, option))
        lane_changes_done += 1

    # Other lane
    distance = 0
    while distance < distance_other_lane:
        next_wps = plan[-1][0].next(step_distance)
        if not next_wps:
            return None, None
        next_wp = next_wps[0]
        distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
        plan.append((next_wp, RoadOption.LANEFOLLOW))

    target_lane_id = plan[-1][0].lane_id

    return plan, target_lane_id

def generate_target_waypoint_list_same_lane(waypoint, distance_same_lane=10, check=True, step_distance=2):
    """
    This methods generates a waypoint list which leads the vehicle to a parallel lane.
    The change input must be 'left' or 'right', depending on which lane you want to change.

    The default step distance between waypoints on the same lane is 2m.
    The default step distance between the lane change is set to 25m.

    @returns a waypoint list from the starting point to the end point on a right or left parallel lane.
    The function might break before reaching the end point, if the asked behavior is impossible.
    """

    plan = []
    plan.append((waypoint, RoadOption.LANEFOLLOW))  # start position

    option = RoadOption.LANEFOLLOW

    # Same lane
    distance = 0
    while distance < distance_same_lane:
        next_wps = plan[-1][0].next(step_distance)
        if not next_wps:
            break
        next_wp = next_wps[0]
        distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
        plan.append((next_wp, RoadOption.LANEFOLLOW))

    return plan, None

def get_actor_control(actor):
    """
    Method to return the type of control to the actor.
    """
    control = actor.get_control()
    actor_type = actor.type_id.split('.')[0]
    if not isinstance(actor, carla.Walker):
        control.steering = 0
    else:
        control.speed = 0

    return control, actor_type


def get_distance(actor1, actor2):
    """
    Get distance between 2 vehicles
    """    
    dist = actor1.get_transform().location - actor2.get_transform().location
    dist = math.sqrt(dist.x ** 2 + dist.y ** 2)
    return dist


def get_velocity(actor):
    """
    Get distance between 2 vehicles
    """    
    dist = actor.get_velocity()
    dist = math.sqrt(dist.x ** 2 + dist.y ** 2)
    return dist

def get_lane(actor):
    """
    Get lane of the actor
    """    
    wp = self.world.get_map().get_waypoint(actor.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
    return wp

# """
#     Get distance between 2 vehicles
# """
# def get_distance(actor1, actor2):
#     dist = actor1.get_transform().location - actor2.get_transform().location
#     dist = math.sqrt(dist.x ** 2 + dist.y ** 2)
#     return dist


def get_controls(obs, wp, target_waypts, target_speed = 5.0):
    arr = []
    for waypts in target_waypts:
        dist = waypts[0].transform.location - wp.transform.location
        dist = math.sqrt(dist.x ** 2 + dist.y ** 2)
        arr.append(dist)
    minind = np.argmin(arr)
    velocity = carla.Vector3D(0, 0, 0)
    angular_velocity = carla.Vector3D(0, 0, 0)
    if minind < len(arr) - 1:
        next_location = target_waypts[minind + 1][0].transform.location
        current_speed = math.sqrt(obs.get_velocity().x**2 + obs.get_velocity().y**2)
        direction = next_location - obs.get_location()
        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
        velocity.x = direction.x / direction_norm * target_speed
        velocity.y = direction.y / direction_norm * target_speed
        print(velocity.x, velocity.y, " target velocity")
        
        # set new angular velocity
        current_yaw = obs.get_transform().rotation.yaw
        delta_yaw = math.degrees(math.atan2(direction.y, direction.x)) - current_yaw

        if math.fabs(delta_yaw) > 360:
            delta_yaw = delta_yaw % 360

        if delta_yaw > 180:
            delta_yaw = delta_yaw - 360
        elif delta_yaw < -180:
            delta_yaw = delta_yaw + 360

        if target_speed == 0:
            angular_velocity.z = 0
        else:
            angular_velocity.z = delta_yaw / (direction_norm / target_speed)

    return velocity, angular_velocity



def compute_ellipse_angles(csm,x_obs_sort,y_obs_sort,psi_obs_sort):
    total_obs = len(x_obs_sort)
    psi_obs = np.zeros(total_obs)

    k=0
    for i in range(0,total_obs):
        vec1 = (x_obs_sort[i] - csm.vehicle.get_location().x,y_obs_sort[i]-csm.vehicle.get_location().y)
        vec2 = (csm.vehicle.get_velocity().x,csm.vehicle.get_velocity().y)
        theta = np.arccos(np.clip(np.dot(vec1,vec2), -1.0, 1.0))
        if (theta<=np.pi/2):
            psi_obs[k] = math.radians(psi_obs_sort[i])
            psi_obs[k] = np.arctan2(np.sin(psi_obs_sort[k]),np.cos(psi_obs_sort[k]))
            k=k+1

    psi_obs = psi_obs[0:k]
    return psi_obs

def apply_control(w,v,prob, csm,target_acc,prev_vel,prev_acc,throttle1,steer):

    physics_control = csm.vehicle.get_physics_control()

    max_steer_angle_list = []
    for wheel in physics_control.wheels:
        max_steer_angle_list.append(wheel.max_steer_angle)
    max_steer_angle = max(max_steer_angle_list)*np.pi/180

    vel = csm.vehicle.get_velocity()
    vel = (vel.x**2 + vel.y**2 + vel.z**2)**0.5

    throttle_lower_border = -(0.01*9.81*physics_control.mass + 0.5*0.3*2.37*1.184*vel**2 + \
        9.81*physics_control.mass*np.sin(csm.vehicle.get_transform().rotation.pitch*2*np.pi/360))/physics_control.mass

    brake_upper_border = throttle_lower_border + -500/physics_control.mass
    csm.pid.setpoint = target_acc

    acc = (vel - prev_vel)/prob.t

    if acc>10:
        control = csm.pid(0)
    else:
        prev_acc = (prev_acc*4 + acc)/5
        control = csm.pid(prev_acc)

    # steer = np.arctan(w*prob.wheel_base/v)
    steer = steer/max_steer_angle
    throttle = 0
    brake = 0

    throttle1 = np.clip(throttle1 + control,-4.0, 4.0)

    if throttle1>throttle_lower_border:
        throttle = (throttle1-throttle_lower_border)/4
        brake = 0
    elif throttle1> brake_upper_border:
        brake = 0
        throttle = 0
    else:
        brake = (brake_upper_border-throttle1)/4
        throttle = 0

    brake = np.clip(brake, 0.0, 1.0)
    throttle = np.clip(throttle, 0.0, 1.0)
    csm.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake)))
   
    prev_vel = vel
    return prev_vel,prev_acc,throttle1,acc

def get_obs_from_gt(csm, prob, x_path_data, y_path_data):
    start_position = csm.vehicle.get_location()
    x_global_init = start_position.x#x_path_data[idx]
    y_global_init = start_position.y#y_path_data[idx]
    total_obs = len(csm.obs_list)
    y_obs = np.zeros(total_obs)
    x_obs = np.zeros(total_obs)
    v_obs = 10.0*np.asarray(np.random.uniform(0, 2, total_obs))
    psi_obs = 0.0*np.asarray(np.random.uniform(0, 2, total_obs))

    k=0
    for j,vehicle in enumerate(csm.obs_list):
        vec1 = (vehicle.get_location().x - start_position.x,vehicle.get_location().y-start_position.y)
        vec2 = (x_path_data[1]-start_position.x,y_path_data[1]-start_position.y)
        theta = np.arccos(np.clip(np.dot(vec1,vec2), -1.0, 1.0))
        if (theta<=np.pi/2):
            x_obs[k] = vehicle.get_location().x 
            y_obs[k] = vehicle.get_location().y
            v_obs[k] = np.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2)
            psi_obs[k] = math.radians(vehicle.get_transform().rotation.yaw)
            psi_obs[k] = np.arctan2(np.sin(psi_obs[k]),np.cos(psi_obs[k]))
            k=k+1
    
    x_obs = x_obs[0:k]
    y_obs = y_obs[0:k]
    v_obs = v_obs[0:k]
    psi_obs = psi_obs[0:k]

    total_obs = np.shape(x_obs)[0]

    if total_obs<prob.num_obs:
        if total_obs==0 or total_obs is None:
            numobs = prob.num_obs
            y_obs = 50000*np.ones(numobs)
            x_obs = 50000*np.ones(numobs)
            v_obs = 0.0*np.asarray(np.random.uniform(0, 2, numobs))
            psi_obs = 0.0*np.asarray(np.random.uniform(0, 2, numobs))

        else:
            for j in range(0,prob.num_obs-total_obs):
                total_obs = np.shape(x_obs)[0]
                x_obs = np.append(x_obs,x_obs[-1])
                y_obs = np.append(y_obs,y_obs[-1])
                v_obs = np.append(v_obs,v_obs[-1])
                psi_obs = np.append(psi_obs,psi_obs[-1])
   
    total_obs = np.shape(x_obs)[0]
    dist_obs = (x_global_init-x_obs)**2+(y_global_init-y_obs)**2
    idx_sort = np.argsort(dist_obs)

    x_obs_sort = x_obs[idx_sort[0:prob.num_obs]]
    y_obs_sort = y_obs[idx_sort[0:prob.num_obs]]
    psi_obs_sort = psi_obs[idx_sort[0:prob.num_obs]]
    v_obs_sort  = v_obs[idx_sort[0:prob.num_obs]]

    return x_obs_sort, y_obs_sort, psi_obs_sort, v_obs_sort

def get_mean_cov(prob, y_lane_lb, y_lane_ub, num_mean_update=8):

    mean_vx_1 = 10
    mean_vx_2 = 10
    mean_vx_3 = 10
    mean_vx_4 = 10

    mean_y_des_1 = 0.0
    mean_y_des_2 = 0.0
    mean_y_des_3 = 0.0
    mean_y_des_4 = 0.0

    mean_param = jnp.hstack(( mean_vx_1, mean_vx_2, mean_vx_3, mean_vx_4, mean_y_des_1, mean_y_des_2, mean_y_des_3, mean_y_des_4))

    diag_param = np.hstack(( 14, 14, 14, 14, 23.0, 23.0, 23.0, 23.0  ))
    cov_param = jnp.asarray(np.diag(diag_param)) 

    neural_output_warmstart = prob.sampling_param(y_lane_lb, y_lane_ub, mean_param, cov_param)
    t_target = (num_mean_update-1)*prob.t

    return mean_param, cov_param, neural_output_warmstart, t_target



def Sort_Tuple(tup):
 
    # getting length of list of tuples
    lst = len(tup)
    for i in range(0, lst):
 
        for j in range(0, lst-i-1):
            if (tup[j][1] > tup[j + 1][1]):
                temp = tup[j]
                tup[j] = tup[j + 1]
                tup[j + 1] = temp
    return tup

def denoise_map(im, min_size=150):
    nb_blobs, im_with_separated_blobs, stats, _n = cv2.connectedComponentsWithStats(im)
    nb_blobs = nb_blobs - 1
    sizes = stats[:, -1]
    sizes = sizes[1:]
    filtered_im = np.zeros_like(im)
    for blob in range(nb_blobs):
        x, y = np.where(im_with_separated_blobs == blob + 1)
        if sizes[blob] < min_size:
            filtered_im[x, y] = 0
        else:
            filtered_im[x, y] = 1
    return filtered_im

def min_rect(im_with_separated_blobs_l, id):
    x, y = np.where(im_with_separated_blobs_l == id + 1)
    x = x.tolist()
    y = y.tolist()
    pts = [x, y]
    pts = np.array(pts)
    pts = pts.T
    box = cv2.minAreaRect(pts)
    points = cv2.boxPoints(box)
    points = points.astype(np.int32)
    points[:,[0, 1]] = points[:, [1, 0]]
    return box, points

def check_in_boundary(x_cen, y_cen):
    return (x_cen in range(0, 10) or x_cen in range(190, 199)) and (y_cen in range(0, 10) or y_cen in range(190, 199))

def add_lost_instances(noiseless, l_pred, ll_pred):
    im = noiseless.cpu().numpy().astype(np.uint8)
    nb_blobs_n, im_with_separated_blobs_n, stats_n, _n = cv2.connectedComponentsWithStats(im)
    nb_blobs_l, im_with_separated_blobs_l, stats_l, _l = cv2.connectedComponentsWithStats(l_pred)

    if nb_blobs_n < nb_blobs_l:
        total_absent = nb_blobs_l - nb_blobs_n
        nb_blobs_n -= 1
        nb_blobs_l -= 1
        id_dist = []

        for blob in range(nb_blobs_l):
            x, y = np.where(im_with_separated_blobs_l == blob + 1)
            x = x.astype(np.float32)
            y = y.astype(np.float32)
            y_cen, x_cen = np.mean(y), np.mean(x)
            dist = 1e15

            for blobn in range(nb_blobs_n):
                xn, yn = np.where(im_with_separated_blobs_n == blobn + 1)
                xn = xn.astype(np.float32)
                yn = yn.astype(np.float32)
                yn_cen, xn_cen = np.mean(yn), np.mean(xn)
                if dist > (x_cen-xn_cen)**2 + (y_cen-yn_cen)**2:
                    dist = (x_cen-xn_cen)**2 + (y_cen-yn_cen)**2

            id_dist.append((blob, dist))

        id_dist = Sort_Tuple(id_dist)
        nb_blobs_ll, im_with_separated_blobs_ll, stats_ll, _ll = cv2.connectedComponentsWithStats(ll_pred)
        nb_blobs_ll -= 1

        for blob_idx in range(len(id_dist)-total_absent, len(id_dist)):
            id, dist = id_dist[blob_idx]
            dist = 1e15
            idll = -1
            for blobn in range(nb_blobs_ll):
                xn, yn = np.where(im_with_separated_blobs_ll == blobn + 1)
                xn = xn.astype(np.float32)
                yn = yn.astype(np.float32)
                yn_cen, xn_cen = np.mean(yn), np.mean(xn)
                if dist > (x_cen-xn_cen)**2 + (y_cen-yn_cen)**2:
                    dist = (x_cen-xn_cen)**2 + (y_cen-yn_cen)**2
                    idll = blobn

            x, y = np.where(im_with_separated_blobs_l == id + 1)
            x = x.tolist()
            y = y.tolist()
            pts = [x, y]
            pts = np.array(pts)
            pts = pts.T
            box_l = cv2.minAreaRect(pts)

            x, y = np.where(im_with_separated_blobs_ll == idll + 1)
            x = x.tolist()
            y = y.tolist()
            pts = [x, y]
            pts = np.array(pts)
            pts = pts.T
            box_ll = cv2.minAreaRect(pts)

            a = np.add(box_l[0], box_ll[0])/2
            b = np.add(box_l[1], np.subtract(box_l[1], box_ll[1]))
            c = np.add(box_l[2], np.subtract(box_l[2], box_ll[2]))
            box = (a, b, c)

            points = cv2.boxPoints(box)
            points = points.astype(np.int32)
            points[:,[0, 1]] = points[:, [1, 0]]
            im = cv2.fillPoly(im, pts =[points], color=(1,1,1))
            noiseless[im==1] = 1

    return noiseless

def add_lost_instances_future(og, noiseless):
    threshold=8
    noiseless = noiseless.numpy().astype(np.uint8)
    nb_blobs_0, im_with_separated_blobs_0, stats_0, _0 = cv2.connectedComponentsWithStats(noiseless[0])
    nb_blobs_0 -= 1

    for i in [1, 2, 3]:
        
        nb_blobs_n, im_with_separated_blobs_n, stats_n, _n = cv2.connectedComponentsWithStats(noiseless[i])
        nb_blobs_n -= 1

        for blob_0 in range(nb_blobs_0):
            x, y = np.where(im_with_separated_blobs_0 == blob_0 + 1)
            x = x.astype(np.float32)
            y = y.astype(np.float32)
            y_cen, x_cen = np.mean(y), np.mean(x)
            dist = 1e15

            for blob_n in range(nb_blobs_n):
                xn, yn = np.where(im_with_separated_blobs_n == blob_n + 1)
                xn = xn.astype(np.float32)
                yn = yn.astype(np.float32)
                yn_cen, xn_cen = np.mean(yn), np.mean(xn)
                if dist > np.sqrt((x_cen-xn_cen)**2 + (y_cen-yn_cen)**2):
                    dist = np.sqrt((x_cen-xn_cen)**2 + (y_cen-yn_cen)**2)

            if dist>threshold:
                x = x.tolist()
                y = y.tolist()
                pts = [x, y]
                pts = np.array(pts)
                pts = pts.T
                pts = pts.astype(np.int32)
                box = cv2.minAreaRect(pts)
                points = cv2.boxPoints(box)
                points = points.astype(np.int32)
                points[:,[0, 1]] = points[:, [1, 0]]
                noiseless[i] = cv2.fillPoly(noiseless[i], pts =[points], color=(1,1,1))

        plt.imshow(noiseless[i], cmap='gray')
        plt.show()

    og[noiseless==1]=1
    return og


def get_pts1():
    bx = np.array([-20.0 + 0.2/5.0, -20.0 + 0.2/5.0])
    dx = np.array([0.2, 0.2])
    w, h = 1.85, 4.084
    pts = np.array([
        [0 / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, -w / 2.],
        [0 / 2. + 0.5, -w / 2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    return pts

def get_pts2():
    bx = np.array([-20.0 + 0.2/5.0, -20.0 + 0.2/5.0])
    dx = np.array([0.2, 0.2])
    w, h = 1.85, 4.084
    pts = np.array([
        [-h / 2. + 0.5, w / 2.],
        [0 / 2. + 0.5, w / 2.],
        [0 / 2. + 0.5, -w / 2.],
        [-h / 2. + 0.5, -w / 2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    return pts
        
# def rotate(gt_x, gt_y,theta):
#     rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
#     gt_final = np.dstack((gt_x, gt_y))[0]
#     gt_final = np.dot(rot, gt_final.T).T
#     gt_x_x = [ (gt_x[k] * np.cos(theta) - gt_y[k] * np.sin(theta))  for k in range(len(gt_x))]
#     gt_y_y = [ (gt_x[k] * np.sin(theta) + gt_y[k] * np.cos(theta))  for k in range(len(gt_x))]
#     gt_x = gt_x_x
#     gt_y = gt_y_y
#     return np.array(gt_x), np.array(gt_y)


def rotate(gt_x, gt_y,theta):
    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    gt_final = np.dstack((gt_x, gt_y))[0]
    gt_final = np.dot(rot, gt_final.T).T
    gt_x = gt_final[:, 0]
    gt_y = gt_final[:, 1]
    return np.array(gt_x), np.array(gt_y)

def jnp_rotate(gt_x, gt_y,theta):
    rot = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    gt_final = jnp.dstack((gt_x, gt_y))
    gt_final = gt_final.reshape(150 * 100, 2)
    gt_final = jnp.dot(rot, gt_final.T).T
    gt_final = gt_final.reshape(150, 100, 2)
    gt_x = gt_final[:, :, 0]
    gt_y = gt_final[:, :, 1]
    return jnp.array(gt_x), jnp.array(gt_y)
