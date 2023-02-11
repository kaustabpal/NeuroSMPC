import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time

def global_to_frenet(obstacle_array, global_path):
    time_array = np.arange(0, global_path.shape[0])
    g_path = np.zeros(global_path.shape)
    g_path[:,0] = global_path[:,0]
    g_path[:,1] = global_path[:,1]
    cs = CubicSpline(time_array, g_path)
    xs = np.arange(0, g_path.shape[0], 0.004)
    g_path = cs(xs)

    x_diff = np.diff(g_path[:,0], prepend=g_path[0][0])
    y_diff = np.diff(g_path[:,1], prepend=g_path[0][1])
    new_g_path = np.sqrt(x_diff**2 + y_diff**2)
    g_path_arc_lengths = np.cumsum(new_g_path)
    new_g_path = np.cumsum(new_g_path)

    new_g_path = np.vstack(((g_path[0][0])*np.ones((new_g_path.shape[0])), new_g_path)).T

    frenet_obs = []
    for i in range(obstacle_array.shape[0]):
        dists_from_path = np.linalg.norm(obstacle_array[i] - g_path, axis=1)
        nearest_point_idx = np.argmin(dists_from_path)
        nearest_point_to_obs = g_path_arc_lengths[nearest_point_idx]
        min_dist_from_path = np.sign(obstacle_array[i][0])*(dists_from_path[nearest_point_idx] + 1*obstacle_array[i][0])
        frenet_obs.append([min_dist_from_path, nearest_point_to_obs])
        # print(min_dist_from_path, nearest_point_to_obs)
    return np.array(frenet_obs), new_g_path

def frenet_to_global(trajectory, controls, global_path):
    trajectory = trajectory.numpy()

    time_array = np.arange(0, global_path.shape[0])
    g_path = np.zeros(global_path.shape)
    g_path[:,0] = global_path[:,0] 
    g_path[:,1] = global_path[:,1] 
    cs = CubicSpline(time_array, g_path)
    xs = np.arange(0, g_path.shape[0], 0.004)
    g_path = cs(xs)

    x_diff = np.diff(g_path[:,0], prepend=g_path[0][0])
    y_diff = np.diff(g_path[:,1], prepend=g_path[0][1])
    theta = np.arctan2(y_diff, x_diff)
    theta_dot = np.diff(theta, prepend=theta[0])
    new_g_path = np.sqrt(x_diff**2 + y_diff**2)
    new_g_path = np.cumsum(new_g_path)

    new_g_path = np.vstack(((g_path[0][0])*np.ones((new_g_path.shape[0])), new_g_path)).T

    mean_controls = np.zeros(controls.shape)
    mean_controls[:,0] = controls[:,0]
    s = time.time()
    traj_dist_from_g_path = cdist(trajectory[1:, :2], new_g_path)
    nearest_point_idxs = np.argmin(traj_dist_from_g_path, axis=1)
    mean_controls[:, 1] = controls[:,1] + theta_dot[nearest_point_idxs]
    return mean_controls