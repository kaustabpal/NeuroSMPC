import torch
import numpy as np
from scipy.interpolate import CubicSpline, BSpline, splrep, splprep, splev
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time

def global_traj(global_path, dt):
    time_array = np.arange(0, global_path.shape[0])
    g_path = np.zeros(global_path.shape)
    g_path[:,0] = global_path[:,0] 
    g_path[:,1] = global_path[:,1] 
    # cs = CubicSpline(time_array, g_path)
    sp_x = splrep(time_array, g_path[:,0], k=3, s=0.0)
    sp_y = splrep(time_array, g_path[:,1], k=3, s=0.0)
    xs = np.arange(0, g_path.shape[0], 0.004)
    # g_path = cs(xs)
    g_path = np.vstack((splev(xs, sp_x, ext=3), splev(xs, sp_y, ext=3))).T
    

    x_diff = np.diff(g_path[:,0])
    y_diff = np.diff(g_path[:,1])
    theta = np.arctan2(y_diff, x_diff)

    new_g_path = np.sqrt(x_diff**2 + y_diff**2)
    new_g_path = np.cumsum(new_g_path)
    new_g_path = np.vstack(((g_path[0][0])*np.ones((new_g_path.shape[0])), new_g_path)).T
    

    return new_g_path, g_path, theta

def global_to_frenet(obstacle_array, new_global_path, g_path):
    path_arc_lengths = new_global_path[:, 1]
    frenet_obs = []
    for i in range(obstacle_array.shape[0]):
        dists_from_path = np.linalg.norm(obstacle_array[i] - g_path, axis=1)
        nearest_point_idx = np.argmin(dists_from_path)
        nearest_point_to_obs = path_arc_lengths[nearest_point_idx]
        min_dist_from_path = np.sign(obstacle_array[i][0])*dists_from_path[nearest_point_idx] + new_global_path[i][0]
        frenet_obs.append([min_dist_from_path, nearest_point_to_obs])
        
    return np.array(frenet_obs)

def frenet_to_global(trajectory, new_global_path, g_path, dt):
    trajectory = trajectory.numpy()
    new_global_path = new_global_path[:-2]
    traj_dist_from_g_path = cdist(trajectory[:, :2], new_global_path)
    nearest_point_idxs = np.argmin(traj_dist_from_g_path, axis=1)
    next_nearest_points_idxs = nearest_point_idxs + 1

    g_path_theta = np.arctan2(g_path[next_nearest_points_idxs][:,1] - g_path[nearest_point_idxs][:,1], g_path[next_nearest_points_idxs][:,0] - g_path[nearest_point_idxs][:,0])
    # for i in range(trajectory.shape[0]-1):
    #     if trajectory[i][0] > 0:
    #         g_path_theta[i] = g_path_theta[i] - np.pi/2
    #     elif trajectory[i][0] < 0:
    #         g_path_theta[i] = g_path_theta[i] + np.pi/2
    #     print(trajectory[i][0], g_path_theta[i])
    
    coord_x = trajectory[:, 0]*np.cos(g_path_theta) + g_path[nearest_point_idxs][:, 0]
    coord_y = trajectory[:, 0]*np.sin(g_path_theta) + g_path[nearest_point_idxs][:, 1]
    coord_theta =  g_path_theta - (np.pi/2 - trajectory[:, 2])

    omega = np.diff(coord_theta)/dt

    return torch.as_tensor(omega)

    # plt.scatter(g_path[nearest_point_idxs][:, 0], g_path[nearest_point_idxs][:, 1], c='b')
    # plt.scatter(new_global_path[nearest_point_idxs][:, 0], new_global_path[nearest_point_idxs][:, 1], c='g')
    # plt.plot(coord_x, coord_y, 'y')
    # plt.plot(trajectory[:,0], trajectory[:,1], 'r')
    # plt.show()
    # quit()
    
    # # mean_controls[:, 1] = controls[:,1] + theta_dot[nearest_point_idxs]
    # # return mean_controls