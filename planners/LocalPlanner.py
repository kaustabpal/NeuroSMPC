#!/usr/bin/env python

import time
import numpy as np
import matplotlib.pyplot as plt

import torch

from nn.model import Model1, Model_Temporal
from dataset_pipeline.goal_sampler_dyn_obs_lane_change import Goal_Sampler as Goal_Sampler_Dyn
from dataset_pipeline.goal_sampler_static_obs import Goal_Sampler
from dataset_pipeline.grad_cem import GradCEM

from dataset_pipeline.utils import rotate, draw_circle

from dataset_pipeline.frenet_transformations import global_traj, global_to_frenet, frenet_to_global

import os

class LocalPlanner:
    def __init__(self, planner="NuroMPPI", expt_type = "temporal") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"

        self.dtype = torch.float32
        np.set_printoptions(suppress=True)

        # Args
        self.seed = 12321
        self.weights_dir = "data/weights/"
        self.exp_id = "dyn1"

        # Set seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        print("Device : ", self.device)

        self.planner_type = planner

        self.expt_type = expt_type
        if self.expt_type == "temporal":
            self.obstacle_array_history = None

        self.temporal_hist_len = 5

        self.model = None
        self.sampler = None
        if self.planner_type == "NuroMPPI":
            if self.expt_type == "static":
                # Load model
                self.model = Model1()
                self.model_weights = torch.load(self.weights_dir+"model_"+self.exp_id+".pt", map_location=torch.device(self.device))
                self.model.load_state_dict(self.model_weights)
                self.model.to(self.device)
                self.model.eval()
            else:
                self.model = Model_Temporal(self.temporal_hist_len)
                self.model_weights = torch.load(self.weights_dir+"model_"+self.exp_id+"_temp.pt", map_location=torch.device(self.device))
                self.model.load_state_dict(self.model_weights)
                self.model.to(self.device)
                self.model.eval()
        
        # TODO :- find a way to not initialize everytime
        #   self.sampler = Goal_Sampler()
        # elif self.planner_type == "MMPI":
        #     self.sampler = Goal_Sampler()
        # elif self.planner_type == "GradCEM":
        #     self.sampler = GradCEM()

        self.visualize = False
        self.save = True

        if self.visualize:
            plt.ion()
        
        self.time_info = {}
        self.time_arr = np.linspace(0, 3.0, 31)

    def to_continuous(self, obs):
        obs_pos = []    
        for j in range(obs.shape[0]):
            for k in range(obs.shape[1]):
                if(obs[j,k]==255):
                    new_x, new_y = rotate([256/2,256/2],[k,j],np.deg2rad(180))
                    obs_pos.append([new_x*30/256,new_y*30/256])
        return obs_pos

    def generate_path(self, obstacle_array, dyn_obs, global_path, current_speed):
        print(self.planner_type)
        if self.expt_type == "temporal":
            if self.planner_type == "NuroMPPI":
                return self.generate_path_nuromppi_dyn(obstacle_array, dyn_obs, global_path, current_speed)
        elif self.expt_type == "static":                
            if self.planner_type == "NuroMPPI":
                return self.generate_path_nuromppi(obstacle_array, global_path, current_speed)
            elif self.planner_type == "MPPI":
                return self.generate_path_mppi(obstacle_array, global_path, current_speed)
            elif self.planner_type == "GradCEM":
                return self.generate_path_gcem(obstacle_array, global_path, current_speed)


    def generate_path_nuromppi_dyn(self, obstacle_array, dyn_obs, global_path, current_speed):
        tic_ = time.time()
        tic = time.time()
        ego_speed = current_speed

        # Process global path
        global_path_orig = np.copy(global_path)        

        global_path[:,0] = -global_path[:,0] + 256 # global path points
        global_path = global_path.astype(np.int32)
        global_path = global_path[np.where((global_path[:, 0] < 256) & (global_path[:, 1] < 256) & (global_path[:, 0] >= 0) & (global_path[:, 1] >= 0) )]
        
        global_path_array = np.zeros((256, 256))
        global_path_array[global_path[:,0], global_path[:,1]] = 255
    
        g_path = global_path_orig - [256/2,256/2] # global path points
        g_path = g_path[:, [1,0]]*30/256

        # Process occupancy map
        obstacle_array = np.copy(obstacle_array)
        if self.obstacle_array_history is None:
            self.obstacle_array_history = np.array(obstacle_array)
        else:
            self.obstacle_array_history = np.dstack((obstacle_array, self.obstacle_array_history))

        print(self.obstacle_array_history.shape, len(self.obstacle_array_history.shape))
        if len(self.obstacle_array_history.shape) == 2:
            self.obstacle_array_history = self.obstacle_array_history.reshape([self.obstacle_array_history.shape[0], self.obstacle_array_history.shape[1], 1])

        print(self.obstacle_array_history.shape, len(self.obstacle_array_history.shape))

        if self.obstacle_array_history.shape[2] >= self.temporal_hist_len:
            self.obstacle_array_history = self.obstacle_array_history[:,:,:self.temporal_hist_len]
        else:
            print("Not enough obstacle history - ", self.obstacle_array_history.shape[2])
            return None, None, -1
        
        bev = np.dstack([self.obstacle_array_history, global_path_array])


        print("BEV shape : ", bev.shape)

        occupancy_map = torch.as_tensor(bev, dtype = self.dtype) # obstacle pos in euclidean space
        # print("Occupancy map shape : ", occupancy_map.shape)
        occupancy_map = torch.permute(occupancy_map, (2,0,1)) / 255.0 
        
        # Process obstacles
        obs_poses = []
        for o in range(len(dyn_obs)):
            y, x = dyn_obs[o][1], dyn_obs[o][0]
            rel_yaw = dyn_obs[o][2] - np.pi/2
            rel_yaw = np.pi/2 - rel_yaw
            new_theta = rel_yaw + np.deg2rad(-dyn_obs[o][4])*self.time_arr
            obs_path_x = y + dyn_obs[o][3]*self.time_arr*np.cos(new_theta)
            obs_path_y = x + dyn_obs[o][3]*self.time_arr*np.sin(new_theta)
            traj = np.vstack((obs_path_x, obs_path_y)).T
            obs_poses.append(traj)
        obstacle_positions_dyn = np.array(obs_poses)
        obstacle_positions = np.array(self.to_continuous(obstacle_array))
        toc = time.time()
        self.time_info["preprocess"] = toc-tic

        tic = time.time()
        with torch.no_grad():
            mean_action = self.model(occupancy_map.unsqueeze(0).to(self.device)).reshape(30,2) # NN output reshaped        
        toc = time.time()
        self.time_info["model"] = toc-tic
        
        tic = time.time()
        mean_action_cpu = mean_action.detach().cpu()
        toc = time.time()
        self.time_info["midprocess"] = toc-tic

        #Finding the best trajectory
        tic = time.time()
        sampler = Goal_Sampler_Dyn(torch.tensor([0, 0, np.deg2rad(90)]), 4.13, 0, obstacles=obstacle_positions_dyn, num_particles = 100)
        sampler.num_particles = 100
        sampler.mean_action = mean_action_cpu
        sampler.infer_traj()
        toc = time.time()
        self.time_info["sampler"] = toc-tic
        
        tic = time.time()
        best_controls = sampler.top_controls[0,:,:] # contains the best v and w
        best_traj = sampler.top_trajs[0,:,:] # contains the best x, y and theta

        best_traj = best_traj.detach().cpu().numpy()
        best_controls = best_controls.detach().cpu().numpy()
        toc = time.time()
        self.time_info["postprocess"] = toc-tic

        tic = time.time()
        if self.visualize or self.save:
            self.plotter(obstacle_positions, g_path, sampler, 2.5, current_speed)
        toc = time.time()
        self.time_info["plot"] = toc-tic

        toc_ = time.time()
        self.time_info["total"] = toc_-tic_
        return best_traj, best_controls, 1
    

    def generate_path_nuromppi(self, obstacle_array, global_path, current_speed):
        tic_ = time.time()
        tic = time.time()
        ego_speed = current_speed

        # Process global path
        global_path_orig = np.copy(global_path)        

        global_path[:,0] = -global_path[:,0] + 256 # global path points
        global_path = global_path.astype(np.int32)
        global_path = global_path[np.where((global_path[:, 0] < 256) & (global_path[:, 1] < 256) & (global_path[:, 0] >= 0) & (global_path[:, 1] >= 0) )]
        
        global_path_array = np.zeros((256, 256))
        global_path_array[global_path[:,0], global_path[:,1]] = 255
    
        g_path = global_path_orig - [256/2,256/2] # global path points
        g_path = g_path[:, [1,0]]*30/256

        # Process occupancy map
        obstacle_array = np.copy(obstacle_array)

        bev = np.dstack([obstacle_array, global_path_array])
        # print("BEV shape : ", bev.shape)

        occupancy_map = torch.as_tensor(bev, dtype = self.dtype) # obstacle pos in euclidean space
        # print("Occupancy map shape : ", occupancy_map.shape)
        occupancy_map = torch.permute(occupancy_map, (2,0,1)) / 255.0 
        
        # Process obstacles
        obstacle_positions = np.array(self.to_continuous(obstacle_array))
        toc = time.time()
        self.time_info["preprocess"] = toc-tic

        tic = time.time()
        with torch.no_grad():
            mean_action = self.model(occupancy_map.unsqueeze(0).to(self.device)).reshape(30,2) # NN output reshaped        
        toc = time.time()
        self.time_info["model"] = toc-tic
        
        tic = time.time()
        mean_action_cpu = mean_action.detach().cpu()
        toc = time.time()
        self.time_info["midprocess"] = toc-tic

        #Finding the best trajectory
        tic = time.time()
        sampler = Goal_Sampler(torch.tensor([0, 0, np.deg2rad(90)]), 4.13, 0, obstacles=obstacle_positions, num_particles = 100)
        sampler.num_particles = 100
        sampler.mean_action = mean_action_cpu
        sampler.infer_traj()
        toc = time.time()
        self.time_info["sampler"] = toc-tic
        
        tic = time.time()
        best_controls = sampler.top_controls[0,:,:] # contains the best v and w
        best_traj = sampler.top_trajs[0,:,:] # contains the best x, y and theta

        best_traj = best_traj.detach().cpu().numpy()
        best_controls = best_controls.detach().cpu().numpy()
        toc = time.time()
        self.time_info["postprocess"] = toc-tic

        tic = time.time()
        if self.visualize or self.save:
            self.plotter(obstacle_positions, g_path, sampler, 2.5, current_speed)
        toc = time.time()
        self.time_info["plot"] = toc-tic

        toc_ = time.time()
        self.time_info["total"] = toc_-tic_
        return best_traj, best_controls, 1
    
    def generate_path_mppi(self, obstacle_array, global_path, current_speed):
        tic_ = time.time()
        tic = time.time()
        ego_speed = current_speed

        # Process global path
        global_path_orig = np.copy(global_path)
        global_path = global_path - [256/2,256/2] # global path points
        global_path = global_path[:, [1,0]]*30/256

        new_global_path, interpolated_global_path, theta = global_traj(global_path, 0.1)

        ego_theta = np.rad2deg(np.pi/2 + (np.pi/2 - theta[0]))

        # Process occupancy map
        obstacle_array = np.copy(obstacle_array)
        obstacle_positions = np.array(self.to_continuous(obstacle_array))
        obstacle_positions_frenet = global_to_frenet(obstacle_positions, new_global_path, interpolated_global_path)
        toc = time.time()
        self.time_info["preprocess"] = toc-tic

        tic = time.time()
        sampler = Goal_Sampler(torch.tensor([0, 0, np.deg2rad(ego_theta)]), 4.13, 0, obstacles=obstacle_positions)
        sampler.plan_traj()
        toc = time.time()
        self.time_info["plan-traj"] = toc-tic

        tic = time.time()
        mean_controls = sampler.mean_action
        mean_traj = sampler.traj_N[-2, :, :]
        cov_controls = sampler.scale_tril
        mean_controls[:, 1] = frenet_to_global(mean_traj.detach().cpu(), new_global_path, interpolated_global_path, 0.1)
        toc = time.time()
        self.time_info["midprocess"] = toc-tic

        tic = time.time()
        sampler.obstacles = obstacle_positions
        sampler.mean_action = torch.as_tensor(mean_controls)
        sampler.c_state = torch.as_tensor([0, 0, np.deg2rad(90)])
        sampler.infer_traj()
        toc = time.time()
        self.time_info["infer-traj"] = toc-tic

        tic = time.time()
        best_controls = sampler.top_controls[0,:,:] # contains the best v and w
        best_traj = sampler.top_trajs[0,:,:] # contains the best x, y and theta

        best_traj = best_traj.detach().cpu().numpy()
        best_controls = best_controls.detach().cpu().numpy()
        toc = time.time()
        self.time_info["postprocess"] = toc-tic

        tic = time.time()
        if self.visualize or self.save:
            self.plotter(obstacle_positions, global_path, sampler, 2.5, current_speed)
        toc = time.time()
        self.time_info["plot"] = toc-tic

        toc_ = time.time()
        self.time_info["total"] = toc_-tic_
        return best_traj, best_controls, 1

    def generate_path_gcem(self, obstacle_array, global_path, current_speed):
        tic_ = time.time()
        tic = time.time()
        ego_speed = current_speed

        # Process global path
        global_path_orig = np.copy(global_path)
        global_path = global_path - [256/2,256/2] # global path points
        global_path = global_path[:, [1,0]]*30/256

        new_global_path, interpolated_global_path, theta = global_traj(global_path, 0.1)

        ego_theta = np.rad2deg(np.pi/2 + (np.pi/2 - theta[0]))

        # Process occupancy map
        obstacle_array = np.copy(obstacle_array)
        obstacle_positions = np.array(self.to_continuous(obstacle_array))
        obstacle_positions_frenet = global_to_frenet(obstacle_positions, new_global_path, interpolated_global_path)
        toc = time.time()
        self.time_info["preprocess"] = toc-tic

        tic = time.time()
        sampler = GradCEM(torch.tensor([0, 0, np.deg2rad(ego_theta)]), 4.13, 0, obstacles=obstacle_positions)
        sampler.plan_traj()
        toc = time.time()
        self.time_info["plan-traj"] = toc-tic

        tic = time.time()
        mean_controls = sampler.mean_action
        mean_traj = sampler.traj_N[-2, :, :]
        cov_controls = sampler.scale_tril
        mean_controls[:, 1] = frenet_to_global(mean_traj.detach().cpu(), new_global_path, interpolated_global_path, 0.1)

        sampler.obstacles = obstacle_positions
        sampler.mean_action = torch.as_tensor(mean_controls)
        sampler.c_state = torch.as_tensor([0, 0, np.deg2rad(90)])
        toc = time.time()
        self.time_info["midprocess"] = toc-tic


        tic = time.time()
        sampler.infer_traj()
        toc = time.time()
        self.time_info["infer-traj"] = toc-tic

        tic = time.time()
        # best_controls = sampler.top_controls[0,:,:] # contains the best v and w
        # best_traj = sampler.top_trajs[0,:,:] # contains the best x, y and theta

        best_controls = sampler.controls_N[-2,:,:] # contains the best v and w
        best_traj = sampler.traj_N[-2,:,:] # contains the best x, y and theta

        print("best traj", len(best_traj))

        best_traj = best_traj.detach().cpu().numpy()[12:]
        best_controls = best_controls.detach().cpu().numpy()[12:]
        toc = time.time()

        tic = time.time()
        if self.visualize or self.save:
            self.plotter(obstacle_positions, global_path, sampler, 2.5, current_speed)
        toc = time.time()
        self.time_info["plot"] = toc-tic

        toc_ = time.time()
        self.time_info["total"] = toc_-tic_
        return best_traj, best_controls, 1

    def plotter(self, obstacle_positions, global_path, sampler, ego_radius, current_speed, filename = None):
        # Clear plot
        plt.clf()
        
        # Car
        x_car, y_car = draw_circle(0, 0, ego_radius)
        plt.plot(x_car,y_car,'g')
        plt.text(0, 0, "v = {}".format(np.round(current_speed, 2)), color='g')

        # Global path
        plt.scatter(global_path[:,0],global_path[:,1],color='blue', alpha=0.1)

        # Obstacles
        plt.plot(obstacle_positions[:,0],obstacle_positions[:,1],'k.')

        # All trajectories
        traj_N = sampler.traj_N.detach().cpu().numpy()
        plt.plot(traj_N[:,:,0], traj_N[:,:,1], '.b', markersize=1, alpha=0.04)

        # Predicted trajectory
        plt.plot(traj_N[-2,12:,0], traj_N[-2,12:,1], 'red', markersize=3, label = "Predicted")

        if self.planner_type != "GradCEM":
            # Best trajectory
            top_trajs = sampler.top_trajs.detach().cpu().numpy()
            plt.plot(top_trajs[0,:,0], top_trajs[0,:,1], 'lime', markersize=2, label = "best traj")
        
        # Set Limits
        plt.ylim(-15,15)
        plt.xlim(-15,15)
        plt.legend(loc="lower center")

        if self.save and filename is not None:
            plt.savefig(filename)
        
        if self.visualize:
            plt.draw()
            plt.pause(0.001)

    def save_plot(self, filename):
        plt.savefig(filename)

    def close_plot(self):
        plt.close()
