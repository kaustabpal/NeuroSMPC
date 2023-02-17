#!/usr/bin/env python

import time
import numpy as np
import matplotlib.pyplot as plt

import torch

from nn.model import Model1
from dataset_pipeline.goal_sampler_static_obs import Goal_Sampler

from dataset_pipeline.utils import rotate, draw_circle

import os

class DeXBee:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

        self.dtype = torch.float32
        np.set_printoptions(suppress=True)

        # Args
        self.seed = 12321
        self.weights_dir = "/scratch/parth.shah/deb/weights/"
        self.exp_id = "exp1-1"

        # Set seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # Load model
        self.model = Model1()
        self.model_weights = torch.load(self.weights_dir+"model_"+self.exp_id+".pt", map_location=torch.device(self.device))
        self.model.load_state_dict(self.model_weights)
        self.model.to(self.device)
        self.model.eval()

        print("Device : ", self.device)

        self.visualize = True
        self.save = False

        if self.visualize:
            plt.ion()

    def to_continuous(self, obs):
        obs_pos = []    
        for j in range(obs.shape[0]):
            for k in range(obs.shape[1]):
                if(obs[j,k]==255):
                    new_x, new_y = rotate([256/2,256/2],[k,j],np.deg2rad(180))
                    obs_pos.append([new_x*30/256,new_y*30/256])
        return obs_pos

    def generate_path(self, obstacle_array, global_path, current_speed):
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


        #Finding the mean actions from the NN
        tic = time.time()
        with torch.no_grad():
            mean_action = self.model(occupancy_map.unsqueeze(0).to(self.device)).reshape(30,2) # NN output reshaped        
        toc = time.time()
        # print("Time taken for NN : ", toc-tic)
        
        tic = time.time()
        mean_action_cpu = mean_action.detach().cpu()
        toc = time.time()
        # print("Time taken for GPU-CPU : ", toc-tic)

        #Finding the best trajectory
        tic = time.time()
        sampler = Goal_Sampler(torch.tensor([0, 0, np.deg2rad(90)]), 4.13, 0, obstacles=obstacle_positions, num_particles = 100)
        sampler.num_particles = 100
        sampler.mean_action = mean_action_cpu
        sampler.infer_traj()
        toc = time.time()
        # print("Time taken for sampling : ", toc-tic)

        best_controls = sampler.top_controls[0,:,:] # contains the best v and w
        best_traj = sampler.top_trajs[0,:,:] # contains the best x, y and theta

        best_traj = best_traj.detach().cpu().numpy()
        best_controls = best_controls.detach().cpu().numpy()
        
        print("Plotting")
        if self.visualize or self.save:
            self.plotter(obstacle_positions, g_path, sampler, 0.5, current_speed)
        print("Done plotting")
        return best_traj, best_controls
    
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
        plt.plot(sampler.traj_N[:,:,0], sampler.traj_N[:,:,1], '.b', markersize=1, alpha=0.04)

        # Predicted trajectory
        plt.plot(sampler.traj_N[-2,:,0], sampler.traj_N[-2,:,1], 'red', markersize=3, label = "Predicted")

        # Best trajectory
        plt.plot(sampler.top_trajs[0,:,0], sampler.top_trajs[0,:,1], 'lime', markersize=2, label = "best traj")
        
        # Set Limits
        plt.ylim(-15,15)
        plt.xlim(-15,15)
        plt.legend(loc="lower center")

        if self.save and filename is not None:
            plt.savefig(filename)
        
        if self.visualize:
            plt.draw()
            plt.pause(0.001)