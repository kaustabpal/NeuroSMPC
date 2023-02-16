#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import torch

from nn.model import Model1
from dataset_pipeline.goal_sampler_static_obs import Goal_Sampler

from dataset_pipeline.utils import rotate

import os

class DeXBee:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        np.set_printoptions(suppress=True)

        # Args
        self.seed = 12321
        self.weights_dir = "../iros_23/weights/"
        self.exp_id = "exp1"

        # Set seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # Load model
        self.model = Model1()
        self.model_weights = torch.load(self.weights_dir+"model_"+self.exp_id+".pt", map_location=torch.device(self.device))
        self.model.load_state_dict(self.model_weights)
        self.model.to(self.device)
        self.model.eval()

    def to_continuous(self, obs):
        obs_pos = []    
        for j in range(obs.shape[0]):
            for k in range(obs.shape[1]):
                if(obs[j,k]==255):
                    new_x, new_y = rotate([256/2,256/2],[k,j],np.deg2rad(180))
                    obs_pos.append([new_x*30/256,new_y*30/256])
        return obs_pos

    def generate_path(self, bev, global_path, current_speed):
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
        occupancy_map = torch.as_tensor(bev, dtype = self.dtype).unsqueeze(0) # obstacle pos in euclidean space
        occupancy_map = torch.permute(occupancy_map, (2,0,1)) / 255.0 
        
        # Process obstacles
        obstacle_positions = np.array(self.to_continuous(bev))

        #Finding the mean actions from the NN 
        with torch.no_grad():
            mean_action = self.model(occupancy_map.unsqueeze(0)).reshape(30,2) # NN output reshaped        
        
        #Finding the best trajectory
        sampler = Goal_Sampler(torch.tensor([0, 0, np.deg2rad(90)]), 4.13, 0, obstacles=obstacle_positions)
        sampler.num_particles = 100
        sampler.mean_action = mean_action
        sampler.infer_traj()

        best_controls = sampler.top_controls[0,:,:] # contains the best v and w
        best_traj = sampler.top_trajs[0,:,:] # contains the best x, y and theta

        return best_traj, best_controls