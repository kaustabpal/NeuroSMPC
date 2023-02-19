import torch
# from goal_sampler_static_obs import Goal_Sampler
from goal_sampler_static_obs import Goal_Sampler
from grad_cem import GradCEM
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import time
import tyro
from dataclasses import dataclass 
import matplotlib.pyplot as plt
import pickle
import os
from utils import rotate
from utils import draw_circle
from frenet_transformations import global_traj, global_to_frenet, frenet_to_global
import argparse
torch.manual_seed(42)

@dataclass
class Args:
    occ_map_dir: str = '/Users/kaustabpal/work/carla_latest/occ_map/' # '/scratch/kaustab.pal/iros_23/dataset/' #'../iros_23/dataset/' #'/scratch/kaustab.pal/iros_23/dataset/' # 'data/dataset_beta/'
    mean_dir: str = '/Users/kaustabpal/work/carla_latest/mean_controls/' #'/scratch/kaustab.pal/iros_23/weights/' #'../iros_23/weights/' #'/scratch/kaustab.pal/iros_23/weights/' 
    plot_im_dir: str = '/Users/kaustabpal/work/carla_latest/plot_im/' #'/scratch/kaustab.pal/iros_23/loss/'  #'../iros_23/loss/' #'/scratch/kaustab.pal/iros_23/loss/' 
    # val_split: float = 0.3
    # num_epochs: int = 500
    # seed: int = 12321
    exp_id: str = 'exp1'
args = tyro.cli(Args)

def to_continuous(obs):
    obs_pos = []    
    for j in range(obs.shape[0]):
        for k in range(obs.shape[1]):
            if(obs[j,k]==255):
                new_x, new_y = rotate([256/2,256/2],[k,j],np.deg2rad(180))
                obs_pos.append([new_x*30/256,new_y*30/256])
    return obs_pos

def run():
    np.set_printoptions(suppress=True)
    dataset_dir = args.occ_map_dir
    plot_im_dir = args.plot_im_dir
    mean_dir = args.mean_dir
    os.makedirs(plot_im_dir, exist_ok=True)
    os.makedirs(mean_dir, exist_ok=True)
    files = os.listdir(dataset_dir)
    print("Dataset size: ", len(files)-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(1255,len(files)):
        t_1 = time.time()
        
        obs_pos = []
        file_name = dataset_dir + "data_" + str(i).zfill(5) + ".pkl"
        plt_save_file_name = plot_im_dir + "data_" + str(i).zfill(5)
        mean_save_filename = mean_dir + "data_" + str(i).zfill(5)
        with open(file_name, "rb") as f:
            data = pickle.load(f)
            
        ego_speed = 0
        iter = 0
        if "speed" in data.keys():
            ego_speed = data["speed"]
        else:
            continue
            # ego_speed = 4.13
        print(i,ego_speed)

        obs = data['obstable_array'] # obstacle pos in euclidean space
        
        g_path = data['g_path'] -[256/2,256/2] # global path points
        g_path = g_path[:, [1,0]]*30/256
        
        obs_pos = np.array(to_continuous(obs))

        new_g_path, interpolated_g_path, theta = global_traj(g_path, 0.1)

        ego_theta = np.rad2deg(np.pi/2 + (np.pi/2 - theta[0]))
        # print(ego_theta)
        
        obs_pos_frenet = global_to_frenet(obs_pos, new_g_path, interpolated_g_path)
        
        t1 = time.time()
        sampler1 = Goal_Sampler(torch.tensor([0,0,np.deg2rad(ego_theta)]), 4, 0, obstacles=obs_pos_frenet)
        sampler1.plan_traj()
        mean_controls1 = sampler1.mean_action
        mean_traj1 = sampler1.traj_N[-2,:,:]
        cov_controls1 = sampler1.scale_tril
        mean_controls1[:,1] = frenet_to_global(mean_traj1, new_g_path, interpolated_g_path, 0.1)
        sampler1.obstacles = obs_pos
        sampler1.mean_action = torch.as_tensor(mean_controls1)
        sampler1.c_state = torch.tensor([0,0,np.deg2rad(90)])
        sampler1.infer_traj()
        print("Sampler1 time:", time.time()-t1)
        
        t2 = time.time()
        sampler2 = GradCEM(torch.tensor([0,0,np.deg2rad(ego_theta)], device = device), 4, 0, obstacles=obs_pos_frenet)
        sampler2.plan_traj()
        mean_controls = sampler2.mean_action
        mean_traj = sampler2.traj_N[-2,:,:]
        cov_controls = sampler2.scale_tril
        mean_controls[:,1] = frenet_to_global(mean_traj.detach(), new_g_path, interpolated_g_path, 0.1)
        sampler2.obstacles = obs_pos
        sampler2.mean_action = torch.as_tensor(mean_controls)
        sampler2.c_state = torch.tensor([0,0,np.deg2rad(90)])
        sampler2.infer_traj()
        print("Sampler2 time:", time.time()-t2)
        # np.save(mean_save_filename,sampler.mean_action)
        
        ## plot
        plt.scatter(g_path[:,0],g_path[:,1],color='blue', alpha=0.1)

        x_car, y_car = draw_circle(0, 0, 1.80)
        plt.plot(x_car,y_car,'g')
                
        plt.plot(obs_pos[:,0], obs_pos[:,1], 'k.')
        # plt.scatter(obs_pos_frenet[:,0], obs_pos_frenet[:,1], color='orange')

        plt.plot(sampler1.traj_N[-2,:,0], sampler1.traj_N[-2,:,1], 'r', label = "mppi")
        plt.plot(sampler2.traj_N[-2,:,0].detach().cpu(), sampler2.traj_N[-2,:,1].detach().cpu(), 'g', label = "grad-cem")
        # plt.plot(sampler.top_trajs[0,:,0], sampler.top_trajs[0,:,1], 'blue')
        print("Total time: ", time.time()-t_1)
        plt.xlim([-15,15])
        plt.ylim([-15,15])
        plt.title("Ego velocity: "+str(round(ego_speed,2)))
        plt.legend(loc="upper left")
        # print(sampler.top_trajs[0,:,:2])
        # plt.savefig(plt_save_file_name)
        plt.show()
        plt.clf()
        quit()




if __name__=='__main__':
    run()
