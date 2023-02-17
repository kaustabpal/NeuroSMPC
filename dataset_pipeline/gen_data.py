import torch
from dataset_pipeline.goal_sampler_static_obs import Goal_Sampler
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import time
# import cv2
import matplotlib.pyplot as plt
import pickle
import os
from dataset_pipeline.utils import rotate
from dataset_pipeline.utils import draw_circle
from dataset_pipeline.frenet_transformations import global_traj, global_to_frenet, frenet_to_global
import argparse
torch.manual_seed(42)

def to_continuous(obs):
    obs_pos = []    
    for j in range(obs.shape[0]):
        for k in range(obs.shape[1]):
            if(obs[j,k]==255):
                new_x, new_y = rotate([256/2,256/2],[k,j],np.deg2rad(180))
                obs_pos.append([new_x*30/256,new_y*30/256])
    return obs_pos

# def get_traj(obstacles):
#     dtype = torch.float32
#     sampler = Goal_Sampler(torch.tensor([0,0,np.deg2rad(90)]), 4.13, 0, obstacles=obstacles)
    
#     sampler.plan_traj()
#     mean_controls = sampler.controls_N[-1,:,:]
#     mean_traj = sampler.traj_N[-1,:,:]
#     cov_controls = sampler.scale_tril
#     return sampler, mean_controls, mean_traj, cov_controls
    

def run():
    np.set_printoptions(suppress=True)
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--dataset_dir", help="dataset_dir")
    argParser.add_argument("-p", "--plot_im_dir", help="plot_image_dir")
    argParser.add_argument("-m", "--mean_dir", help="mean_controls_dir")
    args = argParser.parse_args()
    # dataset_dir = "storm/"
    dataset_dir = args.dataset_dir
    plot_im_dir = args.plot_im_dir
    mean_dir = args.mean_dir
    files = os.listdir(dataset_dir)
    print(len(files)-1)
    for i in range(0,len(files)):
        for velocity in range(1, 6):
            t_1 = time.time()
            print(i)

            data_instance =  str(i).zfill(2) + "_" + str(int(velocity))

            obs_pos = []
            file_name = dataset_dir + "data_" + str(i).zfill(2) + ".pkl"
            plt_save_file_name = plot_im_dir + "data_" + data_instance
            mean_save_filename = mean_dir + "data_" + data_instance
            with open(file_name, "rb") as f:
                data = pickle.load(f)
            obs = data['obstable_array'] # obstacle pos in euclidean space
            
            g_path = data['g_path'] -[256/2,256/2] # global path points
            g_path = g_path[:, [1,0]]*30/256
            
            obs_pos = np.array(to_continuous(obs))

            new_g_path, interpolated_g_path, theta = global_traj(g_path, 0.1)

            ego_theta = np.rad2deg(np.pi/2 + (np.pi/2 - theta[0]))
            # print(ego_theta)
            
            obs_pos_frenet = global_to_frenet(obs_pos, new_g_path, interpolated_g_path)

            sampler = Goal_Sampler(torch.tensor([0,0,np.deg2rad(ego_theta)]), 4.13, 0, obstacles=obs_pos_frenet)
            t1 = time.time()
            sampler.plan_traj()
            # print("Planning time: ", time.time()-t1)

            mean_controls = sampler.mean_action
            mean_traj = sampler.traj_N[-2,:,:]
            cov_controls = sampler.scale_tril
            # print(mean_controls)
            
            mean_controls[:,1] = frenet_to_global(mean_traj, new_g_path, interpolated_g_path, 0.1)
            # print(mean_controls)
            # quit()
            sampler.obstacles = obs_pos
            sampler.mean_action = torch.as_tensor(mean_controls)
            sampler.c_state = torch.tensor([0,0,np.deg2rad(90)])
            sampler.infer_traj()
            np.save(mean_save_filename,sampler.mean_action)
            
            ## plot
            # for k in range(g_path.shape[0]):
            plt.scatter(g_path[:,0],g_path[:,1],color='blue', alpha=0.1)

            x_car, y_car = draw_circle(0, 0, 1.80)
            plt.plot(x_car,y_car,'g')
            
            # for j in range(obs_pos.shape[0]):
            # print(obs_pos[:][:,0])
            
            plt.plot(obs_pos[:,0], obs_pos[:,1], 'k.')
            plt.scatter(obs_pos_frenet[:,0], obs_pos_frenet[:,1], color='orange')
                
            # for j in range(sampler.traj_N.shape[0]):
            plt.plot(sampler.traj_N[:,:,0], sampler.traj_N[:,:,1], 'r', alpha=0.1)
            plt.plot(sampler.traj_N[-2,:,0], sampler.traj_N[-2,:,1], 'g')
            plt.plot(sampler.top_trajs[0,:,0], sampler.top_trajs[0,:,1], 'blue')
            print("Total time: ", time.time()-t_1)
            # print(sampler.top_trajs[0,:,:2])
            plt.savefig(plt_save_file_name)
            # plt.show()
            plt.clf()
            # quit()




if __name__=='__main__':
    run()
