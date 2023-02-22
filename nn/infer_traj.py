import torch
from dataset_pipeline.goal_sampler_static_obs import Goal_Sampler
from dataset_pipeline.grad_cem import GradCEM
from dataset_pipeline.frenet_transformations import global_traj, global_to_frenet, frenet_to_global
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import time
from nn.model import Model1
import matplotlib.pyplot as plt
import pickle
import os
from dataset_pipeline.utils import rotate
from dataset_pipeline.utils import draw_circle
from dataclasses import dataclass 
import tyro
import copy
# from frenet_transformations import global_traj, global_to_frenet, frenet_to_global
# torch.manual_seed(42)

@dataclass
class Args:
    # dataset_dir: str = '/Users/kaustabpal/work/iros_23/sparse_data/test/' # occ_map/' #'../carla_latest/' # 'data/dataset_beta/'
    dataset_dir: str = '/Users/kaustabpal/Downloads/experiments_22-02-2023/NuroMPPI_2-1/' # occ_map/' #'../carla_latest/' # 'data/dataset_beta/'
    weights_dir: str = '../iros_23/weights/' 
    loss_dir: str = '../iros_23/loss/' 
    infer_dir: str = "../iros_23/infer_dir/"
    time_dir: str = "../iros_23/time_dir/"
    val_split: float = 0.3
    num_epochs: int = 1000
    seed: int = 12321
    exp_id: str = 'relu6'
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    np.set_printoptions(suppress=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.makedirs(args.infer_dir, exist_ok=True)
    os.makedirs(args.infer_dir+args.exp_id +"/", exist_ok=True)
    
    model = Model1()
    weights = torch.load(args.weights_dir+"model_"+args.exp_id+".pt", map_location=torch.device(device))
    model.load_state_dict(weights)
    model.to(device)
    occ_map_files = [f for f in os.listdir(args.dataset_dir+"occ_map") if not f.startswith('.')]
    # print(len(files))
    # quit()
    
    nn_time = []
    # torch.save(model.state_dict(), model_path)
    for i in range(30, len(occ_map_files)):
        t_1 = time.time()
        print(i)
        obs_pos = []
        file_name = args.dataset_dir+"occ_map/" + "data_" + str(i).zfill(2) + ".pkl"
        # plt_save_file_name = plot_im_dir + "data_" + str(i)
        # mean_save_filename = args.dataset_dir+"mean_controls/" + "data_" + str(i) + ".npy"
        infer_file_name = args.infer_dir+args.exp_id +"/" + "data_" + str(i).zfill(5)
        # mean_controls_gt = np.load(args.dataset_dir+"mean_controls/" + "data_" + str(i).zfill(5) + ".npy")
        # mean_controls_gt = torch.as_tensor(mean_controls_gt, dtype=dtype, device = device)
        with open(file_name, "rb") as f:
            data = pickle.load(f)
            
        # Plotting g_path on occ_map in separate channel
        obs_array = data['obstable_array']
        g_path_array = np.zeros((256,256))
        g_path = copy.copy(data['g_path'])
        g_path[:,0] = -g_path[:,0] + 256 # global path points
        g_path = g_path.astype(np.int32)
        g_path = g_path[np.where( (g_path[:, 0]<256) & (g_path[:, 1]<256) & (g_path[:, 0]>=0) & (g_path[:, 1]>=0) )]
        g_path_array[g_path[:,0],g_path[:,1]] = 255
        bev = np.dstack([obs_array, g_path_array])


        occ_map = torch.as_tensor(bev, dtype = dtype)
        occ_map = torch.permute(occ_map, (2,0,1)) / 255.0 
        
        g_path = data['g_path'] - [256/2,256/2] # global path points
        g_path = g_path[:, [1,0]]*30/256
        
        obs_pos = np.array(to_continuous(obs_array))
                
        nmppi = Goal_Sampler(torch.tensor([0,0,np.deg2rad(90)]), 4.13, 0, obstacles=obs_pos, num_particles = 100)
        t1 = time.time()
        model.eval()
        with torch.no_grad():
            nmppi.mean_action = model(occ_map.unsqueeze(0)).reshape(30,2) # NN output reshaped
        # print(sampler1.mean_action)
        # quit()
        nmppi.infer_traj()
        t2 = time.time()
        nn_time.append(t2-t1)
        print("Neuro MPPI Inference time: ", t2-t1)

        t1 = time.time()
        new_g_path, interpolated_g_path, theta = global_traj(g_path, 0.1)
        ego_theta = np.rad2deg(np.pi/2 + (np.pi/2 - theta[0]))
        obs_pos_frenet = global_to_frenet(obs_pos, new_g_path, interpolated_g_path)
        mppi = Goal_Sampler(torch.tensor([0,0,np.deg2rad(90)]), 4.13, 0, obstacles=obs_pos_frenet, num_particles = 100)
        mppi.plan_traj()
        mean_controls1 = mppi.mean_action
        mean_traj1 = mppi.traj_N[-2,:,:]
        cov_controls1 = mppi.scale_tril
        mean_controls1[:,1] = frenet_to_global(mean_traj1, new_g_path, interpolated_g_path, 0.1)
        mppi.obstacles = obs_pos
        mppi.mean_action = torch.as_tensor(mean_controls1)
        mppi.c_state = torch.tensor([0,0,np.deg2rad(90)])
        mppi.infer_traj()
        print("MPPI Inference time: ", time.time()-t1)

        t1 = time.time()
        new_g_path, interpolated_g_path, theta = global_traj(g_path, 0.1)
        ego_theta = np.rad2deg(np.pi/2 + (np.pi/2 - theta[0]))
        obs_pos_frenet = global_to_frenet(obs_pos, new_g_path, interpolated_g_path)
        gradcem = GradCEM(torch.tensor([0,0,np.deg2rad(90)]), 4.13, 0, obstacles=obs_pos, num_particles = 100)
        gradcem.plan_traj()
        mean_controls1 = gradcem.mean_action
        mean_traj1 = gradcem.traj_N[-2,:,:]
        cov_controls1 = gradcem.scale_tril
        mean_controls1[:,1] = frenet_to_global(mean_traj1.detach(), new_g_path, interpolated_g_path, 0.1)
        gradcem.obstacles = obs_pos
        gradcem.mean_action = torch.as_tensor(mean_controls1)
        gradcem.c_state = torch.tensor([0,0,np.deg2rad(90)])
        gradcem.infer_traj()
        print("GradCEM Inference time: ", time.time()-t1)
        
        
        # print("Total time: ", time.time()-t_1)
        # quit()

        plt.scatter(g_path[:,0],g_path[:,1],color='blue', alpha=0.1, label = "Global path")

        x_car, y_car = draw_circle(0, 0, 1.80)
        plt.plot(x_car,y_car,'g')
        
        # for j in range(obs_pos.shape[0]):
        # print(obs_pos[:][:,0])
        
        plt.plot(obs_pos[:,0], obs_pos[:,1], 'k.')
            
        # for j in range(sampler.traj_N.shape[0]):
        plt.plot(nmppi.traj_N[:,:,0], nmppi.traj_N[:,:,1], '.b', markersize=1, alpha=0.1)
        # plt.plot(sampler2.traj_N[-2,:,0], sampler2.traj_N[-2,:,1], 'orange', markersize=3,  label = "Ground Truth")
        
        # plt.plot(mppi.traj_N[-2,:,0], mppi.traj_N[-2,:,1], 'green', markersize=3, label = "MPPI")
        # plt.plot(gradcem.traj_N[-2,:,0].detach(), gradcem.traj_N[-2,:,1].detach(), 'blue', markersize=3, label = "GradCEM")
        plt.plot(nmppi.traj_N[-2,:,0], nmppi.traj_N[-2,:,1], 'red', markersize=3.5, label = "Neuro-MPPI")
        plt.plot(nmppi.top_trajs[0,:,0], nmppi.top_trajs[0,:,1], 'green', markersize=2, label = "Best traj")
        # plt.plot(sampler2.top_trajs[0,:,0], sampler2.top_trajs[0,:,1], 'pink', markersize=2, label = "GT best traj")
        
        plt.ylim(-15,15)
        plt.xlim(-15,15)
        plt.legend(loc="lower center")
        plt.xticks([])
        plt.yticks([])
        # print(sampler.top_trajs[0,:,:2])
        # plt.savefig(infer_file_name, dpi=500)
        plt.show()
        plt.clf()
        # quit()
    nn_time = np.array(nn_time)
    print(np.mean(nn_time))
    np.save(args.time_dir+'nn_time', nn_time)




if __name__=='__main__':
    run()
