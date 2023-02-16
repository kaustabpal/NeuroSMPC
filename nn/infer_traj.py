import torch
from dataset_pipeline.goal_sampler_static_obs import Goal_Sampler
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
    dataset_dir: str = '../carla_latest/' # 'data/dataset_beta/'
    weights_dir: str = '../iros_23/weights/' 
    loss_dir: str = '../iros_23/loss/' 
    infer_dir: str = "../iros_23/infer_dir/"
    val_split: float = 0.3
    num_epochs: int = 1000
    seed: int = 12321
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    np.set_printoptions(suppress=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.makedirs(args.infer_dir, exist_ok=True)
    
    model = Model1()
    weights = torch.load(args.weights_dir+"model_"+args.exp_id+".pt", map_location=torch.device(device))
    model.load_state_dict(weights)
    model.to(device)
    occ_map_files = [f for f in os.listdir(args.dataset_dir+"occ_map") if not f.startswith('.')]
    # print(len(files))
    # quit()
    

    # torch.save(model.state_dict(), model_path)
    for i in range(len(occ_map_files)):
        t_1 = time.time()
        print(i)
        obs_pos = []
        file_name = args.dataset_dir+"occ_map/" + "data_" + str(i).zfill(5) + ".pkl"
        # plt_save_file_name = plot_im_dir + "data_" + str(i)
        # mean_save_filename = args.dataset_dir+"mean_controls/" + "data_" + str(i) + ".npy"
        infer_file_name = args.infer_dir + "data_" + str(i).zfill(5)
        mean_controls_gt = np.load(args.dataset_dir+"mean_controls/" + "data_" + str(i).zfill(5) + ".npy")
        mean_controls_gt = torch.as_tensor(mean_controls_gt, dtype=dtype, device = device)
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

        sampler1 = Goal_Sampler(torch.tensor([0,0,np.deg2rad(90)]), 4.13, 0, obstacles=obs_pos, num_particles = 100)
        sampler2 = Goal_Sampler(torch.tensor([0,0,np.deg2rad(90)]), 4.13, 0, obstacles=obs_pos, num_particles = 100)
        sampler2.mean_action = mean_controls_gt
        # sampler1.num_particles = 1
        # sampler2.num_particles = 1
        t1 = time.time()
        model.eval()
        with torch.no_grad():
            sampler1.mean_action = model(occ_map.unsqueeze(0)).reshape(30,2) # NN output reshaped
        t1 = time.time()
        # print(sampler1.mean_action)
        # quit()
        sampler1.infer_traj()
        print("Inference time: ", time.time()-t1)
        sampler2.infer_traj()
        print("Total time: ", time.time()-t_1)
        # quit()

        plt.scatter(g_path[:,0],g_path[:,1],color='blue', alpha=0.1)

        x_car, y_car = draw_circle(0, 0, 1.80)
        plt.plot(x_car,y_car,'g')
        
        # for j in range(obs_pos.shape[0]):
        # print(obs_pos[:][:,0])
        
        plt.plot(obs_pos[:,0], obs_pos[:,1], 'k.')
            
        # for j in range(sampler.traj_N.shape[0]):
        plt.plot(sampler1.traj_N[:,:,0], sampler1.traj_N[:,:,1], '.b', markersize=1, alpha=0.04)
        plt.plot(sampler2.traj_N[-2,:,0], sampler2.traj_N[-2,:,1], 'orange', markersize=3,  label = "Ground Truth")
        plt.plot(sampler1.traj_N[-2,:,0], sampler1.traj_N[-2,:,1], 'red', markersize=3, label = "Predicted")
        plt.plot(sampler1.top_trajs[0,:,0], sampler1.top_trajs[0,:,1], 'lime', markersize=2, label = "best traj")
        
        plt.ylim(-15,15)
        plt.xlim(-15,15)
        plt.legend(loc="lower center")
        # print(sampler.top_trajs[0,:,:2])
        plt.savefig(infer_file_name, dpi=500)
        # plt.show()
        plt.clf()
        # quit()




if __name__=='__main__':
    run()
