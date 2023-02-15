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
# from frenet_transformations import global_traj, global_to_frenet, frenet_to_global
# torch.manual_seed(42)

@dataclass
class Args:
    dataset_dir: str = '../iros_23/dataset/' # 'data/dataset_beta/'
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

    files = os.listdir(args.dataset_dir)
    print(len(files)-1)
    

    # torch.save(model.state_dict(), model_path)
    for i in range(1,len(files)-1):
        t_1 = time.time()
        print(i)
        obs_pos = []
        file_name = dataset_dir + "data_" + str(i).zfill(5) + ".pkl"
        plt_save_file_name = plot_im_dir + "data_" + str(i)
        mean_save_filename = mean_dir + "data_" + str(i)
        infer_file_name =infer_dir + "data_" + str(i)
        
        with open(file_name, "rb") as f:
            data = pickle.load(f)
        obs = data['obstable_array']
        occ_map = torch.as_tensor(data['obstable_array'], dtype = dtype).unsqueeze(0) # obstacle pos in euclidean space
        print(obs.shape)
        g_path = data['g_path'] -[256/2,256/2] # global path points
        g_path = g_path[:, [1,0]]*30/256
        
        obs_pos = np.array(to_continuous(obs))

        # new_g_path, interpolated_g_path, theta = global_traj(g_path, 0.1)

        # ego_theta = np.rad2deg(np.pi/2 + (np.pi/2 - theta[0]))
        # print(ego_theta)
        
        # obs_pos_frenet = global_to_frenet(obs_pos, new_g_path, g_path)

        sampler = Goal_Sampler(torch.tensor([0,0,np.deg2rad(90)]), 4.13, 0, obstacles=obs_pos)
        sampler.num_particles = 100
        t1 = time.time()
        model.eval()
        with torch.no_grad():
            sampler.mean_action = model(occ_map.unsqueeze(0)).reshape(30,2) # NN output reshaped
        t1 = time.time()
        sampler.infer_traj()
        print("Inference time: ", time.time()-t1)
        quit()

        # mean_controls = sampler.mean_action
        # mean_traj = sampler.traj_N[-2,:,:]
        # cov_controls = sampler.scale_tril
        # print(mean_controls)
        
        # mean_controls[:,1] = frenet_to_global(mean_traj, new_g_path, interpolated_g_path, 0.1)
        # print(mean_controls)
        # quit()
        # sampler.obstacles = obs_pos
        # sampler.mean_action = torch.as_tensor(mean_controls)
        # sampler.c_state = torch.tensor([0,0,np.deg2rad(90)])
        # sampler.infer_traj()
        # np.save(mean_save_filename,sampler.best_traj)
        
        ## plot
        # for k in range(g_path.shape[0]):
        plt.scatter(g_path[:,0],g_path[:,1],color='blue', alpha=0.1)

        x_car, y_car = draw_circle(0, 0, 1.80)
        plt.plot(x_car,y_car,'g')
        
        # for j in range(obs_pos.shape[0]):
        # print(obs_pos[:][:,0])
        
        plt.plot(obs_pos[:,0], obs_pos[:,1], 'k.')
            
        # for j in range(sampler.traj_N.shape[0]):
        plt.plot(sampler.traj_N[:,:,0], sampler.traj_N[:,:,1], '.r', markersize=1, alpha=0.05)
        plt.plot(sampler.traj_N[-2,:,0], sampler.traj_N[-2,:,1], '.g', markersize=1)
        plt.plot(sampler.top_trajs[0,:,0], sampler.top_trajs[0,:,1], 'blue', markersize=1)
        print("Total time: ", time.time()-t_1)
        plt.ylim(-15,15)
        plt.xlim(-15,15)
        # print(sampler.top_trajs[0,:,:2])
        plt.savefig(infer_file_name)
        # plt.show()
        plt.clf()
        # quit()




if __name__=='__main__':
    run()
