import torch
from dataset_pipeline.goal_sampler_static_obs import Goal_Sampler
import numpy as np
from nn.model import Model1, Model_Temporal
import matplotlib.pyplot as plt
from dataset_pipeline.utils import rotate
from dataset_pipeline.utils import draw_circle
from dataclasses import dataclass 
import tyro
from nn.dataset import Im2ControlsDataset_Temporal
from torch.utils.data import DataLoader
import os




@dataclass
class Args:
    dataset_dir: str = 'data/carla_dyn_obs_data/' #'../iros_23/dataset/' #'/scratch/kaustab.pal/iros_23/dataset/' # 'data/dataset_beta/'
    weights_dir: str = 'data/weights/' #'../iros_23/weights/' #'/scratch/kaustab.pal/iros_23/weights/' 
    loss_dir: str = 'data/loss/'  #'../iros_23/loss/' #'/scratch/kaustab.pal/iros_23/loss/' 
    infer_dir: str = "data/infer_dir/"
    time_dir: str = "data/time_dir/"
    val_split: float = 0.3
    num_epochs: int = 500
    seed: int = 12321
    past_frames: int = 5
    exp_id: str = 'exp1_temp'
args = tyro.cli(Args)


def to_continuous(obs):
    obs_pos = []    
    for j in range(obs.shape[0]):
        for k in range(obs.shape[1]):
            if(obs[j,k]==255):
                new_x, new_y = rotate([256/2,256/2],[k,j],np.deg2rad(180))
                obs_pos.append([new_x*30/256,new_y*30/256])
    return obs_pos


def infer_dataset():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.makedirs(args.infer_dir, exist_ok=True)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = Im2ControlsDataset_Temporal(dataset_dir=args.dataset_dir, past_frames=args.past_frames)

    model = Model_Temporal(past_frames=args.past_frames)
    weights = torch.load(args.weights_dir+"model_"+args.exp_id+".pt", map_location=torch.device(device))
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    count = -1
    for datapoint in dataset:
        print(count)
        count+=1
        output = model(datapoint['occ_map'].unsqueeze(0).to(device))

        obs_pos = datapoint['occ_map'][0,:,:]*255
        obs_pos = np.array(to_continuous(obs_pos))
        sampler1 = Goal_Sampler(torch.tensor([0,0,np.deg2rad(90)]), 4.13, 0, obstacles=obs_pos, num_particles = 100)
        sampler1.mean_action = output.reshape(30,2).detach().cpu() 
        sampler1.infer_traj()

        sampler2 = Goal_Sampler(torch.tensor([0,0,np.deg2rad(90)]), 4.13, 0, obstacles=obs_pos, num_particles = 100)
        sampler2.mean_action = datapoint['controls'].reshape(30,2)
        sampler2.infer_traj()


        x_car, y_car = draw_circle(0, 0, 1.80)
        plt.plot(x_car,y_car,'g')
        
        # for j in range(obs_pos.shape[0]):
        # print(obs_pos[:][:,0])
        
        plt.plot(obs_pos[:,0], obs_pos[:,1], 'k.')
            
        # for j in range(sampler.traj_N.shape[0]):
        plt.plot(sampler1.traj_N[:,:,0], sampler1.traj_N[:,:,1], '.b', markersize=1, alpha=0.04)
        plt.plot(sampler2.traj_N[-2,:,0], sampler2.traj_N[-2,:,1], 'orange', markersize=3,  label = "Ground Truth")
        plt.plot(sampler1.traj_N[-2,:,0], sampler1.traj_N[-2,:,1], 'red', markersize=3, label = "Predicted")
        plt.plot(sampler1.top_trajs[0,:,0], sampler1.top_trajs[0,:,1], 'lime', markersize=2, label = "NN best traj")
        plt.plot(sampler2.top_trajs[0,:,0], sampler2.top_trajs[0,:,1], 'pink', markersize=2, label = "GT best traj")

        plt.ylim(-15,15)
        plt.xlim(-15,15)
        plt.legend(loc="lower center")
        # print(sampler.top_trajs[0,:,:2])
        infer_file_name = args.infer_dir + "data_" + str(count).zfill(5)
        plt.savefig(infer_file_name, dpi=500)
        # plt.show()
        plt.clf()
    




if __name__=='__main__':
    infer_dataset()
