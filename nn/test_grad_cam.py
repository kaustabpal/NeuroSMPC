import torch
from dataset_pipeline.goal_sampler_static_obs import Goal_Sampler
from dataset_pipeline.grad_cem import GradCEM
from dataset_pipeline.frenet_transformations import global_traj, global_to_frenet, frenet_to_global
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
from torchvision import models
np.set_printoptions(suppress=True)
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
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, RawScoresOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
# from frenet_transformations import global_traj, global_to_frenet, frenet_to_global
# torch.manual_seed(42)

@dataclass
class Args:
    dataset_dir: str = '/Users/kaustabpal/work/iros_23/5k_data/train/' # occ_map/' #'../carla_latest/' # 'data/dataset_beta/'
    # dataset_dir: str = '/Users/kaustabpal/Downloads/experiments_22-02-2023/NuroMPPI_2-1/' # occ_map/' #'../carla_latest/' # 'data/dataset_beta/'
    weights_dir: str = '../iros_23/weights/' 
    loss_dir: str = '../iros_23/loss/' 
    infer_dir: str = "/Users/kaustabpal/work/iros_23/5k_data/train/infer_dir/"
    grad_cam_dir: str = "/Users/kaustabpal/work/iros_23/5k_data/train/grad_cam_dir/"
    time_dir: str = "../iros_23/time_dir/"
    val_split: float = 0.3
    num_epochs: int = 1000
    seed: int = 12321
    exp_id: str = 'exp12'
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
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    os.makedirs(args.grad_cam_dir, exist_ok=True)
    os.makedirs(args.grad_cam_dir+args.exp_id +"/", exist_ok=True)
    
    model = Model1()
    weights = torch.load(args.weights_dir+"model_"+args.exp_id+".pt", map_location=torch.device(device))
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    target_layers = model.fe[-1]
    # quit()
    # model = models.resnet50(pretrained=True)
    # print(model.layer4)
    # quit()
    
    occ_map_files = [f for f in os.listdir(args.dataset_dir+"occ_map") if not f.startswith('.')]
    print("Dataset size: ", len(occ_map_files))
    
    nn_time = []
    mppi_time = []
    gradcem_time = []
    
    for i in range(0,len(occ_map_files)):
        t_1 = time.time()
        print(i)
        obs_pos = []
        file_name = args.dataset_dir+"occ_map/" + "data_" + str(i).zfill(5) + ".pkl"
        # plt_save_file_name = plot_im_dir + "data_" + str(i)
        mean_save_filename = args.dataset_dir+"mean_controls/" + "data_" + str(i).zfill(5) + ".npy"
        grad_cam_file_name = args.grad_cam_dir+args.exp_id +"/" + "data_" + str(i).zfill(5)
        
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

        input_tensor = occ_map.unsqueeze(0)
        targets = []
        for i in range(60):
            targets.append(ClassifierOutputTarget(i))
            
        # targets = [ClassifierOutputTarget(0),ClassifierOutputTarget(1),ClassifierOutputTarget(2),ClassifierOutputTarget(3)]
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        nmppi = Goal_Sampler(torch.tensor([0,0,np.deg2rad(90)]), 4.13, 0, obstacles=obs_pos, num_particles = 100)
        model.eval()
        with torch.no_grad():
            nmppi.mean_action = model(input_tensor).reshape(30,2) # NN output reshaped
            nmppi.mean_action[:,0] = nmppi.mean_action[:,0]*torch.tensor([4.13], device=device)
        nmppi.infer_traj()
        plt.scatter(g_path[:,0]*256/30+256/2, -g_path[:,1]*256/30+256/2,color='yellow', alpha=0.1, label = "Global path")
        
        # plt.show()
        x_car, y_car = draw_circle(256/2, 256/2, 1.80*256/30)
        plt.plot(x_car,y_car,'g')
        plt.plot((nmppi.traj_N[-2,:,0]*256/30+256/2), (-nmppi.traj_N[-2,:,1]*256/30+256/2), 'red', markersize=3.5, label = "NMPPI")
        plt.plot(obs_pos[:,0]*256/30+256/2, -obs_pos[:,1]*256/30+256/2, 'k.')
        plt.imshow(grayscale_cam, alpha=1)
        # plt.legend(loc="lower center")
        plt.savefig(grad_cam_file_name)
        # plt.show()
        plt.clf()
        # quit()





if __name__=='__main__':
    run()
