import torch
from dataset_pipeline.goal_sampler_static_obs import Goal_Sampler
from dataset_pipeline.grad_cem import GradCEM
from dataset_pipeline.frenet_transformations import global_traj,\
    global_to_frenet, frenet_to_global
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision import models
import time
from nn.model import Model1, Model_Temporal
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
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget,\
    RawScoresOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from natsort import natsorted, ns
# from frenet_transformations import global_traj, global_to_frenet,\
#     frenet_to_global
# torch.manual_seed(42)
np.set_printoptions(suppress=True)


@dataclass
class Args:
    dataset_dir: str = '/Users/kaustabpal/Downloads/storm/'
    weights_dir: str = '../iros_23/weights/'
    loss_dir: str = '../iros_23/loss/'
    infer_dir: str = "/Users/kaustabpal/Downloads/storm/infer_dir/"
    grad_cam_dir: str = "/Users/kaustabpal/Downloads/storm/grad_cam_dir/"
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
            if (obs[j, k] == 255):
                new_x, new_y = rotate([256/2, 256/2], [k, j], np.deg2rad(180))
                obs_pos.append([new_x*30/256, new_y*30/256])
    return obs_pos


def _get_idx_map(sequences):
    idx_map = []
    for seq in sequences:
        for i in range(seq[0], seq[1]-3):
            idx_map.append(i)
    return idx_map


def run():
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    os.makedirs(args.grad_cam_dir, exist_ok=True)
    os.makedirs(args.grad_cam_dir+args.exp_id + "/", exist_ok=True)
    sequences = [
            [0,     645],
            [646,  1442],
            [1443,  1939],
            [1940,  2538],
            [2539,  3106],
            [10000, 10633],
            [10634, 11284],
            [11285, 11634],
            [11635, 12067],
            [12068, 12541],
        ]
    past_frames = 5
    occ_map_dir = args.dataset_dir+"occ_map/"

    model = Model_Temporal(past_frames)
    weights = torch.load(args.weights_dir+"model_exp_temp_6-03_aug.pt",
                         map_location=torch.device(device))
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    target_layers = model.fe[-1]

    occ_map_files_unsorted = [f for f in os.listdir(args.dataset_dir+"occ_map")
                              if not f.startswith('.')]
    occ_map_files = natsorted(occ_map_files_unsorted,
                              key=lambda y: y.lower())
    idx_map = _get_idx_map(sequences)
    dataset_len = len(idx_map)
    print("Dataset size: ", len(idx_map))

    nn_time = []
    mppi_time = []
    gradcem_time = []
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    colors = ["mintcream", "lightcyan", "lightblue", "cornflowerblue"]
    for i in range(48, len(occ_map_files)):
        ax = plt.gca()
        idx = idx_map[i]
        print(i)
        occ_map_file = str(idx).zfill(4)+'.pkl'
        with open(occ_map_dir+occ_map_file, "rb") as f:
            data = pickle.load(f)
        grad_cam_file_name = args.grad_cam_dir+args.exp_id + "/" +\
            "data_" + str(i).zfill(4)

        g_path_array = np.zeros((256, 256))
        # place_holder = np.zeros((256,256))
        g_path = data['g_path']
        g_path[:, 0] = -g_path[:, 0] + 256  # global path points
        g_path = g_path.astype(np.int32)
        g_path = g_path[np.where((g_path[:, 0] < 256) & (g_path[:, 1] < 256) &
                                 (g_path[:, 0] >= 0) & (g_path[:, 1] >= 0))]
        g_path_array[g_path[:, 0], g_path[:, 1]] = 255

        obs_array = data['obstable_array']
        bev = np.dstack([obs_array, g_path_array])
        obs_pos = np.array(to_continuous(obs_array))
        plt.scatter(obs_pos[:, 0]*256/30+256/2, -obs_pos[:, 1]*256/30+256/2,
                    color=colors[0], s=1, label='T-4',  alpha=0.1)
        alph = 0.1
        sze = 1
        for j in range(1, past_frames-1):
            alph += 0.1
            sze += 0.2
            occ_map_file = str(idx+j).zfill(4)+'.pkl'
            with open(occ_map_dir+occ_map_file, "rb") as f:
                data = pickle.load(f)
            obs_array = data['obstable_array']
            bev = np.dstack([obs_array, bev])
            obs_pos = np.array(to_continuous(obs_array))
            plt.scatter(obs_pos[:, 0]*256/30+256/2,
                        -obs_pos[:, 1]*256/30+256/2, color=colors[j],
                        label='T-'+str(4-j), s=sze, alpha=alph)
        occ_map_file = str(idx+past_frames-1).zfill(4)+'.pkl'
        with open(occ_map_dir+occ_map_file, "rb") as f:
            data = pickle.load(f)
        obs_array = data['obstable_array']
        bev = np.dstack([obs_array, bev])
        obs_pos = np.array(to_continuous(obs_array))
        plt.scatter(obs_pos[:, 0]*256/30+256/2, -obs_pos[:, 1]*256/30+256/2,
                    color='k', label='T', s=2)

        occ_map = torch.as_tensor(bev, dtype=dtype)
        occ_map = torch.permute(
            occ_map, (2, 0, 1))/255.0  # Use better normalization

        #  Plotting g_path on occ_map in separate channel

        input_tensor = occ_map.unsqueeze(0)
        targets = []
        for i in range(60):
            targets.append(ClassifierOutputTarget(i))

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        nmppi = Goal_Sampler(torch.tensor([0, 0, np.deg2rad(90)]),
                             4.13, 0, obstacles=obs_pos, num_particles=50)
        model.eval()
        with torch.no_grad():
            nmppi.mean_action = model(input_tensor).reshape(
                30, 2)  # NN output reshaped
            nmppi.mean_action[:, 0] = nmppi.mean_action[:, 0]*torch.tensor(
                [4.13], device=device)
        nmppi.infer_traj()
        plt.scatter(g_path[:, 0]*256/30+256/2, -g_path[:, 1]*256/30+256/2,
                    color='yellow', alpha=0.1)

        rect = patches.Rectangle((-0.965*256/30+256/2, -2.345*256/30+256/2),
                                 1.93*256/30, 4.69*256/30, linewidth=1,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.plot((nmppi.traj_N[-2, :, 0]*256/30+256/2), (
            -nmppi.traj_N[-2, :, 1]*256/30+256/2),
                 'red', markersize=3.5, label="NSMPC")
        # plt.plot(obs_pos[:, 0]*256/30+256/2, -obs_pos[:, 1]*256/30+256/2, 'k.', label='T')
        plt.imshow(grayscale_cam, alpha=1)
        plt.legend(loc="upper left", framealpha=0.2)
        plt.savefig(grad_cam_file_name)
        # plt.show()
        plt.clf()
        # quit()


if __name__ == '__main__':
    run()
