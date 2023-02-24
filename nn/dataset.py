import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from natsort import natsorted, ns
from matplotlib import pyplot as plt


class Im2ControlsDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.occ_map_dir = dataset_dir+'occ_map/'
        self.controls_dir = dataset_dir+'mean_controls/'
        self.occ_map_files_unsorted = [f for f in os.listdir(self.occ_map_dir) if not f.startswith('.')] 
        self.controls_files_unsorted = [f for f in os.listdir(self.controls_dir) if not f.startswith('.')]
        self.occ_map_files =natsorted(self.occ_map_files_unsorted, key=lambda y: y.lower())
        self.controls_files =natsorted(self.controls_files_unsorted, key=lambda y: y.lower())
        # print(self.occ_map_files[0], self.controls_files[0])
        # print(self.occ_map_files[1], self.controls_files[1])
        # print(self.occ_map_files[5], self.controls_files[5])
        # quit()
        self.len = len(self.occ_map_files)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # TODO: Normalize controls and inputs
        dtype = torch.float32
        with open(self.occ_map_dir+self.occ_map_files[idx], "rb") as f:
            data = pickle.load(f)

        # Plotting g_path on occ_map in separate channel
        g_path_array = np.zeros((256,256))
        obs_array = data['obstable_array']
        g_path = data['g_path']
        g_path[:,0] = -g_path[:,0] + 256 # global path points
        g_path = g_path.astype(np.int32)
        g_path = g_path[np.where( (g_path[:, 0]<256) & (g_path[:, 1]<256) & (g_path[:, 0]>=0) & (g_path[:, 1]>=0) )]
        g_path_array[g_path[:,0],g_path[:,1]] = 255
        bev = np.dstack([obs_array, g_path_array])


        occ_map = torch.as_tensor(bev, dtype = dtype)
        occ_map = torch.permute(occ_map, (2,0,1)) / 255.0 # Use better normalization
        controls = torch.as_tensor(np.load(self.controls_dir+self.controls_files[idx]), dtype = dtype).flatten()
        sample = {'occ_map': occ_map, 'controls': controls}

        return sample


class Im2ControlsDataset_Temporal(Dataset):
    def __init__(self, dataset_dir, transform=None, past_frames=5):
        self.occ_map_dir = dataset_dir+'temp/storm/'
        self.controls_dir = dataset_dir+'mean_controls/'
        self.past_frames = past_frames
        self.sequences = [
            [0,     1008],
            [1009,  1257],
            [1258,  1683],
            [2367,  3092],
            [3093,  3867],
            [3868,  5066],
            [5067,  5202],
        ]
        self.occ_map_files_unsorted = [f for f in os.listdir(self.occ_map_dir) if not f.startswith('.')] 
        self.controls_files_unsorted = [f for f in os.listdir(self.controls_dir) if not f.startswith('.')]
        self.occ_map_files =natsorted(self.occ_map_files_unsorted, key=lambda y: y.lower())
        self.controls_files =natsorted(self.controls_files_unsorted, key=lambda y: y.lower())
        self.idx_map = self._get_idx_map()
        self.len = len(self.idx_map)

    def _get_idx_map(self):
        idx_map = []
        for seq in self.sequences:
            for i in range(seq[0], seq[1]-3):
                idx_map.append(i)
        return idx_map


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # TODO: Normalize controls and inputs
        idx = self.idx_map[idx]
        dtype = torch.float32
        occ_map_file = 'data_'+str(idx).zfill(2)+'.pkl'
        with open(self.occ_map_dir+occ_map_file, "rb") as f:
            data = pickle.load(f)

        # Plotting g_path on occ_map in separate channel
        g_path_array = np.zeros((256,256))
        # place_holder = np.zeros((256,256))
        g_path = data['g_path']
        g_path[:,0] = -g_path[:,0] + 256 # global path points
        g_path = g_path.astype(np.int32)
        g_path = g_path[np.where( (g_path[:, 0]<256) & (g_path[:, 1]<256) & (g_path[:, 0]>=0) & (g_path[:, 1]>=0) )]
        g_path_array[g_path[:,0],g_path[:,1]] = 255

        obs_array = data['obstable_array']
        bev = np.dstack([obs_array, g_path_array])
        
        for i in range(1, self.past_frames):
            occ_map_file = 'data_'+str(idx+i).zfill(2)+'.pkl'
            with open(self.occ_map_dir+occ_map_file, "rb") as f:
                data = pickle.load(f)
            obs_array = data['obstable_array']
            bev = np.dstack([obs_array, bev])

        # bev_vis = np.dstack([obs_array, g_path_array, place_holder])
        # plt.imshow(bev_vis)
        # plt.show()

        occ_map = torch.as_tensor(bev, dtype = dtype)
        occ_map = torch.permute(occ_map, (2,0,1)) / 255.0 # Use better normalization
        controls_file = 'data_'+str(idx + self.past_frames - 1).zfill(2)+'.npy'
        controls = torch.as_tensor(np.load(self.controls_dir+controls_file), dtype = dtype).flatten()
        sample = {'occ_map': occ_map, 'controls': controls}

        return sample

