import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

class Im2ControlsDataset(Dataset):
    def __init__(self, occ_map_dir, controls_dir, transform=None):
        self.occ_map_dir = occ_map_dir
        self.controls_dir = controls_dir
        self.occ_map_files = [f for f in os.listdir(self.occ_map_dir) if not f.startswith('.')] 
        self.controls_files = [f for f in os.listdir(self.controls_dir) if not f.startswith('.')]
        self.len = len(self.occ_map_files)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        dtype = torch.float32
        with open(self.occ_map_dir+self.occ_map_files[idx], "rb") as f:
            data = pickle.load(f)
        occ_map = torch.as_tensor(data['obstable_array'], dtype = dtype).unsqueeze(0)
        controls = torch.as_tensor(np.load(self.controls_dir+self.controls_files[idx]), dtype = dtype).reshape(-1)
        sample = {'occ_map': occ_map, 'controls': controls}
        return sample

# dataset = Im2ControlsDataset("occ_map/","mean_controls/")
# print(len(dataset))
# print(dataset[0])
# print(dataset[1])
# print(dataset[2])
