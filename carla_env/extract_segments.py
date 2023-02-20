import os
import numpy as np
import json

base_folder = "/scratch/parth.shah/carla_manual_2-20-2023/"

file_path = os.path.join(base_folder, "segments.json")
with open(file_path, "r") as f:
    data = json.load(f)

bev_folder = os.path.join(base_folder, "segments/bev")
data_folder = os.path.join(base_folder, "segments/storm")


print("Creating folders")
print(bev_folder)
os.makedirs(bev_folder, exist_ok=True)
print(data_folder)
os.makedirs(data_folder, exist_ok=True)

i = 0
total = 0
for folder_name in data.keys():
    segments = data[folder_name]
    print("##################################################################")
    print("Folder: {}".format(folder_name))
    for segment in segments:
        for j in range(segment[0], segment[1] + 1):
            curr_bev_path = os.path.join(base_folder, folder_name, "bev/bev_{}.jpg".format(j))
            curr_data_path = os.path.join(base_folder, folder_name, "storm/data_{}.pkl".format(j))
            
            new_bev_path = os.path.join(bev_folder, "bev_{:0>2}.jpg".format(i))
            new_data_path = os.path.join(data_folder, "data_{:0>2}.pkl".format(i))
            print("-----------------------------------------------------------")
            print("Copying {} to {}".format(curr_bev_path, new_bev_path))
            os.system("cp {} {}".format(curr_bev_path, new_bev_path))
            print("--")
            print("Copying {} to {}".format(curr_data_path, new_data_path))
            os.system("cp {} {}".format(curr_data_path, new_data_path))

            i += 1
        total = total + (segment[1] - segment[0] + 1)

print("Total: {}".format(total))