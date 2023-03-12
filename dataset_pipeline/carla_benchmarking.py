import torch
# from goal_sampler_static_obs import Goal_Sampler
from nn.model import Model1
from dataset_pipeline.goal_sampler_dyn_obs_lane_change import Goal_Sampler as gs
from dataset_pipeline.goal_sampler_static_obs import Goal_Sampler
from dataset_pipeline.grad_cem import GradCEM
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import time
# import cv2
import matplotlib.pyplot as plt
import pickle
import os
from dataset_pipeline.utils import rotate
from dataset_pipeline.utils import draw_circle
from dataset_pipeline.frenet_transformations import global_traj, global_to_frenet, frenet_to_global, global_to_frenet_lane, frenet_to_global_with_traj
import copy
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
	
	time_arr = np.linspace(0, 3.0, 31)
	device = "cpu"
	dtype = torch.float32

	t_1 = time.time()
	obs_pos = []
	file_name ="/home/aditya/Documents/DEB/data/experiments/NuroMPPI_temporal_6-1_old/data/" + "data_438.pkl"
	with open(file_name, "rb") as f:
		data = pickle.load(f)
	obs = data['obstable_array'] # obstacle pos in euclidean space
	
	g_path = data['g_path'] -[256/2,256/2] # global path points
	g_path = g_path[:, [1,0]]*30/256

	left_lane = data['left_lane'] -[256/2,256/2] # global path points
	left_lane = left_lane[:, [1,0]]*30/256

	right_lane = data['right_lane'] -[256/2,256/2] # global path points
	right_lane = right_lane[:, [1,0]]*30/256

	dyn_obs = data["dyn_obs"]
	# print(dyn_obs)

	obs_poses = []
	for o in range(len(dyn_obs)):
		y, x = dyn_obs[o][1], dyn_obs[o][0]
		# x = (x - 256/2)*30/256
		# y = (y - 256/2)*30/256
		# print(dyn_obs[o])
		rel_yaw = dyn_obs[o][2] - np.pi/2
		rel_yaw = np.pi/2 - rel_yaw
		# print(rel_yaw)
		new_theta = rel_yaw + np.deg2rad(-dyn_obs[o][4])*time_arr
		obs_path_x = y + dyn_obs[o][3]*time_arr*np.cos(new_theta)
		obs_path_y = x + dyn_obs[o][3]*time_arr*np.sin(new_theta)
		traj = np.vstack((obs_path_x, obs_path_y)).T
		# print(traj)
		# quit()
		obs_poses.append(traj)
	
	obs_pos = np.array(to_continuous(obs))

	new_g_path, interpolated_g_path, theta = global_traj(g_path, 0.1)

	ego_theta = np.rad2deg(np.pi/2 + (np.pi/2 - theta[0]))
	# print(ego_theta)
	obs_pos_frenet = []
	for i in range(len(dyn_obs)):
		obs_traj = global_to_frenet(obs_poses[i], new_g_path, interpolated_g_path)
		obs_pos_frenet.append(obs_traj)
	
	# print(left_lane)
	left_lane_frenet = global_to_frenet_lane(left_lane, new_g_path, interpolated_g_path)
	right_lane_frenet = global_to_frenet_lane(right_lane, new_g_path, interpolated_g_path)

	for i in range(len(obs_pos_frenet)):
		plt.scatter(obs_poses[i][:,0],obs_poses[i][:,1],color='grey')

	obs_pos_frenet = np.array(obs_pos_frenet)


	fn ="/home/aditya/Documents/DEB/data/experiments/NuroMPPI_temporal_6-1_old/run_info/" + "run_info.pkl"
	with open(fn, "rb") as f:
		data_nmppi_dyn = pickle.load(f)
	traj_nmppi_dyn = data_nmppi_dyn["best_path"][434]


	sampler = Goal_Sampler(torch.tensor([0,0,np.deg2rad(ego_theta)]), 4.0, 0.0, obstacles=obs_pos)
	sampler.left_lane_bound = np.median(left_lane_frenet[:, :1])
	sampler.right_lane_bound = np.median(right_lane_frenet[:, :1])
	t1 = time.time()
	sampler.plan_traj()
	print("Planning time: ", time.time()-t1)
	mean_traj = sampler.traj_N[-2,:,:]
	traj, controls = frenet_to_global_with_traj(mean_traj.numpy(), new_g_path, interpolated_g_path, 0.1)
	sampler.obstacles = obs_pos
	sampler.c_state = torch.tensor([0,0,np.deg2rad(90)])
	sampler.infer_traj()
	traj_sampler = sampler.traj_N[-2,:,:]
	
	
	obs_array = data['obstable_array']
	obs_pos = np.array(to_continuous(obs_array))
	cem = GradCEM(torch.tensor([0,0,np.deg2rad(ego_theta)]), 4.0, 0.0, obstacles=obs_pos)
	cem.left_lane_bound = np.median(left_lane_frenet[:, :1])
	cem.right_lane_bound = np.median(right_lane_frenet[:, :1])
	t1 = time.time()
	cem.plan_traj()
	print("Planning time: ", time.time()-t1)
	mean_traj = cem.traj_N[-2,:,:]
	traj, controls = frenet_to_global_with_traj(mean_traj.detach().numpy(), new_g_path, interpolated_g_path, 0.1)
	cem.obstacles = obs_pos
	cem.c_state = torch.tensor([0,0,np.deg2rad(90)])
	cem.infer_traj()
	traj_cem = cem.traj_N[-2,:,:].detach().numpy()



	model = Model1()
	weights = torch.load("data/weights/model_exp19.pt", map_location=torch.device(device))
	model.load_state_dict(weights)
	model.to(device)
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
	t1 = time.time()
	nmppi = Goal_Sampler(torch.tensor([0,0,np.deg2rad(90)]), 4.13, 0, obstacles=obs_pos, num_particles = 100)
	model.eval()
	with torch.no_grad():
		tic = time.time()
		nmppi.mean_action = model(occ_map.unsqueeze(0)).reshape(30,2) # NN output reshaped
		toc = time.time()
		print(toc-tic)
		# quit()
		nmppi.mean_action[:,0] = nmppi.mean_action[:,0]*torch.tensor([4.13], device=device)
	nmppi.infer_traj()
	traj_stat = nmppi.traj_N[-2,:,:]

	plt.scatter(g_path[:,0],g_path[:,1],color='blue', alpha=0.1)

	rect = Rectangle((-0.965, -2.345), 1.93, 4.69, linewidth=1, edgecolor='r', facecolor='none')
	plt.gca().add_patch(rect)
	plt.plot(obs_pos[:,0], obs_pos[:,1], 'k.')
	plt.plot(traj_sampler[:, 0], traj_sampler[:, 1], 'g', label="MPPI")
	plt.plot(traj_cem[:, 0], traj_cem[:, 1], 'orange', label="GradCEM")
	plt.plot(traj_stat[:, 0], traj_stat[:, 1], 'fuchsia', label="NSMPC Static (Ours)")
	plt.plot(traj_nmppi_dyn[:, 1], traj_nmppi_dyn[:, 0], 'b', label="NSMPC Dynamic (Ours)")
	plt.legend(loc="lower center")
	# plt.axis('equal')
	plt.xlim(-15, 15)
	plt.ylim(-15, 15)
	plt.savefig("data/curved.png", dpi=1000)
	plt.show()




if __name__=='__main__':
	run()
