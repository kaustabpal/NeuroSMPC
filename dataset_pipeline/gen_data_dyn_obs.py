import torch
# from goal_sampler_static_obs import Goal_Sampler
from dataset_pipeline.goal_sampler_dyn_obs_lane_change import Goal_Sampler
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import time
# import cv2
import matplotlib.pyplot as plt
import pickle
import os
from utils import rotate
from utils import draw_circle
from frenet_transformations import global_traj, global_to_frenet, frenet_to_global, global_to_frenet_lane
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
	# argParser = argparse.ArgumentParser()
	# argParser.add_argument("-d", "--dataset_dir", help="dataset_dir")
	# argParser.add_argument("-p", "--plot_im_dir", help="plot_image_dir")
	# argParser.add_argument("-m", "--mean_dir", help="mean_controls_dir")
	# args = argParser.parse_args()
	# dataset_dir = "storm/"
	dataset_dir = "/home/aditya/deb_data/segments/storm/"#args.dataset_dir
	plot_im_dir = "/home/aditya/deb_data/segments/plot_im/"#args.plot_im_dir
	mean_dir = "/home/aditya/deb_data/segments/mean_controls/"#args.mean_dir
	files = os.listdir(dataset_dir)
	time_arr = np.linspace(0, 3.0, 31)
	print(len(files)-1)
	for i in range(len(files)):
		t_1 = time.time()
		print(i)
		obs_pos = []
		file_name = dataset_dir + "data_" + str(i).zfill(4) + ".pkl"
		plt_save_file_name = plot_im_dir + "data_" + str(i).zfill(4)
		mean_save_filename = mean_dir + "data_" + str(i).zfill(4)
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

		# plt.scatter(interpolated_g_path[:,0],interpolated_g_path[:,1],color='blue', alpha=0.7)
		# plt.scatter(left_lane[:,0],left_lane[:,1],color='red', alpha=0.7)
		# plt.scatter(right_lane[:,0],right_lane[:,1],color='green', alpha=0.7)
		# plt.show()

		ego_theta = np.rad2deg(np.pi/2 + (np.pi/2 - theta[0]))
		# print(ego_theta)
		obs_pos_frenet = []
		for i in range(len(dyn_obs)):
			obs_traj = global_to_frenet(obs_poses[i], new_g_path, interpolated_g_path)
			obs_pos_frenet.append(obs_traj)
		
		# print(left_lane)
		left_lane_frenet = global_to_frenet_lane(left_lane, new_g_path, interpolated_g_path)
		right_lane_frenet = global_to_frenet_lane(right_lane, new_g_path, interpolated_g_path)

		# print(np.median(left_lane_frenet[:, :1]))
		# print(np.median(right_lane_frenet[:, :1]))
		# print(data["speed"])
		# quit()

		# plt.scatter(new_g_path[:,0],new_g_path[:,1],color='blue', alpha=0.7)
		# plt.scatter(left_lane_frenet[:,0],left_lane_frenet[:,1],color='red', alpha=0.7)
		# plt.scatter(right_lane_frenet[:,0],right_lane_frenet[:,1],color='green', alpha=0.7)
		# obs_pos_frenet = []
		# nxobs = 0.0 + np.cos(np.pi/2)*time_arr
		# nyobs = 5.0 + 1.0*np.sin(np.pi/2)*time_arr
		# traj = np.vstack((nxobs, nyobs)).T
		# obs_pos_frenet.append(traj)

		# nxobs = 3.5 + np.cos(np.pi/2)*time_arr
		# nyobs = 5.0 + 10.0*np.sin(np.pi/2)*time_arr
		# traj = np.vstack((nxobs, nyobs)).T
		# # obs_pos_frenet = traj.reshape(1, 31, 2)
		# obs_pos_frenet.append(traj)

		# print(obs_pos_frenet)

		for i in range(len(obs_pos_frenet)):
			plt.scatter(obs_poses[i][:,0],obs_poses[i][:,1],color='black', alpha=0.7)

		obs_pos_frenet = np.array(obs_pos_frenet)
		# print(obs_pos_frenet[0])
		# print(obs_pos_frenet.shape)

		# print(data["speed"])

		sampler = Goal_Sampler(torch.tensor([0,0,np.deg2rad(ego_theta)]), 4.0, 0.0, obstacles=obs_pos_frenet)
		sampler.left_lane_bound = np.median(left_lane_frenet[:, :1])
		sampler.right_lane_bound = np.median(right_lane_frenet[:, :1])
		# sampler.initialize()

		t1 = time.time()
		sampler.plan_traj()
		# print(sampler.mean_action)

		# plt.plot(sampler.traj_N[:,:,0], sampler.traj_N[:,:,1], '.r', alpha=0.05)
		# plt.plot(sampler.traj_N[-2,:,0], sampler.traj_N[-2,:,1], 'g')
		# plt.plot(sampler.top_trajs[0,:,0], sampler.top_trajs[0,:,1], 'blue')

		# plt.pause(0.001)
		# plt.axis('equal')
		# plt.clf()
		# continue
		print("Planning time: ", time.time()-t1)

		mean_controls = sampler.mean_action
		mean_traj = sampler.traj_N[-2,:,:]
		cov_controls = sampler.scale_tril
		# print(mean_controls)
		
		mean_controls[:,1] = frenet_to_global(mean_traj, new_g_path, interpolated_g_path, 0.1)
		# print(mean_controls)
		# quit()
		sampler.obstacles = obs_pos_frenet
		# sampler.mean_action = torch.as_tensor(mean_controls)
		sampler.c_state = torch.tensor([0,0,np.deg2rad(90)])
		sampler.infer_traj()
		np.save(mean_save_filename,sampler.mean_action)
		
		## plot
		# for k in range(g_path.shape[0]):
		plt.scatter(g_path[:,0],g_path[:,1],color='blue', alpha=0.1)

		x_car, y_car = draw_circle(0, 0, 1.80)
		plt.plot(x_car,y_car,'g')
		
		# for j in range(obs_pos.shape[0]):
		# print(obs_pos[:][:,0])
		
		plt.plot(obs_pos[:,0], obs_pos[:,1], 'k.')
		if len(obs_pos_frenet) > 0:
			plt.scatter(dyn_obs[:,1], dyn_obs[:,0], color='orange')
			
		# for j in range(sampler.traj_N.shape[0]):
		plt.plot(sampler.traj_N[:,:,0], sampler.traj_N[:,:,1], '.r', alpha=0.05)
		plt.plot(sampler.traj_N[-2,:,0], sampler.traj_N[-2,:,1], 'g')
		plt.plot(sampler.top_trajs[0,:,0], sampler.top_trajs[0,:,1], 'blue')
		plt.plot(left_lane[:,0], left_lane[:,1], 'pink')
		plt.plot(right_lane[:,0], right_lane[:,1], 'yellow')
		print("Total time: ", time.time()-t_1)
		# print(sampler.top_trajs[0,:,:2])
		plt.savefig(plt_save_file_name)
		plt.axis('equal')
		plt.pause(0.001)
		# plt.show()
		plt.clf()
		# quit()




if __name__=='__main__':
	run()
