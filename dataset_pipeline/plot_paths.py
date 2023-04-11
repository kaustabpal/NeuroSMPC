import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def rotate(origin, point, angle=180):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return -qx+127, qy-127


def to_continuous(obs):
	obs_pos = []    
	for j in range(obs.shape[0]):
		for k in range(obs.shape[1]):
			if(obs[j,k]==255):
				new_x, new_y = rotate([256/2,256/2],[k,j],np.deg2rad(180))
				obs_pos.append([new_x*30/256,new_y*30/256])
	return obs_pos


fn ="/home/aditya/Documents/DEB/data/experiments/NuroMPPI_temporal_5-3_circle_plot/run_info/" + "run_info.pkl"
with open(fn, "rb") as f:
    data_nmppi_dyn = pickle.load(f)
nsmpc_path = np.array(data_nmppi_dyn["ego_path"][1:])

fn ="/home/aditya/Documents/DEB/data/experiments/MPPI_static_5-3_og/run_info/" + "run_info.pkl"
with open(fn, "rb") as f:
    data_nmppi_dyn = pickle.load(f)
mppi_path = np.array(data_nmppi_dyn["ego_path"][1:])

fn ="/home/aditya/Documents/DEB/data/experiments/MPPI_static_5-3/run_info/" + "run_info.pkl"
with open(fn, "rb") as f:
    data_nmppi_dyn = pickle.load(f)
gradcem_path = np.array(data_nmppi_dyn["ego_path"][1:])

fn ="/home/aditya/Documents/DEB/data/experiments/NuroMPPI_temporal_5-3_circle_plot/data/" + "data_149.pkl"
with open(fn, "rb") as f:
    data = pickle.load(f)
obs_array = data['obstable_array']
obs_pos = np.array(to_continuous(obs_array))

plt.plot(obs_pos[:,0], obs_pos[:,1], 'k.')

# plt.plot(nsmpc_path[:, 0], -1*nsmpc_path[:, 1], 'blue', label="NSMPC (Ours)")
# plt.plot(mppi_path[:520, 0], -1*mppi_path[:520, 1], 'green', label="MPPI")
# plt.plot(gradcem_path[:480, 0], -1*gradcem_path[:480, 1], 'red', label="GradCEM")
plt.legend()
plt.axis('equal')
# plt.savefig("data/path_comp.png", dpi=500)
plt.savefig("data/obs.png", dpi=500)
plt.show()