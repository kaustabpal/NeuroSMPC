import torch
import ghalton
import numpy as np
np.random.seed(42)
torch.manual_seed(42)
import matplotlib.pyplot as plt
import copy
import time
# from behavior_kit import utils
# from behavior_kit.utils import get_dist
from scipy.interpolate import BSpline
import scipy.interpolate as si
from torch import optim

np.set_printoptions(suppress=True)

class GradCEM:
    def __init__(self, c_state, vl, wl, obstacles, num_particles = 1000, device='cpu'):
        # agent info
        self.device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu") # "cpu"
        self.radius = 1.80
        self.c_state = c_state # start state
        # self.c_state.device = self.device
        self.c_state.requires_grad = True
        # self.g_state = g_state # goal state
        self.step_size_mean = 0.7
        self.step_size_cov = 0.2
        self.avoid_obs = False
        self.vl = vl
        self.wl = wl
        self.v_ub = 4.13
        self.v_lb = 0
        self.w_ub = 0.4
        self.w_lb = -0.4
        self.max_ctrl = torch.tensor([self.v_ub, self.w_ub], device=self.device)
        self.min_ctrl = torch.tensor([self.v_lb, self.w_lb], device=self.device)
        self.amin = -3.19
        self.amax = 3.19
        self.jmin = -0.1
        self.jmax = 0.1
        self.init_q = [self.vl, self.wl]

        # obstacle info
        self.obst_radius = 0.083
        self.obstacles = obstacles
        self.n_obst = 0
        
        # MPC params
        # self.N = 2 # Number of samples
        self.dt = 0.1
        self.horizon = 30 # Planning horizon
        self.d_action = 2
        self.knot_scale = 4
        self.n_knots = self.horizon//self.knot_scale
        self.ndims = self.n_knots*self.d_action
        self.bspline_degree = 3
        self.num_particles = int(max(20,num_particles)) #00
        self.top_K = int(0.05*self.num_particles) # Number of top samples
        
        self.null_act_frac = 0.01
        self.num_null_particles = round(int(self.null_act_frac * self.num_particles * 1.0))
        self.num_neg_particles = round(int(self.null_act_frac * self.num_particles)) -\
                                                            self.num_null_particles
        self.num_nonzero_particles = self.num_particles - self.num_null_particles -\
                                                            self.num_neg_particles
        self.sample_shape =  self.num_particles - 1

        if(self.num_null_particles > 0):
            self.null_act_seqs = torch.zeros(self.num_null_particles, self.horizon,\
                                                                    self.d_action)
        # self.initialize_mu()
        # self.initialize_sig()

        # Sampling params
        self.perms = ghalton.EA_PERMS[:self.ndims]
        self.sequencer = ghalton.GeneralizedHalton(self.perms)

        # init_q = torch.tensor(self.c_state)
        self.init_action = torch.zeros((self.horizon, self.d_action)) + torch.tensor(self.init_q)
        self.init_action.to(self.device)
        self.init_mean = self.init_action 
        self.mean_action = self.init_mean.clone()
        self.best_traj = self.mean_action.clone()
        self.init_v_cov = 0.9
        self.init_w_cov = 0.9
        self.init_cov_action = torch.tensor([self.init_v_cov, self.init_w_cov], device=self.device)
        self.cov_action = self.init_cov_action
        self.scale_tril = torch.sqrt(self.cov_action)
        self.full_scale_tril = torch.diag(self.scale_tril)
        
        self.gamma = 0.99
        self.gamma_seq = torch.cumprod(torch.tensor([1]+[self.gamma]*(self.horizon-1)),dim=0).reshape(1,self.horizon)

        self.traj_N = torch.zeros((self.num_particles, self.horizon+1, 3), device=self.device)
        self.controls_N = torch.zeros((self.num_particles, self.horizon, 2), device=self.device)
        self.controls_N.requires_grad = True


        self.top_trajs = torch.zeros((self.top_K, self.horizon+1, 3), device=self.device)
        self.top_controls = torch.zeros((self.top_K, self.horizon, 2), device=self.device)
        self.top_traj = self.c_state.reshape(1,3)*torch.ones((self.horizon+1, 3), device=self.device)

        self.optimizer = optim.RMSprop([self.controls_N], lr=0.1)
        # print(self.device)
        # self.curr_state_N = np.zeros((self.N,1,3))
        # self.V_N_T = np.zeros((self.N, self.horizon))
        # self.W_N_T = np.zeros((self.N, self.horizon))

    # def initialize_mu(self): # tensor contain initialized values'''
    #      self.MU = 0*torch.ones((2,self.horizon)) # 2 dim Mu for vel and Angular velocity
    
    # def initialize_sig(self):
    #     self.SIG = 0.7*torch.ones((2,self.horizon))
    
    def bspline(self, c_arr, t_arr=None, n=30, degree=3):
        sample_device = c_arr.device
        sample_dtype = c_arr.dtype
        cv = c_arr.cpu().numpy()
        if(t_arr is None):
            t_arr = np.linspace(0, cv.shape[0], cv.shape[0])
        # else:
        #     t_arr = t_arr.cpu().numpy()
        spl = si.splrep(t_arr, cv, k=degree, s=0.0)
        #spl = BSpline(t, c, k, extrapolate=False)
        xx = np.linspace(0, n, n)
        # print(xx)
        # quit()
        samples = si.splev(xx, spl, ext=3)
        samples = torch.as_tensor(samples, device=sample_device, dtype=sample_dtype)
        return samples
    
    def scale_controls(self, act_seq):
        return torch.max(torch.min(act_seq, self.max_ctrl),self.min_ctrl)

    def sample_controls(self, inference = False):
        uniform_halton_samples = torch.tensor(self.sequencer.get(self.sample_shape), device=self.device) # samples N control points
        erfinv = torch.erfinv(2 * uniform_halton_samples - 1)
        knot_points = torch.sqrt(torch.tensor([2.0],device=self.device)) * erfinv
        # print(knot_points.shape)
        knot_samples = knot_points.view(self.sample_shape, self.d_action, self.n_knots)
        # print(knot_samples.shape)
        self.samples = torch.zeros((self.sample_shape, self.horizon, self.d_action), device=self.device)
        # print(self.samples.shape)
        for i in range(self.sample_shape):
            for j in range(self.d_action):
                self.samples[i,:,j] = self.bspline(knot_samples[i,j,:],n = self.horizon, \
                                                            degree = self.bspline_degree)
        delta = self.samples
        z_seq = torch.zeros(1,self.horizon,self.d_action, device=self.device)
        delta = torch.cat((delta,z_seq),dim=0)
        scaled_delta = torch.matmul(delta, self.full_scale_tril).view(delta.shape[0],
                                                                    self.horizon,
                                                                    self.d_action)   
        act_seq = self.mean_action.unsqueeze(0).to(self.device) + scaled_delta.to(self.device)
        # if(inference == False):
        #     act_seq = torch.cat((act_seq,torch.(1,self.horizon,self.d_action)),dim=0)
        # if(inference == True):
        #     act_seq = torch.cat((act_seq,self.mean_action.unsqueeze(0)),dim=0)
        act_seq = self.scale_controls(act_seq)
        # append_acts = self.best_traj.unsqueeze(0)
        
        # if(self.num_null_particles > 0):
        #     # negative action particles:
        #     neg_action = torch.tensor([self.v_lb, 0]) * self.mean_action.unsqueeze(0)
        #     # print(neg_action)
        #     neg_act_seqs = neg_action.expand(self.num_neg_particles,-1,-1)
        #     append_acts = torch.cat((append_acts, self.null_act_seqs, neg_act_seqs),dim=0)

        # act_seq = torch.cat((act_seq, append_acts), dim=0)
        # print(self.controls_N.requires_grad)
        self.controls_N = act_seq
        # print(self.controls_N.requires_grad)
        self.controls_N.requires_grad = True
        # print(self.controls_N.requires_grad)
        
        
        # print(act_seq.shape, self.controls_N.shape)
        # return act_seq
        
    def rollout(self, s_o = 1, s_s = 1, s_c = 0.1, s_m = 0):
        # print(self.num_particles)
        # print(self.controls_N.shape[0])
        t_r = time.time()
        self.goal_region_cost_N = torch.zeros((self.traj_N.shape[0]), device=self.device)
        self.left_lane_bound_cost_N = torch.zeros((self.traj_N.shape[0]), device=self.device)
        self.right_lane_bound_cost_N = torch.zeros((self.traj_N.shape[0]), device=self.device)
        left_lane_bound = -4.5
        right_lane_bound = 4.5
        # self.in_balls_cost_N = torch.zeros((self.traj_N.shape[0]))
        self.collision_cost_N = torch.zeros((self.traj_N.shape[0]), device=self.device)
        self.ang_vel_cost_N = torch.zeros((self.controls_N.shape[0]), device=self.device)
        self.dist_to_mean_cost_N = 99*torch.ones((self.controls_N.shape[0]), device=self.device)
        self.center_line_cost_N = torch.zeros((self.controls_N.shape[0]), device=self.device)
        diag_dt = self.dt*torch.ones(self.horizon, self.horizon, device=self.device)
        diag_dt = torch.tril(diag_dt)
        t = []
        t_2 = []
        t_3 = []
        t_4 = []
        self.traj_N[:,0,:] = self.c_state.view(3)
        # print(self.controls_N.shape)
        for i in range(self.num_particles):
            t1 = time.time()
            # self.traj_N[i,0,:] = self.c_state.view(3)
            v = self.controls_N[i,:,0].reshape(-1,1)
            w = self.controls_N[i,:,1].reshape(-1,1)
            # print(self.controls_N.shape)
            w_dt = diag_dt@w.float()
            theta_0 = self.traj_N[i,0,2]*torch.ones(self.horizon,1, device=self.device)
            x_0 = self.traj_N[i,0,0]*torch.ones(self.horizon,1, device=self.device)
            y_0 = self.traj_N[i,0,1]*torch.ones(self.horizon,1, device=self.device)
            theta_new = theta_0 + w_dt
            c_theta = torch.cos(theta_new)
            s_theta = torch.sin(theta_new)
            v_cos_dt = (c_theta.squeeze(1)*diag_dt)@v.float()
            v_sin_dt = (s_theta.squeeze(1)*diag_dt)@v.float()
            x_new = x_0 + v_cos_dt
            y_new = y_0 + v_sin_dt
            self.traj_N[i,1:,:] = torch.hstack((x_new, y_new, theta_new))
            t.append(time.time() - t1)      
            # angular velocity constraints
            self.ang_vel_cost_N[i] = torch.sum(torch.diff(self.controls_N[i,:,1])**2) #torch.norm(self.controls_N[i,:,1])
            
            self.dist_to_mean_cost_N[i] = torch.linalg.norm(self.controls_N[i,:,1] - self.controls_N[-2,:,1]) #torch.norm(self.controls_N[i,:,1])
            
            # center-line cost
            self.center_line_cost_N[i] += torch.linalg.norm(self.traj_N[i,:,0]-0)
            
            # Obstacle avoidance
            t1 = time.time()
            threshold_dist = self.radius + self.obst_radius
            d_to_o = torch.cdist(self.traj_N[i,:,:2], torch.tensor(self.obstacles,dtype=torch.float32,device=self.device), p=2)
            self.collision_cost_N[i] += torch.sum((d_to_o<threshold_dist).type(torch.float32))
            # quit()
            # for o in self.obstacles:
            #     dist = torch.linalg.norm(self.traj_N[i,:,:2]-torch.from_numpy(o)[:2]*torch.ones(self.horizon+1,2),axis = 1)
            #     self.collision_cost_N[i] += torch.sum(500*(dist<=(self.radius + self.obst_radius)).type(torch.float32))
                # print(self.collision_cost_N[i])
            t_4.append(time.time()-t1)
            
                    
        t = np.array(t)
        t_2 = np.array(t_2)
        t_3 = np.array(t_3)
        # print("Rollout time: ",np.sum(t))
        # print("Free balls time: ",np.sum(t_2))
        # print("Lane boundary time: ",np.sum(t_3))
        # print("Obstacle avoidance time: ",np.sum(t_4))
        
                                                        
            # radius = 3.5
            # dist = torch.linalg.norm(self.traj_N[i, self.horizon,:2] - self.g_state[:2])
            # if(dist<=radius):
            #     self.goal_region_cost_N[i] = 0
            # else:
            #     self.goal_region_cost_N[i] = copy.deepcopy(dist)
            
        self.total_cost_N = s_s*self.ang_vel_cost_N + s_o*self.collision_cost_N + s_c*self.center_line_cost_N + \
            s_m*self.dist_to_mean_cost_N
        
    def update_distribution(self):    
        # print(self.controls_N.grad)
        self.optimizer.zero_grad()
        self.total_cost_N.sum().backward()
        # print(self.controls_N.grad)
        self.optimizer.step()
        top_values, top_idx = torch.topk(self.total_cost_N, self.top_K, largest=False, sorted=True)
        self.top_trajs = torch.index_select(self.traj_N, 0, top_idx)
        self.top_controls = torch.index_select(self.controls_N, 0, top_idx)
        self.mean_action = self.top_controls.mean(dim=0, keepdim=True).squeeze(0).detach()
        # print(self.mean_action.shape)
        # print(self.cov_action, self.cov_action.shape)
        self.cov_action =(self.top_controls.std(dim=0, unbiased=False, keepdim=True)).mean(dim=1).data.squeeze(0).detach()
        # print(self.cov_action, self.cov_action.shape)
        self.scale_tril = torch.sqrt(self.cov_action)
        self.full_scale_tril = torch.diag(self.scale_tril)
        
        
        # print(self.cov_action)
        # quit()
        
        # self.best_traj = copy.deepcopy(self.top_controls[0,:,:])
        # top_cost = torch.index_select(self.total_cost_N, 0, top_idx)
        # w = self._exp_util(top_cost)
        
        # return w, self.top_controls
        
    
    
    def plan_traj(self):
        t1 = time.time()
        # self.centers[:,:] = copy.deepcopy(self.top_trajs[0,:,:2])
        # self.get_free_balls()
        # print("Free Balls: ", time.time() - t1)
        # top_w, self.top_controls = self.get_cost()
        self.cov_action = self.init_cov_action
        self.scale_tril = torch.sqrt(self.cov_action)
        self.full_scale_tril = torch.diag(self.scale_tril)
        for i in range(1):
            # print(i)
            # self.scale_tril = torch.sqrt(self.cov_action)
            # self.full_scale_tril = torch.diag(self.scale_tril)
            t1 = time.time()
            # self.get_free_balls()
            # print(self.controls_N.shape)    
            self.sample_controls()
            # print(self.controls_N.shape)    
            # print("Sample Controls: ", time.time() - t1)
            
            t1 = time.time()
            self.rollout()
            # print("Rollout: ", time.time() - t1)
            t1 = time.time()
            self.update_distribution()
            self.controls_N = self.controls_N.detach()
            self.total_cost_N = self.total_cost_N.detach()
            self.traj_N = self.traj_N.detach()
            # self.controls_N.grad.data.zero_()
            # .grad
            # print("Update_Distribution: ", time.time() - t1)
            # print("#######################")
        self.scale_tril = torch.sqrt(self.cov_action)

        self.full_scale_tril = torch.diag(self.scale_tril)
        self.sample_controls()
        self.rollout()
        # print(self.mean_action)
        # self.centers[:,:] = copy.deepcopy(self.top_trajs[0,:,:2])
        # self.get_free_balls()
        # self.mean_action[:-1,:] = self.mean_action[1:,:].clone()
        # self.mean_action[-1,:] = self.init_mean[-1,:].clone()
    
    def infer_traj(self):
        t1 = time.time()
        self.init_cov_action = torch.tensor([0.09, 0.09],device = self.device)
        self.cov_action = self.init_cov_action
        self.scale_tril = torch.sqrt(self.cov_action)
        self.full_scale_tril = torch.diag(self.scale_tril)
        self.sample_controls()
        self.rollout(s_o = 1, s_s = 0, s_c = 0, s_m = 1)   
    
    def get_vel(self, u):
        v1 = self.vl
        w1 = self.wl
        v = torch.zeros(u.shape)
        for i in range(u.shape[1]):
            v[0,i] = v1 + u[0,i]*self.dt
            v1 = v[0,i]
            v[1,i] = w1 + u[0,i]*self.dt
            w1 = v[1,i]       
        return v
                
    
