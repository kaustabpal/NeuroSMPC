# uncompyle6 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.10 (default, Nov 14 2022, 12:59:47) 
# [GCC 9.4.0]
# Embedded file name: /home/aditya/Documents/DEB/dataset_pipeline/casadi_code.py
# Compiled at: 2023-02-28 22:36:49
# Size of source mod 2**32: 11924 bytes
from time import time
import casadi as ca, numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
import os, copy
np.set_printoptions(suppress=True)

def DM2Arr(dm):
    return np.array(dm.full())


def draw_circle(x, y, radius):
    th = np.arange(0, 2 * np.pi, 0.01)
    xunit = radius * np.cos(th) + x
    yunit = radius * np.sin(th) + y
    return (xunit, yunit)


class Agent:

    def __init__(self, agent_id, i_state, g_state, N=50, obstacles=[]):
        self.sensor_radius = 50
        self.id = agent_id
        self.radius = 1.0
        self.obst_radius = 1.0
        self.i_state = np.array(i_state)
        self.g_state = np.array(g_state)
        self.state_init = ca.DM([i_state[0], i_state[1], i_state[2]])
        self.state_target = ca.DM([g_state[0], g_state[1], g_state[2]])
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.theta = ca.SX.sym('theta')
        self.states = ca.vertcat(self.x, self.y, self.theta)
        self.n_states = self.states.numel()
        self.a = ca.SX.sym('a')
        self.j = ca.SX.sym('j')
        self.controls = ca.vertcat(self.a, self.j)
        self.n_controls = self.controls.numel()
        self.obstacles = []
        self.n_obst = 0
        self.avoid_obs = False
        self.N = N
        self.v_ub = 4.13
        self.v_lb = 0
        self.w_ub = 0.5
        self.w_lb = -0.5
        self.amin = -5
        self.amax = 5
        self.jmin = -1.0
        self.jmax = 1.0
        self.right_lane_bound = 4.5
        self.left_lane_bound = -4.5
        self.dt = 0.1
        self.avg_time = []
        self.Q_x = 100
        self.Q_y = 100
        self.Q_theta = 500
        self.X = ca.SX.sym('X', self.n_states, self.N + 1)
        self.U = ca.SX.sym('U', self.n_controls, self.N)
        self.P = ca.SX.sym('P', self.n_states + self.n_states)
        self.Q = ca.diagcat(self.Q_x, self.Q_y, self.Q_theta)
        self.J = ca.vertcat(ca.horzcat(cos(self.theta), 0), ca.horzcat(sin(self.theta), 0), ca.horzcat(0, 1))
        self.RHS = self.J @ self.controls
        self.f = ca.Function('f', [self.states, self.controls], [self.RHS])
        self.OPT_variables = ca.vertcat(self.X.reshape((-1, 1)), self.U.reshape((-1,
                                                                                 1)))
        self.u0 = ca.DM.zeros((self.n_controls, self.N))
        self.X0 = ca.repmat(self.state_init, 1, self.N + 1)
        self.vl = ca.DM(0)
        self.wl = ca.DM(0)

    def get_goal_cost(self):
        st = self.X[:, self.N]
        return (st - self.P[self.n_states:]).T @ self.Q @ (st - self.P[self.n_states:])

    def get_ang_acc_cost(self):
        cost_fn = 0
        for k in range(0, self.N):
            w = self.U[(1, k)]
            cost_fn += w ** 2
        else:
            return cost_fn

    def get_lane_cost(self, lane_x):
        cost_fn = 0
        for k in range(self.N + 1):
            st_x = self.X[(0, k)]
            cost_fn += (st_x - lane_x) ** 2
        else:
            return cost_fn

    def next_state_constraints(self):
        self.g = self.X[:, 0] - self.P[:self.n_states]
        self.lbg = ca.vertcat(0, 0, 0)
        self.ubg = ca.vertcat(0, 0, 0)
        v1 = self.vl
        w1 = self.wl
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            v_ = con[0]
            w_ = con[1]
            v1 += v_ * self.dt
            w1 += w_ * self.dt
            st_next = self.X[:, k + 1]
            st_next_update = st
            st_next_update[(0, 0)] = st[(0, 0)] + v1 * cos(st[2]) * self.dt
            st_next_update[(1, 0)] = st[(1, 0)] + v1 * sin(st[2]) * self.dt
            st_next_update[(2, 0)] = st[(2, 0)] + w1 * self.dt
            self.g = ca.vertcat(self.g, st_next - st_next_update)
            self.lbg = ca.vertcat(self.lbg, 0, 0, 0)
            self.ubg = ca.vertcat(self.ubg, 0, 0, 0)

    def obstacle_constraints(self):
        for o in self.obstacles:
            o_X = o
            for i in range(1, self.N + 1):
                a_st = self.X[:, i]
                o_st = o_X[i, :]
                dist = ca.sqrt((a_st[1] - o_st[1]) ** 2 + (a_st[0] - o_st[0]) ** 2)
                self.g = ca.vertcat(self.g, dist)
                self.lbg = ca.vertcat(self.lbg, 3.0)
                self.ubg = ca.vertcat(self.ubg, ca.inf)

    def lane_boundary_constraints(self):
        self.lbx = ca.vertcat(self.left_lane_bound + self.radius, -ca.inf, -ca.inf)
        self.ubx = ca.vertcat(self.right_lane_bound - self.radius, ca.inf, ca.inf)
        for k in range(self.N):
            self.lbx = ca.vertcat(self.lbx, self.left_lane_bound + self.radius, -ca.inf, -ca.inf)
            self.ubx = ca.vertcat(self.ubx, self.right_lane_bound - self.radius, ca.inf, ca.inf)

    def control_bound_constraints(self):
        for i in range(self.N):
            self.lbx = ca.vertcat(self.lbx, self.amin, self.jmin)
            self.ubx = ca.vertcat(self.ubx, self.amax, self.jmax)

    def vel_bound_constraints(self):
        v1 = copy.deepcopy(self.vl)
        for i in range(self.N):
            v2 = self.U[(0, i)] * self.dt + v1
            self.g = ca.vertcat(self.g, v2)
            self.lbg = ca.vertcat(self.lbg, self.v_lb)
            self.ubg = ca.vertcat(self.ubg, self.v_ub)
            v1 = copy.deepcopy(v2)

    def ang_vel_bound_constraints(self):
        w1 = copy.deepcopy(self.wl)
        for i in range(self.N - 1):
            w2 = self.U[(1, i)] * self.dt + w1
            self.g = ca.vertcat(self.g, w2)
            self.lbg = ca.vertcat(self.lbg, self.w_lb)
            self.ubg = ca.vertcat(self.ubg, self.w_ub)
            w1 = copy.deepcopy(w2)

    def get_vel(self, u):
        v1 = self.vl
        w1 = self.wl
        v = np.zeros(u.shape)
        for i in range(u.shape[1]):
            v[(0, i)] = v1 + u[(0, i)] * self.dt
            v1 = v[(0, i)]
            v[(1, i)] = w1 + u[(1, i)] * self.dt
            w1 = v[(1, i)]
        else:
            return v

    def pred_controls(self):
        cost_fn = self.get_goal_cost() + 100 * self.get_ang_acc_cost()
        cost = cost_fn
        self.next_state_constraints()
        self.lane_boundary_constraints()
        self.control_bound_constraints()
        self.vel_bound_constraints()
        self.ang_vel_bound_constraints()
        self.obstacle_constraints()
        nlp_prob = {'f':cost, 
         'x':self.OPT_variables, 
         'g':self.g, 
         'p':self.P}
        opts = {'ipopt':{
          'max_iter': 2000, 
          'print_level': 0, 
          'acceptable_tol': 1e-08, 
          'acceptable_obj_change_tol': 1e-06}, 
         'print_time':0}
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        args = {'lbg':self.lbg, 
         'ubg':self.ubg, 
         'lbx':self.lbx, 
         'ubx':self.ubx, 
         'p':ca.vertcat(self.state_init, self.state_target), 
         'x0':ca.vertcat(ca.reshape(self.X0, self.n_states * (self.N + 1), 1), ca.reshape(self.u0, self.n_controls * self.N, 1))}
        t1 = time()
        sol = solver(x0=(args['x0']),
          lbx=(args['lbx']),
          ubx=(args['ubx']),
          lbg=(args['lbg']),
          ubg=(args['ubg']),
          p=(args['p']))
        self.avg_time.append(time() - t1)
        self.u0 = ca.reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N)
        self.X0 = ca.reshape(sol['x'][:self.n_states * (self.N + 1)], self.n_states, self.N + 1)