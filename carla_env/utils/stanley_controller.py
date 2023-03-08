import time
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

import carla_env.utils.cubic_spline_planner as csp

from carla_env.utils.custom_pid import PID

class State():
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw  # from IMU
        self.v = v

class Stanley:
    def __init__(self):
        self.k = 0.5  # control gain
        self.dt = 0.1  # [s] time difference
        self.L = 1.9  # [m] Wheel base of vehicle
        self.max_steer = np.radians(30.0)  # [rad] max steering angle

        self.target_speed = 7.2 # m/s
        self.target_speed_history = deque(maxlen=10)

        self.waypoints_list = []
        self.waypoints_type = "dense" # ["dense", "sparse"]

        self.cx = []
        self.cy = []
        self.cyaw = []

        self.state = State(x=0, y=0, yaw=0, v=0)

        self.speed_pid = PID(0.1, 0.0, 0.0, self.target_speed, output_limits=[-1.0, +1.0])

    def stanley_control(self, state, cx, cy, cyaw, last_target_idx):
        """
        Stanley steering control.

        :param state: (State object)
        :param cx: ([float])
        :param cy: ([float])
        :param cyaw: ([float])
        :param last_target_idx: (int)
        :return: (float, int)
        """
        current_target_idx, error_front_axle = self.calc_target_index(state, cx, cy)

        # if last_target_idx >= current_target_idx:
        #     current_target_idx = last_target_idx

        # theta_e corrects the heading error
        theta_e = self.normalize_angle(cyaw[current_target_idx] - state.yaw)
        # theta_d corrects the cross track error
        # theta_d = np.arctan2(self.k * error_front_axle, state.v)
        theta_d = 0.0
        if state.v > 0.5:
            theta_d = np.arctan2(self.k * error_front_axle, state.v)

        # Steering control
        delta = theta_e + theta_d

        return delta, current_target_idx


    def normalize_angle(self, angle):
        """
        Normalize an angle to [-pi, pi].

        :param angle: (float)
        :return: (float) Angle in radian in [-pi, pi]
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle


    def calc_target_index(self, state, cx, cy):
        """
        Compute index in the trajectory list of the target.

        :param state: (State object)
        :param cx: [float]
        :param cy: [float]
        :return: (int, float)
        """
        # Calc front axle position
        fy = state.y + (self.L * np.sin(state.yaw))
        fx = state.x + (self.L * np.cos(state.yaw))

        # Search nearest point index
        dx = [fx - i_cx for i_cx in cx]
        dy = [fy - i_cy for i_cy in cy]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        # TODO : maths behind calculating cross track error??
        # Project RMS error onto front axle vector
        front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                        -np.sin(state.yaw + np.pi / 2)]
        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        return target_idx, error_front_axle

    def set_current_speed(self, speed):
        self.state.v = speed * 3.6 
    
    def set_target_speed(self, speed):
        self.target_speed = speed * 3.6

    def set_waypoints(self, waypoints):
        self.waypoints_list = waypoints
        
        if self.waypoints_type == "dense":        
            self.cx = []
            self.cy = []
            self.cyaw = []
            for wpt in self.waypoints_list:
                self.cx.append(wpt[0])
                self.cy.append(wpt[1])
                self.cyaw.append(wpt[1])
        elif self.waypoints_type == "sparse":
            ax = []
            ay = []

            for wpt in self.waypoints_list:
                ax.append(wpt[0])
                ay.append(wpt[1])
            
            ds = 0.1  # [m] distance of each interpolated points

            cx, cy, cyaw, ck, s = csp.calc_spline_course(ax, ay, ds)


            self.cx = cx
            self.cy = cy
            self.cyaw = cyaw

        self.totalWaypoints = len(self.cx)

    def local_planner_info_callback(self, info):
        speed = info.linear.x
        self.path_curvature = info.angular.z

        self.target_speed_history.append(speed)
        self.target_speed = np.floor(np.mean(np.array(self.target_speed_history)))

    def get_controls(self):
        last_idx = len(self.cx) - 1

        if (len(self.cx) == 0):
            return
        
        steer = 0 
        throttle = 0
        brake = 0
        acc = 0

        target_idx, _ = self.calc_target_index(self.state, self.cx, self.cy)

        if last_idx > target_idx:
            steer, target_idx =  self.stanley_control(self.state, self.cx, self.cy, self.cyaw, target_idx)
            steer = np.clip(steer, -self.max_steer, self.max_steer) / self.max_steer # radians
            
            # PID control
            acc = self.speed_pid.get_output(self.target_speed, self.state.v, dt = 0.1)
            
            if acc > 0:
                throttle = acc
                brake = 0
            else:
                throttle = 0
                brake = abs(acc)

            ret_val = 0
        else:
            steer = 0
            ret_val = -1
        
        action = [throttle, brake, steer, acc]
        return action, ret_val
