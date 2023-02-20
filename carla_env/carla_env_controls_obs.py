import gym
from gym import spaces
import carla
import numpy as np
from carla_env.utils.transforms import se3_to_components
import time
import atexit

import math

import random

import cv2
from scipy.spatial.transform import Rotation as R

# from agents.navigation.basic_agent import BasicAgent
# from agents.navigation.behavior_agent import BehaviorAgent

from carla_env.utils.custom_pid import PID

DEBUG = False


def dummy_function(image):
    pass


process_lidar = True
process_semantic_lidar = True

 
class CarEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CarEnv, self).__init__()

        # Vehicle Details
        self.ego = None
        self.target = None
        self.dummy = None
        self.lidar_sen = None
        self.semantic_lidar_sen = None

        self.ego_path = []
        self.frame = -1
        self.episode = 0

        # self.ego_trans_init = carla.Transform(
        #     carla.Location(x=233.27923583984375,
        #                    y=86.87185668945312, z=2.684593),
        #     carla.Rotation(pitch=-1.183624, yaw=90, roll=-0.084534)
        # )
        # self.ego_trans_init = carla.Transform(
        #     carla.Location(x=-78.316872, y=158.03792, z=0.051953),
        #     carla.Rotation(pitch=0.013059, yaw=-90.464806, roll=-0.005676)
        # )

        ##########
        # GYM env
        ##########

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(
            low=np.array([0.2, -0.3]),
            high=np.array([0.6, 0.3]),
            dtype=np.float32
        )

        # Observatipon Space
            # BEV - (256, 256, 3) - [0, 255]
            # Speed - (1,) - [-inf, +inf]
            # Offset(y, w) - (2,) - [-inf, +inf]
        self.observation_space = spaces.Dict({
            'bev': spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
            'speed': spaces.Box(low=-np.inf, high=+np.inf, shape=(1,), dtype=np.float32),
            'offset': spaces.Box(
                        # Lower Limit
                        low=np.array([-np.inf, -np.inf]),
                        # Upper Limit
                        high=np.array([+np.inf, +np.inf]),
                        dtype=np.float32
                    )
        })

        ##########
        # Carla
        ##########

        # Settin Up the World
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)
        # client.load_world('Town04')
        client.load_world('Town05')
        # client.load_world('Town10HD')
        self.world = client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1  # FPS = 1/0.1 = 10
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        self.map = self.world.get_map()

        self.ego_trans_init = np.random.choice(self.map.get_spawn_points())

        ### Ego-Vehicle
        # Vehicle Transforms
        self.tf_matrix = np.array(self.ego_trans_init.get_matrix())
        self.yaw_init = self.ego_trans_init.rotation.yaw

        self.ego_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        self.ego_bp.set_attribute('role_name', 'ego')
        self.ego_bp.set_attribute('color', '255,0,0')

        self.target_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        self.target_bp.set_attribute('role_name', 'ego')
        self.target_bp.set_attribute('color', '0,255,0')

        ### LIDAR
        lidar_cam = None
        self.lidar_pcd = None
        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('channels', str(16))
        # TODO : fps of sim has been set to 10
        # Set the fps of simulator same as this
        # self.lidar_bp.set_attribute('rotation_frequency', str(20))
        self.lidar_bp.set_attribute('rotation_frequency', str(10)) # changed to 10
        self.lidar_bp.set_attribute('range', str(50))
        self.lidar_bp.set_attribute('lower_fov', str(-15))
        self.lidar_bp.set_attribute('upper_fov', str(15))
        # self.lidar_bp.set_attribute('points_per_second', str(300000))
        self.lidar_bp.set_attribute('points_per_second', str(144000))
        self.lidar_bp.set_attribute('dropoff_general_rate', str(0.0))

        lidar_location = carla.Location(0, 0, 2)
        lidar_rotation = carla.Rotation(0, 0, 0)
        self.lidar_transform = carla.Transform(lidar_location, lidar_rotation)


        ### Semantic LIDAR
        self.semantic_lidar_pcd = None
        self.semantic_lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        self.semantic_lidar_bp.set_attribute('channels', str(16))
        # TODO : fps of sim has been set to 10
        # Set the fps of simulator same as this
        # self.semantic_lidar_bp.set_attribute('rotation_frequency', str(20))
        self.semantic_lidar_bp.set_attribute('rotation_frequency', str(10)) # changed to 10
        self.semantic_lidar_bp.set_attribute('range', str(50))
        self.semantic_lidar_bp.set_attribute('lower_fov', str(-15))
        self.semantic_lidar_bp.set_attribute('upper_fov', str(15))
        # self.semantic_lidar_bp.set_attribute('points_per_second', str(300000))
        self.lidar_bp.set_attribute('points_per_second', str(144000))
        # self.semantic_lidar_bp.set_attribute('dropoff_general_rate', str(0.0))

        semantic_lidar_location = carla.Location(0, 0, 2)
        semantic_lidar_rotation = carla.Rotation(0, 0, 0)
        self.semantic_lidar_transform = carla.Transform(semantic_lidar_location, semantic_lidar_rotation)

        # BEV
        self.bev = None
        self.obstacle_bev = None
        self.next_g_path = None
        self.left_lane_coords = None
        self.right_lane_coords = None
        self.dyn_obs_poses = None
        
        # Global Path
        self.global_path = None
        self.global_path_tf = None
        self.global_path_carla = None
        self.global_path_wps = None

        ### Traffic Manager
        self.traffic_manager = None
        self.number_of_vehicles = 200
        self.number_of_walkers = 0
        self.vehicles = []
        if self.number_of_vehicles > 0:
            self.traffic_manager = client.get_trafficmanager(8000)
            self.traffic_manager.set_global_distance_to_leading_vehicle(5.0)
            self.traffic_manager.set_random_device_seed(3)
            self.traffic_manager_port = self.traffic_manager.get_port()
        
        self.speed_pid = PID(0.1, 0.0, 0.0, SP = 0.0, output_limits=[-1.0, +1.0])

        atexit.register(self.close)

    def _next_observation(self):
        '''
        Odom frame is the car's initial frame
        X is in the direction of car
        Y in towards right of the car
        '''
        observation = {}

        # BEV
        if self.bev is None:
            observation['bev'] = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            observation['bev'] = np.copy(self.bev)

        # Offset
        ego_tf = np.array(self.ego.get_transform().get_matrix())
        # print('Ego Transform: ', self.ego.get_transform())
        ego_tf_odom = np.linalg.pinv(self.tf_matrix) @ ego_tf
        ego_x, ego_y, ego_z, ego_R, ego_P, ego_Y = se3_to_components(
            ego_tf_odom)

        nearest_wpt_tf = self.get_nearest_waypoint_transform(self.ego.get_location())

        waypoint_bl = np.linalg.pinv(ego_tf) @ nearest_wpt_tf.get_matrix()
        wx, wy, wz, wR, wP, wYaw = se3_to_components(waypoint_bl)
        
        ego_y = -ego_y
        wy = -wy
        wYaw = -wYaw

        observation['offset'] = np.array([wy, wYaw], dtype=np.float32)
        self.ego_path.append(np.array([ego_x, ego_y, ego_z]))

        # Speed
        vel_ego = self.ego.get_velocity()
        speed_ego = np.linalg.norm(np.array([vel_ego.x, vel_ego.y]))
        observation['speed'] = np.array([speed_ego], dtype = np.float32)
        self.speed_ego = speed_ego
        return observation

    def step(self, action):

        self.frame += 1

        ##### Acting ##########
        # Acting on Ego
        vel_ego = self.ego.get_velocity()
        speed_ego = np.linalg.norm(np.array([vel_ego.x, vel_ego.y]))
        
        target_speed = action[0]

        acc = self.speed_pid.get_output(target_speed, speed_ego, dt = 0.1)

        if acc > 0:
            throttle = acc
            brake = 0
        else:
            throttle = 0
            brake = abs(acc)

        steer = action[1]
        
        self.ego.apply_control(carla.VehicleControl(
            throttle=float(throttle), steer=float(steer), brake=float(brake)))
        # print(action)
        # self.ego.set_target_velocity(carla.Vector3D(x=-float(action[0])))
        # self.ego.set_target_angular_velocity(carla.Vector3D(z=-float(action[1])))
        # self.ego.apply_control(self.ego_agent.run_step())

        # Tick the world
        self.world.tick()
        spectator = self.world.get_spectator()
        spectator.set_transform(self.dummy.get_transform())
        ######################

        obs = self._next_observation()
        done = False

        ####### Reward Calculation ######
        DISMAX = 1.5
        DESVEL = 3
        SPEED_SCALE = 1
        DIS_SCALE = 1
        ANG_SCALE = 1
        
        speed = obs['speed'][0]
        y_offset = obs['offset'][0]
        yaw_offset = obs['offset'][1]
        if speed <= DESVEL:
            reward_speed = speed / DESVEL
        else:
            reward_speed = (2*DESVEL - speed) / DESVEL
        reward_speed = np.clip(reward_speed, 0, 1)

        reward_dis = - np.abs(y_offset) / DISMAX
        reward_dis = np.clip(reward_dis, -1, 1)
        reward_dis = 1 / (np.abs(reward_dis)+1)

        reward_ang = - np.abs(yaw_offset)
        reward_ang = 1 / (np.abs(reward_ang) + 1)

        reward = SPEED_SCALE*reward_speed + ANG_SCALE*reward_ang + DIS_SCALE*reward_dis

        # TODO : Termination only for lateral_dist
        # TODO : Shouldn't there be a Termination for angle / or total_deviation as well? -review video of bev
        if np.abs(y_offset) > DISMAX:
            print('!!!!!!OUT OF LANE!!!!!!!!!')
            reward = -1000
            done = True

        # Only for training
        if self.frame >= 5000:
            done = True

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.frame = -1

        time.sleep(1.5)
        if self.ego is not None:
            self.lidar_sen.destroy()
            self.semantic_lidar_sen.destroy()
            self.ego.destroy()

        # Spawn Ego
        self.ego_trans_init = np.random.choice(self.map.get_spawn_points())
        self.ego = self.world.spawn_actor(self.ego_bp, self.ego_trans_init)
        # self.ego.set_autopilot(True)

        # Attach Sensors
        ### Lidar
        self.lidar_pcd = None
        self.lidar_sen = self.world.spawn_actor(
            self.lidar_bp, self.lidar_transform, attach_to=self.ego)
        self.lidar_sen.listen(
            lambda point_cloud: self.process_lidar_raw_points(point_cloud))

        ### Semantic Lidar
        self.semantic_lidar_pcd = None
        self.semantic_lidar_sen = self.world.spawn_actor(
            self.semantic_lidar_bp, self.semantic_lidar_transform, attach_to=self.ego)
        self.semantic_lidar_sen.listen(
            lambda point_cloud: self.process_semantic_lidar_raw_points(point_cloud))

        # Attach Dummy Camera
        if self.dummy is not None:
            self.dummy.destroy()
        dummy_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        dummy_transform = carla.Transform(
            carla.Location(x=-6, z=4), carla.Rotation(pitch=10.0))
        self.dummy = self.world.spawn_actor(
            dummy_bp, dummy_transform, attach_to=self.ego, attachment_type=carla.AttachmentType.SpringArm)
        self.dummy.listen(lambda image: dummy_function(image))

        # Wait for the sensors to be ready
        for i in range(10):  # Some buffer time at the start of the episode
            global process_lidar
            process_lidar = False
            self.world.tick()
            spectator = self.world.get_spectator()
            spectator.set_transform(self.dummy.get_transform())
            process_lidar = True

        # Generate a global path
        self.generate_global_path(num_of_waypoints=1000, dist_between_wpts=1)

        # Add traffic
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        self.add_traffic()

        # self.ego_agent = BasicAgent(self.ego)

        # destination = self.global_path_carla[-1]
        # self.ego_agent.set_destination(destination)

        # self.ego_agent.ignore_traffic_lights(active = True)
        # self.ego_agent.ignore_stop_signs(active = True)


        # self.ego.set_autopilot(True)
        # self.traffic_manager.ignore_lights_percentage(self.ego, 100)
        # self.traffic_manager.ignore_signs_percentage(self.ego, 100)

        # route_directions = ["Left", "Straight", "Right"]
        # self.traffic_manager.set_route(self.ego, route_directions)

        # self.traffic_manager.vehicle_percentage_speed_difference(self.ego, -50)
        # self.traffic_manager.set_path(self.ego, self.global_path_carla)

        return self._next_observation()

    def render(self, mode='human', close=False):
        pass

    def close(self):
        things_to_destroy = [self.dummy, self.lidar_sen, self.semantic_lidar_sen, self.ego]
        i = 0
        for thing in things_to_destroy:
            try:
                thing.destroy()
            except Exception as e:
                print("Thing number - ", i)
                print(e)
            i += 1
        self.destroy_all_vehicles()

    def process_lidar_raw_points(self, point_cloud_carla):
        if process_lidar == False:
            return
            
        pcd = np.copy(np.frombuffer(
            point_cloud_carla.raw_data, dtype=np.dtype('float32')))
        self.lidar_pcd = np.reshape(pcd, (int(pcd.shape[0] / 4), 4))


    def process_semantic_lidar_raw_points(self, point_cloud_carla):
        if process_lidar == False:
            return
        
        data = np.frombuffer(
            point_cloud_carla.raw_data, dtype=np.dtype([
                    ('x', np.float32), ('y', np.float32), ('z', np.float32),
                    ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]
                    ))
        self.semantic_lidar_pcd = np.array([data['x'], data['y'], data['z'], data['CosAngle'], data['ObjIdx'], data['ObjTag']]).T

        car_pts_mask = self.semantic_lidar_pcd[:, 5] == 10

        self.car_pts = self.semantic_lidar_pcd[car_pts_mask]
        self.non_car_pts = self.semantic_lidar_pcd[~car_pts_mask]
        
        self.non_car_noise = np.random.normal(0, 0.3, self.non_car_pts[:, :3].shape)
        self.car_noise = np.random.normal(0, 0.1, self.car_pts[:, :3].shape)

        self.non_car_pts[:, :3] += self.non_car_noise
        self.car_pts[:, :3] += self.car_noise

        self.semantic_lidar_pcd = np.concatenate([self.non_car_pts, self.car_pts], axis=0)

        # Generate BEV
        self.bev, self.obstacle_bev, self.next_g_path, self.left_lane_coords, self.right_lane_coords, self.dyn_obs_poses = self.generate_bev((256, 256), sensing_range = 15, z_max=1)
    
    def generate_bev(self, img_size, sensing_range, z_max):
        assert img_size[0] == img_size[1], "BEV should be square"
        
        semantic_pcd = self.semantic_lidar_pcd

        scale = img_size[0] / (2 * sensing_range)
        
        # Creating a blank image
        bev = np.ones([img_size[0], img_size[0], 3], dtype=np.uint8)
        bev[:, :] = [0, 0, 0]

        ### Obstacle Layer
        # finding the points that are not road
        obs_mask = (semantic_pcd[:, 5] != 7 ) & (semantic_pcd[:, 5] != 6) & (semantic_pcd[:, 5] != 18) & (semantic_pcd[:, 5] != 21) # 7 is the tag for road
        obs_pcd = semantic_pcd[obs_mask] 

        # finding the points that are in the vicinity of the ego
        x_mask = np.abs(obs_pcd[:, 0]) < sensing_range
        y_mask = np.abs(obs_pcd[:, 1]) < sensing_range
        z_mask = obs_pcd[:, 2] < z_max
        vicinity_mask = x_mask & y_mask & z_mask
        obs_pcd_near = obs_pcd[vicinity_mask]
        obs_pcd_near[:, 0] = -obs_pcd_near[:, 0]

        # scaling the points to the image size
        obs_pcd_near_scaled = obs_pcd_near[:, :2]*scale + img_size[0]//2
        obs_pixel_coords = np.clip(np.intp(obs_pcd_near_scaled[:, :2]), 0, 255)

        # drawing the points on the image
        bev_obs = np.copy(bev)
        bev_obs.shape
        bev_obs[obs_pixel_coords[:, 0], obs_pixel_coords[:, 1]] = [255, 255, 255]

        '''
        # Dilating the points to make them visible
        kernel = np.ones((5, 5), np.uint8)
        bev_obs = cv2.dilate(bev_obs, kernel, iterations=1)
        '''

        g_path, g_path_wpts = self.get_next_waypoints_in_ego_frame(40, 1)
        # Scaling the points to the image size
        g_path_in_image_scale = g_path[:, :2]*scale + img_size[0]//2

        dyn_obs_poses = self.get_obs_pose(sensing_range)
        #   Scaling Points to image size
        obs_pose_in_image_scale = []
        if len(dyn_obs_poses)>0:
            obs_pose_in_image_scale = dyn_obs_poses[:, :2]*scale + img_size[0]//2
            obs_pose_in_image_scale = np.clip(np.intp(obs_pose_in_image_scale[:, :2]), 0, 255)

        left_lane_pts, right_lane_pts = self.get_lane_boundaries(g_path_wpts)
        # quit()
        #   Scaling Points to image size
        left_lane_pt_in_image_scale = left_lane_pts*scale + img_size[0]//2
        right_lane_pt_in_image_scale = right_lane_pts*scale + img_size[0]//2

        ### Ego Layer
        ego_dim_x, ego_dim_y = 2.1, 4.7 # (in meters) # dimension are for tesla model 3 (in Carla)
        
        # Creatng a rectangle for the ego
        rot_rect = ((127, 127), (ego_dim_x * scale, ego_dim_y * scale), 0)
        box = cv2.boxPoints(rot_rect)
        
        # Drawing the rectangle on the image
        bev_ego = np.copy(bev)
        bev_ego = cv2.drawContours(bev_ego, [box.astype(int)], 0, [255, 255, 255], -1)

        #   Drawing ellipse around obstacles
        bev_dyn_obs_arr = []
        for i in range(len(obs_pose_in_image_scale)):
            bev_dyn_obs = np.copy(bev)
            bev_dyn_obs = cv2.ellipse(bev_dyn_obs, (int(obs_pose_in_image_scale[i][1]), img_size[0] - int(obs_pose_in_image_scale[i][0])), (10,10), np.deg2rad(dyn_obs_poses[i][2]), 0, 360, (255, 255, 255), -1)
            bev_dyn_obs_arr.append(bev_dyn_obs)

        # ### GlobalPath Layer
        # global_path_tf = self.global_path_tf
        
        # # Transforming the global path to the ego's frame
        # ego_tf = np.array(self.ego.get_transform().get_matrix())
        # global_path_relative_tf = np.linalg.pinv(ego_tf) @ global_path_tf

        # # Extracting XYZ coords
        # global_path_relative_coords = global_path_relative_tf[:, 0:3, 3]

        # Finding the points that are in the vicinity of the ego
        x_mask = np.abs(g_path[:, 0]) < sensing_range
        y_mask = np.abs(g_path[:, 1]) < sensing_range
        vicinity_mask = x_mask & y_mask
        global_path_relative_near = g_path[vicinity_mask]

        # # Correcting Axis-Order
        global_path_relative_near = global_path_relative_near[:, [1, 0]]
        # Flipping Y-Axis
        global_path_relative_near[:, 1] = -global_path_relative_near[:, 1]

        # Scaling the points to the image size
        global_path_pixel_coords = global_path_relative_near[:, :2]*scale + img_size[0]// 2
        global_path_pixel_coords = np.clip(np.intp(global_path_pixel_coords[:, :2]), 0, 255)

        #   LEFT LANE
        # Finding the points that are in the vicinity of the ego
        x_mask = np.abs(left_lane_pts[:, 0]) < sensing_range
        y_mask = np.abs(left_lane_pts[:, 1]) < sensing_range
        vicinity_mask = x_mask & y_mask
        left_lane_relative_near = left_lane_pts[vicinity_mask]

        # # Correcting Axis-Order
        left_lane_relative_near = left_lane_relative_near[:, [1, 0]]
        # Flipping Y-Axis
        left_lane_relative_near[:, 1] = -left_lane_relative_near[:, 1]

        # Scaling the points to the image size
        left_lane_pixel_coords = left_lane_relative_near[:, :2]*scale + img_size[0]// 2
        left_lane_pixel_coords = np.clip(np.intp(left_lane_pixel_coords[:, :2]), 0, 255)

        #   RIGHTH LANE
        # Finding the points that are in the vicinity of the ego
        x_mask = np.abs(right_lane_pts[:, 0]) < sensing_range
        y_mask = np.abs(right_lane_pts[:, 1]) < sensing_range
        vicinity_mask = x_mask & y_mask
        right_lane_relative_near = right_lane_pts[vicinity_mask]

        # # Correcting Axis-Order
        right_lane_relative_near = right_lane_relative_near[:, [1, 0]]
        # Flipping Y-Axis
        right_lane_relative_near[:, 1] = -right_lane_relative_near[:, 1]

        # Scaling the points to the image size
        right_lane_pixel_coords = right_lane_relative_near[:, :2]*scale + img_size[0]// 2
        right_lane_pixel_coords = np.clip(np.intp(right_lane_pixel_coords[:, :2]), 0, 255)

        # Drawing the path on the image
        bev_global_path = np.copy(bev)
        if global_path_pixel_coords.shape[0] > 1:
            bev_global_path = cv2.polylines(bev_global_path, [global_path_pixel_coords], False, color=(255, 255, 255), thickness=3)
        
        #   LEFT LANE
        # Drawing the path on the image
        bev_left_lane = np.copy(bev)
        if left_lane_pixel_coords.shape[0] > 1:
            bev_left_lane = cv2.polylines(bev_left_lane, [left_lane_pixel_coords], False, color=(255, 255, 255), thickness=3)
        
        #   RIGHT LANE
        # Drawing the path on the image
        bev_right_lane = np.copy(bev)
        if right_lane_pixel_coords.shape[0] > 1:
            bev_right_lane = cv2.polylines(bev_right_lane, [right_lane_pixel_coords], False, color=(255, 255, 255), thickness=3)

         ### Combining the layers
        obs_mask = bev_obs[:, :, 0] == 255
        ego_mask = bev_ego[:, :, 0] == 255
        global_path_mask = bev_global_path[:, :, 0] == 255
        left_lane_mask = bev_left_lane[:, :, 0] == 255
        right_lane_mask = bev_right_lane[:, :, 0] == 255
        dyn_obs_mask = []
        for i in range(len(bev_dyn_obs_arr)):
            dyn_obs_mask.append(bev_dyn_obs_arr[i][:, :, 0] == 255)

        bev[obs_mask] = [255, 255, 255] # Obstacles - White
        bev[global_path_mask] = [0, 0, 255] # Global Path - Blue
        bev[ego_mask] = [255, 0, 0] # Ego - Red
        bev[left_lane_mask] = [0, 255, 0]
        bev[right_lane_mask] = [0, 255, 0]
        for i in range(len(dyn_obs_mask)):
            bev[dyn_obs_mask[i]] = [255, 255, 0]

        obstacle_bev = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        obstacle_bev[obs_mask] = 255 # Obstacles - White
        
        return bev, obstacle_bev, np.array(g_path_in_image_scale), np.array(left_lane_pt_in_image_scale), np.array(right_lane_pt_in_image_scale), np.array(dyn_obs_poses)
    
    def get_obs_pose(self, obs_range):
        ego_tf = np.array(self.ego.get_transform().get_matrix())
        obs_pose_wrt_ego = []
        for i in range(len(self.vehicles)):
            #   Obstacle traneformation matrix
            vehicle_tf = np.array(self.vehicles[i].get_transform().get_matrix())
            #   Transformation of obs wrt ego
            obs_wrt_ego = np.linalg.pinv(ego_tf) @ vehicle_tf
            #  Rotation matrix
            rot_mat = R.from_matrix(np.array(obs_wrt_ego[:3, :3]))
            rel_yaw = np.rad2deg(np.array(rot_mat.as_rotvec())[-1])
            #  The ego vehicle yaw is 90, hence this is required
            if rel_yaw<0.0:
                rel_yaw = rel_yaw + 360.0
            rel_yaw = rel_yaw + 90.0

            v = self.vehicles[i].get_velocity()
            obs_v = ((v.x)**2 + (v.y)**2)**0.5
            obs_w = self.vehicles[i].get_angular_velocity().z

            #   If obstacle distance less thatn range(of bev), store pose
            if (obs_wrt_ego[0, 3]**2 + obs_wrt_ego[1, 3]**2)**0.5 < obs_range:
                obs_pose_wrt_ego.append([obs_wrt_ego[0, 3], obs_wrt_ego[1, 3], np.deg2rad(rel_yaw), obs_v, obs_w])
        #  Stored as x,y,theta,v,w
        return np.array(obs_pose_wrt_ego)
    
    def get_lane_boundaries(self, g_path_wpts):
        global_path_wps = g_path_wpts
        ego_tf = np.array(self.ego.get_transform().get_matrix())

        left_lane_pts_wrt_ego = []
        right_lane_pts_wrt_ego = []
        for i in range(len(global_path_wps)):
            orientationVec = global_path_wps[i].transform.get_forward_vector()
            length = math.sqrt(orientationVec.y*orientationVec.y+orientationVec.x*orientationVec.x)
            abVec = carla.Location(orientationVec.y,-orientationVec.x,0) / length * 0.5* global_path_wps[i].lane_width
            lane_wpts = []
            next_waypoint = global_path_wps[i]
            #   Get the right most lane
            while next_waypoint.get_right_lane() and next_waypoint.get_right_lane().lane_type == carla.LaneType.Driving:
                next_waypoint = next_waypoint.get_right_lane()
            lane_wpts.append(next_waypoint)
            #   Get all lanes to the left of the right most lane
            while next_waypoint.get_left_lane() and \
                np.sign(next_waypoint.get_left_lane().lane_id) == np.sign(next_waypoint.lane_id) and \
                next_waypoint.get_left_lane().lane_type == carla.LaneType.Driving:

                next_waypoint = next_waypoint.get_left_lane()
                lane_wpts.append(next_waypoint) 

            #   Left most lane
            left_lane_pt_wp = lane_wpts[-1]
            right_lane_pt_wp = lane_wpts[0]
            
            left_lane_pt_wp = left_lane_pt_wp.transform.location + abVec
            left_lane_pt_wp = [left_lane_pt_wp.x, left_lane_pt_wp.y, left_lane_pt_wp.z, 1]
            right_lane_pt_wp = right_lane_pt_wp.transform.location - abVec
            right_lane_pt_wp = [right_lane_pt_wp.x, right_lane_pt_wp.y, right_lane_pt_wp.z, 1]

            #   Right most lane
            left_lane_pt_wrt_ego = np.linalg.pinv(ego_tf) @ left_lane_pt_wp
            right_lane_pt_wrt_ego = np.linalg.pinv(ego_tf) @ right_lane_pt_wp


            # left_lanept = left_lane_pt_wrt_ego[:2, 3]
            # left_lanept[0] = left_lanept[0] + abVec.x
            # left_lanept[1] = left_lanept[0] + abVec.y

            # right_lanept = right_lane_pt_wrt_ego[:2, 3]
            # right_lanept[0] = right_lanept[0] - abVec.x
            # right_lanept[1] = right_lanept[0] - abVec.y

            # left_lane_pts_wrt_ego.append([left_lanept[0], left_lanept[1]])
            # right_lane_pts_wrt_ego.append([right_lanept[0], right_lanept[1]])
            left_lane_pts_wrt_ego.append([left_lane_pt_wrt_ego[0], left_lane_pt_wrt_ego[1]])
            right_lane_pts_wrt_ego.append([right_lane_pt_wrt_ego[0], right_lane_pt_wrt_ego[1]])
            
        return np.array(left_lane_pts_wrt_ego), np.array(right_lane_pts_wrt_ego)

    def generate_global_path(self, num_of_waypoints, dist_between_wpts = 1):
        
        global_path = []
        global_path_wpt_tf = []
        global_path_carla = []
        global_path_wps = []
        # Getting the waypoint of the ego
        carwaypoint = self.map.get_waypoint(self.ego.get_location())

        # Generating the global path
        wpt = carwaypoint
        for _ in range(num_of_waypoints):
            global_path_carla.append(wpt.transform.location)
            global_path_wps.append(wpt)
            global_path.append([wpt.transform.location.x, wpt.transform.location.y, wpt.transform.rotation.yaw])
            global_path_wpt_tf.append(wpt.transform.get_matrix()) 
            wpt = np.random.choice(wpt.next(dist_between_wpts)) # wpt at every 1m
        self.global_path = np.array(global_path)
        self.global_path_tf = np.array(global_path_wpt_tf)
        self.global_path_carla = global_path_carla
        self.global_path_wps = global_path_wps

    def get_nearest_waypoint_transform(self, location):
        global_path = self.global_path
        
        # Getting the nearest waypoint
        curr_pose = np.array([location.x, location.y, location.z])
        
        # Finding the nearest waypoint
        dist = np.linalg.norm(global_path[:, :2] - curr_pose[:2], axis=1)
        nearest_wpt_idx = np.argmin(dist)
        nearest_wpt_ = global_path[nearest_wpt_idx]

        # Creating a transform for the nearest waypoint
        wpt_location = carla.Location(x=nearest_wpt_[0], y=nearest_wpt_[1], z=0)
        wpt_rotation = carla.Rotation(pitch=0, yaw=nearest_wpt_[2], roll=0)
        nearest_wpt_tf = carla.Transform(wpt_location, wpt_rotation)

        return nearest_wpt_tf
    
    def get_next_waypoints_in_ego_frame(self, num_wpts=10, dist_between_wpts=1):
        global_path = self.global_path

        global_path_wp = self.global_path_wps
        
        global_path_tf = self.global_path_tf
        
        # Transforming the global path to the ego's frame
        ego_tf = np.array(self.ego.get_transform().get_matrix())
        global_path_relative_tf = np.linalg.pinv(ego_tf) @ global_path_tf

        ego_location = self.ego.get_transform().location
        ego_yaw = self.ego.get_transform().rotation.yaw

        curr_pose = np.array([ego_location.x, ego_location.y, ego_location.z])

        angles = np.abs(global_path[:, 2] - ego_yaw)
        angles[angles > 180] -= 360
        angles = np.abs(angles)

        # Finding the nearest waypoint
        dist = np.linalg.norm(global_path[:, :2] - curr_pose[:2], axis=1)
        dist[angles > 90] = np.inf
        nearest_wpt_idx = np.argmin(dist)
        # print("Nearest- ", nearest_wpt_idx)

        next_wpts = []
        next_gp_wpts = []
        for i in range(num_wpts):
            if nearest_wpt_idx + i >= len(global_path):
                break
            
            nearest_wpt_tf_in_ego_frame = global_path_relative_tf[nearest_wpt_idx + i]
            coords = nearest_wpt_tf_in_ego_frame[0:3, 3]
            next_gp_wpts.append(global_path_wp[nearest_wpt_idx + i])

            next_wpts.append(coords)

        next_wpts = np.array(next_wpts)
        # next_wpts[:, 1] = -next_wpts[:, 1]
        return np.array(next_wpts), next_gp_wpts

    def add_traffic(self):
        if self.number_of_vehicles == 0:
            return

        if self.traffic_manager is None:
            self.traffic_manager = self.world.get_trafficmanager(8000)
            self.traffic_manager.set_global_distance_to_leading_vehicle(1.0)
            self.traffic_manager.set_random_device_seed(0)
            self.traffic_manager_port = self.traffic_manager.get_port()
        
        self.traffic_manager.set_synchronous_mode(True)

        self.vehicles = []
        # Setting the traffic manager for the ego
        count = self.number_of_vehicles
        while count > 0:
            transform = random.choice(self.vehicle_spawn_points)
            blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=[4])
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                vehicle.set_autopilot(True)
                self.vehicles.append(vehicle)
                # self.traffic_manager.set_desired_speed(vehicle, 3)
                self.traffic_manager.ignore_lights_percentage(vehicle,100)
                count -= 1
        self.traffic_manager.global_percentage_speed_difference(80)
    
    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.

        Args:
            actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
            bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
        
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def destroy_all_vehicles(self):
        for vehicle in self.vehicles:
            try:
                vehicle.destroy()
            except Exception as e:
                pass
        self.vehicles = []

'''
Ego Transform:  Transform(Location(x=8.398685, y=198.185608, z=0.001681), Rotation(pitch=-0.000246, yaw=-179.654129, roll=-0.056671))                 
Ego Transform:  Transform(Location(x=-79.316872, y=148.183792, z=0.001953), Rotation(pitch=0.013059, yaw=-90.464806, roll=-0.005676))                 

'''
