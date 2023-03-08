import carla
import numpy as np
from carla_env.utils.transforms import se3_to_components
import time
import atexit

import json

import math

import random

import cv2
from scipy.spatial.transform import Rotation as R

import logging

from carla_env.utils.custom_pid import PID
from carla_env.utils.stanley_controller import Stanley

from carla_env.utils.carla_utils import generate_target_waypoint_list_same_lane, get_controls

class CarEnv():
    metadata = {'render.modes': ['human']}

    def __init__(self, config_file):
        super(CarEnv, self).__init__()

        print("Initializing Carla Environment")

        # Reading Config File
        self.config = None 
        with open(config_file) as f:
            self.config = json.load(f)
        if self.config is None:
            raise Exception("Error in reading config file")

        # Vehicle Details
        self.ego = None
        self.target = None
        self.dummy = None
        self.lidar_sen = None
        self.semantic_lidar_sen = None
        self.third_person_view = None
        
        self.ego_pose = None
        self.ego_path = []
        self.frame = -1

        ##########
        # Carla
        ##########

        print("--carla client")
        # Settin Up the World
        carla_config = self.config["carla_client"]
        self.client = carla.Client(carla_config["ip"], carla_config["port"])
        self.client.set_timeout(carla_config["timeout"])
        self.client.load_world(carla_config["world"])
        
        print("--world")
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = carla_config["synchronous_mode"]
        settings.fixed_delta_seconds = 1 / carla_config["fps"]  # FPS = 1/0.1 = 10
        settings.no_rendering_mode = carla_config["no_rendering_mode"]

        self.world.apply_settings(settings)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        self.map = self.world.get_map()

        print("--ego setup")
        ### Ego-Vehicle
        # Vehicle Config
        ego_config = self.config["ego"]

        # Vehicle Transforms
        self.ego_trans_init = carla.Transform()
        if ego_config["spawn_type"] == "random":
            self.ego_trans_init = np.random.choice(self.map.get_spawn_points())
        elif ego_config["spawn_type"] == "fixed":
            self.ego_trans_init.location = carla.Location(x = ego_config["spawn_point"][0],
                                                y = ego_config["spawn_point"][1],
                                                z = ego_config["spawn_point"][2])
            wpt = self.map.get_waypoint(self.ego_trans_init.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            self.ego_trans_init = wpt.transform
            self.ego_trans_init.location.z += ego_config["spawn_point"][2]
        

        self.tf_matrix = np.array(self.ego_trans_init.get_matrix())
        self.yaw_init = self.ego_trans_init.rotation.yaw

        # Vehicle Blueprint
        self.ego_bp = self.world.get_blueprint_library().find(ego_config["blueprint"])
        self.ego_bp.set_attribute('role_name', 'ego')
        self.ego_bp.set_attribute('color', '255,0,0')

        print("--sensors setup")
        # Vehicle Sensors
        self._setup_sensors()

        # BEV
        self.bev = None
        self.obstacle_bev = None
        self.next_g_path = None

        # Stanley Controller
        print("--stanley setup")
        self.stanley = Stanley()

        self.left_lane_coords = None
        self.right_lane_coords = None
        self.dyn_obs_poses = None
        
        # Global Path
        self.global_path = None
        self.global_path_tf = None
        self.global_path_carla = None
        self.global_path_wps = None

        ### Traffic Manager
        print("--traffic manager setup")
        self.traffic_manager = None
        self._setup_traffic_manager()        

        self.npcs = {}

        atexit.register(self.close)

    def _next_observation(self):
        '''
        Odom frame is the car's initial frame
        X is in the direction of car
        Y in towards right of the car
        '''
        observation = {}

        # Generate BEV
        self.generate_data((256, 256), sensing_range = 15, z_max=1)

        vel_ego = self.ego.get_velocity()
        self.ego_speed = np.linalg.norm(np.array([vel_ego.x, vel_ego.y]))

        observation["bev"] = self.bev
        observation["obstacle_bev"] = self.obstacle_bev
        observation["next_g_path"] = self.next_g_path
        observation["left_lane_coords"] = self.left_lane_coords
        observation["right_lane_coords"] = self.right_lane_coords
        observation["dyn_obs_poses"] = self.dyn_obs_poses
        
        observation["ego_speed"] = self.ego_speed

        return observation

    def step(self, trajectory, target_speed):
        throttle, brake, steer = 0, 0, 0
        self.frame += 1
        if target_speed != -1:
            
            self.ego_tf = self.ego.get_transform()
            self.ego_path.append(self.ego_tf)
            self.ego_pose = np.array([self.ego_tf.location.x, self.ego_tf.location.y, self.ego_tf.location.z, self.ego_tf.rotation.yaw, self.ego_tf.rotation.pitch, self.ego_tf.rotation.roll])

            vel_ego = self.ego.get_velocity()
            self.ego_speed = np.linalg.norm(np.array([vel_ego.x, vel_ego.y]))
            
            
            self.stanley.set_current_speed(self.ego_speed)
            self.stanley.set_target_speed(target_speed)

            try:
                self.stanley.set_waypoints(trajectory)
            except Exception as e:
                print(e)
                print("Trajectory : ", trajectory)
                print("Velocity : ", np.round(target_speed,2))
                # exit()

            action, ret_val = self.stanley.get_controls()

            throttle = action[0]
            brake = action[1]
            steer = action[2]      

            self.ego.apply_control(carla.VehicleControl(
                throttle=float(throttle), steer=float(steer), brake=float(brake)))

        self.move_npcs()

        # Tick the world
        self.world.tick()

        # Update spectator
        spectator = self.world.get_spectator()
        spectator.set_transform(self.dummy.get_transform())
        ######################

        obs = self._next_observation()
        done = False

        y_offset, yaw_offset = self.get_offset()
        if np.abs(y_offset) > 1.5:
            print('!!!!!!OUT OF LANE!!!!!!!!!')

        done, collided_sensors = self.check_for_collision()

        print("Done  - ", done)

        reward, state = None, None
        action = [throttle, brake, steer]
        return obs, reward, done, state, action

    def reset(self):
        # Reset the state of the environment to an initial state
        print("Resetting the environment")
        self.frame = -1
        self.ego_path = []

        time.sleep(1.5)
        
        print("--spawn ego")
        # Spawn Ego
        self._spawn_ego()

        print("--attach sensors")
        # attach sensors
        self._attach_sensors()

        if self.traffic_manager is None:
            print("--setup traffic manager")
            self._setup_traffic_manager()

        print("--spawn npcs")
        # Spawn NPCs
        self._spawn_npcs()

        print("--setup 3person view")
        # Attach Dummy Camera
        if self.dummy is not None:
            self.dummy.destroy()
        dummy_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        dummy_transform = carla.Transform(
            carla.Location(x=-6, z=4), carla.Rotation(pitch=10.0))
        self.dummy = self.world.spawn_actor(
            dummy_bp, dummy_transform, attach_to=self.ego, attachment_type=carla.AttachmentType.SpringArm)
        self.dummy.listen(lambda image: self.third_person_callback(image))

        print("--wait for sensors")
        # Wait for the sensors to be ready
        global process_lidar
        process_lidar = False
        for i in range(10):  # Some buffer time at the start of the episode
            self.world.tick()
            spectator = self.world.get_spectator()
            spectator.set_transform(self.dummy.get_transform())
        process_lidar = True

        print("--generate global path")
        # Generate a global path
        self.generate_global_path(num_of_waypoints=1000, dist_between_wpts=1)

        print("--Waiting for environment to be ready...")
        for i in range(10):
            self.world.tick()

        return self._next_observation()

    def render(self, mode='human', close=False):
        pass

    def close(self):
        print("Destroying everything")
        things_to_destroy = [self.dummy] + list(self.sensors.values()) + [self.ego] + list(self.npcs.values())
        
        i = 0
        for thing in things_to_destroy:
            try:
                thing.destroy()
            except Exception as e:
                print("T--hing number - ", i)
                print(e)
            i += 1

    def _setup_sensors(self):
        # sensors
        sensors_config = self.config["ego"]["sensors"]
        
        self.sensors = {}
        self.all_sensor_bp = {}
        self.sensor_data = {}
        self.sensor_properties = {}
        for s_conf in sensors_config:
            if s_conf["type"] == "sensor.lidar.ray_cast":
                sensor_bp = self.world.get_blueprint_library().find(s_conf["type"])
                sensor_bp.set_attribute('channels', str(s_conf["channels"]))
                sensor_bp.set_attribute('rotation_frequency', str(s_conf["rotation_frequency"]))
                sensor_bp.set_attribute('range', str(s_conf["range"]))
                sensor_bp.set_attribute('lower_fov', str(s_conf["lower_fov"]))
                sensor_bp.set_attribute('upper_fov', str(s_conf["upper_fov"]))
                sensor_bp.set_attribute('points_per_second', str(s_conf["points_per_second"]))
                sensor_bp.set_attribute('dropoff_general_rate', str(s_conf["dropoff_general_rate"]))

                sensor_transform = carla.Transform(carla.Location(  x = s_conf["location"][0],
                                                                    y = s_conf["location"][1],
                                                                    z = s_conf["location"][2]),
                                                    carla.Rotation( pitch = s_conf["rotation"][0],
                                                                    yaw = s_conf["rotation"][1],
                                                                    roll = s_conf["rotation"][2]))
                self.sensor_properties[s_conf["name"]] = s_conf

                self.all_sensor_bp[s_conf["name"]] = [sensor_bp, sensor_transform]
                self.sensor_data[s_conf["name"]] = None

            elif s_conf["type"] == "sensor.lidar.ray_cast_semantic":
                sensor_bp = self.world.get_blueprint_library().find(s_conf["type"])
                sensor_bp.set_attribute('channels', str(s_conf["channels"]))
                sensor_bp.set_attribute('rotation_frequency', str(s_conf["rotation_frequency"]))
                sensor_bp.set_attribute('range', str(s_conf["range"]))
                sensor_bp.set_attribute('lower_fov', str(s_conf["lower_fov"]))
                sensor_bp.set_attribute('upper_fov', str(s_conf["upper_fov"]))
                sensor_bp.set_attribute('points_per_second', str(s_conf["points_per_second"]))

                sensor_transform = carla.Transform(carla.Location(  x = s_conf["location"][0],
                                                                    y = s_conf["location"][1],
                                                                    z = s_conf["location"][2]),
                                                    carla.Rotation( pitch = s_conf["rotation"][0],
                                                                    yaw = s_conf["rotation"][1],
                                                                    roll = s_conf["rotation"][2]))
            
                
                self.sensor_properties[s_conf["name"]] = s_conf

                self.all_sensor_bp[s_conf["name"]] = [sensor_bp, sensor_transform]
                self.sensor_data[s_conf["name"]] = None
            
            elif s_conf["type"] == "sensor.other.collision":
                sensor_bp = self.world.get_blueprint_library().find('sensor.other.collision')

                self.all_sensor_bp[s_conf["name"]] = [sensor_bp, carla.Transform()]
                self.sensor_properties[s_conf["name"]] = s_conf
                self.sensor_data[s_conf["name"]] = []

    def _spawn_ego(self):
        if self.ego is not None:
            self._destroy_all_sensors()
            try:
                self.ego.destroy()
            except Exception as e:
                print(e)
        
        # Spawn Ego
        if self.config["ego"]["spawn_type"] == "random":        
            self.ego_trans_init = np.random.choice(self.map.get_spawn_points())
        self.ego = self.world.spawn_actor(self.ego_bp, self.ego_trans_init)
        
        print("----Ego spawned at: ", self.ego_trans_init.location)

        self.ego.set_autopilot(False)

    def _destroy_all_sensors(self):
        for sensor in self.ego.get_all_sensors():
            try:
                sensor.destroy()
            except Exception as e:
                print(e)

    def _attach_sensors(self):
        for sensor_name in self.all_sensor_bp.keys():
            if self.sensor_properties[sensor_name]["type"] == "sensor.lidar.ray_cast":
                lidar_bp, lidar_tf = self.all_sensor_bp[sensor_name][0], self.all_sensor_bp[sensor_name][1]
                
                self.sensors[sensor_name] = self.world.spawn_actor(
                    lidar_bp, lidar_tf, attach_to=self.ego
                    )
                
                self.sensors[sensor_name].listen(
                    lambda point_cloud: self.process_lidar_raw_points(point_cloud))
            elif self.sensor_properties[sensor_name]["type"] == "sensor.lidar.ray_cast_semantic":
                lidar_bp, lidar_tf = self.all_sensor_bp[sensor_name][0], self.all_sensor_bp[sensor_name][1]

                self.sensors[sensor_name] = self.world.spawn_actor(
                    lidar_bp, lidar_tf, attach_to=self.ego
                    )
                
                self.sensors[sensor_name].listen(
                    lambda point_cloud: self.process_semantic_lidar_raw_points(point_cloud))
            
            elif self.sensor_properties[sensor_name]["type"] == "sensor.other.collision":
                collision_bp, collision_tf = self.all_sensor_bp[sensor_name][0], self.all_sensor_bp[sensor_name][1]
                self.sensors[sensor_name] = self.world.spawn_actor(
                    collision_bp, collision_tf, attach_to=self.ego
                    )
                self.sensors[sensor_name].listen(
                    lambda event: self.process_collision_data(event, sensor_name))

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

    def process_collision_data(self, event, sensor_name):
        sensor_conf = self.sensor_properties[sensor_name]

        impulse = event.normal_impulse
        intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.sensor_data[sensor_name].append(intensity)

        if len(self.sensor_data[sensor_name])>sensor_conf["history_length"]:
            self.sensor_data[sensor_name].pop(0)

    def third_person_callback(self, image_data):
        image_no = image_data.frame
        
        image = np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8"))
        image = np.reshape(image, (image_data.height, image_data.width, 4))

        self.third_person_view = image
        

    def _setup_traffic_manager(self):
        traffic_manager_config = self.config["traffic_manager"]
        self.traffic_manager_port = traffic_manager_config["port"]
        
        self.traffic_manager = self.client.get_trafficmanager(self.traffic_manager_port)
        self.traffic_manager.set_random_device_seed(traffic_manager_config["random_device_seed"])


    def _spawn_npcs(self):
        self.traffic_manager.set_synchronous_mode(False)

        npc_config = self.config["npc"]
        
        ego_tf = self.ego_trans_init
        ego_location = ego_tf.location
                
        self.npcs = {}
        for i in range(len(npc_config)):
            npc_conf = npc_config[i]

            if npc_conf["spawn_type"] == "random":
                npc_tf = random.choice(self.map.get_spawn_points())
                spawn_point = [npc_tf.location.x, npc_tf.location.y, npc_tf.location.z]
            elif npc_conf["spawn_type"] == "fixed":
                spawn_point = npc_conf["spawn_point"]
                npc_tf = carla.Transform()
                npc_tf.location = carla.Location(x=spawn_point[0], y=spawn_point[1], z=spawn_point[2])
                
                wpt = self.map.get_waypoint(npc_tf.location, project_to_road = True, lane_type=carla.LaneType.Driving)
                npc_tf = wpt.transform
                npc_tf.location.z = spawn_point[2]

            elif npc_conf["spawn_type"] == "relative":
                spawn_point = npc_conf["spawn_point"]
                
                npc_tf = carla.Transform()
                npc_tf.location = carla.Location(x=spawn_point[0], y=spawn_point[1], z=spawn_point[2])
                
                wpt = self.map.get_waypoint(ego_location, project_to_road = True, lane_type=carla.LaneType.Driving)
                
                relative_x = spawn_point[0]
                wpt = wpt.next(relative_x)[0]
                
                relative_y = spawn_point[1]
                while relative_y != 0:
                    if relative_y > 0:
                        if wpt.get_right_lane() is None:
                            break
                        wpt = wpt.get_right_lane()
                        relative_y -= 1
                    elif relative_y < 0:
                        if wpt.get_left_lane() is None:
                            break
                        wpt = wpt.get_left_lane()
                        relative_y += 1

                npc_tf = wpt.transform
                npc_tf.location.z += spawn_point[2]
 
            number_of_tries = 10
            npc = None 
            while npc is None:
                npc_bp = self.world.get_blueprint_library().find(npc_conf["blueprint"])

                try:
                    npc = self.world.spawn_actor(npc_bp, npc_tf)
                except Exception as e:
                    npc = None
                    print("----Failed to spawn NPC - ", i, " - ", npc_conf["blueprint"], " - at ", npc_tf.location)
                    print(e)

                if (npc is None) and number_of_tries > 0:
                    number_of_tries -= 1
                    continue
                elif npc is None:
                    print("----Failed to spawn NPC - ", i, " - ", npc_conf["blueprint"], " - at ", npc_tf.location)
                    break

                print("----Spawned NPC - ", i, " - ", npc_conf["blueprint"], " - at ", spawn_point)
                if npc_conf["target_speed"] > 0.0:
                    npc.set_autopilot(True)
                
                self.traffic_manager.distance_to_leading_vehicle(npc, npc_conf["distance_to_leading_vehicle"])
                self.traffic_manager.ignore_lights_percentage(npc, npc_conf["ignore_lights_percentage"])
                self.traffic_manager.ignore_signs_percentage(npc, npc_conf["ignore_signs_percentage"])
                
                # self.traffic_manager.set_desired_speed(npc, npc_conf["desired_speed"])
                # self.traffic_manager.vehicle_percentage_speed_difference(npc, npc_conf["speed_difference_percentage"])

                self.npcs[npc_conf["id"]] = npc
                print("List of npcs - ", self.npcs.keys())
        self.traffic_manager.set_synchronous_mode(True)


    def move_npcs(self):
        npcs = self.npcs
        
        npc_config = self.config["npc"]
        for npc_conf in npc_config:
            if npc_conf["control_type"] == "autopilot":
                continue
            elif npc_conf["control_type"] == "waypoint":
                npc = npcs[npc_conf["id"]]
                if not npc.is_alive:
                    continue
                target_speed = npc_conf["target_speed"]
                if target_speed == 0.0:
                    continue
                wpt = self.map.get_waypoint(npc.get_location(), project_to_road = True, lane_type = carla.LaneType.Driving)

                target_wpts = generate_target_waypoint_list_same_lane(wpt, distance_same_lane=10, step_distance=1)[0]

                v, w = get_controls(npc, wpt, target_wpts, target_speed)

                npc.set_target_velocity(v)
                npc.set_target_angular_velocity(w)

    def generate_data(self, img_size, sensing_range, z_max):
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

        g_path, g_path_wpts = self.get_next_waypoints_in_ego_frame(40, 1)
        # Scaling the points to the image size
        g_path_in_image_scale = g_path[:, :2]*scale + img_size[0]//2

        dyn_obs_poses = self.get_npc_poses(sensing_range)
        #   Scaling Points to image size
        obs_pose_in_image_scale = []
        if len(dyn_obs_poses)>0:
            obs_pose_in_image_scale = dyn_obs_poses[:, :2]*scale + img_size[0]//2
            obs_pose_in_image_scale = np.clip(np.intp(obs_pose_in_image_scale[:, :2]), 0, 255)

        left_lane_pts, right_lane_pts = self.get_lane_boundaries(g_path_wpts)
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
        
        self.bev = bev
        self.obstacle_bev = obstacle_bev
        self.next_g_path = np.array(g_path_in_image_scale)
        self.left_lane_coords = np.array(left_lane_pt_in_image_scale)
        self.right_lane_coords = np.array(right_lane_pt_in_image_scale)
        self.dyn_obs_poses = np.array(dyn_obs_poses)
    
    def get_offset(self):
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
        
        return wy, wYaw

    def check_for_collision(self):
        collision_sensors = []
        for s_conf in self.config["ego"]["sensors"]:
            if s_conf["type"] == "sensor.other.collision":
                collision_sensors.append(s_conf["name"])
        
        collision = False
        collided_sensors = []
        for c_s in collision_sensors:          
            collision_hist = self.sensor_data[c_s]
            if len(collision_hist) > 0: 
                collision = collision or True
                collided_sensors.append(c_s)
                print('Collision with ', c_s)
            
        return collision, collided_sensors
        
    def get_npc_poses(self, obs_range):
        ego_tf = np.array(self.ego.get_transform().get_matrix())
        
        npc_poses_wrt_ego = []
        
        for id in list(self.npcs.keys()):
            #   Obstacle traneformation matrix
            npc = self.npcs[id]
            npc_tf = np.array(npc.get_transform().get_matrix())

            #   Transformation of obs wrt ego
            obs_wrt_ego = np.linalg.pinv(ego_tf) @ npc_tf

            #  Rotation matrix
            rot_mat = R.from_matrix(np.array(obs_wrt_ego[:3, :3]))
            rel_yaw = np.rad2deg(np.array(rot_mat.as_rotvec())[-1])
            
            #  The ego vehicle yaw is 90, hence this is required
            if rel_yaw<0.0:
                rel_yaw = rel_yaw + 360.0
            rel_yaw = rel_yaw + 90.0

            v = self.npcs[id].get_velocity()
            obs_v = ((v.x)**2 + (v.y)**2)**0.5
            obs_w = self.npcs[id].get_angular_velocity().z

            #   If obstacle distance less thatn range(of bev), store pose
            if (obs_wrt_ego[0, 3]**2 + obs_wrt_ego[1, 3]**2)**0.5 < obs_range:
                npc_poses_wrt_ego.append([obs_wrt_ego[0, 3], obs_wrt_ego[1, 3], np.deg2rad(rel_yaw), obs_v, obs_w])
        
        #  Stored as x,y,theta,v,w
        return np.array(npc_poses_wrt_ego)
    
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

    def destroy_all_npcs(self):
        for npc in self.npcs:
            try:
                npc.destroy()
            except Exception as e:
                pass
        self.npcs = []

'''
Ego Transform:  Transform(Location(x=8.398685, y=198.185608, z=0.001681), Rotation(pitch=-0.000246, yaw=-179.654129, roll=-0.056671))                 
Ego Transform:  Transform(Location(x=-79.316872, y=148.183792, z=0.001953), Rotation(pitch=0.013059, yaw=-90.464806, roll=-0.005676))                 

'''
