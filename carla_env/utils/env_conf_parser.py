#!/usr/bin/env python

import json
import os

class ConfigParser:
    def __init__(self, config_file):
        self.config = self.parse_config(config_file)    

    def parse_config(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)

        return config

    def get_config(self):
        return self.config
    
    def get_config_value(self, key):
        if key not in self.config:
            raise ValueError("Key {} not found in config file".format(key))
        return self.config[key]

    def get_ego_config(self):
        if 'ego' not in self.config:
            raise ValueError("Ego not found in config file")
        return self.config['ego']
    
    def get_all_obstacle_config(self):
        if 'obstacles' not in self.config:
            raise ValueError("Obstacles not found in config file")
        return self.config['obstacles']
    
    def get_obstacle_config(self, obstacle_id):
        if obstacle_id not in self.config['obstacles']:
            raise ValueError("Obstacle id {} not found in config file".format(obstacle_id))
        return self.config['obstacles'][obstacle_id]
    
    def get_all_pedestrian_config(self):
        if 'pedestrians' not in self.config:
            raise ValueError("Pedestrians not found in config file")
        return self.config['pedestrians']
    
    def get_save_path(self):
        if 'save_path' not in self.config:
            raise ValueError("Save path not found in config file")
        return self.config['save_path']
    

