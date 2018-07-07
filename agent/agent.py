# -*- coding: utf-8 -*-
# @Time    : 2018/7/3 16:19
# @Author  : LI Jiawei
# @Email   : jliea@connect.ust.hk
# @File    : agent.py
# @Software: PyCharm

'''
.........

'''
import numpy as np
import pandas as pd
import random

class Agent():
    def __init__(self, config, model=None, replay_buffer=None, noise=None, record=None, item_set=None, verbose=1):
        ### Basic Info of agent
        self.config = config
        self.model = model
        self.replay_buffer = replay_buffer
        self.noise = noise
        self.discount_factor = config["normal_args"]["discount_factor"]
        self.verbose = verbose

        self.state_dim = config["normal_args"]["state_dim"]
        self.action_dim = config["normal_args"]["action_dim"]

        ### Eev relative args
        self.record = record
        self.item_set = item_set

        self.categries_list_of_item_set = self.create_categries_list_of_item_set()

    def get_discount_factor(self):
        return self.discount_factor

    def get_replay_buffer(self):
        return self.replay_buffer

    def get_verbose(self):
        return self.verbose

    def create_categries_list_of_item_set(self):
        return self.item_set["item_category"].unique()

    def predict_action(self, input):
        return self.model.predict_action(input)

    def choose_action(self, input, p=None):
        prediction = self.model.predict_action(input)
        noise_process = self.noise.return_noise()
        if p is not None:
            return p * prediction + (1-p) * noise_process
        else:
            return prediction + noise_process

    def choose_action_random(self):
        return random.sample(list(self.categries_list_of_item_set),self.action_dim)

    def get_reward_from_actions(self):
        reward = 0
        actions = self.choose_action_random()
        for action in actions:
            matched = self.record[self.record["item_category"] == action]
            re = matched["behavior_type"].values
            if len(matched) != 0:
                reward = matched["behavior_type"].sum()
        return reward
