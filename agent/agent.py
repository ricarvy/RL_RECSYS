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
    def __init__(self, config, model=None, replay_buffer=None, noise=None, record=None,
                 item_set=None, user_item_data=None, verbose=1):
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
        self.user_item_data = user_item_data

        self.categries_list_of_item_set = self.create_categries_list_of_item_set()
        self.relative_users = self.get_relative_users(sample_user_num=1000)

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

    def get_reward_from_actions(self, discount_factor = 0.01):
        reward = 0
        actions = self.choose_action_random()
        for action in actions:
            # print(action)
            # print(action in self.record["item_category"].tolist())
            if action in self.record["item_category"].tolist():
                # print(self.record[self.record["item_category"] == action]["behavior_type"])
                reward += np.sum(self.record[self.record["item_category"] == action]["behavior_type"])
            else:
                for relative_user in self.relative_users:
                    print(f"{action}/{len(actions)}  relative user : {relative_user}")
                    relative_record = self.user_item_data[self.user_item_data["user_id"] == relative_user[0]]
                    if action in relative_record:
                        reward += discount_factor * \
                                  np.sum(relative_record[relative_record["item_category"] == action]["behavior_type"])
        return reward

    def get_relative_users(self, sample_user_num=1000, user_id=15330397, theta=0.01, threshold=50):
        user_set = self.user_item_data["user_id"].unique()[:sample_user_num]  ### 20000
        user = 15330397
        print(user)
        user_cluster = []
        user_data = self.user_item_data[self.user_item_data["user_id"] == user]["item_category"].values
        for _, other_user in enumerate(user_set):
            other_user_data = self.user_item_data[self.user_item_data["user_id"] == other_user]["item_category"].values
            intersection = np.intersect1d(user_data, other_user_data)
            if len(intersection) > threshold:
                user_cluster.append([other_user, len(intersection)])
        return user_cluster


    def update_states(self):
        pass