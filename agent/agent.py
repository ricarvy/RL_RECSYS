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

        self.cur_state = self.record.head(self.state_dim).loc[:, ["item_id", "item_category"]]
        self.cur_action = self.choose_action_random().loc[:, ["item_id", "item_category"]]

    def get_discount_factor(self):
        return self.discount_factor

    def get_replay_buffer(self):
        return self.replay_buffer

    def get_verbose(self):
        return self.verbose

    def get_record(self):
        return self.record

    def get_cur_state(self):
        return self.cur_state

    def get_cur_action(self):
        return self.cur_action

    def get_item_set(self):
        return self.item_set

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
        self.cur_action = self.item_set.sample(self.action_dim)
        return self.cur_action

    # def choose_action_cheat(self):
    #     cur_action_a = [53918331, 315215324, 58770666, 92663497, 57273542, 288257305, 237754076, 282989515]
    #     cur_action_b = [8595, 8595, 8595, 8595, 8595, 8595, 8595, 8595]
    #     self.cur_action = pd.DataFrame({"item_id": cur_action_a, "item_category": cur_action_b})
    #     return self.cur_action

    def get_reward_from_actions(self, discount_factor = 0.01):
        self.choose_action_random()
        reward = 0
        for action in self.cur_action.values:
            if action[0] in self.cur_state["item_id"].tolist():
                reward += np.sum(self.record[self.record["item_id"] == action[0]]["behavior_type"])
            elif action[1] in self.cur_state["item_category"].tolist():
                reward += discount_factor * np.sum(self.record[self.record["item_category"] == action[1]]["behavior_type"])
            self.update_states(action, reward)
        return reward

    # def get_relative_users(self, sample_user_num=1000, user_id=15330397, theta=0.01, threshold=50):
    #     user_set = self.user_item_data["user_id"].unique()[:sample_user_num]  ### 20000
    #     user = 15330397
    #     print(user)
    #     user_cluster = []
    #     user_data = self.user_item_data[self.user_item_data["user_id"] == user]["item_category"].values
    #     for _, other_user in enumerate(user_set):
    #         other_user_data = self.user_item_data[self.user_item_data["user_id"] == other_user]["item_category"].values
    #         intersection = np.intersect1d(user_data, other_user_data)
    #         if len(intersection) > threshold:
    #             user_cluster.append([other_user, len(intersection)])
    #     return user_cluster


    def update_states(self,action, reward):
        # print(action[0])
        if reward == 0:
            # print(f"del {self.item_set[self.item_set['item_id'] == action[0]].shape[0]} rows")
            self.item_set = self.item_set[self.item_set["item_id"] != action[0]]
        else:
            self.cur_state = self.cur_state.append(pd.DataFrame([action[[0,2]]], columns=["item_id", "item_category"]).fillna(0))
            self.cur_state = self.cur_state[1:]