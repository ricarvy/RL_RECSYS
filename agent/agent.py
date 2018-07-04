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

class Agent():
    def __init__(self, config, model=None, replay_buffer=None, noise=None, discount_factor=0.5, verbose=1):
        self.model = model
        self.replay_buffer = replay_buffer
        self.noise = noise
        self.discount_factor = config["normal_args"]["discount_factor"]
        self.verbose = verbose

    def get_discount_factor(self):
        return self.discount_factor

    def get_replay_buffer(self):
        return self.replay_buffer

    def get_verbose(self):
        return self.verbose
