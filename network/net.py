# -*- coding: utf-8 -*-
# @Time    : 2018/7/3 15:15
# @Author  : LI Jiawei
# @Email   : jliea@connect.ust.hk
# @File    : net.py
# @Software: PyCharm

'''
This file is used for network implement, It include two types of
net work named Actor-Network and Critic-Network. Each type of network
can seperate into two types network : online network and target network.
So In this part we only implement A and C network because the o and t
net work are initialized by A and C
'''

import tensorflow as tf
from tensorflow.contrib import slim

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr = 0.001):
        pass

    def learn(self, s, a, td):
        pass

    def choose_action(self, s):
        pass

    def build_network(self, data):
        net = slim.conv2d(data, 64, 7, 2, padding='SAME', scope='conv2')

class Critic(object):
    def __init__(self):
        pass

    def learn(self):
        pass

    def build_network(self):
        pass