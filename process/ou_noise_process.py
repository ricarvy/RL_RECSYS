# -*- coding: utf-8 -*-
# @Time    : 2018/7/4 17:32
# @Author  : LI Jiawei
# @Email   : jliea@connect.ust.hk
# @File    : ou_noise_process.py
# @Software: PyCharm

'''
refer to openai
https://github.com/rll/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
'''

import numpy as np
import logging

class OU_Process(object):
    def __init__(self, config):
        logging.info("create OU process of agent ..............................")
        self.action_dim = config["normal_args"]["action_dim"]
        self.theta = config["ou"]["theta"]
        self.mu = mu=config["ou"]["mu"]
        self.sigma = sigma=config["ou"]["sigma"]
        self.current_x = None

        self.init_process()
        logging.info("The params are : action_dim, theta, mu, sigma, current_x")
        logging.info(f"OU process's action_dim is {self.action_dim}")
        logging.info(f"OU process's theta is {self.theta}")
        logging.info(f"OU process's mu is {self.mu}")
        logging.info(f"OU process's sigma is {self.sigma}")
        logging.info(f"OU process's cureent_x is {self.current_x}")

        logging.info("OU process creation ended ..............................")

    def init_process(self):
        self.current_x = np.ones(self.action_dim) * self.mu

    def update_process(self):
        dx = self.theta * (self.mu - self.current_x) + self.sigma * np.random.randn(self.action_dim)
        self.current_x = self.current_x + dx

    def return_noise(self):
        self.update_process()
        return self.current_x

if __name__ == "__main__":
    ou = OU_Process(3, theta=0.15, mu=0, sigma=0.2)
    states = []
    for i in range(10000):
        states.append(ou.return_noise()[0])
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()

