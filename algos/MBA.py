# -*- coding: utf-8 -*-
# @Time    : 2018/7/5 11:24
# @Author  : LI Jiawei
# @Email   : jliea@connect.ust.hk
# @File    : MBA.py
# @Software: PyCharm

import numpy as np

'''
This file implement some normal algorithms that 
solve the EE problem, including 
Naive-bandit, Thompson-sampling-bandit, UCB-bandit, Epsilon-greedy-bandit

'''

import numpy as np
import random

class Bandit:
    def __init__(self, K):
        self.buffer_size = 50
        self.K = K
        self.beta_params = self.build_beta_params(self.K)
        self.total_count = 0
        self.name = ""

    def build_beta_params(self, K):
        # lose/win/total_count
        beta_params = np.zeros((K,3))
        beta_params[:,2] = 1
        return beta_params

    def update_arm(self, choosen_arm, result):
        self.beta_params[choosen_arm,result] += 1
        self.beta_params[choosen_arm,2] += 1
        self.total_count += 1

    def get_beta_params(self):
        return self.beta_params

    def get_name(self):
        return self.name

'''
Naive Bandit will choose the arm with higest mean reward

Hint : In order to avoid 0-division, total count will be 
set as 1 initially
'''
class NaiveBandit(Bandit):
    def __init__(self, K):
        Bandit.__init__(self,K)
        self.name = "NaiveBandit"

    def choose_arm(self):
        return np.argmax(self.beta_params[:,1]/self.beta_params[:,2])
        self.total_count += 1

class ThompsonSamplingBandit(Bandit):
    def __init__(self, K):
        Bandit.__init__(self, K)
        self.name = "ThompsonSamplingBandit"

    def choose_arm(self):
        pass

class UCBBandit(Bandit):
    def __init__(self, K):
        Bandit.__init__(self, K)
        self.name = "UCBBandit"

    def choose_arm(self):
        t = np.sum(self.beta_params, axis=0)[2]
        means = self.beta_params[:,1] / self.beta_params[:, 2]
        variance = means - means ** 2
        UCB = means + np.sqrt(
            np.minimum(variance +
                       np.sqrt(2 * np.log(t) / self.beta_params[:,2]), 0.25) *
                        np.log(t) / self.beta_params[:,2])
        return np.argmax(UCB)

    def get_name(self):
        return self.name

class EpsilonGreedyBandit(Bandit):
    def __init__(self, K, epsilon = 0.5):
        Bandit.__init__(self,K)
        self.name = "EpsilonGreedyBandit"
        self.epsilon = epsilon

    def choose_arm(self):
        if random.random() > self.epsilon:
            return self.choose_max_arm()
        else:
            return self.choose_random_arm()

    def choose_max_arm(self):
        return np.argmax(self.beta_params[:, 1] / self.beta_params[:, 2])

    def choose_random_arm(self):
        return random.randint(0,self.K)




if __name__ == '__main__':
    bandit = EpsilonGreedyBandit(K = 3)
    bandit.update_arm(choosen_arm=1, result=1)
    bandit.update_arm(choosen_arm=2, result=1)
    bandit.update_arm(choosen_arm=1, result=1)
    # print(bandit.get_beta_params())
    print(bandit.choose_arm())