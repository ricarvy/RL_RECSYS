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


'''
Naive Bandit will choose the arm with higest mean reward

Hint : In order to avoid 0-division, total count will be 
set as 1 initially
'''
class NaiveBandit:
    def __init__(self, K):
        self.buffer_size = 50
        self.K = K
        self.beta_params = self.build_beta_params(self.K)
        self.total_count = None

    def choose_arm(self):
        return np.argmax(self.beta_params[:,1]/self.beta_params[:,2])

    def update_arm(self, choosen_arm, result):
        self.beta_params[choosen_arm,result] += 1
        self.beta_params[choosen_arm,2] += 1

    def build_beta_params(self, K):
        # lose/win/total_count
        beta_params = np.zeros((K,3))
        beta_params[:,2] = 1
        return beta_params

    def get_beta_params(self):
        return self.beta_params

class ThompsonSamplingBandit:
    def __init__(self):
        pass

class UCBBandit:
    def __init__(self):
        pass

class EpsilonGreedyBandit:
    def __init__(self):
        pass



if __name__ == '__main__':
    naiveBandit = NaiveBandit(3)
    naiveBandit.update_arm(choosen_arm=1, result=1)
    naiveBandit.update_arm(choosen_arm=2, result=1)
    naiveBandit.update_arm(choosen_arm=1, result=1)
    print(naiveBandit.get_beta_params())
    print(naiveBandit.choose_arm())