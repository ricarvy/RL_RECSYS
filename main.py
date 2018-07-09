# -*- coding: utf-8 -*-
# @Time    : 2018/7/3 14:09
# @Author  : LI Jiawei
# @Email   : jliea@connect.ust.hk
# @File    : main.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging

from argparse import ArgumentParser
from datetime import datetime

from utils.test import Data
from utils.json_loader import config_load
from agent.agent import Agent
from buffer.replay_buffer import Replay_Buffer
from process.ou_noise_process import OU_Process
from network.net import Model

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode",
                        dest="mode",
                        help="")
    parser.add_argument("--process",
                        dest="process",
                        help="")
    return parser

def create_df(path, name):
    logging.info(f"create {name} of agent : ..............................")
    o_t = datetime.now()

    df = pd.read_csv(path)
    logging.info(f"the shape of df is {df.shape}")

    n_t = datetime.now()
    logging.info(f"Time consuming is {(o_t-n_t).microseconds} ms")
    logging.info(f"{name} creation is ended : ..............................")

    return df

def create_agent(config, session):
    logging.info("Create agent : ====================================================================")
    model = Model(config=config,
                  sess=session)
    replay_buffer = Replay_Buffer(config)
    ou_process = OU_Process(config)
    record = create_df("data/temp/user_15330397.csv", "record")
    item_set = create_df("data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv", "item_set")
    user_item_data = create_df("data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv", "user_set")
    agent = Agent(config=config ,
                  model=model,
                  replay_buffer=replay_buffer,
                  noise=ou_process,
                  record=record,
                  item_set=item_set,
                  user_item_data = user_item_data,
                  verbose=1)
    logging.info("End creating agent : ====================================================================")
    return agent

def main():
    logging.basicConfig(level=logging.DEBUG)
    agent = None
    session = tf.Session()
    parser = build_parser()
    option = parser.parse_args()
    config = config_load("config.json")

    if option.mode == "train":
        print("train")
    if option.process == "create_agent":
        agent = create_agent(config=config, session=session)
        # for i in range(10):
        #     print(f"Num {i} epoch : ")
        #     print(agent.get_reward_from_actions())
        print(f"total item_set is {agent.get_item_set().shape[0]}")
        reward_count = 0
        for epoch in range(1000):
            reward = agent.get_reward_from_actions()
            print(f"Now is num {epoch} epoch: "
                    f"current_state is {agent.get_cur_state()['item_id'].tolist()}, "
                    f"curretn_action is {agent.get_cur_action()['item_id'].tolist()},"
                    f"current reward is {reward}")
            print(f"Now's length of item_set is {agent.get_item_set().shape[0]}")
            reward_count += reward
        print(reward_count)



if __name__ == '__main__':
    main()