# -*- coding: utf-8 -*-
# @Time    : 2018/7/3 14:09
# @Author  : LI Jiawei
# @Email   : jliea@connect.ust.hk
# @File    : main.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser

from utils.test import Data
from utils.json_loader import config_load
from agent.agent import Agent

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--processes")

def main():
    # config = config_load('config.json')
    # agent = Agent(config)
    # df = pd.read_csv("data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv")
    # df = df[df['user_id'] == 15330397]
    # df.to_csv("data/temp/user_15330397.csv")
    # df_group_by_item_category = df.groupby(by=["item_category"])['user_id'].count()

    df_user = pd.read_csv("data/temp/user_15330397.csv")
    item_category_user = df_user["item_category"].values
    # # print(df[df['item_id'] == 2538])
    # df_group_by_item_cat = df.groupby(by=["item_category"])["user_id"].count().sort_values(ascending=False)
    # print(df_group_by_item_cat)
    # print(df[df["item_id"] == 54031954])
    # print(df_group_by_item_id)
    # plot_index = df_group_by_item_id.index
    # plot_value = df_group_by_item_id.values
    # print(plot_index[np.argmax(plot_value)])  # 54031952

    # df = pd.read_csv("data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv")
    # print(df[df["item_category"] == 54031952])

    # plt.bar(range(len(plot_value)),plot_value,tick_label = plot_index)
    # plt.show()

    df_item =pd.read_csv("data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv")
    item_category_item = df_item["item_category"].values
    print(np.intersect1d(item_category_user,item_category_item))

if __name__ == '__main__':
    main()