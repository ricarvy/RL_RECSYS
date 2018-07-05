# -*- coding: utf-8 -*-
# @Time    : 2018/7/3 14:09
# @Author  : LI Jiawei
# @Email   : jliea@connect.ust.hk
# @File    : main.py
# @Software: PyCharm

import pandas as pd

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
    data = pd.read_csv("data/user_logs_from_mangodb.csv")
    data = data[data['event_name'] != "af_process_to_pay"]
    item_set = data[data['sku'] == "264826011"]
    print(item_set)


if __name__ == '__main__':
    main()