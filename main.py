# -*- coding: utf-8 -*-
# @Time    : 2018/7/3 14:09
# @Author  : LI Jiawei
# @Email   : jliea@connect.ust.hk
# @File    : main.py
# @Software: PyCharm

from argparse import ArgumentParser

from utils.test import Data
from utils.json_loader import config_load
from agent.agent import Agent

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--processes")

def main():
    config = config_load('config.json')
    agent = Agent(config)



if __name__ == '__main__':
    main()