# -*- coding: utf-8 -*-
# @Time    : 2018/7/4 17:25
# @Author  : LI Jiawei
# @Email   : jliea@connect.ust.hk
# @File    : json_loader.py
# @Software: PyCharm

import json

def config_load(path):
    with open(path, 'r') as file:
        return json.load(file)