# -*- coding: utf-8 -*-
# @Time    : 2018/7/7 15:05
# @Author  : LI Jiawei
# @Email   : jliea@connect.ust.hk
# @File    : cluster.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import json

threshold = 50
data = pd.read_csv("../data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv")

user_set = data["user_id"].unique()[:1000]  ### 20000
item_set = data["item_id"].unique()  ###
category_set = data["item_category"].unique()  ### 9557
user_interaction_dict = dict()
# for user in user_set:
user = 15330397
print(user)
user_cluster = []
user_data = data[data["user_id"] == user]["item_category"].values
for id,other_user in enumerate(user_set):
    # print(f"{user} / {id} / {other_user}")
    print(id)
    other_user_data = data[data["user_id"] == other_user]["item_category"].values
    intersection = np.intersect1d(user_data, other_user_data)
    if len(intersection) > threshold:
        user_cluster.append([other_user,len(intersection)])
user_interaction_dict[str(user)] = user_cluster
# with open("test_15330397.json",'w') as file:
#     json.dump(user_interaction_dict,file)
#     print("done!")
print(user_cluster)

# user_10001082_data = data[data["user_id"] == 10001082]
# user_10001082_cate = user_10001082_data["item_category"].values
# user_100029775_data = data[data["user_id"] == 100029775]
# user_100029775_cate = user_100029775_data["item_category"].values
# print(np.intersect1d(user_10001082_cate, user_100029775_cate))