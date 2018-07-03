# -*- coding: utf-8 -*-
# @Time    : 2018/7/3 14:05
# @Author  : LI Jiawei
# @Email   : jliea@connect.ust.hk
# @File    : test.py
# @Software: PyCharm

import pandas as pd

class Data:
    def __init__(self, file_path):
        self.__name = 'data'

        self.file_path = file_path

        self.dataframe = self.createDataFrame()

    def createDataFrame(self):
        dataframe = pd.read_csv(self.file_path, sep=';')
        return dataframe

    def data_cleaning(self, dataframe):
        pass

    def split_df_to_train_validate_test_set(self, split_rate = 0.5):
        pass
