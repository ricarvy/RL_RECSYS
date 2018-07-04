# -*- coding: utf-8 -*-
# @Time    : 2018/7/4 8:54
# @Author  : LI Jiawei
# @Email   : jliea@connect.ust.hk
# @File    : fake_data_gen.py
# @Software: PyCharm

'''
Assume that we have one specific cusumer that have about 1000 browsing
records for 100 goods

Item ID bounded in [0,50]
Operation ID bounded in [0,3]

'''

import pandas as pd
import numpy as np

UID = 123456789

I_arr = np.random.randint(0,50,size=2000)
O_arr = np.random.randint(0,3,size=2000)

df = pd.DataFrame({
    'UID' : UID,
    'IID' : pd.Series(I_arr,index=list(range(2000)), dtype='float32'),
    'OID' : pd.Series(O_arr,index=list(range(2000)), dtype='float32')
})
df.to_csv('../data/temp/fake_data.csv')
