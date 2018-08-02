# -*- coding: utf-8 -*-
# @Time    : 2018/7/24 9:34
# @Author  : LI Jiawei
# @Email   : jliea@connect.ust.hk
# @File    : db_hive_temp.py
# @Software: PyCharm

from pyhive import  hive


class db_hive(object):
    def __init__(self,name='hive',password='hive',port=10000):
        self.cursor=hive.Connection(host="172.31.57.86", port=port, username=name, auth='CUSTOM', password=password).cursor()


    def excute_sql(self,sqlstr):
        self.cursor.execute(sqlstr)

    @property
    def fetchall(self):
        return  self.cursor.fetchall()

if __name__ == '__main__':

    test=db_hive()
    test.excute_sql('select * from apl_gb_recommend_10002_dev_db.student')
    for x in test.fetchall:
        print(x)