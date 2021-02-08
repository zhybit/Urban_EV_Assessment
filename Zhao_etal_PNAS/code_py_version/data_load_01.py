#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: YZ
"""
import sys
import pandas as pd
import os
sys.path.append("./code")
import data_load_02

#%%
def get_operating_data(num):
    list_operation = []
    for i in range(42):
        list_operation.append('data_operate_' + str(i) + '.csv')
    file_path= '../ev_data/data_vehicle_operation/' + list_operation[num]
    d1 = pd.read_csv(file_path)
    return d1

def get_operating_data_auto():
    data_op = pd.DataFrame()
    for i in range(42):
        name = 'data_operate_' + str(i) + '.csv'
        file_path= '../ev_data/data_vehicle_operation/' + name
        d_tmp = pd.read_csv(file_path)
        data_op = pd.concat([data_op, d_tmp])
    return data_op

#%%
if __name__ == '__main__':
    data_op = pd.DataFrame()
    data_op = get_operating_data_auto()
    data_info = data_load_02.get_veh_info() # data specification data
    data_op_p1 = pd.merge(data_op, data_info,left_on='b.vin', right_on='vin') # merge dynamic and static data
