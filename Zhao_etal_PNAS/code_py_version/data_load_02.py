#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: YZ
"""
import pandas as pd
import os
import numpy as np
#%%
def get_veh_info():
    data_path = '../ev_data/data_vehicle_feature/'
    names = ['data_models.csv', 'data_vehicles.csv']
    data_mod = pd.read_csv(data_path + names[0])
    data_veh = pd.read_csv(data_path + names[1])
    data_veh_mod = pd.merge(data_veh, data_mod, 
                            left_on=data_veh.columns[6], 
                            right_on = data_mod.columns[0])
    data_veh_mod_p = data_veh_mod.iloc[:, [2, 3, 4, 5, 6, 9, 12, 15]]
    data_veh_mod_p.columns  = ['vin', 'fleet_type', 'city', 'province', 
                               'veh_model', 'common_name', 'brand', 'fuel_type']
    return data_veh_mod_p

def get_ambient_temp():
    data_path = '../ev_data/data_temp/Region_temp.xlsx'
    data_temp = pd.read_excel(data_path)
    return data_temp

def get_daily_dist():
    data_path = '../ev_data/data_sta_sets/daily_sets_df.csv'
    data_dist = pd.read_csv(data_path)
    return data_dist

def fleet_class(cell): # classify fleet type into pri/pub
    if cell == 'private':
        return 'pr'
    elif cell in ['taix', 'rental']:
        return 'pu'
    else: # official
        return cell
    
def get_monthly_dist():
    data_dist = get_daily_dist()
    data_dist['time_dt'] = pd.to_datetime(data_dist['time'], format='%Y%m%d')
    data_dist['month'] = data_dist['time_dt'].dt.month
    data_dist['fleet_type2'] = data_dist['fleet_type'].apply(fleet_class)
    tbl = pd.pivot_table(data_dist, 
                         index=['vehicle', 'month', 'region', 'fleet_type2'],
                         values='dert_dist(km)', aggfunc=np.sum)
    dict_mon_dict = {}
    for reg in ['BJ', 'SH', 'GZ']:
        for ft in ['pr', 'pu']:
            tbl_tmp = tbl.loc(axis=0)[:, :, reg, ft].loc[:, 'dert_dist(km)']
            dict_mon_dict.update({'%s%s'%(reg, ft): tbl_tmp})
    return dict_mon_dict

if __name__ == '__main__':
    data_veh_mod_p =  get_veh_info()
    dict_mon_dict = get_monthly_dist()
