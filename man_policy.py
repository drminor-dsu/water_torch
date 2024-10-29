#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 06:58:52 2019

@author: drminor
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from water_gym import WaterGym, Actions, pumps
#from perform_evaluate import plot_performance

actions = Actions

def select_pumps(pump_pol, elev):
    """
    현재의 펌프 상태에서 유수지 수위를 보고 펌프 조합 갱신
    반드시 이전 pump_pol이 지시하는 펌프상태 입력이 있어야 함(초기엔 모두 False)

    Parameters
    ----------
    pump_pol : TYPE
        DESCRIPTION.
    elev : TYPE
        DESCRIPTION.

    Returns
    -------
    pump_pol : TYPE
        Newly updated pump combination considering the current water level.
    action : TYPE
        Determine the aciton to preceed gym from pump_pol.

    """
        
    if elev >= 6.2:
        pump_pol[0] = True
    elif elev <= 5.5:
        pump_pol[0] = False
    else:
        pass
    
    if elev >= 6.3:
        pump_pol[1] = True
    elif elev <=5.6:
        pump_pol[1] = False
    else:
        pass
    
    if elev >= 6.4:
        pump_pol[2] = True
    elif elev <= 5.7:
        pump_pol[2] = False
    else:
        pass
    
    if elev >= 6.5:
        pump_pol[3] = True
    elif elev <= 5.8:
        pump_pol[3] = False
    else:
        pass
    
    if elev >= 6.6:
        pump_pol[4] = True
    elif elev <= 5.9:
        pump_pol[4] = False
    else:
        pass
    
    # 현재의 펌프 조합과 일치하는 action으로 변환 
    action = 0
    for i in range(len(actions)):
        if np.sum(np.array(actions[i]) ^ np.array(pump_pol)) == 0:
            action = i
            break
        
    return pump_pol, action


def operation(test_inp, minutes):    
    gym = WaterGym(test_inp, minutes=minutes)
    init_state_, reward, done, info = gym.reset()
    
    action_list = [0]
    state_list = [init_state_]
    reward_list = [reward]
    info_list = [info]
    
    # g_times = []
    # g_inflows = []
    # g_outflows = []
    # g_rainfalls = []
    # g_before_elevations = []
    # g_after_elevations = []
    # g_before_volumes =[]
    # g_after_volumes = []
    # g_changes = []
    # g_actions = []
    
    pump_pol = [False] * 5
    pump_pol, action = select_pumps(pump_pol, init_state_[-1])
    
    while (1):
        state_, reward, done, info = gym.step(action)
        
        # for performance evaluations
        state_list.append(state_)
        reward_list.append(reward)
        info_list.append(info)
        
        pump_pol, action = select_pumps(pump_pol, state_[-1])
        action_list.append(action)
        
        if done == True:
            break
    
    return action_list, state_list, reward_list, info_list     


def plot_performance(ftitle, state_list, actions, 
                     kind=['rain', 'inflow', 'outflow', 'vol', 'level']):
    rain_list = []
    inflow_list = []
    outflow_list = []
    vol_list = []
    level_list = []
    
    
    for state in state_list:
        rain, inflow, outflow, vol, level = state
        rain_list.append(rain)
        inflow_list.append(inflow)
        outflow_list.append(outflow)
        vol_list.append(vol)
        level_list.append(level)
    
    kind_dic = {'rain': rain_list, 'inflow': inflow_list, 
                'outflow': outflow_list, 'vol': vol_list, 'level': level_list}
    color = ['black', 'blue', 'red', 'green', 'yellow']
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    
    fig.suptitle(ftitle, y=0.92, fontsize=20)
    for i in range(len(kind)-1):
        ax[0].plot(range(len(kind_dic[kind[i]])), 
                kind_dic[kind[i]], 
                color=color[i],
                label=list(kind_dic.keys())[i])
    ax[0].legend()
    
    ax[1].plot(range(len(actions)),
                 kind_dic['rain'],
                 color=color[0],
                 label='rain/m')
    ax[1].plot(range(len(actions)),
                 kind_dic['level'],
                 color=color[1],
                 label='level')
    ax[1].scatter(range(len(actions)),
                 actions,
                 color=color[2],
                 label='actions')
    ax[1].legend()
    
    fig.show()


if __name__ == '__main__':
    test_list = ['/home/drminor/projects/water/torch/data/gasan/30year/30yr_0060m_h150.inp']
    action_list, state_list, reward_list, info_list = operation(test_list[0])
    plt.figure()
    fig_title =  test_list[0].split('/')[-1]
    plot_performance(fig_title, state_list, action_list, ['rain', 'inflow', 'outflow', 'vol', 'level'])