#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 02:20:12 2019

@author: drminor
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
import torch

regions = ('naechon', 'bongdong', 'dooyeo', 'gasan')


def pump_info(region:str):
    '''
    주어진 지역에 대한 펌프 및 펌프 운용용 정보

    Parameters
    ----------
    region : str
        DESCRIPTION.

    Returns
    -------
    pumpq : TYPE
        DESCRIPTION.
    pumps : TYPE
        DESCRIPTION.
    actions : TYPE
        DESCRIPTION.

    '''
    global regions
    
    if region == regions[0]:
        pumpq = (80, 200, 200)
        pumps = 3
        actions = [[False, False, False], [True, False, False], 
                   [False, True, False], [True, True, False],  
                   [False, True, True], [True, True, True]]
        # state: [rainfall, inflow, prev_vol, current_vol, current_water_level,
        #          pprev_action, prev_action]
        #minv = [0.0, 0.0, 0.0, 0.0, 3.999999407833506, 0, 0]
        #maxv = [5.170001, 1007.5366199953078, 126784.02756489342, 127391.45984611512,
        # 50.0494643130072, 5, 5]
        minval = {'rain_f': 0.0, 'c_vol': 0.0, 'p_vol': 0.0, 'ele': 4., 'flow_in':0.0}
        maxval = {'rain_f': 6.0, 'c_vol': 130000., 'p_vol': 130000., 'ele': 7.61, 'flow_in': 1100.}
    elif region == regions[1]:
        pumpq = (70, 70, 70)
        pumps = 3
        actions = [[False, False, False], [True, False, False],  
                   [True, True, False], [True, True, True]]
        #minv [0.0, 0.0, 0.0, 0.0, 2.9521518627299304, 0, 0]
        #maxv [5.170001, 541.4456896012902, 96695.53479340575, 97096.80381995169, 35.856316741235155, 3, 3]
        minval = {'rain_f': 0.0, 'c_vol': 0.0, 'p_vol': 0.0, 'ele': 2.95, 'flow_in':0.0}
        maxval = {'rain_f': 6.0, 'c_vol': 100000., 'p_vol': 100000., 'ele': 8., 'flow_in': 600.}
    elif region == regions[2]:
        pumpq = (90, 90)
        pumps = 2
        actions = [[False, False], [True, False], [True, True]]
        #minv [0.0, 0.0, 0.0, 0.0, 6.417812277561131, 0, 0]
        #maxv [5.170001, 75.38075029037512, 2253.414924831332, 2328.795675121707, 10.236152242404184, 2, 2]
        minval = {'rain_f': 0.0, 'c_vol': 0.0, 'p_vol': 0.0, 'ele': 6.4, 'flow_in':0.0}
        maxval = {'rain_f': 6.0, 'c_vol': 2500., 'p_vol': 2500., 'ele': 10.7, 'flow_in': 85.}
    elif region == regions[3]:
        #Elevation = (4.7, 5.8, 6.3, 7.0, 8.0, 8.2, 10) # unit: m,  elevation of reservoir 6.3을지켜라!
        #Volume = (0.0, 1000.0, 2407.0, 4452.0, 8105.0, 10008.0, 16000.0) # unit: m^3, volume of reservoir
        pumpq = (100, 100, 100, 170, 170) # unit: cmm(cubic meter per minute), outflow rate of pump
        pumps = 5

        actions = [[False, False, False, False, False],
                    [True, False, False, False, False],
                    [True, True, False, False, False],
                    [True, True, True, False, False],
                    [True, True, True, True, False],
                    [True, True, True, True, True]] # selected pumps

        # max and min data for standardization of state values
        maxval = {'rain_f': 5.55, 'c_vol': 32522., 'p_vol': 32249., 'ele': 10., 'flow_in': 1750.}
        minval = {'rain_f': 0.0, 'c_vol': 0.0, 'p_vol': 0.0, 'ele': 4.7, 'flow_in':0.0}

    else:
        print(f'Please confirm region name {region}')
        sys.exit()
    
    return pumpq, pumps, actions, minval, maxval


def volume_to_waterlevel(vol:float, region:str) -> float:
    global regions
    
    elev = 0.0
    if region == regions[0]:
        elev = 0.00036148 * vol + 3.999999407833506
    elif region == regions[1]:
        elev = 0.00033888 * vol + 2.9521518627299304
    elif region == regions[2]:
        elev = 0.00163962 * vol + 6.417812277561131
    else:
        print(f'Please confirm region name {region}')
        sys.exit()
    
    return elev


def data_load(region:str) -> './data/naechon/10year/10yr_0720m_h30.inp, \
                              ./data/rainfall/10year/10yr_0720m_h30.txt':
    '''
    주어진 데이터 파일 시스템으로부터 지정된 지역의 inp 파일 리스트와 rainfall 파일 리스트 획득

    Parameters
    ----------
    region : str
        'naechon', 'bongdong', 'dooyeo'

    Returns
    -------
    inp_file : list
    rain_file : list

    '''
    
    inp_path = f'./data/{region}/'
    rain_path = f'./data/rainfall/'
    sub_dirs = os.listdir(rain_path)
    sub_dirs = ['10year'] # temporary for test
    print(sub_dirs)
    
    inp_files = []
    for dirs in sub_dirs:
        files = os.listdir(inp_path+dirs)
        files = [dirs+'/'+ file for file in files if not file.strip().split('.')[1] in ['out', 'rpt']]
        inp_files.extend(files)
    
    rain_files = []
    for dirs in sub_dirs:
        files = os.listdir(rain_path+dirs)
        files = [dirs+'/'+ file for file in files]
        rain_files.extend(files)
    
    inp_files.sort()
    rain_files.sort()
    
    # data shuffling
    indices = np.arange(len(inp_files))
    np.random.shuffle(indices)
    inp_file = [inp_path + inp_files[i] for i in indices]
    rain_file = [rain_path + rain_files[i] for i in indices]
    
    return inp_file, rain_file


def rainfall_read(fpath):

    with open(fpath, 'r') as fd:
        cum_rains = fd.read().strip().split()
        cum_rains = np.array(cum_rains, dtype=np.float32) # Cumulative rainfalls

        # rainfall per 1 minute
        first = cum_rains[0]
        incremental = np.array([cum_rains[i+1]-cum_rains[i] \
                                for i in np.arange(cum_rains.shape[0]-1)])
        rainfalls_per_min = np.array([first] + list(incremental))

    return rainfalls_per_min, cum_rains


def compute_inflow(nodes):
    inflow = 0.0
    for node in nodes:
        inflow += node.total_inflow * 60
    
    return inflow


def rainfall(rain, index):
    if index < len(rain):
        rainfall = rain[index] if rain[index] >= 0.0 else 0.0 #!! i don't know why rainfall has minus values
    else:
        rainfall = 0.0 # when rain stops
    
    return rainfall

#def _compute_reward(prev_volume, current_volume, pprev_action, prev_action):
#    w1 = 0.9
#    w2 = 0.0
#    
#    level_change = volume_to_waterlevel(prev_volume) - volume_to_waterlevel(current_volume)
#    action_change = abs(pprev_action - prev_action)
#    reward = w1 * level_change/(10. - 4.7) - w2 * action_change / 5.
#    
#    #outflow = np.array(pumpQ)[actions[prev_action]].sum()
#    #if outflow <= prev_volume:
#    #    reward += 0.5 
#    
#    return reward * 5

def _compute_reward_no_prev(prev_volume, current_volume, prev_action, region, pumpq, actions):
    w1 = 0.8
    
    level_change = volume_to_waterlevel(prev_volume) - volume_to_waterlevel(current_volume)
    reward = w1 * level_change/(10. - 4.7)
    
    outflow = np.array(pumpq)[actions[prev_action]].sum()
    if outflow <= prev_volume:
        reward += 0.2 
    
    return reward * 5


def _compute_reward(prev_volume, current_volume, 
                    pprev_action, prev_action, 
                    region:str, pumpq, actions):
    '''
    Reward function: computing reward

    Parameters
    ----------
    prev_volume : TYPE
        DESCRIPTION.
    current_volume : TYPE
        DESCRIPTION.
    pprev_action : TYPE
        DESCRIPTION.
    prev_action : TYPE
        DESCRIPTION.
    region : str
        DESCRIPTION.
    pumpq : TYPE
        DESCRIPTION.
    actions : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    w1 = 0.8
    w2 = 0.0
    
    max_level = {'naechon': 7.61, 'bonddong': 8., 'dooyeo': 10.7}
    min_level = {'naechon': 4., 'bonddong': 2.95, 'dooyeo': 6.4}
    level_change = volume_to_waterlevel(prev_volume, region) - volume_to_waterlevel(current_volume, region)
    action_change = abs(pprev_action - prev_action)
    reward = w1 * level_change/(max_level[region] - min_level[region]) - w2 * action_change / 5.
    
    outflow = np.array(pumpq)[actions[prev_action]].sum()
    if outflow <= prev_volume:
        reward += 0.2 
    
    return reward * 5


def env_reset_no_prev():
    '''
    Environment reset: set 5-dimension input value to zero 

    Returns
    -------
    initial_states : TYPE
        DESCRIPTION.

    '''
    initial_states = np.array([0.0]*5)
    initial_states = np.expand_dims(initial_states, 0)
    
    return initial_states

def env_reset(nactions) -> 'np.array':
    '''
    전, 전전에 선택한 action을 input에 포함한 vertion

    Parameters
    ----------
    nactions : TYPE, optional
        The number of possible actions.

    Returns
    -------
    initial_states : np.array
        DESCRIPTION.

    '''
    enc_pp = np.eye(nactions)[0]
    enc_p = np.eye(nactions)[0]
    remain = np.array([0.0]*5)
    initial_states = np.concatenate((remain, enc_pp, enc_p))
    initial_states = np.expand_dims(initial_states, 0)
    
    return initial_states


def state_transform_no_prev(rainfall, inflow, prev_volume, current_volume, 
                    waterlevel, actions, minval, maxval):
    '''
    Input data normalization

    Parameters
    ----------
    rainfall : TYPE
        DESCRIPTION.
    inflow : TYPE
        DESCRIPTION.
    prev_volume : TYPE
        DESCRIPTION.
    current_volume : TYPE
        DESCRIPTION.
    waterlevel : TYPE
        DESCRIPTION.
    actions : TYPE
        DESCRIPTION.

    Returns
    -------
    input_states : TYPE
        DESCRIPTION.

    '''
    
    std_rain_m = (rainfall - minval['rain_f'])/(maxval['rain_f']-minval['rain_f'])
    std_prev_vol = (prev_volume - minval['p_vol'])/(maxval['p_vol']-minval['p_vol'])
    std_current_vol = (current_volume - minval['c_vol'])/(maxval['c_vol']-minval['c_vol'])
    std_level = (waterlevel - minval['ele'])/(maxval['ele']-minval['ele'])
    std_inflow = (inflow - minval['flow_in'])/(maxval['flow_in']-minval['flow_in'])

    input_states = np.array([std_rain_m, std_inflow, std_prev_vol, std_current_vol, std_level])
    input_states = np.expand_dims(input_states, 0)
                    
    return input_states


def state_transform(rainfall, inflow, prev_volume, current_volume, waterlevel,\
                      pprev_action, prev_action, actions, minval, maxval):
    
    std_rain_m = (rainfall - minval['rain_f'])/(maxval['rain_f']-minval['rain_f'])
    std_prev_vol = (prev_volume - minval['p_vol'])/(maxval['p_vol']-minval['p_vol'])
    std_current_vol = (current_volume - minval['c_vol'])/(maxval['c_vol']-minval['c_vol'])
    std_level = (waterlevel - minval['ele'])/(maxval['ele']-minval['ele'])
    std_inflow = (inflow - minval['flow_in'])/(maxval['flow_in']-minval['flow_in'])
    
    enc_pp = np.eye(actions)[pprev_action]
    enc_p = np.eye(actions)[prev_action]
    remain = np.array([std_rain_m, std_inflow, std_prev_vol, std_current_vol, std_level])
    input_states = np.concatenate((remain, enc_pp, enc_p))
    input_states = np.expand_dims(input_states, 0)
                    
    return input_states


def env_step_no_prev(action, rainfall, inflow, current_vol, region, pumpq, actions):   
    # compute state using selected action and current state variables
    prev_action = action
    
    prev_vol = current_vol  
    outflow = np.array(pumpq)[actions[action]].sum()
    current_vol = current_vol + inflow - outflow
    current_vol = current_vol if current_vol >= 0.0 else 0.0
    
    #prev_water_level = volume_to_waterlevel(prev_vol)
    current_water_level = volume_to_waterlevel(current_vol, region)
    
    state = (rainfall, inflow, prev_vol, current_vol, current_water_level)
    # compute rewards
    reward = _compute_reward(prev_vol, current_vol, prev_action, region, pumpq, actions)
    
    return state, reward, outflow


def env_step(action, prev_action, rainfall, inflow, current_vol, region, pumpq, actions):   
    # compute state using selected action and current state variables
    pprev_action = prev_action
    prev_action = action
    
    prev_vol = current_vol  
    outflow = np.array(pumpq)[actions[action]].sum()
    current_vol = current_vol + inflow - outflow
    current_vol = current_vol if current_vol >= 0.0 else 0.0
    
    #prev_water_level = volume_to_waterlevel(prev_vol)
    current_water_level = volume_to_waterlevel(current_vol, region)
    
    state = (rainfall, inflow, prev_vol, current_vol, current_water_level,\
              pprev_action, prev_action)
    # compute rewards
    reward = _compute_reward(prev_vol, current_vol, pprev_action, prev_action, region, pumpq, actions)
    
    return state, reward, outflow


def print_state_reward_no_prev(state, reward):
    outputs = np.concatenate((state, [[reward]]), axis=1)
    df = pd.DataFrame(outputs, columns=['rainfall', 'inflow', 'pvol', 'cvol', 'level', 'reward'])
    #print(df)

def print_state_reward(state, reward):
    outputs = np.concatenate((state, [[reward]]), axis=1)
    df = pd.DataFrame(outputs, columns=['rainfall', 'inflow', 'pvol', 'cvol', 'level', \
                    'ppact_0', 'ppact_1', 'ppact_2', 'ppact_3', 'ppact_4', 'ppact_5',\
                    'pact_0', 'pact_1', 'pact_2', 'pact_3', 'pact_4', 'pact_5', 'reward'])
    #print(df)
    
def file_remove(inp_file):
    file_list = ['.rpt', '.out']
    head, tail = os.path.split(inp_file)
    tail = tail.split('.')[0]
    for ext in file_list:
        file = os.path.join(head, tail + ext)
        if os.path.isfile(file):   
            os.remove(file)
            #print(file + ' was removed')

            
def graph_drawing(df, cum_rainfall, filename):
    df['cum_rain'] = df['rainfall'].cumsum()
    cols = df.columns.tolist()
    cols.insert(2, cols.pop())
    df = df[cols]
    df['change'] = df['change'].cumsum()
    size = df.shape
    #print(df)
    
    #['time',  'rainfall', 'cum_rain', 'inflow', 'outflow', 'volume', 'level', 'change', 'action']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(27, 20)
    ticks = [(size[0]//5) * x for x in range(5)]
    ticks.append(size[0] - 1)
    
    label_size = 20
    tick_size = 17
    legend_size = 17
    
    ax[0,0].set_xlabel('Time($m$)', fontsize=label_size, fontweight='bold')
    ax[0,0].set_ylabel('Volume($m^3$)', fontsize=label_size, fontweight='bold')
    ax[0,0].tick_params(labelsize=tick_size)
    ax[0,0].plot(df['inflow'], color='b', marker='o', label= 'Inflow to reservoir')
    ax[0,0].plot(df['outflow'], color='r', marker='D', label= 'Outflow from reservoir')
    ax[0,0].plot(df['rainfall'], color='y', linewidth=5, label='Rainfall')
    ax[0,0].plot(df['cum_rain'], color='g', linewidth=5, label= 'Cumulative rainfalls')
    ax[0,0].legend(loc='best', fontsize=legend_size)

    ax[0,1].set_xlabel('Time($m$)', fontsize=label_size, fontweight='bold')
    ax[0,1].set_ylabel('Water level($m$)', fontsize=label_size, fontweight='bold')
    ax[0,1].tick_params(labelsize=tick_size)
    ax[0,1].plot(df['level'], color='b', marker='o', label='Water level of reservoir')
    ax[0,1].legend(loc='best', fontsize=legend_size)

    ax[1,0].set_xlabel('Time($m$)', fontsize=label_size, fontweight='bold')
    ax[1,0].set_ylabel('Action($number$)/Changes($frequecy$)', fontsize=label_size, fontweight='bold')
    ax[1,0].tick_params(labelsize=tick_size)
    ax[1,0].plot(df['action'], marker='o', color='b', linestyle='None', label='Selected pumping options')
    ax[1,0].plot(df['change'], marker='D', color='g', label='Cumulative changes of pumps')
    #ax[2].text(10,5, str(cum_changes[-1]), color='r', fontsize=20, fontweight='bold')
    ax[1,0].legend(loc='best', fontsize=legend_size)
    
    ax[1,1].set_xlabel('Time($m$)', fontsize=label_size, fontweight='bold')
    ax[1,1].set_ylabel(r'Volume($m^3$)', fontsize=label_size, fontweight='bold')
    ax[1,1].tick_params(labelsize=tick_size)
    ax[1,1].plot(df['volume'], color='b', marker='o', label='Volume of reservoir')
    ax[1,1].plot(df['outflow'], color='g', marker='D', label= 'Outflow')
    ax[1,1].legend(loc='best', fontsize=legend_size)
    
    #fig.show()
    fig.savefig(filename, bbox_inches='tight', dpi=150)

if __name__ == '__main__':
    # for reg in regions:
    #     pumpQ, n_pump = pump_info(reg)
    #     print(f'{pumpQ}, {n_pump}')
    
    inp, rain = data_load('naechon')
