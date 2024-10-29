#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:18:59 2024

@author: drminor
"""

import os, sys
import math
import numpy as np
import matplotlib.pyplot as plt

from sim_swmm import SimSwmm


Regions = ('naechon', 'bongdong', 'dooyeo', 'gasan')
Level = (4.7, 5.8, 6.3, 7.0, 8.0, 8.2, 10) # unit: m,  elevation of reservoir 6.3을지켜라!
Volume = (0.0, 1000.0, 2407.0, 4452.0, 8105.0, 10008.0, 16000.0) # unit: m^3, volume of reservoir
pumpq = (100, 100, 100, 170, 170) # unit: cmm(cubic meter per minute), outflow rate of pump
pumps = 5

Actions = [[False, False, False, False, False],
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True]] # selected pumps

# Actions = [[False, False, False, False, False],
#             [True, False, False, False, False],
#             [False, False, False, True, False],
#             [True, True, False, False, False],
#             [True, False, False, True, False],
#             [True, True, True, False, False],
#             [False, False, False, True, True],
#             [True, True, False, True, False],
#             [True, False, False, True, True],
#             [True, True, True, True, False],
#             [True, True, False, True, True],
#             [True, True, True, True, True]] # selected pumps

# max and min data for standardization of state values
maxval = {'rain_f': 5.55, 'c_vol': 32522., 'p_vol': 32249., 'ele': 10., 'flow_in': 1750.}
minval = {'rain_f': 0.0, 'c_vol': 0.0, 'p_vol': 0.0, 'ele': 4.7, 'flow_in':0.0}


class WaterGym:
    """
        
    """
    def __init__(self, inp, minutes:int=1, region:str='gasan'):
        self.inp = inp
        self.minutes = minutes # 의사 결정 분단위 기간 
        self.region = region
        
        # simswmm = SimSwmm(self.inp, self.region)
        # self.rains, self.outfalls = simswmm.execute()
        # self.clock = 0
        # self.reservior_vol = 0.0
        # self.reservior_level = 0.0
        
        
    def reset(self):
        
        # Simulate SWMM using the given inp file
        simswmm = SimSwmm(self.inp, self.minutes, self.region)
        self.rains, self.outfalls, self.length = simswmm.execute()
        
        self.clock = 0 # as an indicator which preceeds the simulation
        self.inflow = [0.0]
        self.outflow = [0.0]
        self.reservior_vol = [0.0]
        self.reservior_level = [4.7]
        self.acts = [0]
        self.rewards = [0.0]
        
        # print(self.rains[self.clock], self.outfalls[self.clock], \
        #       self.outflow[self.clock], self.reservior_vol[self.clock], \
        #           self.reservior_level[self.clock], self.rewards[0], False, (0.0, 0.0, 0.0))

        return [self.rains[self.clock], 
                self.outfalls[self.clock], 
                self.outflow[self.clock],
                self.reservior_vol[self.clock],
                self.reservior_level[self.clock]],\
                self.rewards[0], False, (0.0, 0.0, 0.0)
    
    
    def step(self, action:int):
        """
        선택한 액션에 대해 현재의 rain, inflow, outflow, reservoir_vol ... 등 반환
3) Àü·Â ¼Ò¸ð ÃÖ¼ÒÈ­
        Parameters
        ----------
        action : int
            Selected action by the agent

        Returns
        -------
        rains : float
            precipitaion per minute
        inflow: float
            reservior inflow per minute
        outflow: float
            reservior outflow per minute
        reservior_vol : TYPE
            current reservior volume
        reservior_level : float
            current reservior level
        cur_reward : float
            reward for the selected action
        done : bool
            whether simulation is over or not.
        info : tuple
            compostion of reward (vol_reward, act_reward, energy_reward)

        """
        done = False
        self.clock += 1
        if self.clock == self.length - 1:
            done = True
        if self.clock >= self.length: # the simulation is finished
            return (-1., -1., -1., -1., -1.), -1., True, None#, -1
        
        # renewing states and reward using selected action
        self.acts.append(action)
        
        # Compute inflow into the reservoir
        cur_inflow = self.outfalls[self.clock]
        self.inflow.append(cur_inflow)
        
        # Compute outflow from the reservoir, 
        # current volume and level of the reservoir
        cur_outflow = np.array(pumpq)[Actions[action]].sum() # the attempt to disperse the water in the reservoir as much as possible
        cur_vol = self.reservior_vol[self.clock-1] + cur_inflow - cur_outflow
        excess_pump = False # Penalty for excess pumping 
        if cur_vol < 0.0:
            cur_vol = 0.0
            excess_pump = True
            cur_outflow = self.reservior_vol[self.clock-1] + cur_inflow
        self.outflow.append(cur_outflow)
        self.reservior_vol.append(cur_vol)
        cur_level = self.vol2level(self.reservior_vol[self.clock])
        self.reservior_level.append(cur_level)
        
        # Compute reword for a selected action
        # abs_diff, cur_reward, info = self.reward(self.clock, excess_pump)
        cur_reward, info = self.reward(self.clock, excess_pump)
        self.rewards.append(cur_reward)
        
        # print(self.rains[self.clock], cur_inflow, cur_outflow, 
        #       cur_vol, cur_level, 
        #       cur_reward, done, info)

        return [self.rains[self.clock], 
                cur_inflow, 
                cur_outflow, 
                cur_vol, 
                cur_level],\
                cur_reward, done, info#, abs_diff
    
    
    def vol2level(self, vol:float)->float:
        
        wlevel = 0.0
        #print(vol)
        
        if vol < 0:
            #print("Warning: Current Volume of the Reservoir minus")
            return Level[0]
        for index in range(len(Volume)-1):
            if vol >= Volume[index+1]:
                continue
            else:
                break
        else:
            index += 1
                
        if index < 6:
            wlevel = Level[index] + (Level[index+1]-Level[index]) / (Volume[index+1]-Volume[index]) \
                * (vol-Volume[index])
        else:
            wlevel = Level[index]
        
        return wlevel
    
    
    def reward(self, clock:int, excess_pump:bool)->float: 
        """
        vol_reward : Reward for pumping quantity of action, as the volume of pumping increases the reward increases
        act_reward : Reward for the consistency of actions
        energy_reward : Reward for reducing energy consumption 
        penalty
        
        !! 보상의 가중치를 학습 파라미터로 내재화 하는 방법?
        
        Parameters
        ----------
        clock : int
            DESCRIPTION.
        over_pump : bool
            If over_pump is True, it represents the action (combination of pumps) tries more pumping than reservior volume + inflow.
            It occurs penalty.

        Returns
        -------
        t_reward : TYPE
            DESCRIPTION.
        (vol_reward, act_reward, energy_reward) :
            for reward analysis

        """
        w1 = 1.0
        w2 = 0.0
        w3 = 0.0
        w4 = 0.0
        
        # vol_reward:
        vol_reward = (self.outflow[clock]/np.array(pumpq).sum()) * (self.reservior_level[clock]/Level[-1])
        
        # act_reward:
        # 이전 펌핑 방식과 동일하면 1 아니면 0의 보상
        first = Actions[self.acts[clock]]
        second = Actions[self.acts[clock-1]]
        changes = (np.array(first) & np.array(second)).sum() # 변경되지 않은 펌프의 개수
        act_reward = changes / len(first)
        
        # energy_reward:
        # 최대 펌핑시의 에너지에 대한 현재 펌핑의 에너지 소모량의 비를 이용
        # 소모량이 작을 수록 보상은 1에 가까워지고 최대 소비량에 가까워지면 0에 근접
        energy_reward = 1.0 - np.array(pumpq)[Actions[self.acts[clock]]].sum() / np.array(pumpq).sum()
        excess_pump_penalty = -10.0
        
        t_reward = w1 * vol_reward\
                + w2 * act_reward\
                + w3 * energy_reward\
                + w4 * excess_pump * excess_pump_penalty    
        
        return t_reward, (w1*vol_reward, w2*act_reward, w3*energy_reward, w4*excess_pump * excess_pump_penalty)
        # return abs_diff, t_reward, (w1*vol_reward, w2*act_reward, w3*energy_reward, w4*excess_pump * excess_pump_penalty)
    
    
    
if __name__ == '__main__':
    sim = WaterGym('/home/drminor/projects/water/torch/data/gasan/100year/100yr_1440m_h1239.inp',
                   minutes=5)
    state = sim.reset()
    rewards = []
    actions = []
    diffs = []
    for i in range(len(sim.rains)):
        act = np.random.randint(0, 6)
        actions.append(act)
        # you need including diff related codes  
        state, reward, done, info = sim.step(act)
        rewards.append(info)
        #diffs.append(diff)
    
    fig, ax = plt.subplots(1, 1)
    x = np.arange(0, 5000)
    y1 = 2*np.log(1.0/x)
    y2 = 2*np.log10(1.0/x)
    y3 = 2*np.log2(1.0/x)
    
    ax.plot(x, y1, label='log e')
    ax.plot(x, y2, label='log 10')
    ax.plot(x, y3, label='log 2')
    ax.legend()
    
    fig.show()    