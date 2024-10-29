#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:52:45 2024

@author: drminor
"""

import os, sys
import glob
import numpy as np
from pyswmm import Simulation
from pyswmm.nodes import Nodes

import matplotlib.pyplot as plt


class SimSwmm:
    def __init__(self, inp, minutes:int=1, region:str='gasan'):
        
        # Road inp file and corresponding rain file 
        # (inp sample:) /home/drminor/projects/water/torch/data/gasan/10year/10yr_0010m_h051.inp
        self.minutes = minutes # 의사결정 간격 분 단위 
        self.inp_file = inp
        inter_name =  '/'.join(inp.split('.')[0].split('/')[-2:])
        self.rain_file = '/home/drminor/projects/water/torch/data/rainfall/' + inter_name + '.txt'
        #print(self.inp_file+'\n', self.rain_file, end='\n', flush=True)
    
    
    def cum_rains_return(self)->list:
        """
        단순히 누적 강우량만 리턴

        Returns
        -------
        None.

        """
        with open(self.rain_file, 'r') as fd:
            cum_rains = fd.read().strip().split()
            cum_rains = np.array(cum_rains, dtype=np.float32) # Cumulative rainfalls
       
        return list(cum_rains)
        
        
    def rainfall_read(self)->list:
        """
        아래 두 문제를 고려해서 Read rainfall file and return rainfalls per miniute
        1.원 rainfall file에는 누적 강우량으로 되어 있음.
        2. 강우생성 수식에 오류가 있음. 음의 강우량 발생  

        Returns
        -------
        list
            DESCRIPTION.

        """

        with open(self.rain_file, 'r') as fd:
            cum_rains = fd.read().strip().split()
            cum_rains = np.array(cum_rains, dtype=np.float32) # Cumulative rainfalls

            # rainfall per self.minutes 
            first = cum_rains[0]
            # print(cum_rains.shape[0])
            # print(cum_rains.shape[0]//self.minutes)
            incremental = np.array([cum_rains[(i+1)*self.minutes]-cum_rains[i*self.minutes]
                                    for i in range(cum_rains.shape[0]//self.minutes-1)])
        # self.cum_rains = cum_rains
        rainfalls_per_min = np.array([first] + list(incremental))
        
        assert all(rainfalls_per_min >= 0.0), f'{self.rain_file}: There are minus rains.'
        
        return list(rainfalls_per_min)
    
    
    def swmm_execute(self)->list:
        """
        Execute SWMM simulation and return outfalls per minute using inp file
        1 분당 유수지로의 유입량 계산
        
        Returns
        -------
        outfalls : TYPE
            DESCRIPTION.

        """    
        time_unit = self.minutes * 60 # seconds
        with Simulation(self.inp_file) as sim:
            nodes = [] # outfall nodes
            for node in Nodes(sim):
                if node.is_outfall():
                    nodes.append(node)
            #print(nodes[0].nodeid)
            sim.step_advance(time_unit)
            outfalls = []
            times = []
            total = 0.0
            for ind, step in enumerate(sim):
                #print(step.getCurrentSimulationTime())
                times.append(step.getCurrentSimulationTime())
                
                summation = 0.
                for nd in nodes:
                    summation += nd.total_inflow*time_unit
                outfalls.append(summation)
                total = total + summation
                
                #print('node0: {:.3f}'.format(nodes[0].total_inflow*60))
                #print('node1: {:.3f}'.format(nodes[1].total_inflow*60))
                #print('sum: {:.3f}'.format(summation), end='\n\n')
                #print('node1: {:.3f}'.format(nodes[1].total_inflow))
                #print(nodes[1].total_outflow)
            #print('\n\ntotal: '+str(total))
            sim.report()
        
        return outfalls
            
    
    def execute(self)->(list, list):
        """
        return cumulative rains, rains per minute and outfalls lists with the same lenght

        Returns
        -------
        rains : TYPE
            DESCRIPTION.
        outfalls : TYPE
            DESCRIPTION.
        len(rains) : int
            the total number of steps of simulation

        """
        rains = self.rainfall_read()
        outfalls =self.swmm_execute()
        
        diff = len(outfalls) - len(rains)
        if diff > 0:
            zeros = [0. for _ in range(diff)]
            rains += zeros
        assert len(rains) == len(outfalls), "There is difference between len(rains) and len(outfalls)"
        
        return rains, outfalls, len(rains)


if __name__ == '__main__':
    sim = SimSwmm('/home/drminor/projects/water/torch/data/gasan/30year/30yr_0060m_h143.inp'
                  , minutes=1)
    cum_rains = sim.cum_rains_return()
    rains, outfalls, _ = sim.execute()
    
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['axes.grid'] = True
    fig, ax = plt.subplots(1,2, sharex=True)
    fig.set_size_inches(11, 13)
    
    label_size = 25
    tick_size = 20
    legend_size = 25
    
    
    ax[1].set_ylabel('Volume($m^3$)', fontsize=label_size, fontweight='bold')
    ax[1].plot(outfalls, color='r', marker='D', label= 'Outfall into the reservoir')
    ax[1].tick_params(labelsize=tick_size)
    ax[1].legend(loc='best', fontsize=legend_size)
    
    ax[0].set_xlabel('Time($m$)', fontsize=label_size, fontweight='bold')
    ax[0].set_ylabel('Rainfall(mm)', fontsize=label_size, fontweight='bold')
    ax[0].plot(cum_rains, color='b', marker='o', label= 'Cumulative rainfall')
    ax[0].legend(loc='best', fontsize=legend_size)
    
    plt.tight_layout()
    plt.show()